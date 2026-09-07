"""Causal support-set output-trust update for the COQMF local model."""

import numpy as np


NUMERICAL_RIDGE = 1e-12
MAX_TRUST_BACKTRACKS = 60


def _sigmoid(values):
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def output_trust_budget(predictions, targets):
    """Return the parameter-free support-only RMS output-change budget."""
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if predictions.shape != targets.shape or predictions.ndim != 1:
        raise ValueError("predictions and targets must be equal-length vectors")
    if len(targets) == 0:
        return 0.0
    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(targets)):
        raise ValueError("support predictions and targets must be finite")
    return float(np.median(np.abs(predictions - targets)) *
                 np.sqrt(len(targets) / (len(targets) + 1.0)))


def support_prediction_rms_change(before_user_vector, before_user_bias,
                                  before_service_vectors, before_service_biases,
                                  after_user_vector, after_user_bias,
                                  after_service_vectors, after_service_biases):
    """Measure actual, rather than parameter-space, support prediction change."""
    before = _sigmoid(np.asarray(before_service_vectors) @ np.asarray(before_user_vector) +
                      float(before_user_bias) + np.asarray(before_service_biases))
    after = _sigmoid(np.asarray(after_service_vectors) @ np.asarray(after_user_vector) +
                     float(after_user_bias) + np.asarray(after_service_biases))
    if before.shape != after.shape:
        raise ValueError("before and after support predictions must have equal shape")
    return float(np.sqrt(np.mean((after - before) ** 2))) if len(before) else 0.0


def _solve_block(curvature, gradient, eta):
    identity = np.eye(len(gradient), dtype=float)
    system = curvature + NUMERICAL_RIDGE * identity
    try:
        solved = np.linalg.solve(system, gradient)
    except np.linalg.LinAlgError:
        # At l2=fedprox_mu=0 a rank-one block can remain numerically singular
        # when its scale is so large that the fixed ridge is below machine
        # resolution.  Preserve the original solve for every nonsingular
        # block and use the deterministic minimum-norm solution only for the
        # exceptional singular block.
        solved = np.linalg.lstsq(
            np.asarray(system, dtype=np.float64),
            np.asarray(gradient, dtype=np.float64),
            rcond=None,
        )[0]
        if not np.all(np.isfinite(solved)):
            raise FloatingPointError("nonfinite singular-block fallback solution")
    return -float(eta) * solved


def _solve_service_blocks(service_weights, service_jacobian,
                          service_gradient, eta, l2, fedprox_mu):
    """Solve the independent service blocks with a rank-one closed form.

    Each service block has the form ``gamma I + w_i j_i j_i.T`` after the
    same numerical ridge used by :func:`_solve_block` is included, where
    ``gamma = l2 + fedprox_mu + NUMERICAL_RIDGE``.  Sherman--Morrison gives

    ``(gamma I + w j j.T)^-1 g =
       (g - w*j*(j.T@g)/(gamma + w*j.T@j)) / gamma``.

    The computation is batched over support rows and therefore avoids one
    small dense factorization per service.  The inputs are intentionally not
    mutated; validation remains owned by ``support_output_trust_update``.
    """
    service_weights = np.asarray(service_weights, dtype=float)
    service_jacobian = np.asarray(service_jacobian, dtype=float)
    service_gradient = np.asarray(service_gradient, dtype=float)
    base_regularization = float(l2) + float(fedprox_mu)
    if base_regularization == 0.0:
        # With only the 1e-12 numerical ridge, the rank-one closed form can
        # lose several digits through cancellation.  Preserve the exact
        # pre-change dense path for this allowed but non-frozen edge case.
        dimension = service_jacobian.shape[1]
        service_delta = np.empty_like(service_gradient)
        for index in range(len(service_weights)):
            curvature = (service_weights[index] *
                         np.outer(service_jacobian[index],
                                  service_jacobian[index]) +
                         base_regularization * np.eye(dimension))
            service_delta[index] = _solve_block(
                curvature, service_gradient[index], eta)
        return service_delta
    gamma = base_regularization + NUMERICAL_RIDGE
    jacobian_norm_sq = np.einsum("ni,ni->n", service_jacobian,
                                 service_jacobian)
    jacobian_gradient = np.einsum("ni,ni->n", service_jacobian,
                                  service_gradient)
    correction = (service_weights * jacobian_gradient /
                  (gamma + service_weights * jacobian_norm_sq))
    solved = (service_gradient - correction[:, None] * service_jacobian) / gamma
    return -float(eta) * solved


def support_output_trust_update(targets, user_vector, user_bias, user_error,
                                service_vectors, service_biases, service_errors,
                                eta, l2, beta, local_epochs=1, fedprox_mu=0.0,
                                reference_vectors=None, reference_biases=None):
    """Apply order-invariant support-set block GN steps under an output trust region.

    All weights, gradients, curvature terms, and error-state innovations in an
    epoch are computed from the same pre-update support snapshot.
    """
    targets = np.asarray(targets, dtype=float)
    user_vector = np.asarray(user_vector, dtype=float).copy()
    service_vectors = np.asarray(service_vectors, dtype=float).copy()
    service_biases = np.asarray(service_biases, dtype=float).copy()
    service_errors = np.asarray(service_errors, dtype=float).copy()
    user_bias, user_error = float(user_bias), float(user_error)
    reference_vectors = (service_vectors.copy() if reference_vectors is None else
                         np.asarray(reference_vectors, dtype=float))
    reference_biases = (service_biases.copy() if reference_biases is None else
                        np.asarray(reference_biases, dtype=float))

    if targets.ndim != 1 or service_vectors.ndim != 2:
        raise ValueError("targets must be a vector and service_vectors a matrix")
    if service_vectors.shape != reference_vectors.shape:
        raise ValueError("reference_vectors must match service_vectors")
    if not (len(targets) == len(service_vectors) == len(service_biases) ==
            len(service_errors) == len(reference_biases)):
        raise ValueError("all support-aligned arrays must have equal length")
    if service_vectors.shape[1:] != user_vector.shape:
        raise ValueError("user and service latent dimensions must match")
    values = (targets, user_vector, service_vectors, service_biases, service_errors,
              reference_vectors, reference_biases)
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("CSOTU inputs must be finite")
    if np.any(targets <= 0) or np.any(targets >= 1):
        raise ValueError("CSOTU targets must lie strictly inside (0, 1)")
    if user_error <= 0 or np.any(service_errors <= 0):
        raise ValueError("AMF reliability errors must be positive")
    if eta < 0 or l2 < 0 or not 0 <= beta <= 1 or local_epochs < 1 or fedprox_mu < 0:
        raise ValueError("invalid CSOTU optimization constants")
    if len(targets) == 0:
        return (user_vector, user_bias, user_error, service_vectors,
                service_biases, service_errors)

    dimension = len(user_vector)
    for _ in range(local_epochs):
        old_user_vector, old_user_bias = user_vector.copy(), user_bias
        old_vectors, old_biases = service_vectors.copy(), service_biases.copy()
        old_user_error, old_errors = user_error, service_errors.copy()
        predictions = _sigmoid(old_vectors @ old_user_vector + old_user_bias + old_biases)
        residuals = predictions - targets
        probability_slopes = predictions * (1.0 - predictions)
        relative_errors = np.abs(residuals) / targets
        denominators = old_user_error + old_errors
        user_weights = old_user_error / denominators
        service_weights = 1.0 - user_weights

        user_features = np.column_stack((old_vectors, np.ones(len(targets))))
        user_jacobian = probability_slopes[:, None] * user_features
        user_gradient = np.mean(
            user_weights[:, None] * residuals[:, None] * user_jacobian, axis=0)
        user_gradient += l2 * np.concatenate((old_user_vector, [old_user_bias]))
        user_curvature = np.einsum(
            "n,ni,nj->ij", user_weights, user_jacobian, user_jacobian
        ) / len(targets) + l2 * np.eye(dimension + 1)
        user_delta = _solve_block(user_curvature, user_gradient, eta)

        service_features = np.column_stack(
            (np.repeat(old_user_vector[None, :], len(targets), axis=0),
             np.ones(len(targets))))
        service_jacobian = probability_slopes[:, None] * service_features
        service_state = np.column_stack((old_vectors, old_biases))
        reference_state = np.column_stack((reference_vectors, reference_biases))
        service_gradient = (service_weights[:, None] * residuals[:, None] *
                            service_jacobian + l2 * service_state +
                            fedprox_mu * (service_state - reference_state))
        service_delta = _solve_service_blocks(
            service_weights, service_jacobian, service_gradient, eta, l2,
            fedprox_mu)

        budget = output_trust_budget(predictions, targets)
        alpha = 0.0 if budget == 0.0 else 1.0
        for _ in range(MAX_TRUST_BACKTRACKS + 1):
            candidate_user = old_user_vector + alpha * user_delta[:-1]
            candidate_user_bias = old_user_bias + alpha * user_delta[-1]
            candidate_vectors = old_vectors + alpha * service_delta[:, :-1]
            candidate_biases = old_biases + alpha * service_delta[:, -1]
            change = support_prediction_rms_change(
                old_user_vector, old_user_bias, old_vectors, old_biases,
                candidate_user, candidate_user_bias, candidate_vectors, candidate_biases)
            if change <= budget + 16 * np.finfo(float).eps:
                break
            alpha *= 0.5

        user_vector, user_bias = candidate_user, float(candidate_user_bias)
        service_vectors, service_biases = candidate_vectors, candidate_biases
        user_error = float(old_user_error + beta * np.mean(
            user_weights * (relative_errors - old_user_error)))
        service_errors = old_errors + beta * service_weights * (relative_errors - old_errors)

    outputs = (user_vector, user_bias, user_error, service_vectors,
               service_biases, service_errors)
    if not all(np.all(np.isfinite(value)) for value in outputs):
        raise FloatingPointError("CSOTU produced a non-finite state")
    return outputs


def csotu_extra_bytes(participants=1):
    """CSOTU changes local computation only and adds no communication fields."""
    if participants < 0:
        raise ValueError("participants must be nonnegative")
    return 0, 0
