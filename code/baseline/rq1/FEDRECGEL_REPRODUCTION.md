# FedRecGEL-QoS causal adaptation

This directory contains an independent NumPy adaptation of FedRecGEL for the
WSDREAM online federated QoS task.  It is not a copy or a derivative of the
public PyTorch repository.

## Source boundary

The source method is *Sharpness-Aware Minimization for Generalized Embedding
Learning in Federated Recommendation*, WWW 2026, pp. 6630--6641.  The public
implementation audited for this reconstruction was:

`https://github.com/anonymifish/FedRecGEL`

The implementation snapshot used for the audit is commit
`c7f2d929b49bb4112d8623d284c4c521663b0785`.  The repository contains a
GPL-3.0 license.  This QoS implementation was written independently with
NumPy; no public source file is imported or copied.

The original repository is an implicit recommender.  It uses binary
positive/negative interactions, negative sampling, BCE-with-logits loss and
HR/NDCG evaluation.  Those task-specific pieces are not suitable for the
continuous WSDREAM QoS target and are replaced below.

## Retained FedRecGEL mechanism

For a user/client (u) and service (s), the model has two interaction
branches:

\[
h_{GMF}=p_u^g\odot q_s^g,
\qquad
h_{MLP}=\operatorname{MLP}([p_u^m;q_s^m]),
\]

\[
\hat y_{u,s}=\sigma\!\left(w^\top[h_{GMF};h_{MLP}]+b\right).
\]

The user embeddings (p_u^g,p_u^m) are private client state.  The service
embeddings (q_s^g,q_s^m), MLP parameters and regression head are shared
server state.

For normalized QoS target (y), the local support objective is

\[
L_u=\frac{1}{2|D_u|}\sum_{(u,s)\in D_u}
(\hat y_{u,s}-y_{u,s})^2.
\]

The two SAM groups follow the source implementation.  If (g_{sh}) and
(g_{ns}) are the ordinary shared and non-shared gradients, the perturbed
points are

\[
\epsilon_{sh}=\rho_{sh}\frac{g_{sh}}{\|g_{sh}\|_2+\varepsilon},
\qquad
\epsilon_{ns}=\rho_{ns}\frac{g_{ns}}{\|g_{ns}\|_2+\varepsilon}.
\]

The shared SAM gradient is evaluated at
\(\theta_{sh}+\epsilon_{sh}\), the private SAM gradient at
\(\theta_{ns}+\epsilon_{ns}\), and both perturbations are removed before the
parameter update.  The forward/backward pass receives the complete local
shared-parameter bundle explicitly, including the MLP and regression head;
therefore local shared Adam updates are used by every subsequent local epoch.
The server averages one final-epoch shared SAM gradient equally over the
participating users and performs one Adam update.  Private user embeddings and
their optimizer states remain client-local.

Communication accounting treats one server-to-client and one client-to-server
message as the complete shared parameter state in float32.  Private user
embeddings are not counted as transmitted.

## Causal protocol adaptation

The runner enforces the project's frozen ordering:

```text
history state + support[t]
    -> user-local update + SAM
    -> equal-client shared-gradient aggregation
    -> read-only prediction of query[t]
```

Only support values are passed to `FedRecGELQoS.update_round`.  Query labels
are used only by `evaluate`, and future labels are never passed to the model.
The query call is audited with `state_copy()` and must leave all model and
optimizer arrays, the round counter, the server Adam step, and the RNG state
byte-for-byte/logically unchanged.  Empty support is a valid no-op round.

QoS values are normalized with the fixed, ex-ante `log1p` transform

\[
y=\frac{\log(1+x)}{\log(1+x_{max})},
\]

and predictions are mapped back before reporting MAE, RMSE, MRE and NPRE.
RT and TP use their own manifest upper bounds.

Because the adapted target lies in `[0,1]`, the regression head uses a stable
sigmoid output.  Its bias is initialized exactly once from the mean of the
first non-empty support slice (`logit(mean(support))`).  This initialization
never reads query or future labels.  Per-slice artifacts record support
prediction minima, maxima and boundary-saturation fractions so a finite but
collapsed model cannot be mistaken for a valid convergence result.

## Files and commands

- `methods/fedrecgel.py`: independent NumPy model, manual backpropagation,
  explicit shared forward bundle, SAM, private/shared state and communication
  result.
- `run_fedrecgel_temporal.py`: manifest/data loading, causal slices, fixed
  preprocessing, read-only audit, metrics and artifacts.

Dataset execution follows the other RQ1 runners:

```powershell
python code/baseline/rq1/run_fedrecgel_temporal.py `
  --qos rt `
  --data data/WSDream-QoSDataset3/rtRate `
  --mask experiments/protocol_fixed_seed_unified/rt_density_0.05_seed_42_masks.npz `
  --manifest experiments/protocol_fixed_seed_unified/rt_density_0.05_seed_42_manifest.json `
  --output experiments/baseline/fedrecgel_rt_d05_seed42_smoke_t02 `
  --seed 42 --slices 2
```

The runner writes `per_time.csv` and `summary.json`.  The output is a causal
QoS adaptation result and must not be described as a reproduction of the
original implicit-recommendation HR/NDCG table.

## Implementation assumptions

The public code uses PyTorch Adam, zero dropout, factor size 32, three MLP
layers, local learning rate 0.1 and SAM radii 0.02.  This reconstruction keeps
those architectural defaults and uses an explicit NumPy Adam implementation.
The public binary implicit-feedback experiment configuration passes
`global_lr=1.5`, `local_lr=0.1`, `fraction=0.1`, `local_epochs=1`, and
`rho_sh=rho_ns=0.02`.  Directly transferring the two learning rates to the
continuous-MSE NumPy adaptation caused the shared MLP/head to diverge while
remaining finite.  The QoS regression therefore uses `global_lr=0.001` and
`local_lr=0.01`; both the source-task values and the adapted executable values
are recorded in `summary.json`.  Client sampling is applied to users that have support in
the current slice; the seeded RNG selects `max(1, floor(fraction*n))` clients,
matching the source server while making the small-client boundary explicit.

The public implementation evaluates each client's loss in chunks of 128 and
sums the batch losses.  The current NumPy adaptation evaluates all of one
client's current-slice support together.  This is a remaining numerical
adaptation, not a change to the shared/non-shared SAM parameter grouping.

The source code also samples random negatives and evaluates test negatives;
these are intentionally removed because WSDREAM supplies continuous QoS
observations and the current protocol evaluates a hidden current-slice query.
