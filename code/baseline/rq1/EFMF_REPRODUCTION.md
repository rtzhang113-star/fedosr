# FMF/EFMF independent reconstruction

## Source and code availability

Primary source: Yilei Zhang, Xiao Zhang, and Xinyuan Li, *Towards Efficient
and Privacy-Preserving Service QoS Prediction with Federated Learning*,
CollaborateCom 2020, LNICST 350, pp. 37–57, 2021,
https://doi.org/10.1007/978-3-030-67540-0_3.

An earlier related ICWS 2020 paper is *Efficient and Privacy-Preserving
Federated QoS Prediction for Cloud Services*, DOI
10.1109/ICWS49710.2020.00079.

No official implementation or software license was located. GitHub searches
for the full title, DOI, `EFMF QoS`, and algorithm text returned no algorithm
repository. The corresponding public author account `stonezyl` contains only
the author's homepage repository; its publication entry has no code link.
Therefore, every source file here is an independent implementation based on
the paper, not copied third-party code.

## Paper-defined method

- Dataset: 142 users, 4,532 services, 64 consecutive 15-minute slices; RT and
  TP are evaluated.
- Each user keeps its QoS records and user latent factor locally.
- The server maintains a shared service latent matrix.
- Prediction is `sigmoid(U_i^T S_j)` after mapping QoS to `[0,1]`.
- FMF trains local user/service factors and averages the shared service part.
- EFMF retains user factors across rounds and applies incremental SGD only to
  newly observed values.
- EFMF uploads only touched service updates, may randomly omit update
  coordinates, uses stochastic b-bit quantization, and delays updates below a
  threshold.
- The paper reports `l=10`, omitted proportion `r=20%`, `b=8`; the threshold
  delta is only described as tuned to an optimal value.
- Paper preprocessing uses Box-Cox with alpha `-0.007` for RT and `-0.005` for
  TP, followed by fixed lower/upper-bound normalization.

## Current protocol adaptation

The project uses the actual cleaned tensor `64 x 142 x 4500` and strict
`history + support[t] -> query[t]` evaluation. At slice `t`, only support tuples
are passed to `update_round`; query values are read after the update solely for
metrics. Model arrays, pending delayed updates, round index, and RNG state are
included in the query-read-only audit.

### Support-only scale adaptation and convergence audit

The paper does not publish a factor initializer.  Under the project's fixed
`log1p` target space, zero-centred factors make `sigmoid(U_i^T S_j)` almost
exactly `0.5`, which maps to about `3.58` seconds for RT and `99.0` kbps for
TP.  This is badly mismatched to the observed support scale.  Before the first
update, the causal runner therefore computes only the normalized mean of
`support[0]` and aligns the initial user/service factor magnitudes and sign so
their typical dot product matches its logit.  The prediction equation is not
changed, initialization is allowed exactly once, and no query or future label
is inspected.  This is an explicit reconstruction assumption.

For every slice, the runner records normalized support MSE and prediction mean
both before and after the update.  Formal promotion requires the support-only
audit to show positive aggregate loss reduction; query MAE/RMSE are never used
to initialize, tune, or select the run.

The runner offers two preprocessing modes:

1. `manifest_fixed_log1p` (default): uses the same frozen transform and domain
   bounds as the other RQ1 methods. This is the fair main-table configuration.
2. `paper_boxcox`: uses the paper's fixed alpha and an explicitly supplied
   positive lower and upper bound. It never estimates bounds or alpha from
   query/future data. Defaults are the paper's reported ranges (`qmax=20` for
   RT and `qmax=1000` for TP). The paper does not report its exact positive
   `qmin`; the default `1e-6` is therefore a disclosed reconstruction
   assumption. Values outside these bounds are rejected rather than silently
   clipped.

The implementation derives gradients from the stated squared-error objective.
With `error = prediction - target`, one gradient step reduces a controlled
synthetic loss. This also resolves the paper's printed update-sign ambiguity.

## Deliberately incomplete security layer

The paper sketches Laplace noise plus CRT-style encrypted aggregation, but does
not specify the sensitivity, privacy budget used in experiments, prime/key
sizes, fixed-point encoding, overflow rules, or security parameters. Simulating
these operations with arbitrary settings would not be a reproducible security
implementation. The current baseline therefore implements the predictive and
communication mechanisms only, writes
`privacy_cryptographic_layer_implemented=false` to every summary, and must not
be described as providing formal differential privacy or secure aggregation.

## Files

- `methods/efmf.py`: FMF/EFMF state, incremental client updates, aggregation,
  stochastic sparsification/quantization, threshold delay, communication
  accounting, predictions, and state audit.
- `run_efmf_temporal.py`: frozen manifest/mask loading, causal stream,
  preprocessing, RT/TP metrics, and artifacts.

## Example

```powershell
python code/baseline/rq1/run_efmf_temporal.py `
  --qos rt `
  --data data/WSDream-QoSDataset3/rtRate `
  --mask experiments/protocol_fixed_seed_unified/rt_density_0.10_seed_42_masks.npz `
  --manifest experiments/protocol_fixed_seed_unified/rt_density_0.10_seed_42_manifest.json `
  --output experiments/baseline/efmf_rt_d10_seed42 `
  --seed 42 --slices 64 --mode efmf
```

Formal runs must freeze source hashes, masks, configuration, interpreter,
status matrix, and completion markers before launch. Smoke output is not a
paper result.
