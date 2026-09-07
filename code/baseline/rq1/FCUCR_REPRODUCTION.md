# FCUCR-QoS reproduction

This directory contains an independent NumPy adaptation of FCUCR for the
WSDream user--service--time QoS stream.  It is not a copy of the public
repository because the upstream repository does not declare a software
license.

## Provenance

The source paper is:

> Chunxu Zhang, Zhiheng Xue, Guodong Long, Weipeng Zhang, and Bo Yang,
> “Learning Evolving Preferences: A Federated Continual Framework for
> User-Centric Recommendation,” Proceedings of the ACM Web Conference 2026,
> pp. 6609--6620, DOI: [10.1145/3774904.3792575](https://doi.org/10.1145/3774904.3792575).

The implementation reference is the public repository
[Poizoner/code4FCUCR_www2026](https://github.com/Poizoner/code4FCUCR_www2026),
`main` commit
`1ba1f054d94eac15a6a19965cf8e45db78a94782` (verified against the remote branch
on 2026-08-26).  The repository metadata reports no declared license and the
tree contains no `LICENSE` or `requirements.txt`.  Accordingly, this baseline
uses the paper's equations and an independent implementation rather than
copying source code.  The upstream implementation is only a provenance and
algorithm reference.

## FCUCR mechanism retained

FCUCR separates a shared representation (f_\phi) from a private client
prediction module (p_{\psi_u}).  For a current support slice
\(\mathcal S_u^{(t)}\), its representation prototype is

\[
h_u^{(t)} = f_\phi(\mathcal S_u^{(t)}).
\]

The time-aware self-distillation term is

\[
\mathcal L_{dist} =
\operatorname{MSE}\left(f_{\phi_u^{(t)}}(\mathcal S_u^{(t)}),
f_{\phi_u^{(t-1)}}(\mathcal S_u^{(t)})\right).
\]

The server retrieves the top-(k) prototypes from other clients in a strict
past knowledge base and fuses them with cosine-similarity softmax weights:

\[
\rho_u^{(t)} = \operatorname{softmax\mbox{-}mean}
\left(\operatorname{TopK}\{h_v^{(t')}:v\ne u,\ t'<t\}\right).
\]

The QoS head receives the local support representation and the transferred
prototype.  For this regression migration, the local objective uses the
official FCUCR scaling pattern with the fixed-log1p QoS mean-squared base
loss:

\[
\mathcal L = \mathcal L_{qos} + \lambda\min\left(
\frac{\mathcal L_{qos}}{\mathcal L_{dist}+10^{-8}},\ C_{dist}\right)
\mathcal L_{dist},
\qquad C_{dist}=10.
\]

The local optimizer is Adam with \(\beta_1=0.9,\beta_2=0.98\), matching the
public client.  Private user Adam moments persist across slices and are never
averaged.  Only the shared service/time representation is FedAvg-aggregated;
equal-client aggregation is the default, with sample-weighted aggregation
available as an explicit control.

Because this is a fixed-log1p regression head, each client is scale-initialized
once, at its first support-only participation.  Let \(\bar y_u\) be that
client's mean normalized support target and let \(\overline{z}_{u,\mathrm{nb}}\)
be the current mean logit with the private bias removed.  The private bias is
initialized as

\[
b_u \leftarrow \operatorname{logit}(\operatorname{clip}(\bar y_u))
      - \overline{z}_{u,\mathrm{nb}}.
\]

The initialized flag and calibration diagnostics are part of the model state
used by the read-only audit.  No query or future label participates in this
calibration.  Under the current teacher/student semantics, the first local
epoch starts with identical student and previous-slice teacher factors; a
positive TASD term therefore requires at least two local epochs.  The runner
defaults to four local epochs and explicitly rejects a later non-empty slice
with positive distillation weight when fewer than two epochs are requested.

## Causal WSDream adaptation

For every time slice (t), the runner performs the following order:

1. Read only the support mask and transform its raw QoS values with the frozen
   fixed-log1p transform.
2. Compute each participating user's support prototype using the current
   shared service/time representation, then perform the one-time support-only
   scale initialization described above.
3. On the server, retrieve only entries with timestamp `< t` and a different
   user ID, then perform top-k cosine-softmax fusion.  The current slice has
   no KB entries yet, so neither a current prototype nor a self-prototype can
   be selected.  Each client receives only one fused prototype vector; the
   individual top-k KB entries are not downloaded.
4. Update private user state and a local copy of shared factors using support
   regression plus the ratio/cap-scaled previous-slice teacher MSE.  The
   default/frozen configuration uses four local epochs so that the teacher
   loss is active on later slices.
5. Equal-average only shared service/time factors at the server by default
   (`aggregation_mode=equal`).
6. Recompute each post-update support prototype and append it to the KB.
7. Freeze state and predict the disjoint query entries.  Query labels are used
   only by the final MAE/RMSE/MRE/NPRE calculation.

The implementation uses a compact NumPy factorized representation:

* shared service factors and shared time factors form the representation;
* private user factors, user bias, and prototype gates form the QoS head;
* a prototype is the mean of service-plus-time representations over the
  support entries of one user and slice;
* the model output is a sigmoid-normalized QoS value, inverted to the raw
  domain only for evaluation.
* `kb_window=0` retains all past prototypes; a positive `kb_window` keeps
  only the most recent window of time slices after each append.

This is a task migration, not a claim that the original recommendation code
solves QoS regression unchanged.  Ranking loss, negative item sampling,
HR/NDCG evaluation, and validation/test-session prototype selection from the
upstream code are intentionally not used here.

The source code's default `KB_len` is zero, while its README experiment
command uses `KB_len=20`; this runner therefore defaults to `top_k=20` to match
the documented experiment setting.  The source `KB_all=0` means an unlimited
KB; this migration exposes the additional time-based `kb_window` control.

## Files and artifacts

* `methods/fcucr.py`: independent model, causal KB, teacher snapshots,
  support-only scale initialization, private/shared Adam state, prediction,
  state audit, and byte counters.
* `run_fcucr_temporal.py`: manifest/data/mask runner, fixed-log1p transform,
  query read-only audit, and `per_time.csv`/`summary.json` writers.

The official repository does not implement a byte ledger and averages the
complete client model, then sends all non-`mlp4logfeats` parameters back to
clients.  The QoS migration instead uses an explicit shared/private boundary:
its communication ledger intentionally retains the current symmetric count:
one float32 shared representation payload plus one float32 prototype payload
in each direction for every participating client.  Semantically, retrieval
and top-k fusion happen at the server and the client downloads exactly one
fused prototype vector; no individual top-k entries are downloaded.  The
prototype term in the uplink counter is retained for comparability with the
existing ledger and is not a claim that clients upload the KB entries.  Private
user state and raw support/query values are not counted as network payloads.
The implementation does not provide cryptographic privacy, secure
aggregation, or a formal DP guarantee.

Every time slice must contain at least one query observation.  The runner
rejects an empty query slice before mutating the model; an empty support slice
is allowed and performs a no-client update before read-only query prediction.

## Example

```powershell
python code/baseline/rq1/run_fcucr_temporal.py `
  --qos rt `
  --data <frozen-data.txt> `
  --mask <frozen-mask.npz> `
  --manifest <frozen-manifest.json> `
  --output <output-directory> `
  --seed 42 `
  --slices 64 `
  --rank 10 `
  --learning-rate 0.01 `
  --local-epochs 4 `
  --distillation-weight 1.0 `
  --distil-cap 10.0 `
  --top-k 20 `
  --aggregation-mode equal
```

The supplied manifest must identify the same QoS type and seed and must use
the frozen `fixed_log1p` preprocessing contract.
