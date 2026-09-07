# PerFed baseline reconstruction

This directory contains the PerFed baseline runner for RQ1.  It is an
independent reconstruction of Li et al., *A Personalized Federated Tensor
Factorization Framework for Distributed IoT Services QoS Prediction From
Heterogeneous Data*, IEEE Internet of Things Journal, 2022,
DOI: `10.1109/JIOT.2022.3197172`.

The paper describes an offline static tensor-completion experiment.  The
project's executable entry point below is a separately labelled causal online
adaptation.  Its output must not be reported as a reproduction of the paper's
Table II unless a separate static random-deletion runner and protocol are
implemented.

## Source and evidence boundary

The paper PDF is present at the repository root.  The project audit recorded
checks of the authors' publication/lab pages, the GTML-Lab GitHub presence,
the author pages, and the WS-DREAM data repository, but no corresponding
official PerFed implementation was found.  Therefore `methods/perfed.py` is
an independent reconstruction from the paper's equations and Algorithm 1,
not a wrapper around author code.  This boundary is important: matching a
paper number cannot be claimed from this implementation alone.

The paper does provide the RTdata dimensions and several parameter sweeps,
but it does not provide the complete preprocessing, initialization,
partition, optimizer, or stopping recipe needed for bitwise reproduction.
Those missing choices are written into every causal `summary.json`.

## Paper method reconstructed

For edge server \(k\), the QoS tensor is represented by a rank-\(R\) CP model

\[
\widehat X_k(u,s,t)=\sum_{r=1}^{R} A^{(1)}_k(u,r)
 B^{(2)}_k(s,r) B^{(3)}_k(t,r).
\]

The user factor \(A^{(1)}_k\) is private to the edge server.  The service and
time factors are public in the communication sense, but each edge server
keeps a local personalized copy.  A global service/time copy is maintained
by the parameter master.  The local objective in Eq. (13) is the CP
reconstruction loss plus a quadratic coupling between local public factors and
the global public factors, controlled by `gamma`.  The update is implemented
with sparse SGD/EASGD-style steps, so only supplied support tuples are sent to
the numerical method.

The causal runner uses a deterministic IID user partition into `K=6` edge
servers by default.  The partition is generated from the configured seed and
only the user count; it does not inspect QoS values or masks.  A current slice
is processed as follows:

1. Build one batch per participating edge server from `support[t]` only.
2. Normalize support values with the manifest's frozen `fixed_log1p`
   transform and update local PerFed state.
3. Predict `available[t] & ~support[t]` by a read-only model call.
4. Compute MAE, RMSE, MRE, and NPRE against query labels only.
5. Record elapsed time and public-factor upload/download bytes.

The preferred method interface is `sync_round(client_batches, epochs=...)`,
followed by `predict(users, services, time_slices, client_ids=...)` and
`state_arrays()`.  The runner also accepts `update_round` and a sequential
`update_client` plus `aggregate` fallback for small independent
implementations, and records the selected update path in `summary.json`.
Query labels are never passed to either update method.

## Formula corrections and implementation choices

The PDF has two notation/gradient issues that must be kept visible in any
report:

* Eq. (17) prints a plus-gradient update, although Eq. (15), the stated
  gradient-descent procedure, and the loss minimization require a negative
  gradient step.  The reconstruction uses the standard subtract-gradient
  update.
* Eq. (18) has an incomplete summation/index expression.  It does not state
  the client index or an aggregation weight clearly.  The reconstruction uses
  the support-weighted elastic public-factor center over participating clients,
  matching the \(p_k\) weighting in Eq. (13) and the EASGD interpretation.

These are implementation decisions, not claims that the PDF contains the
corrected equations.  They should remain in the experiment record.

## Parameters: known versus assumed

| Item | Paper evidence | Causal runner default/status |
|---|---|---|
| Tensor shape | RTdata is 142 users x 4500 services x 64 time slots | Loaded from the manifest; shape is checked |
| CP rank \(R\) | Sweep 3--20; final text selects \(R=5\) | `--rank 5` |
| Edge servers \(K\) | Static study reports \(K=3,6,10\) | `--num-edge-servers 6` for the historical smoke convention |
| `gamma` | Sensitivity sweep 0.001--0.1; no unique final value is reported | `--gamma 0.01`, explicitly a reconstruction assumption |
| Learning rate | Not reported | `--learning-rate 0.001`, reconstruction assumption |
| Local epochs/iterations | Not specified as a reproducible final setting | `--local-epochs 1`, reconstruction assumption |
| Initialization | Not specified | rank-scaled positive-uniform initialization in the method, reconstruction assumption |
| Random seed | Not specified | CLI default `42`; must be recorded and matched to the manifest |
| Client partition | IID random assignment and QoS-clustered Non-IID are described, but exact procedure is absent | deterministic IID user split from seed; no query/value inspection |
| Preprocessing | Static experiment says randomly remove values; complete transform is absent | causal runner requires `fixed_log1p` and uses `log1p(values) / log1p(manifest.upper_bound)`; fixed ex-ante bound |
| Stopping/round schedule | Not fully reported | one local update round per causal time slice; `--slices` controls prefix length |
| Communication dtype | Complexity gives factor counts, not a wire dtype | method-reported bytes, otherwise factor-count x model dtype item size |

The old two-slice RT smoke convention was `seed=42`, `K=6`, `R=5`,
`learning_rate=0.001`, `gamma=0.01`, and one local epoch.  It is a
runability check only; it is not a performance result or a paper-number
replication.

The preserved historical smoke record reports support counts `21,821/22,487`,
query counts `415,881/428,551`, and two-slice mean MAE/RMSE
`2.251497/3.041680`.  These values are retained as runability evidence only;
the current runner does not treat them as a formal result and this task does
not create a new result directory.

## Commands

Run a two-slice smoke (use an output directory reserved for smoke artifacts):

```powershell
python code/baseline/rq1/run_perfed_temporal.py `
  --qos rt `
  --data data/WSDream-QoSDataset3/rtRate `
  --mask experiments/protocol_fixed_seed_unified/rt_density_0.05_seed_42_masks.npz `
  --manifest experiments/protocol_fixed_seed_unified/rt_density_0.05_seed_42_manifest.json `
  --output experiments/baseline/perfed_rt_d05_seed42_smoke_t02 `
  --seed 42 --slices 2
```

The TP command changes `--qos tp`, the data file to `tpRate`, and the mask and
manifest to the matching TP density/seed pair.  A full causal prefix uses
`--slices 64`; formal multi-seed execution should freeze the exact command,
source snapshot, mask/manifest pair, and output directory before launch.

The runner writes exactly two primary outputs:

* `per_time.csv`: one row per processed time slice with support/query counts,
  four query metrics, timing, communication bytes, and the query read-only
  audit flag.
* `summary.json`: aggregate metrics, method/protocol labels, all reconstruction
  assumptions, partition hash, update path, and total communication.

No result files are created by importing the module.  `run_arrays(...)` is
provided for causal audits that perturb query or future labels in memory; it
does not reload or rewrite the 160-MB raw tensor.

## Reporting restrictions

Report causal outputs as **PerFed causal adaptation** and keep them separate
from the paper's offline random-deletion evaluation.  Do not use query labels
to tune `gamma`, learning rate, initialization, partitioning, or update
schedule.  Do not promote the historical smoke metrics to a baseline result
without a frozen protocol, source snapshot, and the project's formal
validation gates.
