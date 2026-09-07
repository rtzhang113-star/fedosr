# FedEM-QoS reproduction note

## Scope and provenance

This implementation follows the mechanism of *Breaking the Aggregation
Bottleneck in Federated Recommendation: A Personalized Model Merging
Approach* (AAAI 2026), while adapting its recommendation objective to the
frozen WSDream Dataset 3 causal QoS protocol.

The official-source audit is pinned to:

* repository: `https://github.com/jundongchen13/FedEM`;
* commit: `747ed85b17156ef62b3813a34f08c7ad626c2cf0`;
* no `LICENSE` file was found in that repository snapshot.

No reference source is copied into this repository.  The NumPy implementation
is an independent reconstruction of the official mechanism.  The executable
source is `methods/fedem.py`; the causal protocol wrapper is
`run_fedem_temporal.py`.

## Official mechanism retained

The reference repository defines one FCF recommender per user: a user vector,
an item embedding table, and a sigmoid dot-product score.  It uses a private
adapter to merge each local item vector with a personalized server item
vector.  Its default adapter is the MLP

```text
input  = [local_item_factor; server_item_factor - local_item_factor]
MLP    = 32 -> 16 -> 8 -> 1
hidden = ReLU, ReLU
output = Sigmoid(omega)
merge  = local_factor + omega * (server_factor - local_factor)
```

The QoS implementation keeps that adapter shape and merge semantics.  With
the default `rank=16`, the factor-only input is 32-dimensional and the
adapter sizes are exactly `[32, 16, 8, 1]`.  For other ranks, the default
architecture is `[2*rank, rank, max(1, rank//2), 1]`.  Service bias is merged
using the same `omega`, but is deliberately not provided to the MLP input.
Each client owns its own adapter parameters.

The reference trains the adapter first and the user/item embeddings second on
each local batch.  The QoS implementation follows this order with two real
Adam updates per seeded-shuffled mini-batch: a support regression-loss Adam
step for the adapter, followed by a fresh forward pass and one embedding Adam
step for the private user and dense service parameters.  Service factor and
bias moments are maintained independently for every service row, matching the
dense ``torch.nn.Embedding`` semantics.  Adam moments are reset when a new
client local round is started, matching the reference's construction of fresh
optimizers per client round.  The support residual and both optimizer steps
use only the current support slice.

## QoS-required task migration

The reference task is implicit recommendation: positives are binarized,
negative items are sampled, and `BCELoss` is evaluated with HR/NDCG ranking.
The WSDream task instead uses fixed-log1p-normalized RT/TP regression and
reports MAE/RMSE/MRE/NPRE after inverse transformation.  Accordingly:

* a WSDream user is one client;
* item factors become persistent service factors;
* private user factors and user biases are retained, and service biases are
  added for QoS regression;
* no negative sampling, leave-one-out split, or future/query training is used;
* `update_client` accepts exactly one time slice and rejects a mixed-time batch;
* support aggregation counts are raw support samples, not multiplied by local
  epochs;
* the runner executes `support[t] -> local update -> aggregation -> query[t]`
  and snapshots all state before read-only query prediction.

These are task/protocol substitutions, not claims that the original paper
used QoS data.

## Selected official-aligned-v1 profile

The user-selected formal profile restores the configuration that produced
`official_code_aligned_smoke_20260826`: one local epoch, full-client support
batches, factor and adapter regularization of `0.0003`, and support-only
residual service signatures for client similarity. The later experimental
profile (`10` epochs, batch size `256`, zero regularization, and PCA over the
complete persistent service-factor table) remains preserved in the historical
smoke artifacts but is no longer the default formal profile.

This is an RQ1 causal QoS adaptation profile, not a claim that the original
recommendation experiment or every unpublished training default was reproduced
verbatim. The support-signature similarity is the earlier causal adaptation;
the paper/code-aligned PCA implementation remains selectable with
`similarity_source="persistent_pca"` for audits and controlled comparisons.

The selected profile is source-frozen in
`methods/fedem_official_aligned_v1.py` and executed by
`run_fedem_official_aligned_v1_temporal.py`. The D05--D25 formal launcher uses
these files explicitly. The later repaired implementation remains in
`methods/fedem.py` and `run_fedem_temporal.py`, preventing silent mixing while
preserving both implementations for audit.

## PCA/L2 similarity and official aggregation modes

The reference computes one similarity value per pair of participating clients
from the complete persistent item-factor matrix.  It centers each client's
matrix, projects onto its first PCA component, then uses the default

```text
similarity(i,j) = 1 / (L2_distance(PCA_i, PCA_j) + 1).
```

Cosine similarity is available as the reference's optional alternative, with
negative values clipped to zero.  Similarities greater than `0.9` are set to
`1`.  The QoS implementation applies the same operation to the complete
persistent service-factor table of the aggregation participants; the
support values only determine which clients can participate and their sample
counts, never the query/future labels.

The aggregation modes are:

* `avg`: every target row uses support-sample proportions `p`;
* `sim`: every target row uses its normalized similarity row;
* `both` (default): the reference QP

  ```text
  min_x ||x||^2 + (2 alpha s - 2 p)^T x
  s.t. x >= 0, sum(x)=1
  ```

  is solved exactly by the equivalent Euclidean simplex projection
  `x = Proj_simplex(p + alpha * similarity_row)` because the reference sets
  `s = -similarity`.  The default is `alpha=0.9`.  The implementation uses a
  NumPy simplex projection rather than adding a CVXPY runtime dependency.

Each target in the aggregation subset receives its own weighted service
factor/bias table.  Clients outside that subset do not receive a new
personalized server model in that round.  A client-specific copy of its
adapter may be kept in the server participant state for official state
semantics, but adapter parameters are never averaged across clients.

## Sampling and defaults

The default online-compatible sampling settings match the reference:

```text
clients_sample_ratio = 1.0
agg_clients_ratio    = 0.1
agg_mode              = both
sim_mode              = l2
rank                 = 16
local_epochs          = 10
batch_size           = 256
learning_rate        = 0.1
adapter_learning_rate = 0.1
alpha                = 0.9
regularization       = 0.0
adapter_regularization = 0.0
```

The aggregation subset is selected with the model seed.  For a small online
slice, at least one participating client is retained and at least one client
is selected for aggregation so that a valid server state exists.
The two regularization values remain configurable QoS options; zero is the
official-code profile because the reference parses `weight_decay` but does
not pass it to Adam and its loss is unregularized `BCELoss`.

## Dense communication accounting

The reference aggregates dense item embedding tables.  This adaptation uses
the fixed **official dense QoS adaptation accounting** convention:

* upload: all service factors and service biases, plus the client adapter MLP
  parameters in the participant state;
* download: a dense personalized service-factor/service-bias table for each
  aggregation target;
* user factor and user bias: never uploaded;
* adapter parameters: copied to server participant state but never mixed;
* service indices: implicit in dense tables, so no sparse index metadata is
  charged.

The byte counts are FP32 counts.  They are an explicit QoS adaptation of the
reference's dense item-table communication semantics, not a claim about the
original repository's transport implementation.

## Causal evaluation contract

For each time slice `t`:

```text
support[t] -> seeded shuffle/mini-batches
            -> local adapter Adam -> local embedding Adam
            -> random aggregation subset -> query[t] prediction (read only)
```

The runner verifies the fixed-log1p manifest and dataset hash.  The query audit
snapshots all numeric arrays, all client/server adapter arrays, server-model
availability flags, `round_index`, sampling records, and RNG state before
prediction.  Query labels are used only for the current metric; they cannot
influence a later update.

## Reproduction command

```powershell
python code/baseline/rq1/run_fedem_temporal.py `
  --qos rt `
  --data data/WSDream-QoSDataset3/rtRate `
  --mask experiments/protocol_fixed_seed_unified/rt_density_0.10_seed_42_masks.npz `
  --manifest experiments/protocol_fixed_seed_unified/rt_density_0.10_seed_42_manifest.json `
  --output experiments/baseline/fedem_qos_rt_d10_s42 `
  --seed 42
```

The runner writes `per_time.csv` and `summary.json` to the selected output
directory.
