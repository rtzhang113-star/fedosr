# Baseline implementations

Baseline source for the FedOSR online QoS prediction study is organized under
`code/baseline/`.  The current release keeps the method implementations,
executable temporal runners, and method-specific reproduction notes together
under `rq1/`.

The layout is:

- `rq1/methods/`: numerical baseline implementations;
- `rq1/run_*`: one-run executable entry points;
- `rq1/*_REPRODUCTION.md`: paper-to-code assumptions and reporting limits.

Large run artifacts remain outside this release under the original
`experiments/baseline/<method_and_protocol>/` paths, so source code and
generated results are not mixed.

Implemented baselines currently include FedAvg, FedProx, PerFed (PFTF),
FL-LFT, and the independent causal FMF/EFMF reconstruction.  EFMF source,
runner, and paper-to-code notes are respectively in
`rq1/methods/efmf.py`, `rq1/run_efmf_temporal.py`, and
`rq1/EFMF_REPRODUCTION.md`.

The 2026 recommendation-method adaptations are also kept here:

- FCUCR: `rq1/methods/fcucr.py`, `rq1/run_fcucr_temporal.py`,
  and `rq1/FCUCR_REPRODUCTION.md`;
- FedEM: `rq1/methods/fedem.py`, `rq1/run_fedem_temporal.py`,
  and `rq1/FEDEM_REPRODUCTION.md`;
- FedRecGEL: `rq1/methods/fedrecgel.py`, `rq1/run_fedrecgel_temporal.py`,
  and `rq1/FEDRECGEL_REPRODUCTION.md`.

These three entries use the authors' pinned GitHub implementations as the
primary executable specification and independently adapt those mechanisms to
QoS regression.  They do not claim that the original recommendation tasks or
reported HR/NDCG results were reproduced unchanged.  Their reproduction notes
freeze the upstream commit, license status, task mapping, causal restrictions,
privacy boundary, and communication-accounting assumptions.
