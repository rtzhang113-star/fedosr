# FedOSR implementation

This directory contains the core implementation of FedOSR for online QoS
prediction and the baseline implementations used for comparison.  It is a
compact source distribution: generated experiment results and the dataset are
kept in the sibling `results/` and `data/` directories.

## Layout

- `fed_online_qos/`: FedOSR prediction, aggregation, residual calibration,
  communication, and protocol preparation modules.
- `baseline/rq1/methods/`: baseline method implementations.
- `baseline/rq1/run_*.py`: executable temporal baseline runners.
- `baseline/rq1/*_REPRODUCTION.md`: method-specific reproduction notes,
  assumptions, and reporting boundaries.

Run commands from the package root (the directory containing `code/`,
`results/`, and `data/`). The dataset is expected at
`data/WSDream-QoSDataset3/`.

The source copy preserves the original research tree's implementation
boundary; the original `code/` directory remains unchanged.
