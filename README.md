# FedOSR

This repository maintains the implementation of **FedOSR**, a federated
online method for QoS prediction. FedOSR performs causal, time-slice-wise
model updates without using query labels or future observations during
training.

FedOSR contains three main components:

- **C-OMF**: causal online matrix factorization with output-stability control;
- **FedSRA**: service-level sparse residual-aware aggregation and calibration;
- **TSD**: tolerance-aware sparse downlink with bounded differential encoding.

## Dependencies

- Python 3.11
- NumPy

Install the dependency with:

```bash
python -m pip install -r requirements.txt
```

## Repository structure

```text
code/
  fed_online_qos/    FedOSR implementation
  baseline/rq1/      baseline implementations and runners
data/
  WSDream-QoSDataset3/readme.txt
results/             reserved for future public results
```

The public package intentionally excludes experiment outputs, checkpoints,
per-time-slice records, plotting scripts, and internal experiment-management
code.

## Dataset

The experiments use **WS-DREAM Dataset 3**, which contains time-aware response
time (RT) and throughput (TP) observations collected from 142 service users
over 64 time intervals.

The raw `rtRate` and `tpRate` files are not redistributed in this repository.
Please obtain the dataset from its source and place the two files under
`data/WSDream-QoSDataset3/`:

- [Official WS-DREAM dataset page](https://www.inpluslab.com/wsdream/)
- [Official Dataset 3 download](https://inpluslab.com/wsdream/download/WSDream-QoSDataset3.zip)
- [Author-associated GitHub dataset archive](https://github.com/LanternYing/Dataset)

The GitHub archive preserves the dataset description but does not contain the
large Dataset 3 RT/TP files. The original dataset description and citation are
also retained in `data/WSDream-QoSDataset3/readme.txt`.

## Usage

First generate a support/query protocol. For example, for RT at 10% training
density with seed 42:

```bash
python code/fed_online_qos/prepare_rt_protocol.py \
  --data-file data/WSDream-QoSDataset3/rtRate \
  --density 0.10 \
  --seed 42 \
  --output-dir protocols
```

Then run FedOSR with the generated mask and manifest:

```bash
python code/fed_online_qos/run_fed_amf.py \
  --qos-type rt \
  --data-file data/WSDream-QoSDataset3/rtRate \
  --mask-file protocols/rt_density_0.10_seed_42_masks.npz \
  --manifest protocols/rt_density_0.10_seed_42_manifest.json \
  --run-dir output/fedosr_rt_d10_seed42 \
  --dimension 10 \
  --eta 0.8 \
  --l2 0.0003 \
  --beta 0.3 \
  --local-epochs 1 \
  --aggregation stca \
  --stca-component evidence \
  --service-raw-residual-calibration \
  --log-domain-service-calibration \
  --dual-constrained-log-calibration \
  --event-sparse-service-downlink \
  --causal-support-output-trust-update \
  --output-trust-bounded-compression \
  --communication-parameter-bytes 4
```

For TP, use `prepare_tp_protocol.py`, `--qos-type tp`, and the `tpRate` file.
Baseline-specific runners and implementation notes are provided under
`code/baseline/rq1/`.

## Related project

The concise organization of this repository follows the release style of the
official [AMF repository](https://github.com/wsdream/AMF), which provides the
online matrix-factorization foundation related to this work.

## License

No project license has been added yet. The WS-DREAM dataset remains subject to
the terms stated by its provider and in the included dataset readme.
