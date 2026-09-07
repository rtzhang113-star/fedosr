========================Time-Aware Web Service QoS Dataset========================

The main goal of this dataset is to offer real-world data for future research.
The provider collected 4,532 Web services from public sources on the Web. A
total of 142 distributed computers from PlanetLab evaluated the QoS performance
of these services over 64 time intervals.

The dataset is freely available for research purposes under the terms stated
by its provider. Redistribution of the raw dataset to another party or over the
Web is not permitted. The raw rtRate and tpRate files are therefore not included
in this repository.

If you use this dataset, please cite:

Yilei Zhang, Zibin Zheng, and Michael R. Lyu, "WSPred: A Time-Aware
Personalized QoS Prediction Framework for Web Services," Proceedings of the
22nd IEEE International Symposium on Software Reliability Engineering
(ISSRE), 2011.

@inproceedings{Zhang11issre,
  title={WSPred: A Time-Aware Personalized QoS Prediction Framework for Web Services},
  author={Yilei Zhang and Zibin Zheng and Michael R. Lyu},
  booktitle={Proceedings of IEEE Symposium on Software Reliability Engineering (ISSRE'11)},
  year={2011}
}

The original package contains:

- rtRate: response-time observations in seconds.
- tpRate: throughput observations in kbps.

Record format:

Time Interval ID | Web Service ID | Service User ID | QoS value

===============================Source Links================================

Author-associated GitHub dataset archive:
https://github.com/LanternYing/Dataset

Official WS-DREAM dataset page:
https://www.inpluslab.com/wsdream/

Official WS-DREAM Dataset 3 download:
https://inpluslab.com/wsdream/download/WSDream-QoSDataset3.zip

The GitHub archive preserves the Dataset 3 description but does not contain
the large rtRate and tpRate files. Obtain the raw files from the official
WS-DREAM source and follow the provider's terms.
