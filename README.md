# FLAD: Byzantine-robust Federated Learning Based on Gradient Feature Anomaly Detection


## Introduction

This is the official repository of FLAD: Byzantine-robust Federated Learning Based on Gradient Feature Anomaly Detection.

We propose FLAD, a novel Byzantine-robust FL approach based on gradient feature anomaly detection, which uses neural networks to adaptively learn gradient features and measure feature similarity to counteract various types of poisoning attacks. Specifically, FLAD employs a small clean dataset to bootstrap trust and trains Feature Extraction Models (FEM). With FEM and DBSCAN clustering, abnormal gradients from malicious clients are detected and eliminated. Extensive experiments on both Non-IID and IID datasets demonstrate that FLAD achieves superior accuracy, robustness, efficiency, and generalizability compared to state-of-the-art approaches. Additionally, we implement privacy-preserving FLAD (PFLAD) using CKKS and Random Permutation techniques to ensure transmitted gradient privacy.

## Project Directory Structure
```shell
|-- FLAD # XXX
|   |-- XXX


```

## Requirements

We have tested all the code in this repository on a server with the following configuration:
- CPU: XXX
- GPU: XXX
- OS: XXX

The code of this repository is written in Python. We use conda to manage the Python dependencies. Please download and install conda first.

After cloning this repository, change the working directory to the cloned directory.

We recommend using Python version XXX or higher. Create a new conda environment and install the dependencies:


## Citation

Please cite it if you find the repository helpful. Thank you!

```
@article{FLAD2025,
  title={{FLAD}: Byzantine-robust Federated Learning Based on Gradient Feature Anomaly Detection},
  author={Tang, Peng and Zhu, Xiaoyu and Qiu, Weidong and Huang, Zheng and Mu, Zhenyu and Li, Shujun},
  journal={IEEE Transactions on Dependable and Secure Computing}, 
  year={2025},
  doi={10.1109/TDSC.2025.3542437},
}
```

