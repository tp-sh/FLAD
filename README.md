# FLAD: Byzantine-robust Federated Learning Based on Gradient Feature Anomaly Detection


## Introduction

This is the official repository of FLAD: Byzantine-robust Federated Learning Based on Gradient Feature Anomaly Detection.

We propose FLAD, a novel Byzantine-robust FL approach based on gradient feature anomaly detection, which uses neural networks to adaptively learn gradient features and measure feature similarity to counteract various types of poisoning attacks. Specifically, FLAD employs a small clean dataset to bootstrap trust and trains Feature Extraction Models (FEM). With FEM and DBSCAN clustering, abnormal gradients from malicious clients are detected and eliminated. Extensive experiments on both Non-IID and IID datasets demonstrate that FLAD achieves superior accuracy, robustness, efficiency, and generalizability compared to state-of-the-art approaches. Additionally, we implement privacy-preserving FLAD (PFLAD) using CKKS and Random Permutation techniques to ensure transmitted gradient privacy.

## Project Directory Structure
```shell
|-- 1.FLAD # FLAD Source Code
|   |-- checkpoints # The trained model is saved to this directory
|   |-- Attack.py # Attack Methods
|   |-- clients.py # Client Processing Code
|   |-- getData.py # Getting Data
|   |-- main.py # FLAD Main Function
|   |-- Models.py # Global Model and Feature Extraction Model Structures
|-- 2.FLTrust
|   |-- checkpoints # The trained model is saved to this directory
|   |-- Attack.py # Attack Methods
|   |-- clients.py # Client Processing Code
|   |-- getData.py # Getting Data
|   |-- FLTrustServer.py # FLTrust Main Function
|   |-- Models.py # Global Model Structures
|-- 3.Others_Defences
|   |-- checkpoints # The trained model is saved to this directory
|   |-- Attack.py # Attack Methods
|   |-- clients.py # Client Processing Code
|   |-- getData.py # Getting Data
|   |-- main.py # FedAvg, Krum, Bulyan and Median Main Function
|   |-- Models.py # Global Model Structures
|-- 4.FLAME
|   |-- Attack.py # Attack Methods
|   |-- clients.py # Client Processing Code
|   |-- getData.py # Getting Data
|   |-- main.py # FLAME Main Function
|   |-- Models.py # Global Model Structures
|   |-- test_HDBSCAN.py # Test if the HDBSCAN library is successfully installed
|-- 5.PFLAD # Privacy-Preserving FLAD
|   |-- Attack.py # Attack Methods
|   |-- clients.py # Client Processing Code
|   |-- encrypted_CKKS.py # CKKS cryptographic processing code
|   |-- getData.py # Getting Data
|   |-- main.py # PFLAD Main Function (For MNIST dataset only)
|   |-- Models.py # Global Model and Feature Extraction Model Structures
|   |-- test_agg.py # Test if the tenseal library (for CKKS) is successfully installed
|-- data # Source Data
|   |-- MNIST 
|   |-- CIFAR_10
```

## Requirements

The code of this repository is written in Python. We use conda to manage the Python dependencies. Please download and install conda first.

After cloning this repository, change the working directory to the cloned directory.

We use Python version 3.7.8 and Cuda 11.0. Create a new conda environment and install the dependencies:

```shell
Package            Version
------------------ ------------
pandas             0.23.4
numpy              1.16.5
torch              1.10.1
torchvision        0.11.2
scikit-learn       0.22.1
opencv-python      3.4.3.18
hdbscan            0.8.26 # For HDBSCAN in FLAME
tenseal            0.3.14 # For CKKS in PFLAD
```

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

