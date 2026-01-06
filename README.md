DiSE
==============================
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.placeholder.svg)](https://doi.org/10.5281/zenodo.18137788)

## 📋 Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Data Availability](#data-availability)
4. [Usage](#usage)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Overview
### A multimodal model for automatic structure elucidation of organic compounds

### *<ins>Di</ins>ffusion-based <ins>S</ins>tructure <ins>E</ins>lucidation (DiSE)*

*DiSE* is a diffusion probabilistic model for automatic structure elucidation of organic compounds. DiSE follows a two-stage workflow—edge-noise injection and subsequent denoising—and employs a tailored graph representation that fully leverages all available spectra, including MS, <sup>1</sup>H and <sup>13</sup>C chemical shifts, HSQC and COSY.

![DiSE](https://github.com/user-attachments/assets/e2ae3a50-e473-419c-8f76-ba948152be75)

Details can be found at https://arxiv.org/abs/2510.26231.




## Installation
We recommend using `conda` to manage the environment to ensure reproducibility.
```bash
# 1. Clone the repository
git clone https://github.com/Anan-Wu-XMU/DiSE.git
cd DiSE
cd src # Move to the src directory


# 2. Create the conda environment
conda create -n DiSE_env python=3.10
conda activate DiSE_env

pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install hydra-core --upgrade
pip install pytorch_lightning packaging tqdm protobuf wandb pandas imageio rdkit

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install torch_geometric

```

## Data Availability
QM9-NMR Dataset: The dataset used in this work is publicly available at [Zenodo](https://doi.org/10.5281/zenodo.18137788) and as raw data at [moldis-group.github.io](https://moldis-group.github.io/qm9nmr/).

Pre-trained Weights: Download the model checkpoints (DiSE_best.ckpt) from [Link to Zenodo](https://doi.org/10.5281/zenodo.18137788).

Directory Structure: After downloading, the data/model folder as follows:

```text
DiSE/
├── configs/
│   ├── bellissinanes/  # Configuration files for natural products
│   ├── datasets/      # Dataset default configurations
│   ├── experiments/   # Experiment configurations
│   ├── general/      # General configurations
│   ├── model/       # Model default configurations
│   ├── PA/         # Configurations for total synthesis molecules
│   └── train/      # Training configurations
├── data/
│   ├── qm9_nmr/      # QM9-NMR dataset
│
├── src/
│   ├── analysis/        # Analysis scripts
│   ├── datasets/       # Dataset handling scripts
│   ├── diffusion/      # Diffusion model implementation
│   ├── metrics/       # Evaluation metrics
│   ├── models/       # Model architectures
│   ├── Pretrained/   # Pretrained model utilities
│       ├── pcqm4mv2nmr-advanced-best
│           └── checkpoints
│               └── pcqm4mv2nmr-advanced  # !!! This folder should contain the downloaded model checkpoint DiSE_best.ckpt
│   ├── utils/        # Utility functions
│   └── main.py      # !!! Main entry point for training and inference
└── README.md


```
## Usage
### 1. Breif Introduction of Input
Examples can be found in the `configs/bellissinanes` directory (with the file extension *.yaml). Users only need to input the molecular formula (inferred from MS), the corresponding 13C and 1H chemical shifts, and the HH COSY data. 

!!! It's important to note that when inputting the HH COSY data, you should input the chemical shifts of the carbons directly bonded to the hydrogens, not the chemical shifts of the hydrogens themselves.

### 2. Inference (Predicting Structure from Spectra)
To elucidate a structure (example for Bellissinanes-1):

!!! sure that you have placed the downloaded model checkpoint `DiSE_best.ckpt` in the correct directory as shown above.

!!! must change the .yaml configuration files according to your actual absolute paths before running the commands.

```bash
python main.py +bellissinanes=1-639.yaml 
```

### 3. Training
To train DiSE on the QM9-NMR dataset:
```bash
python main.py
```

## Citation
If you find DiSE useful in your research, please consider citing the following paper:
```
@misc{chen2025disediffusionprobabilisticmodel,
      title={DiSE: A diffusion probabilistic model for automatic structure elucidation of organic compounds}, 
      author={Haochen Chen and Qi Huang and Anan Wu and Wenhao Zhang and Jianliang Ye and Jianming Wu and Kai Tan and Xin Lu and Xin Xu},
      year={2025},
      eprint={2510.26231},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2510.26231}, 
}
```


## Acknowledgements
Code development based on
- [DiGress](https://github.com/cvignac/DiGress)


