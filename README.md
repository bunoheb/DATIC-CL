DATIC-CL: Difficulty-Aware Textual Image Classification with Curriculum Learning
Official implementation of the paper:
"DATIC-CL: Difficulty-Aware Textual Image Classification with Curriculum Learning"

This repository provides a unified framework for curriculum learning strategies in document image classification:
Base (no curriculum)
PreCL (predefined schedules: step, linear, root)
AutoCL (dynamic difficulty measurer and schedules: ACL, SPL)
All methods share a common data loader, backbone networks, and trainer implementation (curriculum/algorithms).
---
Experimental Environment
All experiments in the paper were conducted on the following hardware and software setup:
Hardware: Windows 11 Home (64-bit), NVIDIA RTX 3090 GPU, AMD Ryzen 7 7800X3D 8-Core CPU (4.20 GHz), 32 GB RAM
Software: Python 3.9.18, PyTorch 2.5.1+cu121, CUDA 12.1
Note: Other platforms (Linux, macOS) are supported as long as the dependencies are satisfied.
---
Installation
Using Conda (recommended)
```
conda create -n datic-cl python=3.9 -y
conda activate datic-cl 
pip install -r requirements.txt
```
Ensure PyTorch matches your CUDA version.
For CUDA 12.1, install PyTorch as follows:
```
pip install torch==2.5.1+cu121 torchvision==0.16.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```
If using a different CUDA version, adjust the version and index URL accordingly. See:
https://pytorch.org/get-started/locally/
Note: requirements.txt does not include the torch and torchvision lines by default, because they vary by CUDA version. Instead, they are installed separately as shown above.
---
Dataset
This project uses the RVL-CDIP dataset.
Main experiments: 100,000 sampled images with difficulty scores  
→ `ref/data\\\_with\\\_combined\\\_difficulty.csv` (included in supplementary material)
Quick testing: 160-sample lightweight CSV  
→ `data/data\\\_with\\\_combined\\\_difficulty.csv` (used by default)
You can also load RVL-CDIP directly from Hugging Face:  
https://huggingface.co/datasets/aharley/rvl_cdip
Once the sampled dataset (e.g., 100K stratified subset) is prepared,
run `preprocessing.py` to compute difficulty scores and generate the training CSV.
All training scripts support `--use-huggingface` to bypass local CSVs.
---
Directory Structure
project_root/
├── train_WithoutCL.py         # Base (no curriculum) training (CNN, ResNet34)
├── train_PreCL.py             # Predefined curriculum (step, linear, root)
├── train_AutoCL.py            # Auto curriculum (SPL, ACL)
├── evaluate_compare.py        # Evaluation script for all methods
│
├── curriculum/                # Core library implementation for curriculum strategies
│   ├── datasets/		# Dataset loaders
│   │   ├── init.py
│   │   ├── rvlcdip.py
│   │   ├── custom_dataset.py
│   │   ├── document_dataset.py
│   │   └── utils.py
│   ├── trainers/		 # Classifier trainer
│   │   ├── init.py
│   │   └── image_classifier.py
│   ├── algorithms/		# Curriculum algorithm
│   │   ├── init.py
│   │   ├── base.py
│   │   ├── predefined.py
│   │   ├── self_paced.py
│   │   └── adaptive.py
│   ├── backbones/			# CNN and ResNet backbones
│   │   ├── init.py
│   │   ├── resnet.py
│   │   └── convnet.py
│   └── utils/			 # Logging, random seed utils
│       ├── init.py
│       ├── log.py
│       └── rand.py
│
├── data/
│   └── data_with_combined_difficulty.csv     # 160-sample version for quick reproduction
│
├── results/                 # Evaluation outputs
├── runs/                    # Training logs and model checkpoints
├── temps/                   # Temporary files (replay buffer, metadata, etc.)
│
├── preprocessing.py         # Tool to compute difficulty scores and generate dataset
├── README.md                # Main documentation
└── requirements.txt         # Python dependencies
This structure supports modular training and evaluation of multiple curriculum learning strategies for document image classification.
---
.gitignore (Optional but Recommended)
To keep your repository clean and focused, you may want to ignore the following files:
Python artifacts
pycache/
*.py[cod]
*.ipynb_checkpoints/
Data and logs
runs/
results/
temps/
*.log
*.pth
*.csv
*.xlsx
OS metadata
.DS_Store
---
Example Commands
Below are example commands to run each curriculum strategy. All scripts support common arguments such as --seed, --backbone, and --use-huggingface.
Preprocessing:
```
python preprocessing.py
```
Training
(1) Run baseline (e.g., convnet):
```
python train\\\_WithoutCL.py --data rvl --net resnet34 --seed 42 --epochs 150 --batch-size 64 --lr 1e-4
```
(2) Run PreCL (e.g., step):
```
python train\\\_PreCL.py --method PreCL --data rvl --net convnet --seed 42 --epochs 150 --batch-size 64 --lr 1e-4 --func-type step --num-steps 50 --epochs-per-step 3
```
(3) Run AutoCL (e.g., SPL):
```
python train\\\_AutoCL.py --variant SPL --data rvl --net convnet --seed 42 --epochs 150 --batch-size 64 --lr 1e-4 --start-rate 0.5 --grow-epochs 149 --grow-fn linear --weight-fn hard
```
3-1. Evaluate Best Models (Recommended)
To evaluate the best checkpoint from each training run:
```
python evaluate\\\_compare.py --root runs --method-filter Base,PreCL,AutoCL --data-filter rvl --which best --latest-only
```
3-2. Custom Evaluation with Specified Model Files
You can also manually evaluate saved models (e.g., from ref/ref_models/):
```
python evaluate\\\_compare.py --root runs --method-filter Base,PreCL,AutoCL --data-filter rvl --latest-only --which best --manifest "ref/ref\\\_models.csv"
```
This is useful for verifying results without requiring retraining.
---
Environment
matplotlib==3.6.3  
numpy==1.24.0  
pandas==1.5.3  
scikit-learn==1.1.3  
torch==2.5.1+cu121  
torchvision==0.16.1+cu121
---
Acknowledgements
This project extends and builds upon the CurML library. CurML provided the foundational curriculum learning framework, which was adapted and expanded for textual image classification tasks in this project.
If you find the original CurML code helpful, please cite the following paper:
@inproceedings{zhou2022curml,
title={CurML: A Curriculum Machine Learning Library},
author={Zhou, Yuwei and Chen, Hong and Pan, Zirui and Yan, Chuanhao and Lin, Fanqi and Wang, Xin and Zhu, Wenwu},
booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
pages={7359--7363},
year={2022}
}
---
Supplementary Material
Full training code and a lightweight dataset are included in this repository.
Pretrained models will be made available after the review process.
