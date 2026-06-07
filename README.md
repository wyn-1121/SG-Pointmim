# SG-PointMIM: Spectral Self-Guided Masking and Dual-Target Reconstruction for Point Cloud Representation Learning

This repository contains the official implementation of **SG-PointMIM**.

---

## 🌟 Overview

Masked Point Modeling (MPM) is a powerful paradigm for self-supervised 3D representation learning. However, existing methods are fundamentally bottlenecked by content-agnostic random masking, which ignores the irregular topology of point clouds, and single reconstruction objectives, which fail to balance fine-grained geometry with high-level semantics. 

To overcome these limitations, we propose **SG-PointMIM**, a unified pre-training framework that integrates **Spectral Self-Guided Masking (SGM)** and **Dual-Target Reconstruction**.

### Key Contributions
1. **Spectral Self-Guided Masking (SGM):** Leveraging spectral graph theory, SGM constructs a patch affinity graph and identifies topologically salient regions via spectral partitioning, forcing the encoder to perform global structural reasoning.
2. **Dual-Target Reconstruction:** Our dual-target decoder jointly optimizes coordinate-level geometric regression and teacher-guided semantic alignment, achieving a powerful synergy between spatial precision and high-dimensional semantic discriminability.
3. **State-of-the-Art Performance:** SG-PointMIM achieves a state-of-the-art accuracy of **88.1%** on the challenging real-world ScanObjectNN benchmark.

---

## 🛠️ Installation

### Environment Setup
```bash
# Clone the repository
git clone [https://github.com/wyn-1121/SG-Pointmim.git](https://github.com/wyn-1121/SG-Pointmim.git)
cd SG-Pointmim

# Create a virtual environment
conda create -n sg_pointmim python=3.8 -y
conda activate sg_pointmim

# Install PyTorch (adjust according to your CUDA version)
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 itsdangerous cudatoolkit=11.3 -c pytorch

# Install extensions
pip install -r requirements.txt
