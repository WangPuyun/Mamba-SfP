<div align="center">

# Mamba-SfP

**A Mamba-based framework for underwater Shape-from-Polarization (SfP) normal reconstruction**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/WangPuyun/Mamba-SfP.git)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![arXiv](https://img.shields.io/badge/arXiv-TBD-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/TBD)
[![Elsevier](https://img.shields.io/badge/Elsevier-TBD-FF6C00?logo=elsevier&logoColor=white)](https://www.sciencedirect.com/science/article/pii/TBD)

</div>

<p align="center">
  <img src="./README_img/3D.gif" alt="Mamba-SfP Demo" width="92%" />
</p>

---

## 🌊 Overview

`Mamba-SfP` is a research codebase for 3D shape from polarization (SfP) normal estimation via selective state space models.


## 🧠 Architecture

<p align="center">
  <img src="./README_img/Network.png" alt="Mamba-SfP Network Architecture" width="95%" />
</p>

## ✨ Highlights

- Explored the application of Mamba in the field of shape from polarization.
- Achieves high computational efficiency with only 33.24M parameters and 13.96 GFLOPs.


## ⚙️ Environment Setup

### Option 1: Conda (recommended)

```bash
conda create -n mamba-sfp python=3.10 -y
conda activate mamba-sfp
pip install -r requirements.txt
```

### Option 2: Use provided `environment.yml`

```bash
conda env create -f environment.yml
conda activate tongyong
```

## 🧩 Data Preparation

Put dataset files in:

```text
./Underwater Dataset/Baseline_Data/
```

Required CSV index files:

- `Dataset/train_list.csv`
- `Dataset/val_list.csv`
- `Dataset/test_list.csv`

## 🚀 Training

Run distributed training:

```bash
python train.py --train_batch_size 24 --val_batch_size 4 --epochs 1000 --checkpoints_dir ./pt/Mamba/
```

Resume from a checkpoint:

```bash
python train.py --model_name /700.pth --checkpoints_dir ./pt/Mamba/
```

TensorBoard logs are written to `./runs` by default.

## 🔍 Evaluation and Visualization

Generate predicted normal maps and angular error maps:

```bash
python Angle_error_map.py --ckpt_path ./pt/Mamba/700.pth --results_dir ./results_sfp --error_maps_dir ./error_maps --summary_path ./table1_metrics.txt --nprocs 1
```

Outputs:

- Predicted normal maps: `./results_sfp`
- Error heatmaps: `./error_maps`
- Quantitative summary: `./table1_metrics.txt`

Generate Grad-CAM visualizations:

```bash
python CAM_map.py
```

Outputs:

- Predicted normal maps: `./results_sfp`
- Grad-CAM maps: `./error_maps/*_gradcam.png`
- Grad-CAM overlay images: `./error_maps/*_gradcam_overlay.png`


## 📚 Citation

Paper link: `TBD`

```bibtex
@article{mamba_sfp_2026,
  title   = {TBD},
  author  = {TBD},
  journal = {TBD},
  year    = {2026}
}
```
