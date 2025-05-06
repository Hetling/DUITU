## Real-Time Image Segmentation on CamVid via U-Net Pruning
#### Optimizing inference speed for autonomous driving applications using pruning techniques.

### 📌 Overview
This project focuses on accelerating U-Net-based image segmentation on the CamVid dataset to achieve real-time performance (≥30 FPS) using pruning. The pruned model reduces computational overhead while maintaining competitive accuracy, making it suitable for autonomous driving systems.

Key Features:

✅ U-Net variants with configurable kernel sizes and upscaling.

✅ Magnitude-based pruning for model compression.

✅ Inference time benchmarking scripts.

✅ Pre-trained models (pruned and quantized).

📂 Repository Structure
```bash
.
├── data/                   # CamVid dataset (train/val/test splits + labels)
│   ├── class_dict.csv      # Class-ID mappings
│   ├── train/              # Training images
│   ├── train_labels/       # Training masks
│   └── ...                 
├── models/                 # U-Net implementations
│   ├── Unet.py             # Base U-Net
│   ├── UNetKernelSize.py   # U-Net with customizable kernels
│   └── UnetUpscale.py      # U-Net with upscaling tweaks
├── scripts/                # Core scripts
│   ├── dataloader.py       # Data loading and preprocessing
│   ├── trainUnet.py        # Training script
│   ├── inferencetimer.py   # Benchmark inference speed
│   ├── quantizeUnet.py     # Post-training quantization
│   └── visualisations.py   # Visualization utilities
├── playground/             # Experimental notebooks/scripts
├── quantized_unet.pth      # Quantized model weights
├── requirements.txt        # Python dependencies
└── 
```

## 🚀 Quick Start
1. Install Dependencies
bash
pip install -r requirements.txt
1. Train U-Net
Train the baseline model:

```bash
python scripts/trainUnet.py --data_dir ./data --epochs 50 --save_path ./scripts/unet_model.pth
```
3. Prune the Model
Run iterative pruning (example):

```python
from scripts.prune import prune_unet
prune_unet(
    model_path="./scripts/unet_model.pth",
    target_sparsity=0.3,  # Remove 30% of filters
    save_path="./scripts/pruned_unet.pth"
)
```
4. Benchmark Inference
```bash
python scripts/inferencetimer.py --model_path ./scripts/pruned_unet.pth --data_dir ./data/val
```
Outputs FPS and mIoU metrics.

## 🔧 Key Scripts
Script	Purpose
trainUnet.py	Train U-Net on CamVid.
inferencetimer.py	Measure inference speed (FPS) and accuracy.
quantizeUnet.py	Apply 8-bit quantization for further speedup.
visualisations.py	Generate segmentation masks overlays.
## 📊 Results
Model	mIoU (%)	Inference Time (ms)	FPS
Baseline U-Net	75.0	20	50
Pruned U-Net	72.3	10	100
Quantized U-Net	71.8	8	125
Pruning achieves a 2× speedup with minimal accuracy drop.

## 📖 References
CamVid Dataset

U-Net Paper

PyTorch Pruning Tutorial