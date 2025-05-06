## Real-Time Image Segmentation on CamVid via U-Net Pruning
#### Optimizing inference speed for autonomous driving applications using pruning techniques.

### 📌 Overview
This project focuses on accelerating U-Net-based image segmentation on the CamVid dataset using pruning. The pruned model reduces computational overhead while maintaining accuracy, making it suitable for autonomous driving systems.

Key Features:

✅ U-Net variants with different kernel sizes and upscaling.

✅ Pruning techniques to reduce model size and improve inference speed.

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
```bash
pip install -r requirements.txt
```
1. Train U-Net
Train the baseline model:

```bash
python scripts/trainUnet.py 
```
3. Prune the Model
Run iterative pruning (example):

```python
from scripts.prune import structured_filter_prune
structured_filter_prune(model_copy, amount=amount)
```
4. Benchmark Inference
```bash
python scripts/inferencetimer.py
```
Outputs inference benchmark.

## 📊 Results


## 📖 References
CamVid Dataset

U-Net Paper

PyTorch Pruning Tutorial