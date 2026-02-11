<h1 align="center">Evaluating Nuisance Novelty to Expose Gaps in<br>Open Set Reliability</h1>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/ECCV-2026-blue.svg" alt="ECCV 2026"></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.8+-green.svg" alt="Python 3.8+"></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-1.10+-orange.svg" alt="PyTorch"></a>

</p>

<p align="center">
  <b>A dataset-agnostic framework for generating Locally Nuisanced (LN) benchmarks<br>that isolate novelty detector failure from classifier failure in Open Set Recognition.</b>
</p>

<p align="center">
  <a href="#">[Paper]</a> •
  <a href="#">[Project Page]</a> •
  <a href="#datasets">Datasets</a> •
  <a href="#pretrained-models">Models</a>
</p>

---


<p align="center"><i>
<b>Nuisance Novelty</b> arises when a known input is correctly classified but incorrectly rejected as unknown. Our LN framework generates benchmarks that preserve classifier accuracy while exposing detector-specific failures invisible in prior evaluations.
</i></p>

---

## Abstract

Open Set Recognition (OSR) systems pair a closed-set classifier with a novelty detector (post-processor) that decides whether to accept or reject predictions. Existing benchmarks rely on global corruptions that degrade both components simultaneously, **conflating classifier failure with detector failure**.

We introduce **nuisance novelty**—a failure mode where the classifier remains correct, but the novelty detector erroneously flags the sample as unknown. Our framework generates **Locally Nuisanced (LN)** variants by:

1. Identifying classifier-competent regions via reconstruction-based competency
2. Applying controlled local corruptions to those regions
3. Retaining only samples that preserve strict closed-set correctness

We release three benchmarks—**ImageNet-LN**, **Cars-LN**, and **CUB-LN**—exposing widespread nuisance novelty invisible in prior evaluations.

---

## News

---

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/anonymous/Local-Nuisance-Generation.git
cd Local-Nuisance-Generation

# Create conda environment (recommended)
conda create -n ln python=3.8 -y
conda activate ln

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=1.10
torchvision>=0.11
numpy>=1.20
Pillow>=8.0
PyYAML>=5.4
scikit-image>=0.18
opencv-python>=4.5
matplotlib>=3.4
tqdm>=4.60
```

---

## Data Format

This framework requires image lists in the **imglist format**. Each line contains a relative image path followed by its class label:

```
path/to/image1.jpg 0
path/to/image2.jpg 0
path/to/image3.jpg 1
...
```

You can use the [OpenOOD benchmark](https://github.com/Jingkang50/OpenOOD) data structure directly, or convert your dataset to this format.

### Example imglist file

```
imagenet/val/n01440764/ILSVRC2012_val_00000293.JPEG 0
imagenet/val/n01440764/ILSVRC2012_val_00002138.JPEG 0
imagenet/val/n01443537/ILSVRC2012_val_00000236.JPEG 1
```

---

### Directory Structure

After downloading, organize your data as follows:

```
data/
├── images_largescale/
│   └── imagenet/
│       ├── train/
│       └── val/
├── benchmark_imglist/
│   └── imagenet/
│       ├── train_imagenet.txt
│       └── val_imagenet.txt
```

---

## Quick Start

### Generate LN Variants (Single Image)

```python
import torch
from torchvision import transforms
from PIL import Image

from ln_dataset.core.autoencoder import StandardAE
from ln_dataset.core.masks import generate_reconstruction_mask
from ln_dataset.nuisances import LocalNoiseNuisance

# Load image
img = Image.open("sample.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
img_tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]

# Load autoencoder
ae = StandardAE()
ae.load_state_dict(torch.load("assets/imagenet_ae/model.pth"))
ae.eval()

# Generate competency mask
mask = generate_reconstruction_mask(ae, img_tensor, target_area=0.33)

# Apply local nuisance
nuisance = LocalNoiseNuisance()
ln_image = nuisance.apply(img_tensor, mask, manual_param=0.5)  # 50% severity
```

### Run Full Pipeline

```bash
python -m ln_dataset.core.run_pipeline --config configs/imagenet.yaml
```

---

## Full Pipeline

The LN generation pipeline consists of four steps:

### Step 1: Train Autoencoder (or download pre-trained)

```bash


python -m ln_dataset.core.train_ae \
    --config path_to_config \
    --data path_to_data \
    --imglist path_to_imglist \
    --epochs 50 \
    --save_dir assets/imagenet_ae/
```

### Step 2: Calibrate PaRCE

Calibrate the competency scoring on training data:

```bash
python -m ln_dataset.core.calibrate_parce \
    --config path_to_config \
    --data path_to_data  \
    --imglist path_to_imglist \
    --ae_weights assets/imagenet_ae/model.pth \
    --save_path assets/imagenet_ae/parce_calib.pth \
    --samples 20000
```

### Step 3: Calibrate Quantile Bins

Compute severity bin edges from training distribution:

```bash
python -m ln_dataset.core.calibrate_bins \
    --config path_to_config \
    --data path_to_data \
    --imglist path_to_imglist
    --ae_weights assets/imagenet_ae/model.pth \
    --parce_calib assets/imagenet_ae/parce_calib.pth \
    --save_json assets/imagenet_ae/bin_edges.json \
    --samples 5000
```

### Step 4: Generate LN Dataset

Generate locally nuisanced variants on validation data:

```bash
python -m ln_dataset.core.generate_ln \
    --config path_to_config \
    --data path_to_data \
    --imglist path_to_imglist \
    --ae_weights assets/imagenet_ae/model.pth \
    --parce_calib assets/imagenet_ae/parce_calib.pth \
    --bin_edges_json assets/imagenet_ae/bin_edges.json \
    --out_dir data/ln_benchmarks/imagenet_ln \
    --debug_max 10
```


## Citation

If you find this work useful, please cite:

```bibtex

```

