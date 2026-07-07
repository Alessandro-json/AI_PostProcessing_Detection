# AI_PostProcessing_Detection
# Joint Detection of AI-Generated Images and Post-Processing Alterations

**Computer Vision — Prof. Irene Amerini, Spring 2026**
**Sapienza University of Rome**

Alessandro Esposito · Alessandro Piccolino · Annalisa Verrando

---

## Overview

This project implements a unified multi-task deep learning framework that simultaneously addresses two forensic questions given a single input image:

1. **Real or fake?**: is the image a genuine photograph or AI-generated?
2. **What post-processing was applied?**: original, internet-transmitted, or re-digitized?

The core observation motivating this work is that AI-generated image detectors are typically evaluated on clean, uncompressed images. In practice, images circulating online have almost always undergone some form of post-processing, such as compression, re-uploading, re-digitization, that disrupts detector performance. This project investigates whether jointly predicting authenticity and post-processing type leads to more robust detection.

---
## Experimental roadmap

We first trained RGB single-task baselines for fake detection and transformation classification, then compared them with RGB multi-task models using different loss weightings. This allowed us to evaluate whether the two tasks benefit from being learned jointly.

Since the RGB multi-task experiments showed trade-offs between fake detection and transformation classification, we extended the multi-task setting with additional cues. Frequency information was introduced to better capture post-processing traces, while depth maps were added to provide geometric information about scene structure. We then combined depth and frequency in a shared multimodal model and tested a gated fusion variant, where modality weights are learned separately for the two heads.

Finally, ViT-based models were evaluated as an alternative to the ResNet backbone, in order to test whether global patch-level attention could improve the joint detection task.

## Dataset

**RRDataset**: a real-world robustness benchmark combining real photographs and AI-generated images across three post-processing conditions:

| Split | Description |
|---|---|
| Original | Images closest to ideal benchmark conditions |
| Internet-transmitted | Platform compression, resizing, social-media artifacts |
| Re-digitized | Re-photographed, scanned, printed or displayed |

### Subset used

A balanced subset of 3,000 images was selected, preserving both supervised dimensions:

```
Train:      2,100 images  (350 per fake × transform cell)
Validation:   450 images  (75 per cell)
Test:         450 images  (75 per cell)

Fake balance:      50% real / 50% AI-generated
Transform balance: 1,000 images per condition
```

---

## Repository structure

```
src/
├── dataset.py                        # RGB dataset and transforms
├── dataset_freq.py                   # RGB + FFT frequency dataset
├── dataset_depth_frequency.py        # RGB + depth + frequency dataset
│
├── model_RGB.py                      # ResNet18 baseline (single/multi-task)
├── model_freq.py                     # ResNet18 + FreqEncoder (FFT branch)
├── model_depth_frequency.py          # ResNet18 + depth + frequency
├── model_depth_frequency_gated.py    # Resnet18 gated RGB + depth + frequency 
├── model_vit_RGB.py                  # ViT-Small RGB baseline
├── model_vit_depth_frequency.py      # ViT-Small + depth + frequency
│
├── train_RGB.py                      # Train RGB baseline (all tasks/weights)
├── train_depth_frequency.py          # Train RGB + depth + frequency
├── train_depth_frequency_gated.py    # Train gated RGB + depth + frequency 
├── FrequencyAugumented.py            # Train frequency model (cosine + learned weights)
├── train_vit_RGB.py                  # Train ViT RGB baseline
├── train_vit_RGB_1_2.py              # Train ViT RGB — λ_fake=1.0, λ_transform=2.0
├── train_vit_depth_frequency.py      # Train ViT + depth + frequency
├── train_vit_depth_frequency_1_2.py  # Train ViT D+F — λ_fake=1.0, λ_transform=2.0
│
├── evaluate_RGB.py                   # Evaluate RGB baseline
├── evaluate_freq.py                  # Evaluate frequency model
├── evaluation_depth_frequency.py     # Evaluate RGB + depth + frequency
├── evaluate_vit_RGB.py               # Evaluate ViT RGB
├── evaluate_vit_depth_frequency.py   # Evaluate ViT + depth + frequency
├── evaluation_depth_frequency_gated.py     # Evaluate gated RGB + depth + frequency model
│
├── loss.py                           # UncertaintyWeightedLoss (Kendall et al. 2018)
├── balanced_db.py                    # Stratified subset sampling
├── make_csv.py                       # CSV split generation
├── generate_depth_map.py             # MiDaS depth map precomputation
│
├── results_freq.py                   # Display frequency results in notebook
├── results_vit.py                    # Display ViT results in notebook
└── compare_all.py                    # Full comparison table and plots

data/
└── splits/
    ├── train_balanced.csv
    ├── val_balanced.csv
    └── test_balanced.csv

checkpoints/                          # Saved model checkpoints
results/                              # Evaluation outputs (metrics, confusion matrices)
notebook.ipynb                        # Main Colab notebook
requirements.txt
```

---

## Models

### RGB baseline

A pretrained ResNet18 extracts a shared feature vector. Two independent classification heads predict authenticity and transformation type jointly.

```
Input: RGB image [3, 224, 224]
  ↓ ResNet18 (pretrained ImageNet)
  ↓ 512-dim feature vector
  ├── fake_head  → real / AI-generated
  └── transform_head → original / transmitted / re-digitized
```

Supports three modes: `--task fake`, `--task transform`, `--task multitask`.

### Frequency branch (RGB + FFT)

An FFT log-magnitude spectrum is computed on-the-fly from each image and processed by a lightweight CNN encoder (`FreqEncoder`). The frequency features are fused with the RGB backbone features before the two heads.

```
Input: RGB image
  ├── ResNet18 → 512-dim RGB features
  └── compute_fft_map() → FreqEncoder → 128-dim frequency features
        ↓ concatenation + fusion layer + attention gate
        ├── fake_head
        └── transform_head
```

**Why FFT?** AI-generated images (diffusion models, GANs) leave periodic artifacts in the frequency domain that are invisible in RGB but appear clearly in the log-magnitude spectrum.

The FFT map is computed as: grayscale conversion → `torch.fft.fft2` → `fftshift` → log magnitude → normalization to [0, 1].

### RGB + Depth + Frequency

A three-branch model combining RGB (ResNet18), estimated depth (MiDaS → SmallMapEncoder), and FFT frequency (SmallMapEncoder). Features from all three branches are concatenated, fused, and optionally passed through a sigmoid attention gate before the two heads.

Depth maps are precomputed with MiDaS and stored as `.npy` files. Frequency maps are computed on-the-fly.

### RGB + Depth + Frequency Gated

As a final ResNet-based multimodal experiment, we introduced a gated fusion mechanism on top of the RGB + Depth + Frequency architecture.

The non-gated model uses a single shared fused representation for both tasks. In contrast, the gated model learns task-specific modality weights for RGB, depth, and frequency. This means that the fake detection head and the transformation classification head can rely on different combinations of the three modalities.

For each task, a small gating network outputs three softmax-normalized weights, one for RGB, one for depth, and one for frequency. These weights are used to compute a weighted sum of the modality feature vectors before classification.

This experiment tests whether task-specific multimodal fusion is more effective than a shared multimodal representation. However, the gated model did not improve the overall performance, suggesting that the added complexity did not provide a clear advantage in this setting.

### ViT-Small variants

`vit_small_patch16_224` (via `timm`) replaces ResNet18 as the RGB backbone, keeping the rest of the architecture identical. ViT divides the image into 196 patches of 16×16 pixels and applies global self-attention, allowing each patch to attend to all others — potentially better at capturing long-range periodic artifacts left by generative models.

Available as drop-in replacements for both the RGB baseline and the RGB + Depth + Frequency model.

---

## Multi-task loss

The combined loss is a weighted sum of two cross-entropy terms:

```
L_total = λ_fake · CE_fake + λ_transform · CE_transform
```

Three weighting strategies are supported:

| Strategy | Flag | Description |
|---|---|---|
| Manual 1/1 | `--lambda_fake 1.0 --lambda_transform 1.0` | Equal weight |
| Manual 1/2 | `--lambda_fake 1.0 --lambda_transform 2.0` | More weight on transform |
| Manual 2/1 | `--lambda_fake 2.0 --lambda_transform 1.0` | More weight on fake |
| Learned | `--loss_weighting learned` | Kendall et al. 2018 uncertainty weighting |

---

## Setup

```bash
git clone https://github.com/Alessandro-json/AI_PostProcessing_Detection
cd AI_PostProcessing_Detection
pip install -r requirements.txt
pip install timm   # required for ViT models
```

The project is designed to run on **Google Colab free tier** (T4 GPU). 
---

## Usage

### Prepare data

```bash
# Generate balanced CSV splits
python src/balanced_db.py --image_root data/raw/RRDataset_subset

# Precompute depth maps (required for depth models only)
python src/generate_depth_map.py \
    --image_root data/raw/RRDataset_subset \
    --output_root data/depth_maps
```

### Train

```bash
# RGB multitask baseline (λ=1/1)
python src/train_RGB.py \
    --task multitask \
    --train_csv data/splits/train_balanced.csv \
    --val_csv   data/splits/val_balanced.csv \
    --image_root data/raw/RRDataset_subset \
    --epochs 10 --batch_size 32 --num_workers 0 \
    --lambda_fake 1.0 --lambda_transform 1.0 \
    --checkpoint_name best_rgb_multitask_1_1.pt

# Frequency model (cosine scheduler + learned weights)
python src/FrequencyAugumented.py \
    --train_csv data/splits/train_balanced.csv \
    --val_csv   data/splits/val_balanced.csv \
    --image_root data/raw/RRDataset_subset \
    --epochs 10 --batch_size 32 --num_workers 0 \
    --scheduler cosine --warmup_epochs 2 \
    --loss_weighting learned \
    --checkpoint_name best_freq_learned_cosine.pt

# RGB + Depth
python src/train_depth.py \
  --train_csv data/splits/train_balanced.csv \
  --val_csv   data/splits/val_balanced.csv \
  --image_root data/raw/RRDataset_subset \
  --depth_root drive/MyDrive/CV_Project/depth_maps \
  --checkpoint_name best_depth_uncertainty.pt \
  --epochs 10 --batch_size 32 --num_workers 0 \
  --use_uncertainty_weighting \
  --no_edge

# RGB + Depth + Frequency
python src/train_depth_frequency.py \
  --train_csv data/splits/train_balanced.csv \
  --val_csv   data/splits/val_balanced.csv \
  --image_root data/raw/RRDataset_subset \
  --depth_root drive/MyDrive/CV_Project/depth_maps \
  --checkpoint_name best_depth_frequency_uncertainty.pt \
  --epochs 10 --batch_size 32 --num_workers 0 \
  --use_uncertainty_weighting 

# ViT RGB multitask (λ=1/2) 
python src/train_vit_RGB_1_2.py \
    --train_csv data/splits/train_balanced.csv \
    --val_csv   data/splits/val_balanced.csv \
    --image_root data/raw/RRDataset_subset \
    --epochs 10 --batch_size 16 --num_workers 0 \
    --checkpoint_name best_vit_rgb_multitask_1_2.pt
```

### Evaluate

```bash
# RGB
python src/evaluate_RGB.py \
    --task multitask \
    --csv_path data/splits/test_balanced.csv \
    --image_root data/raw/RRDataset_subset \
    --checkpoint checkpoints/best_rgb_multitask_1_1.pt \
    --output_dir results/rgb_multitask_1_1 \
    --batch_size 32 --num_workers 0

# Frequency
python src/evaluate_freq.py \
    --csv_path data/splits/test_balanced.csv \
    --image_root data/raw/RRDataset_subset \
    --checkpoint checkpoints/best_freq_learned_cosine.pt \
    --output_dir results/freq_learned_cosine \
    --batch_size 32 --num_workers 0

# RGB + Depth
python src/evaluate_depth.py \
  --csv_path data/splits/test_balanced.csv \
  --image_root data/raw/RRDataset_subset \
  --depth_root drive/MyDrive/CV_Project/depth_maps \
  --checkpoint checkpoints/best_depth_uncertainty.pt \
  --output_dir results/depth_only \
  --batch_size 32 --num_workers 0

# RGB + Depth + Frequency
python src/evaluation_depth_frequency.py \
  --csv_path data/splits/test_balanced.csv \
  --image_root data/raw/RRDataset_subset \
  --depth_root drive/MyDrive/CV_Project/depth_maps \
  --checkpoint checkpoints/best_depth_frequency_uncertainty.pt \
  --output_dir results/depth_frequency_uncertainty \
  --batch_size 32 --num_workers 0

# ViT
python src/evaluate_vit_RGB.py \
    --task multitask \
    --csv_path data/splits/test_balanced.csv \
    --image_root data/raw/RRDataset_subset \
    --checkpoint checkpoints/best_vit_rgb_multitask_1_2.pt \
    --output_dir results/vit_rgb_multitask_1_2 \
    --batch_size 16 --num_workers 0
```


---

## Key results

| Model | Fake acc. | Transform acc. |
|---|---|---|
| RGB single-task (fake-only) | 92.67% | — |
| RGB single-task (transform-only) | — | 81.33% |
| RGB multitask 1/1 | 92.22% | 78.22% |
| RGB multitask 1/2 (best RGB) | 92.67% | 82.22% |
| RGB multitask 2/1 | 92.47% | 80.22% |
| RGB multitask learned | 92.22% | 78.34% |
| Frequency 1/1 | 90.22% | 82.44% |
| RGB + Depth | 93.56% | ~80% |
| RGB + Depth + Frequency | 92.44% | 81.67% |
| ViT RGB | 93.56% | 86.22% |
| ViT RGB + Depth + Frequency | 92.44% | 83.33% |
| RGB + Depth + Frequency Gated | 91.56% | 76.00% |
Re-digitization is consistently the hardest condition for real/fake detection across all models. The two tasks show mild competition: configurations that improve transformation accuracy tend to slightly decrease fake accuracy.

---

## Ablation study

The ablation study on loss weights (λ_fake / λ_transform) shows that the 1/2 configuration is the best RGB setting, improving transformation classification while preserving fake detection performance. Learned uncertainty weighting did not consistently improve results over the best manual configuration.

The depth branch produces a strong and stable multimodal baseline. The explicit edge-consistency branch (RGB/depth disagreement) improved transformation accuracy but decreased fake accuracy, and was not selected as the primary model.

The ViT-based experiments were introduced as an architectural comparison with the ResNet-based models. Overall, ViT achieved stronger results, especially on transformation classification, suggesting that global patch-level attention is more effective for capturing long-range visual and forensic patterns in this joint detection setting.

---

## References

- Li, Chunxiao, et al. "Bridging the Gap Between Ideal and Real-world Evaluation: Benchmarking AI-Generated Image Detection in Challenging Scenarios." ICCV 2025.
- Shao, Rui, Tianxing Wu, and Ziwei Liu. "Detecting and grounding multi-modal media manipulation." CVPR 2023.
- Kendall, Alex, Yarin Gal, and Roberto Cipolla. "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics." CVPR 2018.
