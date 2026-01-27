# HypCBM: Hyperbolic Concept Bottleneck Models

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

This repository contains the official implementation of **Hyperbolic Concept Bottleneck Models (HypCBM)**, a novel approach that leverages hyperbolic geometry to encode hierarchical concept relationships for interpretable image classification.

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training Models](#training-models)
  - [Training HypCBM (Entailment-based)](#training-hypcbm-entailment-based)
  - [Training PCBM Baseline](#training-pcbm-baseline)
- [Encoding Concepts](#encoding-concepts)
- [Evaluation](#evaluation)
  - [Hierarchical Consistency](#hierarchical-consistency)
  - [OOD Robustness](#ood-robustness)
- [Interactive Interventions](#interactive-interventions)
- [Supported Datasets](#supported-datasets)
- [Example SLURM Jobs](#example-slurm-jobs)
- [Citation](#citation)
- [License](#license)

---

## Overview

HypCBM extends Concept Bottleneck Models to hyperbolic space, enabling:

- **Hierarchical Concept Representations**: Concepts are embedded in hyperbolic space where parent-child relationships naturally emerge through entailment cones
- **Entailment-based Activations**: Image-concept similarity is computed via hyperbolic entailment, capturing "is-a" relationships
- **Propagated Interventions**: When a parent concept is modified, child concepts are automatically updated following the hyperbolic hierarchy
- **Improved Interpretability**: The learned concept hierarchies provide intuitive explanations for model predictions

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU support)
- Conda (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/daniuyter/hypcbm.git
cd hypcbm

# Create conda environment
conda create -n hypcbm python=3.10
conda activate hypcbm

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Download HyCoCLIP/MERU/CLIP checkpoints (place in hycoclip/)
# See hycoclip/model-zoo.md for download links
```

## Project Structure

```
hypcbm/
├── concepts/               # Concept bank utilities
├── data/                   # Dataset loaders (CIFAR, ImageNet, CUB, SUN397)
├── eval/                   # Evaluation scripts
│   ├── eval_ood_robustness.py
│   └── hierarchical_consistency_eval.py
├── hycoclip/               # HyCoCLIP/MERU backbone models
│   ├── configs/            # Model configuration files
│   ├── hycoclip/           # Core hyperbolic CLIP implementation
│   └── *.pth               # Pre-trained checkpoints
├── models/                 # Model definitions
│   ├── entailment_cbm.py   # Entailment cone feature computation
│   ├── model_zoo.py        # Backbone loading utilities
│   └── pcbm_utils.py       # Post-hoc CBM utilities
├── training_tools/         # Training utilities
├── utils/                  # General utilities
│   ├── common.py           # Shared helper functions
│   ├── concept_utils.py    # Concept loading/encoding
│   ├── hyperbolic_interventions.py
│   └── intervention_utils.py
├── vocab/                  # Concept vocabulary files
│   ├── concept_names_cifar10_hycoclip.txt
│   ├── concept_names_cifar100_hycoclip.txt
│   ├── concept_names_imagenet_hycoclip.txt
│   └── concept_names_sun397_hycoclip.txt
├── encode_concepts.py      # Encode concepts
├── train_hypcbm.py         # Train HypCBM (entailment-based)
├── train_pcbm_baseline.py  # Train Euclidean PCBM baseline
├── interactive_interventions.ipynb  # Interactive demo notebook
└── requirements.txt
```

## Quick Start

### 1. Encode Concepts

First, encode your concept vocabulary into hyperbolic embeddings:

```bash
python encode_concepts.py \
  --concepts-txt vocab/concept_names_cifar100_hycoclip.txt \
  --out-dir outputs \
  --backbone-name hycoclip \
  --scenario cifar100 \
  --min-norm 0.27
```

### 2. Train HypCBM

Train the entailment-based HypCBM:

```bash
python train_hypcbm.py \
  --concept-bank outputs/multimodal_concept_hycoclip_cifar100_minnorm:0.27.pkl \
  --backbone-name hycoclip \
  --out-dir outputs \
  --dataset cifar100 \
  --entail-eta 1.5 \
  --lam 1e-5
```

### 3. Explore Interactively

Launch the interactive notebook:

```bash
jupyter notebook interactive_interventions.ipynb
```

---

## Training Models

### Training HypCBM (Entailment-based)

The main training script for HypCBM uses entailment cone activations:

```bash
python train_hypcbm.py \
  --concept-bank <path-to-concept-bank.pkl> \
  --backbone-name hycoclip \
  --out-dir <output-directory> \
  --dataset <dataset-name> \
  --entail-eta <eta-value> \
  --lam <regularization> \
  [--viz-concept-accuracy] \
  [--viz-lambdas 1e-3 1e-4 1e-5]
```

**Key Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--concept-bank` | Path to encoded concept bank (.pkl) | Required |
| `--backbone-name` | Backbone model (`hycoclip`, `meru`, `clip`, `clipvitb`) | `hycoclip` |
| `--dataset` | Dataset name (`cifar10`, `cifar100`, `imagenet`, `sun397`, `cub`) | `cifar100` |
| `--entail-eta` | Entailment cone scaling factor (smaller = stricter) | `1.4` |
| `--lam` | Elastic-net regularization strength | `1e-5` |
| `--alpha` | Elastic-net L1 ratio (1.0 = pure L1) | `0.99` |
| `--train-fraction` | Fraction of training data to use | `1.0` |
| `--viz-concept-accuracy` | Enable accuracy vs. sparsity visualization | `False` |
| `--viz-lambdas` | Lambda values to sweep for visualization | `None` |

clipvitb is the original OpenAI CLIP. for this, you have to uncomment the code below the 'for OpenAI CLIP:' comments.
**Example: CIFAR-100**

```bash
python train_hypcbm.py \
  --concept-bank outputs/multimodal_concept_hycoclip_cifar100_minnorm:0.27.pkl \
  --backbone-name hycoclip \
  --out-dir outputs_cifar100 \
  --dataset cifar100 \
  --entail-eta 1.5 \
  --lam 1e-5 \
  --viz-concept-accuracy \
  --viz-lambdas 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6
```

**Example: ImageNet**

```bash
python train_hypcbm.py \
  --concept-bank outputs_imagenet/multimodal_concept_hycoclip_imagenet_minnorm:0.27.pkl \
  --backbone-name hycoclip \
  --out-dir outputs_imagenet \
  --dataset imagenet \
  --imagenet-root /path/to/imagenet/torchvision_ImageFolder \
  --embedding-cache-dir outputs/visual_embeddings \
  --entail-eta 1.5 \
  --lam 3e-5
```

**Example: SUN397**

```bash
python train_hypcbm.py \
  --concept-bank outputs_sun397/multimodal_concept_hycoclip_sun397_minnorm:0.27.pkl \
  --backbone-name hycoclip \
  --out-dir outputs_sun397 \
  --dataset sun397 \
  --sun397-root data/sun397/SUN_dataset/data \
  --entail-eta 1.4 \
  --lam 1e-5
```

### Training PCBM Baseline

Train the Euclidean PCBM baseline for comparison:

```bash
python train_pcbm_baseline.py \
  --concept-bank <path-to-concept-bank.pkl> \
  --backbone-name clip \
  --out-dir <output-directory> \
  --dataset <dataset-name> \
  --lam <regularization>
```

**Example:**

```bash
python train_pcbm_baseline.py \
  --concept-bank outputs/multimodal_concept_clip_cifar100_minnorm:0.00.pkl \
  --backbone-name clip \
  --out-dir outputs_baseline \
  --dataset cifar100 \
  --lam 1e-5
```

---

## Encoding Concepts

The `encode_concepts.py` script encodes concept names into hyperbolic/Euclidean embeddings using the text encoder.

```bash
python encode_concepts.py \
  --concepts-txt <path-to-concepts.txt> \
  --out-dir <output-directory> \
  --backbone-name <backbone> \
  --scenario <scenario-name> \
  [--min-norm <threshold>]
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--concepts-txt` | Path to text file with one concept per line | Required |
| `--out-dir` | Output directory for the concept bank | Required |
| `--backbone-name` | Backbone to use (`hycoclip`, `meru`, `clip`) | `meru` |
| `--scenario` | Scenario name for output filename | `custom` |
| `--min-norm` | Filter concepts with embedding norm below this | `0.0` |

**Concept File Format:**

Create a text file with one concept per line:

```text
a dog
a cat
fur
whiskers
four legs
domestic animal
```

---

## Evaluation

### Hierarchical Consistency

Evaluate hierarchical consistency:

```bash
python eval/hierarchical_consistency_eval.py \
  --artifact-path outputs/entailment_cbm_cifar100_hycoclip_1frac.pkl \
  --concept-bank outputs/multimodal_concept_hycoclip_cifar100_minnorm:0.27.pkl \
  --dataset cifar100 \
  --eta 1.2
```

### OOD Robustness

Evaluate out-of-distribution robustness:

```bash
python eval/eval_ood_robustness.py \
  --artifact-path outputs/entailment_cbm_cifar100_hycoclip_1frac.pkl \
  --concept-bank outputs/multimodal_concept_hycoclip_cifar100_minnorm:0.27.pkl \
  --ood-dataset cifar100-c \
  --corruption-types gaussian_noise shot_noise impulse_noise
```

---

## Interactive Interventions

The `interactive_interventions.ipynb` notebook provides an interactive interface for:

1. **Visualizing Predictions**: See which concepts contribute most to each prediction
2. **Performing Interventions**: Modify concept activations and observe downstream effects
3. **Comparing Propagation**: See how hierarchical propagation differs from single-concept clamping

To use:

```bash
jupyter notebook interactive_interventions.ipynb
```

Update the `DATASET_CONFIGS` in the first cell to point to your trained artifacts.

---

## Supported Datasets

| Dataset | Classes | Description |
|---------|---------|-------------|
| `cifar10` | 10 | CIFAR-10 image classification |
| `cifar100` | 100 | CIFAR-100 fine-grained classification |
| `imagenet` | 1000 | ImageNet-1K (requires `--imagenet-root`) |
| `sun397` | 397 | SUN397 scene classification |
| `cub` | 200 | CUB-200-2011 bird classification (requires `--cub-root`) |
---
Any other datasets can be added with some modifications.

## Example Jobs

### Concept Encoding

```bash
python encode_concepts.py \
  --concepts-txt vocab/concept_names_imagenet_hycoclip.txt \
  --out-dir outputs_imagenet \
  --backbone-name hycoclip \
  --scenario imagenet \
  --min-norm 0.27
```
### CIFAR-100 Training

```bash
python train_hypcbm.py \
  --concept-bank outputs/multimodal_concept_hycoclip_cifar100_minnorm:0.27.pkl \
  --backbone-name hycoclip \
  --out-dir outputs_cifar100 \
  --dataset cifar100 \
  --entail-eta 1.5 \
  --lam 1e-5 \
  --viz-concept-accuracy \
  --viz-lambdas 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6
```

### ImageNet Training

```bash
python train_hypcbm.py \
  --concept-bank outputs_imagenet/multimodal_concept_hycoclip_imagenet_minnorm:0.27.pkl \
  --backbone-name hycoclip \
  --out-dir outputs_imagenet \
  --dataset imagenet \
  --imagenet-root /path/to/imagenet \
  --embedding-cache-dir outputs/visual_embeddings \
  --entail-eta 1.5 \
  --lam 3e-5 \
  --viz-concept-accuracy \
  --viz-lambdas 3e-5 1e-5 1e-6 1e-7
```

### SUN397 Training

```bash
# Step 2: Train HypCBM
python train_hypcbm.py \
  --concept-bank outputs_sun397/multimodal_concept_hycoclip_sun397_minnorm:0.27.pkl \
  --backbone-name hycoclip \
  --out-dir outputs_sun397 \
  --dataset sun397 \
  --entail-eta 1.4 \
  --lam 1e-5 \
  --viz-concept-accuracy \
  --viz-lambdas 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5
```

---

## Citation

If you find this work useful, please cite:

```
🤞
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This work builds upon:
- [Post-hoc Concept Bottleneck Models](https://github.com/mertyg/post-hoc-cbm) by Yuksekgonul et al.
- [HyCoCLIP/MERU](https://github.com/PalAvik/hycoclip) for hyperbolic vision-language models (Pal et al.)
