This repository contains the official implementation to reproduce the experiments in the paper: [Towards Better Generalization and Interpretability in Unsupervised Concept-Based Models](https://arxiv.org/abs/2506.02092)
Published at **ECML PKDD 2025 (Research Track)**.

---

## 📄 Paper Summary

We introduce **LCBM (Learnable Concept-Based Model)**, a novel **unsupervised concept-based model** for image classification that improves both **accuracy** and **interpretability**.

Unlike prior approaches, LCBM learns a compact set of concepts as **Bernoulli latent variables with embeddings**, enabling richer representations without human supervision. It achieves strong performance across multiple datasets and supports **interpretable predictions** via linear combinations of concept activations.

### 🔍 Key Features
- **Improved generalization**: Matches or exceeds prior unsupervised CBMs and approaches black-box performance.
- **Human-aligned concepts**: Concepts are more intuitive and better aligned, as shown through F1 scores, CAS, and a user study.
- **Faithful explanations**: Supports concept interventions and visual dictionaries for transparent decision-making.

We evaluated the model on both toy and real datasets: MNIST Even/Odd, MNIST Addition, CIFAR-10, CIFAR100, Tiny ImageNet, Skin Lesions and CUB-200.

## Reproducibility Instructions

Follow the steps below to reproduce the results reported in the paper.

### 1. Environment Setup

Make sure you have [conda](https://docs.conda.io/en/latest/) installed.

```bash
conda env create -f environment.yaml
conda activate lcbm
```

### 2. Training Models
To train and save all models used in the paper:

- LCBM (ours):
```bash
bash run_LCBM.sh
```

Baselines:
- BlackBox:
```bash
bash run_E2E.sh
```
- Label-Free CBM:
```bash
bash run_LF_CBM.sh
```
For this baseline, the concept have been extracted from GPT-4o and are reported into the `concept_list` folder.

⚠️ For other baselines (SENN, BotCL, etc.), we used the official repositories. We recommend doing the same for faithful reproduction.

### 3. Evaluation & Explanations
To compute all the metrics and generate explanations:
```bash
bash compute_concept_metrics.sh
bash run_explanations.sh
```

These scripts cover concept alignment, interpretability metrics, and qualitative visualizations such as concept dictionaries and Grad-CAMs.