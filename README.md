This repository contains the official implementation to reproduce the experiments in the paper: [Towards Better Generalization and Interpretability in Unsupervised Concept-Based Models](https://arxiv.org/abs/2506.02092)
Published at **ECML PKDD 2025**.

---

## 📄 Paper Summary

We present **LCBM (Learnable Concept-Based Model)**, a novel **unsupervised concept-based model** for image classification that advances both **generalization** and **interpretability**.

🔍 **Key Contributions**:
- Introduces a **Bernoulli latent space with learnable concept embeddings**, allowing richer, more expressive unsupervised concepts.
- Achieves **superior accuracy** over previous unsupervised CBMs and nearly closes the gap with black-box models.
- Produces **interpretable and human-aligned concepts**, supported by strong F1 and CAS metrics, and a user study.
- Maintains **local interpretability** by combining concept activations linearly for final predictions.

📊 **Highlights**:
- Validated across seven datasets (MNIST, CIFAR-10/100, Tiny ImageNet, Skin Lesions, CUB-200, etc.).
- Outperforms baselines in **reconstruction error**, **mutual information retention**, and **human-concept alignment**.
- Enables intuitive interventions and concept visualizations with concept dictionaries and counterfactual analysis.

This work demonstrates that **unsupervised CBMs** can achieve both **high performance and strong interpretability**, setting a new standard for scalable explainable models.

