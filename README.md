# Paraphrase Detection with Siamese Neural Networks

> **Deep Learning & NLP Project** | ESI-SBA | 2024/2025

## Overview

This project implements a Siamese BiLSTM neural network with multi-head attention for semantic similarity detection between sentence pairs. The model determines whether two sentences convey the same meaning despite different wording or structure — a critical task for plagiarism detection, question answering, and chatbot systems. Trained on combined PAWS and Quora Question Pairs datasets with advanced data augmentation.

## Architecture

```
Input Sentence Pair
    │
    ├── Embedding Layer (384d, subword vocabulary)
    ├── BiLSTM Encoder (3 layers, 512 hidden units)
    ├── Multi-Head Attention (4 heads)
    ├── Handcrafted Features (Jaccard, containment, POS similarity, n-gram overlap, length ratio)
    └── Feature Fusion → Residual Classifier → Similarity Score
```

## Datasets

| Dataset | Size | Purpose |
|---------|------|---------|
| PAWS | ~50K pairs | Adversarial paraphrase detection |
| Quora Question Pairs | ~400K pairs | Real-world semantic similarity |

## Key Techniques

- **Siamese Architecture:** Shared weights for efficient sentence comparison
- **Multi-Head Attention:** 4-head self-attention for contextual understanding
- **Data Augmentation:** WordNet synonym replacement (30% probability)
- **Focal Loss:** Handles class imbalance (α=0.75, γ=2)
- **Feature Engineering:** 5 handcrafted linguistic features fused with neural representations

## Hyperparameters

```python
CONFIG = {
    "max_sequence_length": 128,
    "embedding_dim": 384,
    "hidden_dim": 512,
    "num_layers": 3,
    "dropout": 0.4,
    "batch_size": 64,
    "epochs": 20,
    "lr": 2e-4,
    "weight_decay": 0.05,
    "focal_alpha": 0.75,
    "focal_gamma": 2,
    "augment_prob": 0.3,
    "vocab_size": 35000
}
```

## Tech Stack

- Python 3.8+, PyTorch
- HuggingFace Datasets, spaCy, NLTK
- Jupyter Notebooks (Kaggle T4×2 GPU recommended)

## Repository Structure

```
.
├── trainingCode.ipynb    # Full training pipeline
└── README.md             # This file
```

## Usage

```bash
# Install dependencies
pip install torch datasets huggingface-hub spacy nltk

# Download spaCy model
python -m spacy download en_core_web_sm

# Open notebook
jupyter notebook trainingCode.ipynb
```

## Outputs

- `best_model.pth` — Best checkpoint with vocabulary
- `confusion_matrix.png` — Prediction performance visualization
- `test_results.pkl` — Detailed metrics and classification report

## Authors

**Adem Toumi** — ESI-SBA, Engineering Degree in AI & Data Science  
**Ahmed Saadi** — ESI-SBA, Engineering Degree & M2 in AI & Data Science

## License

MIT
