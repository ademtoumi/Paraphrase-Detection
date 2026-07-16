# Paraphrase Detection with Siamese Neural Networks

**NLP & Deep Learning Project — ESI-SBA, 2024–2025**

---

## Overview

This project addresses the task of paraphrase detection — determining whether two sentences convey the same meaning — using a Siamese neural network architecture combining bidirectional LSTMs with multi-head attention. Paraphrase detection is a fundamental problem in natural language processing with direct applications in plagiarism detection, question answering, and information retrieval. The model integrates both learned semantic representations and handcrafted linguistic features to capture surface-level and semantic similarity simultaneously.

---

## Objectives

- Design and train a Siamese BiLSTM network with attention for sentence-level semantic similarity detection.
- Augment learned representations with handcrafted linguistic features to improve robustness across paraphrase types.
- Evaluate the approach on large-scale benchmark datasets combining both human-annotated and adversarially constructed paraphrase pairs.

---

## Methodology

### Architecture

The model follows a Siamese structure in which both input sentences share identical encoder weights. Each sentence is encoded by a three-layer bidirectional LSTM with 512 hidden units per direction. A four-head multi-head attention mechanism is applied over the BiLSTM output to produce a context-aware sentence representation.

In parallel, a set of handcrafted linguistic features is computed for each sentence pair: Jaccard similarity, containment ratio, part-of-speech similarity, n-gram overlap at multiple granularities, and length ratio. These features are concatenated with the attention-weighted representations and passed through a residual classifier that produces the final binary prediction.

### Training

The model is trained on two datasets: PAWS (approximately 50,000 pairs), which contains adversarially constructed paraphrases requiring semantic rather than lexical matching, and the Quora Question Pairs dataset (approximately 400,000 pairs). Data augmentation is applied via WordNet synonym substitution with a 30% probability, improving generalization across lexical variations. Training uses Focal Loss (α = 0.75, γ = 2) to address class imbalance and focus learning on difficult examples.

---

## Technologies

Python · PyTorch · HuggingFace Datasets · spaCy · NLTK · Jupyter Notebook

---

## Repository Structure

```
paraphraseDetection-master/
├── trainingCode.ipynb
└── README.md
```

---

## Authors

Adem Toumi — ESI-SBA  
Ahmed Saadi — ESI-SBA
