# Sentiment Analysis of Movie Reviews Using Deep Learning 🎬

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.0%2B-D00000.svg)](https://keras.io/)
[![Dataset](https://img.shields.io/badge/Dataset-IMDb%20Movie%20Reviews-yellow.svg)](https://ai.stanford.edu/~amaas/data/sentiment/)
[![Execution](https://img.shields.io/badge/Offline-100%25%20Compatible-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

An end-to-end Deep Learning project for binary text classification and natural language processing (NLP). The model automatically reads English movie reviews and classifies the underlying sentiment as **Positive (1)** or **Negative (0)** using an optimized Neural Network with custom Word Embeddings and Global Average Pooling.

---

## 📌 Table of Contents

- [Overview & Objectives](#-overview--objectives)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Deep Learning Model Architecture](#-deep-learning-model-architecture)
- [Dataset Mechanics & Preprocessing](#-dataset-mechanics--preprocessing)
- [Installation & Setup](#-installation--setup)
- [How to Run the Project](#-how-to-run-the-project)
- [Experimental Results & Evaluation](#-experimental-results--evaluation)
- [Custom Sentiment Prediction](#-custom-sentiment-prediction)
- [Tech Stack](#-tech-stack)

---

## 🎯 Overview & Objectives

In the digital era, millions of user reviews are generated daily across platforms like IMDb, Rotten Tomatoes, and social media. Manual sentiment inspection at scale is impractical.

This project implements a lightweight yet high-performance Deep Neural Network that:
1. **Processes raw text reviews** into dense mathematical vectors.
2. **Learns semantic relationships** between words in a 16-dimensional continuous vector space.
3. **Classifies reviews** with high accuracy (~85.1%) while keeping the model memory-efficient (~626 KB) and ultra-fast (~7 seconds training time on CPU).
4. **Operates 100% offline** via local NumPy dataset arrays and a JSON word dictionary.

---

## ✨ Key Features

- 🌐 **100% Offline Capability:** Includes a standalone dataset builder (`prepare_dataset.py`) that stores the benchmark IMDb dataset locally as binary `.npy` arrays and `word_index.json`.
- ⚡ **Lightweight & Fast Architecture:** Avoids computationally expensive RNNs/LSTMs, opting for `Embedding` + `GlobalAveragePooling1D`, training 10 epochs in under 10 seconds on standard CPU.
- 🎯 **High Accuracy:** Achieves **~85.10% accuracy** and a low **test loss of 0.3703** on 25,000 unseen test reviews.
- 🛠️ **Custom Inference Pipeline:** Provides an intuitive `predict_sentiment()` function that handles text normalization, punctuation removal, tokenization, unknown word mapping, padding, and sigmoid confidence scoring.

---

## 📂 Repository Structure

```text
Sentiment_Analysis_DL/
│
├── dataset/                    # Local offline dataset directory (Generated)
│   ├── word_index.json         # IMDb word to integer mapping dictionary (~1.6 MB)
│   ├── x_train.npy             # Training review integer sequences (25,000 reviews)
│   ├── y_train.npy             # Training binary sentiment labels (0/1)
│   ├── x_test.npy              # Testing review integer sequences (25,000 reviews)
│   └── y_test.npy              # Testing binary sentiment labels (0/1)
│
├── prepare_dataset.py          # Script to download & save dataset locally for offline use
├── Sentiment_Analysis.ipynb    # Main Jupyter Notebook (Model building, training & evaluation)
└── README.md                   # Project documentation
```

---

## 🧠 Deep Learning Model Architecture

The neural network is built using the Keras `Sequential` API with explicit input dimension specifications.

```text
[Input Sequence (256,)] 
         │
         ▼
[Embedding Layer (10000 -> 16)]  ──► Output: (None, 256, 16) | 160,000 Params
         │
         ▼
[GlobalAveragePooling1D]         ──► Output: (None, 16)      | 0 Params
         │
         ▼
[Dense Layer (16, ReLU)]         ──► Output: (None, 16)      | 272 Params
         │
         ▼
[Dense Layer (1, Sigmoid)]       ──► Output: (None, 1)       | 17 Params
```

### Detailed Layer Breakdown & Parameter Math

| # | Layer Type | Output Shape | Param # | Purpose / Description |
|---|------------|--------------|---------|-----------------------|
| 0 | `Input` | `(None, 256)` | 0 | Fixed input sequence length of 256 integer tokens. |
| 1 | `Embedding` | `(None, 256, 16)` | 160,000 | Maps word IDs to 16-dim dense semantic vector space ($10,000 \times 16$). |
| 2 | `GlobalAveragePooling1D` | `(None, 16)` | 0 | Averages the 256 vectors into a single 16-dim review summary vector. |
| 3 | `Dense` (Hidden) | `(None, 16)` | 272 | Non-linear feature extraction with ReLU ($16 \times 16 + 16$). |
| 4 | `Dense` (Output) | `(None, 1)` | 17 | Binary sentiment probability with Sigmoid ($16 \times 1 + 1$). |

- **Total Parameters:** **160,289** (100% Trainable, ~626.13 KB memory footprint).
- **Optimizer:** Adam ($\text{learning\_rate} = 0.001$)
- **Loss Function:** Binary Cross-Entropy
- **Evaluation Metric:** Accuracy

---

## 📊 Dataset Mechanics & Preprocessing

- **Source:** IMDb Benchmark Movie Reviews Dataset (Stanford / Andrew Maas).
- **Dataset Size:** 50,000 reviews (25,000 Train / 25,000 Test).
- **Class Balance:** Perfectly balanced (50% Positive / 50% Negative).


### Preprocessing & Tokenization Details

1. **Vocabulary Size (`vocab_size = 10000`):** Restricted to the top 10,000 most frequent words according to Zipf's Law, capturing >95% of textual signal while eliminating noise/typos.
2. **Special Tokens & Offset:**
   - `0` ➔ `<PAD>` (Padding token)
   - `1` ➔ `<START>` (Start of sequence marker)
   - `2` ➔ `<UNK>` (Unknown / Out-of-vocabulary word)
   - Real words start at token index `4` ($\text{Token ID} = \text{word\_index}[\text{word}] + 3$).
3. **Sequence Standardization (`max_length = 256`):**
   - Reviews > 256 words are post-truncated.
   - Reviews < 256 words are post-padded with zeros.

---

## ⚙️ Installation & Setup

### Prerequisites

Ensure you have **Python 3.10+** installed on your system.

### Install Required Dependencies

Run the following command in your terminal/command prompt:

```bash
pip install tensorflow keras numpy matplotlib
```

---

## 🚀 How to Run the Project

### Step 1: Prepare Dataset for Offline Execution

Run `prepare_dataset.py` once to download the IMDb benchmark dataset and save it into the local `dataset/` directory:

```bash
python prepare_dataset.py
```

*Output:*
```text
Downloading IMDb dataset with top 10000 vocabulary words...
Saving train and test sets to 'dataset/'...
Downloading and saving IMDb word index...

Dataset preparation complete! All files saved locally in 'dataset/':
  - x_train.npy (13640.0 KB)
  - y_train.npy (195.4 KB)
  - x_test.npy (13182.5 KB)
  - y_test.npy (195.4 KB)
  - word_index.json (1595.2 KB)
```

### Step 2: Open & Run the Jupyter Notebook

Launch Jupyter Notebook or VS Code / Antigravity IDE and run `Sentiment_Analysis.ipynb`:

```bash
jupyter notebook Sentiment_Analysis.ipynb
```

Execute all cells sequentially. The notebook will load all data directly from `dataset/` without needing an internet connection.

---

## 📈 Experimental Results & Evaluation

### Training Metrics (10 Epochs, Batch Size = 512)

| Epoch | Train Accuracy | Train Loss | Val Accuracy | Val Loss |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 59.79% | 0.6888 | 60.50% | 0.6816 |
| 3 | 72.97% | 0.6403 | 76.22% | 0.6171 |
| 5 | 80.81% | 0.5299 | 82.26% | 0.5043 |
| 8 | 85.99% | 0.3809 | 84.86% | 0.3872 |
| 10 | **87.74%** | **0.3278** | **86.06%** | **0.3509** |

### Test Dataset Evaluation

```text
Test Loss    : 0.3703
Test Accuracy: 85.10%
Training Time: ~7 seconds (CPU)
```

---

## 🔮 Custom Sentiment Prediction

You can test arbitrary movie review strings using the custom prediction function in Python:

```python
predict_sentiment("This movie was absolutely wonderful with great acting and a brilliant plot!")
# Output:
# Review: "This movie was absolutely wonderful with great acting and a brilliant plot!"
# Score : 0.6681
# Result: POSITIVE

predict_sentiment("Terrible film, completely boring, horrible acting and total waste of time.")
# Output:
# Review: "Terrible film, completely boring, horrible acting and total waste of time."
# Score : 0.2077
# Result: NEGATIVE
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Deep Learning Framework:** TensorFlow 2.x & Keras 3.x
- **Data Manipulation:** NumPy
- **Data Visualization:** Matplotlib
- **Notebook Environment:** Jupyter Notebook / VS Code / Antigravity IDE

---

⭐ *If you find this repository helpful, feel free to star it!*
