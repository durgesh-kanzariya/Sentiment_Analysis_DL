# Complete Viva Preparation Guide
## Sentiment Analysis of Movie Reviews Using Deep Learning

This document is your master cheat sheet for your oral examination (viva). It covers every concept, formula, architectural decision, code detail, parameter calculation, model comparison, line-by-line code explanation, and potential examiner question.

---

## Table of Contents
1. **High-Level Project Summary**
2. **Dataset Mechanics & Preprocessing**
3. **Layer-by-Layer Architecture & Math**
4. **Why This Architecture? Comparison with Alternative Models**
5. **Compilation & Training Strategy**
6. **Result Interpretation & Graphs**
7. **The "Predict Sentiment" Function & Edge Cases**
8. **Top 25 Rapid-Fire Viva Questions & Answers**
9. **Exhaustive Line-by-Line Code Explanation (Every Line Explained)**

---

## 1. High-Level Project Summary
* **Goal:** Automatically classify natural language movie reviews into **Positive (1)** or **Negative (0)** sentiment.
* **Problem Type:** Binary Text Classification (Natural Language Processing + Deep Learning).
* **Dataset:** IMDb benchmark dataset (50,000 balanced movie reviews: 25,000 training, 25,000 testing).
* **Architecture:** `Input(256)` → `Embedding(10000, 16)` → `GlobalAveragePooling1D` → `Dense(16, ReLU)` → `Dense(1, Sigmoid)`.
* **Total Parameters:** **160,289** (100% trainable, ~626 KB memory footprint).
* **Performance:** **~85.10% test accuracy**, test loss of **0.3703**, training time of **~7 seconds** on CPU.
* **Offline Execution:** 100% offline-compatible using local `.npy` arrays and `word_index.json`.

---

## 2. Dataset Mechanics & Preprocessing

### Why IMDb Benchmark?
* It is the gold standard NLP sentiment benchmark created by Andrew Maas et al. (Stanford).
* It is **perfectly balanced** (50% positive, 50% negative), meaning random guessing gives 50% accuracy.
* No class imbalance techniques (like SMOTE or class weights) are needed.

### Vocabulary Size (`vocab_size = 10000`)
* **Examiner Question:** *"Why restrict vocabulary to 10,000 words instead of all 88,584 words in the dataset?"*
* **Answer:** According to **Zipf's Law**, the top 10,000 words account for >95% of word occurrences in English text. The remaining ~78,000 words are rare misspellings, typos, character names, or obscure words that appear only once or twice. Keeping them would massively increase the embedding matrix (from 160k parameters to 1.4 million) causing severe overfitting and memory waste without adding meaningful signal.

### Special Tokens & Offset by 3
* In the Keras IMDb dataset, the first 3 indices are reserved:
  - `0`: `<PAD>` (padding token to make sequences equal length)
  - `1`: `<START>` (marks the beginning of a review)
  - `2`: `<UNK>` (unknown word / out of vocabulary token)
  - `3`: Reserved / unused
* **Actual words start at index 4**:
  $$\text{Dataset Token ID} = \text{word\_index}[\text{word}] + 3$$
* This is why in `predict_sentiment()` we add 3 to dictionary lookups, and assign `2` to unknown words.

### Sequence Padding (`max_length = 256`)
* **Examiner Question:** *"Why do we need `pad_sequences`?"*
* **Answer:** Deep learning models require rectangular matrix tensors of uniform shape `(batch_size, sequence_length)` to perform parallel matrix multiplication on CPU/GPU.
* Reviews vary from 20 words to over 1,000 words.
* We choose **256 words**:
  - Reviews $> 256$ words are truncated (`truncating='post'`).
  - Reviews $< 256$ words are padded with `0` at the end (`padding='post'`).
  - 256 captures the vast majority of review sentiment while keeping computation very fast.

---

## 3. Layer-by-Layer Architecture & Math

### Layer 0: Input Layer (`keras.Input(shape=(256,))`)
* Defines the expected input shape: a 1D sequence of 256 integers.
* **Parameters:** `0` (does not learn weights).
* Explicitly setting this ensures `model.summary()` calculates and displays output dimensions and parameter counts upfront before fitting with **zero Keras warnings**.

---

### Layer 1: Embedding Layer (`layers.Embedding(input_dim=10000, output_dim=16)`)
* **Examiner Question:** *"What is an Embedding layer, and why not use One-Hot Encoding?"*
* **Answer:**
  1. **One-Hot Encoding** represents each word as a sparse vector of 10,000 zeros and a single 1. This causes a massive matrix (`256 × 10000 = 2.56` million numbers per review), consumes huge memory, and treats all words as completely orthogonal (e.g., "good" and "great" have 0 similarity).
  2. An **Embedding Layer** is a trainable lookup table of shape `(10000, 16)`. It maps each word ID to a dense, continuous 16-dimensional vector space where semantically related words end up close to each other (via cosine similarity).
* **Output Shape:** `(None, 256, 16)`
* **Parameter Calculation:**
  $$\text{Params} = \text{vocab\_size} \times \text{embedding\_dim} = 10,000 \times 16 = \mathbf{160,000}$$

---

### Layer 2: GlobalAveragePooling1D (`layers.GlobalAveragePooling1D()`)
* **Examiner Question:** *"What is GlobalAveragePooling1D and why use it instead of Flatten?"*
* **Answer:**
  - It takes the `(256, 16)` matrix and computes the mathematical average across the 256 sequence timesteps for each of the 16 features:
    $$\text{pooled}[j] = \frac{1}{256} \sum_{i=1}^{256} \text{Embedding}[i, j]$$
  - Output is a single summary vector of shape `(None, 16)`.
  - **Flatten()** would produce $256 \times 16 = 4,096$ values, requiring $(4096 \times 16) + 16 = 65,552$ dense weights, leading to severe overfitting.
  - **Global Average Pooling** uses **0 parameters**, reduces dimensions drastically, and ensures the model is fast on CPU.

---

### Layer 3: Dense Hidden Layer (`layers.Dense(16, activation='relu')`)
* **Activation Function:** **ReLU** (Rectified Linear Unit):
  $$f(x) = \max(0, x)$$
* **Why ReLU?**
  - Solves the vanishing gradient problem (derivative is 1 for positive inputs).
  - Extremely fast to compute compared to sigmoid/tanh.
  - Introduces non-linearity so the network can learn complex feature interactions.
* **Output Shape:** `(None, 16)`
* **Parameter Calculation:**
  $$\text{Params} = (\text{input\_units} \times \text{neurons}) + \text{biases} = (16 \times 16) + 16 = 256 + 16 = \mathbf{272}$$

---

### Layer 4: Dense Output Layer (`layers.Dense(1, activation='sigmoid')`)
* **Activation Function:** **Sigmoid**:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
* **Why Sigmoid?**
  - S-shaped curve that squashes any real value into the range `[0.0, 1.0]`.
  - Exactly corresponds to the probability $P(\text{Sentiment} = \text{Positive} \mid x)$.
  - A threshold of `0.5` gives the binary decision:
    $$\hat{y} = \begin{cases} 1 \text{ (Positive)}, & \text{if } \sigma(z) \ge 0.5 \\ 0 \text{ (Negative)}, & \text{if } \sigma(z) < 0.5 \end{cases}$$
* **Output Shape:** `(None, 1)`
* **Parameter Calculation:**
  $$\text{Params} = (\text{input\_units} \times 1) + 1 \text{ bias} = (16 \times 1) + 1 = \mathbf{17}$$

---

### Total Parameter Summary Table

| Layer | Type | Output Shape | Formula | Total Params |
| :--- | :--- | :---: | :--- | :---: |
| `input_layer` | `Input` | `(None, 256)` | None | 0 |
| `embedding` | `Embedding` | `(None, 256, 16)` | $10,000 \times 16$ | 160,000 |
| `global_avg_pool` | `GlobalAveragePooling1D` | `(None, 16)` | Non-parametric | 0 |
| `dense_hidden` | `Dense (ReLU)` | `(None, 16)` | $(16 \times 16) + 16$ | 272 |
| `dense_output` | `Dense (Sigmoid)` | `(None, 1)` | $(16 \times 1) + 1$ | 17 |
| **Total** | | | **All Trainable** | **160,289** |

---

## 4. Why This Architecture? Comparison with Alternative Models
*(The "Why Didn't You Use X?" Section — Crucial for Examiners)*

### Comprehensive Model Comparison Table

| Model Architecture | Parameter Count | Training Time (CPU) | IMDb Accuracy | Primary Strength | Main Weakness / Why Not Chosen |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Our Model (Embedding + GlobalAvgPool + Dense)** | **~160,000** | **~7 seconds** | **~85.1%** | **Instant CPU training, zero dependencies, highly interpretable, lightweight.** | Ignores sequential word order and contrastive words like *"but"*. |
| **Flatten + Dense (`Embedding` → `Flatten` → `Dense`)** | ~225,000 | ~12 seconds | ~81.0% | Simple. | **Severe overfitting** (4,096 flattened inputs create rigid, positional weights). |
| **Traditional ML (TF-IDF + Naive Bayes / Logistic Regression)** | ~10,000–50,000 | ~5 seconds | ~84–86% | Simple baseline, fast. | **No semantic representation**; treats words as isolated symbols with zero cosine vector similarity. |
| **Simple RNN (`Embedding` → `SimpleRNN(64)` → `Dense`)** | ~165,000 | ~45 seconds | ~81–83% | Captures sequential flow. | **Vanishing/Exploding gradients** over 256 steps; inferior accuracy to pooling. |
| **LSTM (`Embedding` → `LSTM(64)` → `Dense`)** | ~181,000 | ~3–5 minutes | ~87–88% | Handles long-term context, negations (*"not good"*). | **4x heavier computation** per step (4 internal gates); slow on CPU without GPU. |
| **Bidirectional LSTM (`BiLSTM(64)`)** | ~202,000 | ~6–8 minutes | ~89.0% | Reads text forwards and backwards. | High computational complexity; prone to overfitting on small vocabulary without heavy dropout. |
| **1D CNN (`Embedding` → `Conv1D` → `MaxPool` → `Dense`)** | ~170,000 | ~20 seconds | ~86–87% | Fast; captures local phrases (e.g. 3-word n-grams). | Introduces multiple hyperparameters (kernel size, filters, stride); less intuitive as an introductory baseline. |
| **Pretrained Transformers (BERT / DistilBERT / RoBERTa)** | **110,000,000+** | **1+ hour (requires GPU)** | **~93–95%** | State-of-the-art context & attention. | **Massive size (~440 MB)**, requires high-end GPU with 16GB+ VRAM, cannot run offline in a lightweight assignment setting. |

---

### Detailed Justifications Against Common Alternatives

#### 1. Why Deep Learning instead of Traditional ML (Naive Bayes / SVM)?
* Traditional ML relies on **manual feature engineering** (TF-IDF bag-of-words) which treats words as independent, orthogonal discrete tokens ("king" and "queen" share no geometric similarity).
* Deep Learning automatically learns a **continuous vector space (word embeddings)** where semantic relationships and analogies are captured mathematically through backpropagation.
* This architecture serves as the fundamental stepping stone to modern neural NLP and representation learning.

#### 2. Why not an LSTM or GRU?
* **Computational Efficiency:** An LSTM cell has 4 internal neural gates (Forget gate, Input gate, Candidate state, Output gate). Across 256 time steps and 25,000 reviews, training takes several minutes on CPU, whereas Global Average Pooling trains in **~7 seconds**.
* **Diminishing Returns:** For general sentiment classification, an LSTM only increases accuracy from **~85% to ~87–88%** on IMDb, at the cost of a 30x increase in training time.
* **Overfitting Risk:** LSTMs are prone to overfitting on medium datasets without complex recurrent dropout tuning.
* Our objective was to demonstrate a clean, fast, and accessible deep learning baseline that runs reliably on standard classroom laptops without GPU requirements.

#### 3. Why not 1D Convolutional Neural Networks (Conv1D)?
* While 1D CNNs are faster than LSTMs and capture local multi-word patterns through sliding kernels, they introduce multiple extra hyperparameters (kernel size, filter counts, pooling strides).
* Global Average Pooling achieves comparable overall accuracy (~85%) with fewer hyperparameters and maximum architectural transparency.

#### 4. Why not Pre-trained Transformers (BERT / DistilBERT)?
* **Model Footprint:** BERT-Base contains **110 million parameters** (~440 MB file size) compared to our model's **160,289 parameters** (~626 KB).
* **Hardware Dependency:** Fine-tuning BERT requires a dedicated CUDA GPU with 16GB+ VRAM; on a standard CPU, fine-tuning takes over an hour per epoch.
* **Offline Criteria:** Our assignment strictly requires 100% self-contained offline execution inside the local project folder without downloading heavy external pretrained weights from Hugging Face.

---

## 5. Compilation & Training Strategy

### Optimizer: Adam (Adaptive Moment Estimation)
* **Examiner Question:** *"Why use Adam instead of basic Stochastic Gradient Descent (SGD)?"*
* **Answer:**
  - Adam combines the advantages of **Momentum** (smooths out noisy gradients by keeping a moving average of past gradients) and **RMSProp** (adapts the learning rate individually for each parameter based on historical squared gradients).
  - It converges faster, requires minimal hyperparameter tuning, and handles sparse gradients from embedding lookups effectively.

### Loss Function: Binary Cross-Entropy
* **Formula:**
  $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
* **Why not Mean Squared Error (MSE)?**
  - MSE on sigmoid outputs causes vanishing gradients during backpropagation when predictions are confidently wrong (derivatives near 0).
  - Cross-entropy penalizes wrong, confident predictions exponentially ($-\log(0) \to \infty$), giving strong gradient updates for faster learning.

### Training Settings
* `batch_size = 512`: Large batch size allows vectorized CPU matrix operations, keeps gradient estimates stable, and completes an epoch in <1 second.
* `epochs = 10`: Model reaches optimal convergence around epoch 6–8.
* `validation_split = 0.2`: 20% (5,000 reviews) of training data is set aside to validate generalization during training without peeking at the test set.

---

## 6. Result Interpretation & Graphs

### Accuracy & Loss Curves Explained
* **Training Accuracy:** Starts at ~56% and increases smoothly to ~87%.
* **Validation Accuracy:** Starts at ~68% and reaches ~85.5%.
* **Loss Behavior:** Training loss decreases steadily from ~0.69 to ~0.34. Validation loss decreases to ~0.36 around epoch 6–8, then plateaus.
* **Overfitting Observation:** The gap between training accuracy (~87%) and validation accuracy (~85.5%) is small (~1.5%), showing the model generalizes well without severe overfitting.

---

## 7. The "Predict Sentiment" Function & Edge Cases

### Function Pipeline
1. **Cleaning:** Lowercases input string and strips special punctuation using `char.isalnum()`.
2. **Input Guard:** Validates that `words` list is non-empty; returns an informative prompt if the user inputs an empty string or symbols.
3. **Token Mapping:** Prepends `<START>` token (`1`). For each word, checks if `word_index[word] + 3 < 10000`; if yes, appends ID; otherwise appends `<UNK>` (`2`).
4. **Padding:** `pad_sequences(..., maxlen=256, padding='post')`.
5. **Inference:** `model.predict(padded)` yields float probability $\hat{y} \in [0, 1]$.
6. **Classification:** $\hat{y} \ge 0.5 \implies$ `POSITIVE`, otherwise `NEGATIVE`.

### Viva Trap Question: The "Crazy Good" Edge Case
* **Question:** *"Why did 'This movie was absolutely long but very interesting and crazy good' get a score of 0.4463 (NEGATIVE)?"*
* **Answer (Speak this confidently!):**
  1. **Bag-of-Words Averaging:** `GlobalAveragePooling1D` averages word vectors without tracking word order or grammar. It doesn't know that `"but"` overrides `"long"`, nor that `"crazy"` is an intensifier slang for `"good"`.
  2. **Dataset Bias:** In classic IMDb reviews, `"long"` and `"crazy"` heavily appear in negative reviews (*"too long and boring"*, *"crazy mess"*). Their negative embeddings pull down the average.
  3. **Zero Padding Dilution:** The 11 words are averaged across 256 positions (with 244 zeros), compressing the score close to the neutral baseline (`0.50`). A score of `0.4463` is only `0.05` away from neutral, allowing `"long"` and `"crazy"` to tip it just under `0.50`.
  4. **Solution in Advanced NLP:** Recurrent networks (LSTM/GRU) or Transformer models (BERT) track syntax, negation, and multi-word phrases.

---

## 8. Top 25 Rapid-Fire Viva Questions & Answers

**Q1: What is the goal of this project?**  
*A: To build a Deep Learning model that classifies IMDb English movie reviews as either Positive or Negative.*

**Q2: What is the difference between supervised and unsupervised learning? Which one is this?**  
*A: Supervised learning trains on labeled data (inputs + ground truth outputs). This project is supervised because each review has a binary label (0 or 1).*

**Q3: What does the Embedding layer do?**  
*A: It maps sparse integer word IDs to dense 16-dimensional continuous vectors where words with similar semantic meanings are positioned close to one another.*

**Q4: How many parameters are in the Embedding layer?**  
*A: 160,000 parameters ($10,000 \text{ vocabulary words} \times 16 \text{ embedding dimensions}$).*

**Q5: Why did you choose 16 dimensions for the embedding?**  
*A: 16 dimensions provide enough representational power to separate positive and negative sentiments while keeping computation fast and preventing overfitting on a small dataset.*

**Q6: What is the purpose of GlobalAveragePooling1D?**  
*A: It takes the average across the 256 word vectors to produce a single 16-dimensional summary vector representing the entire review.*

**Q7: How many parameters does GlobalAveragePooling1D have?**  
*A: Zero. It is a fixed mathematical averaging operation, not a layer with trainable weights.*

**Q8: What is the activation function in the hidden layer, and why?**  
*A: ReLU ($f(x) = \max(0, x)$). It introduces non-linearity, is computationally very fast, and prevents the vanishing gradient problem.*

**Q9: What is the activation function in the final output layer, and why?**  
*A: Sigmoid ($\sigma(z) = 1 / (1 + e^{-z})$). It maps the final logit into a probability between 0.0 and 1.0, ideal for binary classification.*

**Q10: What is the decision threshold for classification?**  
*A: 0.5. A probability $\ge 0.5$ is classified as Positive; $< 0.5$ is classified as Negative.*

**Q11: What loss function did you use, and why?**  
*A: Binary Cross-Entropy. It measures the divergence between true binary labels and predicted probabilities, penalizing confident wrong predictions exponentially.*

**Q12: What optimizer did you use?**  
*A: Adam (Adaptive Moment Estimation). It dynamically adjusts the learning rate for each parameter using first and second moments of the gradients.*

**Q13: Why do we pad sequences to 256 words?**  
*A: Neural networks require uniform rectangular input tensors to perform efficient batch matrix multiplication. Longer reviews are truncated; shorter ones are padded with 0.*

**Q14: What are tokens 0, 1, and 2 reserved for in Keras IMDb?**  
*A: 0 is `<PAD>`, 1 is `<START>`, and 2 is `<UNK>` (unknown / out-of-vocabulary word).*

**Q15: Why is vocabulary restricted to 10,000 words?**  
*A: Top 10,000 words cover ~95%+ of English words in the dataset. Filtering the tail 78,000 rare words reduces noise, speeds up training, and prevents overfitting.*

**Q16: Why didn't you use an LSTM or GRU?**  
*A: LSTMs have 4 complex internal gates requiring 30x more training time on CPU for only a marginal ~2% gain on IMDb. Average Pooling provides ~85% accuracy in 7 seconds.*

**Q17: Why didn't you use BERT or Transformers?**  
*A: BERT has 110M+ parameters, requires a GPU with 16GB+ VRAM, and requires heavy external downloads (~440MB), violating our lightweight, offline assignment requirement.*

**Q18: Why didn't you use Flatten() instead of GlobalAveragePooling1D?**  
*A: Flatten() creates 4,096 features ($256 \times 16$), leading to over 65,000 dense weights, severe overfitting, and position-rigid weights. Average Pooling uses 0 parameters.*

**Q19: How does this project run 100% offline?**  
*A: We downloaded the dataset once using `prepare_dataset.py` and saved arrays as `x_train.npy`, `y_train.npy`, `x_test.npy`, `y_test.npy`, and `word_index.json`. The notebook loads strictly from disk using NumPy and JSON with zero network calls.*

**Q20: What was the final accuracy on the test set?**  
*A: ~85.10% accuracy and ~0.37 test loss on 25,000 completely unseen test reviews.*

**Q21: What is the total parameter count of your model?**  
*A: 160,289 parameters: 160,000 (Embedding) + 0 (GlobalAveragePooling1D) + 272 (Dense 16) + 17 (Dense 1).*

**Q22: How would you prevent overfitting if the model overfits?**  
*A: By adding a Dropout layer (e.g., `Dropout(0.2)`), applying L2 weight regularization, using Early Stopping, or reducing embedding dimensions.*

**Q23: What does the batch size of 512 do?**  
*A: It defines how many reviews are processed together before updating weights via backpropagation. 512 stabilizes gradient estimates and accelerates CPU computation via SIMD vectorization.*

**Q24: What is the main weakness of GlobalAveragePooling1D?**  
*A: It ignores word order and syntactic structure (like negation *"not good"* or contrastive conjunctions *"long but good"*).*

**Q25: When would you upgrade to an LSTM or BERT in a production environment?**  
*A: In production applications where handling subtle negation, sarcasm, complex multi-clause sentences, and conversational nuances justifies the higher GPU compute cost.*

---

## 9. Exhaustive Line-by-Line Code Explanation

This section takes every single line of code across all 12 code cells in `Sentiment_Analysis.ipynb` (plus `prepare_dataset.py`) and explains what it does in clear, plain language.

---

### Code Cell [1]: Import Required Libraries

```python
# Import core deep learning libraries
import tensorflow as tf
import keras
from keras import layers
from keras.preprocessing.sequence import pad_sequences
```
* `import tensorflow as tf`: Imports Google's open-source machine learning and deep learning framework under the standard alias `tf`.
* `import keras`: Imports Keras, the high-level neural network API running on top of TensorFlow that provides simple abstractions for layers, models, and optimizers.
* `from keras import layers`: Imports neural network building blocks such as `Embedding`, `GlobalAveragePooling1D`, and `Dense`.
* `from keras.preprocessing.sequence import pad_sequences`: Imports the utility function that pads or truncates variable-length lists of numbers into a uniform 2D NumPy array.

```python
# Import numerical array and plotting libraries
import numpy as np
import matplotlib.pyplot as plt
```
* `import numpy as np`: Imports NumPy, the core library for handling multidimensional numerical arrays and matrix operations in Python.
* `import matplotlib.pyplot as plt`: Imports Matplotlib's pyplot interface to plot training graphs, curves, and charts.

```python
# Import json for reading the local offline word dictionary
import json
```
* `import json`: Imports Python's built-in JSON parser, which we use to load `dataset/word_index.json` from the local disk into a Python dictionary.

```python
# Print installed TensorFlow and Keras versions to confirm environment setup
print("TensorFlow Version:", tf.__version__)
print("Keras Version     :", keras.__version__)
```
* `print(...)`: Outputs the exact installed library versions (e.g., TensorFlow 2.21.0 and Keras 3.15.1) to verify that the environment is correctly set up.

---

### Code Cell [2]: Load Dataset from Local Disk

```python
# Define maximum vocabulary size (top 10,000 most frequent words)
vocab_size = 10000
```
* `vocab_size = 10000`: Sets a constant limiting our dictionary to the top 10,000 most common words, filtering out rare words to conserve memory and prevent overfitting.

```python
# Load training reviews and binary sentiment labels from local offline dataset files
x_train = np.load("dataset/x_train.npy", allow_pickle=True)
y_train = np.load("dataset/y_train.npy")
```
* `x_train = np.load("dataset/x_train.npy", allow_pickle=True)`: Loads the 25,000 training reviews from local disk. `allow_pickle=True` is required by NumPy because before padding, `x_train` contains variable-length Python lists of integers stored as an object array.
* `y_train = np.load("dataset/y_train.npy")`: Loads the 25,000 binary training ground-truth labels (`1` for positive, `0` for negative).

```python
# Load testing reviews and binary sentiment labels from local offline dataset files
x_test = np.load("dataset/x_test.npy", allow_pickle=True)
y_test = np.load("dataset/y_test.npy")
```
* Loads the 25,000 completely separate test reviews and labels from disk to evaluate generalization after training.

```python
# Confirm dataset successfully loaded from disk
print("Dataset loaded successfully from local offline files!")
print("Number of training reviews:", len(x_train))
print("Number of testing reviews :", len(x_test))
```
* `len(...)`: Checks and prints the length of the arrays, confirming both train and test sets have exactly 25,000 reviews.

---

### Code Cell [3]: Understand the Dataset & Decode Words

```python
# Display dataset shapes and counts
print("Training labels shape:", y_train.shape)
print("Testing labels shape :", y_test.shape)
```
* `y_train.shape`: Prints `(25000,)`, proving `y_train` is a 1D vector containing 25,000 scalar labels.

```python
# Inspect first review in encoded integer format
print("\nFirst review (first 15 word IDs):", x_train[0][:15])
print("First review label (1 = Positive, 0 = Negative):", y_train[0])
```
* `x_train[0][:15]`: Slices and prints the first 15 numbers of the very first review (e.g. `[1, 14, 22, 16, ...]`), demonstrating to the reader that raw reviews are already converted to integer token IDs.
* `y_train[0]`: Prints the integer `1`, showing that review #1 is positive.

```python
# Load word dictionary from local JSON file (100% offline)
with open("dataset/word_index.json", "r", encoding="utf-8") as f:
    word_index = json.load(f)
```
* `with open(..., encoding="utf-8")`: Safely opens the local JSON file. Explicitly specifying `utf-8` ensures compatibility across Windows, Mac, and Linux.
* `word_index = json.load(f)`: Reads the file into a Python dictionary where keys are English words (e.g. `'film'`) and values are their popularity ranks (e.g. `19`).

```python
# Invert dictionary to map integer IDs back to words
# Note: IDs 0, 1, 2 are reserved for padding, start token, and unknown words (offset by 3)
reverse_word_index = {value: key for key, value in word_index.items()}
```
* Dictionary comprehension that swaps `{word: integer_id}` into `{integer_id: word}` so we can look up an integer and get the original English word back.

```python
# Decode first 50 integer tokens back to human-readable English text
decoded_review = " ".join([reverse_word_index.get(i - 3, "?") for i in x_train[0][:50]])
print("\nDecoded review text (first 50 words):")
print(decoded_review)
```
* `i - 3`: Subtracts 3 from each token ID because Keras offsets vocabulary words by 3 to reserve indices 0 (`<PAD>`), 1 (`<START>`), and 2 (`<UNK>`).
* `.get(i - 3, "?")`: Looks up the word in the dictionary; if not found, returns a question mark `"?"`.
* `" ".join(...)`: Concatenates the decoded words into a single readable English string.

---

### Code Cell [4]: Data Preprocessing (Padding)

```python
# Set a fixed sequence length for all reviews
max_length = 256
```
* `max_length = 256`: Defines the fixed sequence length for every review tensor.

```python
# Standardize training reviews to exactly 256 tokens (truncate longer, pad shorter with 0)
x_train = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')

# Standardize testing reviews to exactly 256 tokens
x_test = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')
```
* `pad_sequences(...)`:
  - `maxlen=256`: Enforces an exact sequence length of 256.
  - `padding='post'`: Appends `0` at the end of the review if it has fewer than 256 words.
  - `truncating='post'`: Drops words from the end if the review is longer than 256 words.
* Converted from a Python list of variable lengths into a uniform 2D numerical matrix.

```python
# Verify that all review sequences now have an identical shape
print("Shape of x_train after padding:", x_train.shape)
print("Shape of x_test after padding :", x_test.shape)
```
* Prints `(25000, 256)` for both, proving the dataset is now a 2D matrix ready for batch training.

---

### Code Cell [5]: Build the Deep Learning Model

```python
# Build a simple sequential Deep Learning model
model = keras.Sequential([
```
* `keras.Sequential([...])`: Initializes a feedforward linear stack of neural network layers where data flows sequentially from the first layer to the last.

```python
    # Explicit input layer defines sequence dimensions upfront so model summary builds cleanly without warnings
    keras.Input(shape=(max_length,)),
```
* `keras.Input(shape=(256,))`: Tells Keras upfront that each input sample is a vector of 256 numbers. This allows Keras 3 to pre-calculate all shapes and parameter counts in `model.summary()` without throwing any warnings.

```python
    # Layer 1: Embedding Layer - converts word IDs into 16-dimensional continuous vectors
    layers.Embedding(input_dim=vocab_size, output_dim=16),
```
* `layers.Embedding(input_dim=10000, output_dim=16)`:
  - `input_dim=10000`: We have 10,000 unique word IDs in our vocabulary.
  - `output_dim=16`: Each word ID is mapped to a trainable 16-dimensional continuous vector.
  - Transforms input from shape `(None, 256)` to `(None, 256, 16)`.
  - Learnable parameters: $10,000 \times 16 = 160,000$.

```python
    # Layer 2: Global Average Pooling - averages vectors across sequence to produce a 16-dim summary
    layers.GlobalAveragePooling1D(),
```
* `layers.GlobalAveragePooling1D()`: Averages the 256 word vectors across the sequence length dimension, collapsing shape `(None, 256, 16)` into a single 16-dimensional summary vector `(None, 16)`. Has 0 trainable parameters.

```python
    # Layer 3: Dense Hidden Layer - learns non-linear sentiment patterns with ReLU activation
    layers.Dense(16, activation='relu'),
```
* `layers.Dense(16, activation='relu')`:
  - A fully connected layer with 16 artificial neurons.
  - `activation='relu'`: Applies $f(x) = \max(0, x)$ to introduce non-linearity and prevent vanishing gradients.
  - Parameters: $(16 \text{ inputs} \times 16 \text{ neurons}) + 16 \text{ biases} = 272$.

```python
    # Layer 4: Output Layer - single neuron with Sigmoid activation (outputs probability 0.0 to 1.0)
    layers.Dense(1, activation='sigmoid')
])
```
* `layers.Dense(1, activation='sigmoid')`:
  - Single output neuron representing the binary sentiment prediction.
  - `activation='sigmoid'`: Squashes linear combinations into $[0.0, 1.0]$, representing the probability of a Positive review.
  - Parameters: $(16 \text{ inputs} \times 1 \text{ neuron}) + 1 \text{ bias} = 17$.

```python
# Display full model summary showing layer names, output shapes, and parameter counts
model.summary()
```
* Prints the formatted summary table displaying all 4 layers, output dimensions, and total parameter count (**160,289**).

---

### Code Cell [6]: Compile the Model

```python
# Configure learning settings for training
model.compile(
    # Adam optimizer: efficiently adjusts learning rate during training
    optimizer='adam',
    # Binary Crossentropy: standard loss function for two-class (positive/negative) problems
    loss='binary_crossentropy',
    # Accuracy: evaluates percentage of reviews correctly classified
    metrics=['accuracy']
)
```
* `model.compile(...)`: Configures how the network learns before training starts.
  - `optimizer='adam'`: Adaptive learning rate optimization combining Momentum and RMSProp.
  - `loss='binary_crossentropy'`: Mathematical loss function quantifying how far predicted probabilities are from true labels (0 or 1).
  - `metrics=['accuracy']`: Tells Keras to compute and display percentage classification accuracy after each epoch.

```python
print("Model successfully compiled!")
```
* Simple confirmation log that the model graph is compiled.

---

### Code Cell [7]: Train the Model

```python
# Set number of full training cycles over dataset
epochs = 10
# Set number of samples processed per gradient update
batch_size = 512
```
* `epochs = 10`: The model will pass through the entire training dataset 10 times.
* `batch_size = 512`: 512 reviews are grouped together per forward/backward pass. Larger batch sizes stabilize gradients and execute quickly on multi-core CPUs via SIMD vectorization.

```python
# Train the neural network
history = model.fit(
    x_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    # Set aside 20% of training data (5,000 reviews) for real-time validation
    validation_split=0.2,
    verbose=1
)
```
* `model.fit(...)`: The core training loop that executes forward passes, loss calculation, backpropagation, and weight updates.
  - `validation_split=0.2`: 20% (5,000 reviews) is held out as a validation set to evaluate unseen performance at the end of each epoch.
  - `verbose=1`: Displays live progress bars and metrics for each epoch.
  - `history`: Returns a History object storing training and validation loss/accuracy across all 10 epochs.

---

### Code Cell [8]: Plot Training and Validation Accuracy

```python
# Create figure for accuracy curves
plt.figure(figsize=(8, 5))
```
* `plt.figure(figsize=(8, 5))`: Initializes a new Matplotlib figure window with width 8 inches and height 5 inches.

```python
# Plot training accuracy across epochs
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o', color='royalblue')

# Plot validation accuracy across epochs
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s', color='darkorange')
```
* `history.history['accuracy']`: Extracts the list of 10 training accuracy values recorded during `model.fit()`.
* `marker='o'`: Adds circular markers on each epoch point.
* `history.history['val_accuracy']`: Extracts the list of 10 validation accuracy values.
* `marker='s'`: Adds square markers on validation points.

```python
# Format plot title, axis labels, legend, and grid
plt.title('Training and Validation Accuracy vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
```
* Labels the axes, title, adds a legend identifying the blue vs. orange lines, enables a background grid, and renders the plot.

---

### Code Cell [9]: Plot Training and Validation Loss

```python
# Create figure for loss curves
plt.figure(figsize=(8, 5))

# Plot training loss across epochs
plt.plot(history.history['loss'], label='Training Loss', marker='o', color='crimson')

# Plot validation loss across epochs
plt.plot(history.history['val_loss'], label='Validation Loss', marker='s', color='darkorange')

# Format plot title, axis labels, legend, and grid
plt.title('Training and Validation Loss vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
```
* Identical plotting structure as Cell [8], but extracts `'loss'` (crimson line) and `'val_loss'` (orange line) to verify that loss decreases smoothly over training.

---

### Code Cell [10]: Evaluate the Model on Unseen Test Data

```python
# Evaluate final trained model on unseen test set (25,000 reviews)
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
```
* `model.evaluate(x_test, y_test, verbose=0)`:
  - Runs all 25,000 unseen test reviews through the trained model.
  - Computes the exact test loss and test accuracy.
  - `verbose=0`: Suppresses progress bars for clean printing.

```python
# Print final evaluation metrics
print("=" * 35)
print(f"Final Test Loss    : {test_loss:.4f}")
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
print("=" * 35)
```
* Prints formatted summary showing test loss (~`0.3703`) and accuracy (~`85.10%`).

---

### Code Cell [11]: Generate Predictions on 5 Test Dataset Reviews

```python
# Select first 5 reviews from the test dataset
sample_indices = [0, 1, 2, 3, 4]
sample_reviews = x_test[sample_indices]
sample_labels = y_test[sample_indices]
```
* Takes the first 5 padded review vectors and their ground-truth labels from `x_test`.

```python
# Generate model probability predictions
predictions = model.predict(sample_reviews, verbose=0)
```
* `model.predict(...)`: Feeds the 5 samples forward through the network and returns a 2D NumPy array of 5 probability values between 0.0 and 1.0.

```python
# Display predictions alongside actual labels
print("--- Predictions on 5 Test Dataset Reviews ---\n")
for i, (prob, actual) in enumerate(zip(predictions, sample_labels)):
    predicted_label = "Positive" if prob[0] >= 0.5 else "Negative"
    actual_label = "Positive" if actual == 1 else "Negative"
    status = "CORRECT" if predicted_label == actual_label else "INCORRECT"
    
    print(f"Sample #{i + 1}:")
    print(f"  Confidence Score   : {prob[0]:.4f}")
    print(f"  Predicted Sentiment: {predicted_label}")
    print(f"  Actual Sentiment   : {actual_label}")
    print(f"  Result             : {status}\n")
```
* `zip(predictions, sample_labels)`: Pairs each predicted probability with its true label.
* `prob[0] >= 0.5`: If probability is 0.5 or higher, sets prediction to `"Positive"`, otherwise `"Negative"`.
* Checks if `predicted_label == actual_label` and prints `"CORRECT"` or `"INCORRECT"`.

---

### Code Cell [12]: Custom Review Prediction Function

```python
# Define beginner-friendly helper function to test arbitrary review text
def predict_sentiment(review_text):
```
* Defines the inference function accepting any arbitrary English string `review_text`.

```python
    # Step 1: Convert review text to lowercase and remove punctuation
    clean_text = ""
    for char in review_text.lower():
        if char.isalnum() or char.isspace():
            clean_text += char
        else:
            clean_text += " "
```
* Iterates through characters in the lowercased review string:
  - `char.isalnum()`: Keeps letters and numbers.
  - `char.isspace()`: Keeps spaces.
  - Replaces all punctuation (`!`, `.`, `?`, `,`, `-`, etc.) with spaces to separate words cleanly.

```python
    # Split cleaned string into individual words
    words = clean_text.split()
    
    # Validate that input contains readable words
    if not words:
        print(f"Review: \"{review_text}\"")
        print("Result: Please enter a valid text review with words.\n")
        return
```
* `clean_text.split()`: Splits the string by whitespace into a Python list of words.
* `if not words`: Input validation guard. If the input is empty or contains only symbols, prints a polite message and safely exits.

```python
    # Step 2: Convert words to integer token IDs using the IMDb word dictionary
    # Offset rules: 0=<PAD>, 1=<START>, 2=<UNK> (unknown word)
    tokens = [1]  # Begin sequence with start token
    for word in words:
        # Check if word exists in dictionary and falls within vocabulary size
        if word in word_index and (word_index[word] + 3) < vocab_size:
            tokens.append(word_index[word] + 3)
        else:
            tokens.append(2)  # Assign unknown token if word not recognized
```
* Initializes `tokens = [1]` because index `1` is the standard `<START>` token in IMDb.
* For each word in the review:
  - Checks if word exists in `word_index` and its adjusted index `(word_index[word] + 3)` is within our 10,000 vocabulary limit.
  - If valid, appends `word_index[word] + 3`.
  - If the word is unknown or rare ($> 10000$), appends `2` (`<UNK>`).

```python
    # Step 3: Pad the sequence to fixed length 256
    padded = pad_sequences([tokens], maxlen=max_length, padding='post', truncating='post')
```
* `[tokens]`: Wraps the 1D token list into a 2D batch list of shape `(1, num_tokens)`.
* `pad_sequences`: Pads trailing zeros up to 256 tokens, matching the exact shape expected by `model.predict()`.

```python
    # Step 4: Generate prediction probability score (0.0 to 1.0)
    score = float(model.predict(padded, verbose=0)[0][0])
    
    # Classify sentiment using 0.5 threshold
    sentiment = "POSITIVE" if score >= 0.5 else "NEGATIVE"
```
* `model.predict(padded, verbose=0)[0][0]`: Feeds the padded tensor into the model, extracting the single scalar probability value.
* `sentiment = "POSITIVE" if score >= 0.5 else "NEGATIVE"`: Assigns final binary sentiment.

```python
    # Print formatted, beginner-friendly output
    print(f"Review: \"{review_text}\"")
    print(f"Score : {score:.4f}")
    print(f"Result: {sentiment}\n")
```
* Displays the review, confidence score formatted to 4 decimal places, and the classification result.

```python
# Test the function with sample positive and negative reviews
print("--- Custom Review Predictions ---\n")
predict_sentiment("This movie was absolutely wonderful with great acting and a brilliant plot!")
predict_sentiment("Terrible film, completely boring, horrible acting and total waste of time.")
```
* Tests the function with a clear positive and negative example to demonstrate working real-time inference.

---

### Dataset Preparation Script: `prepare_dataset.py`

```python
import os
import json
import numpy as np
from keras.datasets import imdb
```
* Imports file path utilities (`os`), JSON serializer (`json`), array handler (`numpy`), and Keras's built-in IMDb downloader (`imdb`).

```python
def prepare_offline_dataset():
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
```
* `os.path.abspath(__file__)`: Determines the absolute directory where the script is located.
* `os.makedirs(dataset_dir, exist_ok=True)`: Creates the `dataset/` directory if it does not already exist without throwing an error.

```python
    vocab_size = 10000
    print(f"Downloading IMDb dataset with top {vocab_size} vocabulary words...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
```
* `imdb.load_data(num_words=10000)`: Downloads the IMDb dataset once from storage.googleapis.com, pre-filtered to the top 10,000 words.

```python
    # Paths for saving
    x_train_path = os.path.join(dataset_dir, "x_train.npy")
    y_train_path = os.path.join(dataset_dir, "y_train.npy")
    x_test_path = os.path.join(dataset_dir, "x_test.npy")
    y_test_path = os.path.join(dataset_dir, "y_test.npy")
    word_index_path = os.path.join(dataset_dir, "word_index.json")
    
    print("Saving train and test sets to 'dataset/'...")
    np.save(x_train_path, x_train, allow_pickle=True)
    np.save(y_train_path, y_train)
    np.save(x_test_path, x_test, allow_pickle=True)
    np.save(y_test_path, y_test)
```
* `np.save(...)`: Serializes the arrays into binary `.npy` format on disk. Once written, the final notebook loads them without ever touching `imdb.load_data()`.

```python
    print("Downloading and saving IMDb word index...")
    word_index = imdb.get_word_index()
    with open(word_index_path, "w", encoding="utf-8") as f:
        json.dump(word_index, f, ensure_ascii=False)
```
* `imdb.get_word_index()`: Downloads the IMDb word-to-integer dictionary mapping.
* `json.dump(...)`: Writes the dictionary to `dataset/word_index.json` in UTF-8 encoding.

```python
    print("\nDataset preparation complete! All files saved locally in 'dataset/':")
    for fname in ["x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy", "word_index.json"]:
        fpath = os.path.join(dataset_dir, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  - {fname} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    prepare_offline_dataset()
```
* Loops through the generated files and prints their file sizes in kilobytes, confirming successful offline preparation.
* `if __name__ == "__main__":` ensures the function executes only when run directly from the terminal.
