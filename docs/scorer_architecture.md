# Phoneme Scorer Architecture Design

## 1. Overview
The Scorer is an independent, lightweight neural network that operates on the offline-extracted feature caches. By decoupling it from the heavy HuBERT backbone, we can train it extremely fast on a single GPU. It predicts multi-dimensional pronunciation scores at the phoneme level for a given target word.

## 2. Input Representation
For each phoneme in a word, the model receives a concatenated feature vector:

**Acoustic Features (Precomputed):**
- `mean_embedding`: 768-dim (HuBERT last hidden state mean)
- `variance`: 1-dim (HuBERT hidden state variance)
- `energy_mean`: 1-dim (RMS energy over the span)
- `duration_z_score`: 1-dim (Normalized duration relative to canonical expectation)
- `starts_late`: 1-dim (Binary flag for gap before phoneme)

**Linguistic Features:**
- `target_phoneme`: Embedded into a dense vector (e.g., 32-dim). This tells the network *what* phoneme it should be listening for, allowing it to learn phoneme-specific acoustic distributions.

*Input Dimension per Phoneme:* $768 + 1 + 1 + 1 + 1 + 32 = 804$

## 3. Network Architecture

Modern Goodness of Pronunciation (GOP) approaches use Contextual Encoders to model co-articulation (how a phoneme sounds depends on the phonemes before and after it).

### Step 3.1: Input Projection
A linear layer reduces the 804-dim input down to a more compact representation (e.g., `d_model = 256`) and applies Layer Normalization and Dropout to prevent overfitting.

### Step 3.2: Contextual Encoder
Since words are short sequences (typically 3 to 15 phonemes), a **Lightweight Transformer Encoder** or a **Bi-directional LSTM (BiLSTM)** is optimal. 
*Recommendation:* A 2-layer Transformer Encoder (4 heads, `d_model=256`). It captures contextual dependencies perfectly and is faster to train than RNNs.

### Step 3.3: Multi-Task Scoring Heads
The output of the Contextual Encoder for each phoneme is passed through parallel shallow Multi-Layer Perceptrons (MLPs) to predict distinct aspects of pronunciation.

1.  **Match/Quality Head:** Predicts the phonetic accuracy (0-100). How close did the speaker's acoustics match the expected target phoneme?
2.  **Duration Head:** Predicts the rhythm/duration score (0-100). Was the phoneme held for the correct relative amount of time?
3.  **Presence Head:** Binary classification (Logits). Was the phoneme actually spoken, or was it omitted/skipped?

## 4. Loss Functions (Multi-Task Learning)

The model is trained to minimize a combined loss function:

$$ L_{total} = \lambda_1 L_{match} + \lambda_2 L_{duration} + \lambda_3 L_{presence} $$

-   **$L_{match}$ (MSE or Huber Loss):** Regression loss against the target phonetic quality score.
-   **$L_{duration}$ (MSE or Huber Loss):** Regression loss against the target duration score.
-   **$L_{presence}$ (BCE With Logits Loss):** Binary Cross-Entropy against the omission label.

## 5. Training Strategy (The Data Gap)
Because we are currently precomputing **LibriTTS** (which contains high-quality native speech), all phonemes implicitly have a `match_score=100`, `duration_score=100`, and `presence=1`. 

To make the model learn what "bad" pronunciation sounds like before we introduce the **SpeechOcean762** dataset (which has real human-annotated errors), we will use **Self-Supervised Negative Sampling** during training:
1.  **Phoneme Substitution (Simulated Mispronunciation):** Randomly change the `target_phoneme` input for a frame to a different phoneme, and set the target `match_score` to a low value. The model learns: "I am told to expect /P/, but the acoustics sound like /B/. Score = Low."
2.  **Simulated Omission:** Drop the acoustic features of a phoneme entirely (or replace with silence) and train the Presence Head to output 0.
3.  **Real Errors (Later):** Fine-tune the network on the `speechocean762` dataset.