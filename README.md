# Sequence Labeling Assignment

This project contains implementations for sequence labeling using:

- **Part 1**: Emission-based tagging (baseline)
- **Part 2**: HMM + Viterbi decoding (generative baseline)
- **Part 3**: 4th-best Viterbi decoding
- **Part 4**: Structured Perceptron (discriminative model)

## 📁 File Descriptions

| **File** | **Description** |
|----------|-----------------|
| `part1.py` | Emission-based baseline tagging |
| `p2_train_hmm.py` | Train HMM parameters (emission + transition) |
| `p2_viterbi.py` | Viterbi decoding for HMM |
| `part3.py` | 4th-best Viterbi decoding |
| `p4_perceptron_train.py` | Train structured perceptron |
| `p4_perceptron_decode.py` | Decode with perceptron + Viterbi |
| `preprocess_train.py` | Preprocess training data for unknown words |
| `train.unk`, `dev.in`, `dev.out`, `train`, `test.in` | Data files |
| `dev.p1.out`, `dev.p2.out`, `dev.p3.out`, `dev.p4.out`, `test.p4.out` | Model output files |
| `perceptron_weights.json` | Weights for Part 4 |

## Preprocess

### Preprocess Step

Before training either model, we first preprocess the original `train` file to replace rare words with a special `#UNK#` token. This helps handle unseen words during decoding.
```
python preprocess_train.py
```
This script reads `train`, replaces words that occur once with `#UNK#`, and writes the result to `train.unk`. This file is used for both part 2 and part 4.

## ✅ Part 1: Emission-Based Tagging

Train a simple emission model and tag the data:

```
python part1.py
```
This will:
- Create a modified training set
- Train the emission probabilities
- Save output predictions for dev.in to dev.p1.out

### 2. Evaluate p1

```
cd ../EvalScript
python evalResult.py ../EN/dev.out ../EN/dev.p1.out
```

## ✅ Part 2: HMM + Viterbi

### 1. Train the HMM Model

Run the training script to estimate emission and transition probabilities from `train.unk`.

```
python p2_train_hmm.py
```
This generates:
- emission_probs.txt
- transition_probs.txt

### 2. Run Viterbi Decoding
Use the learned model to tag `dev.in`.

```
python p2_viterbi.py
```
This writes the output to `dev.p2.out`.

### 3. Evaluate p2

```
cd ../EvalScript
python evalResult.py ../EN/dev.out ../EN/dev.p2.out
```

## ✅ Part 3: 4th-Best Viterbi Tagging

### 1. Preprocess and Train

Use a modified Viterbi algorithm to find the 4th-best tag sequence for each sentence:

```
python part3.py
```

This will:
- Replace rare words in the training data
- Estimate emission and transition probabilities
- Find the 4th-best tag sequence using Viterbi
- Save predictions to dev.p3.out

### 2. Evaluate p3

```
cd ../EvalScript
python evalResult.py ../EN/dev.out ../EN/dev.p3.out
```

## ✅ Part 4

### 1. Train the perceptron model

This will train on `train.unk` and save learned feature weights to `perceptron_weights.json`.

```
python p4_perceptron_train.py
```

### 2. Decode with Perceptron + Viterbi
This will tag each sentence in `dev.in` using the trained model and output to `dev.p4.out`.
```
python p4_perceptron_decode.py
```

### 3. Evaluate p4

```
cd ../EvalScript
python evalResult.py ../EN/dev.out ../EN/dev.p4.out
```

