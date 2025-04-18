# Sequence Labeling Assignment — Part 2 & Part 4

This project contains implementations for sequence labeling using:

- **Part 2**: HMM + Viterbi decoding (generative baseline)
- **Part 4**: Structured Perceptron (discriminative model)

## 📁 File Structure


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

