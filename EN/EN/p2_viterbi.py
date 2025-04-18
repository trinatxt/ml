import math
from collections import defaultdict

emission_file = "emission_probs.txt"
transition_file = "transition_probs.txt"
dev_in_file = "dev.in"
dev_out_file = "dev.p2.out"

# Read emission probs
emissions = defaultdict(dict)
tags = set()
with open(emission_file, "r", encoding="utf-8") as f:
    for line in f:
        tag, word, prob = line.strip().split()
        emissions[tag][word] = float(prob)
        tags.add(tag)
UNK = "#UNK#"
for tag in tags:
    if "#UNK#" not in emissions[tag]:
        print(f"⚠️ Missing #UNK# emission for tag: {tag}")

# Read transition probs
transitions = defaultdict(dict)
with open(transition_file, "r", encoding="utf-8") as f:
    for line in f:
        prev, curr, prob = line.strip().split()
        transitions[prev][curr] = float(prob)
        tags.add(curr)

START = "<s>"
STOP = "</s>"
UNK = "#UNK#"

# BIO sequence fixer
def fix_bio_sequence(tag_seq):
    fixed = []
    prev = "O"
    for tag in tag_seq:
        if tag.startswith("I-"):
            if prev == "O" or (prev[2:] != tag[2:]):
                tag = "B-" + tag[2:]
        fixed.append(tag)
        prev = tag
    return fixed

# Viterbi decoding
def viterbi(sentence):
    n = len(sentence)
    v = [{} for _ in range(n)]
    bp = [{} for _ in range(n)]

    for tag in tags:
        if tag == STOP:
            continue  
        trans_p = transitions[START].get(tag, 1e-10)
        emis_p = emissions[tag].get(sentence[0], emissions[tag].get(UNK, 1e-10))
        v[0][tag] = math.log(trans_p) + math.log(emis_p)
        bp[0][tag] = START

    for i in range(1, n):
        for tag in tags:
            max_prob = float("-inf")
            best_prev = None
            emis_p = emissions[tag].get(sentence[i], emissions[tag].get(UNK, 1e-10))
            if emis_p == 1e-10:
                print(f"[WARN] Word '{sentence[i]}' unseen for tag '{tag}' — falling back to #UNK# or tiny prob")
            for prev_tag in tags:
                if prev_tag == STOP:
                    continue
                trans_p = transitions[prev_tag].get(tag, 1e-10)
                prob = v[i-1][prev_tag] + math.log(trans_p) + math.log(emis_p)
                if prob > max_prob:
                    max_prob = prob
                    best_prev = prev_tag
            v[i][tag] = max_prob
            bp[i][tag] = best_prev

    # Termination
    max_prob = float("-inf")
    best_last_tag = None
    for tag in tags:
        trans_p = transitions[tag].get(STOP, 1e-10)
        prob = v[n-1][tag] + math.log(trans_p)
        if prob > max_prob:
            max_prob = prob
            best_last_tag = tag

    # Backtrack
    tags_sequence = [best_last_tag]
    for i in range(n-1, 0, -1):
        tags_sequence.insert(0, bp[i][tags_sequence[0]])

    # Fix BIO consistency
    return fix_bio_sequence(tags_sequence)

# Predict
with open(dev_in_file, "r", encoding="utf-8") as fin, open(dev_out_file, "w", encoding="utf-8") as fout:
    sentence = []
    for line in fin:
        line = line.strip()
        if line == "":
            if sentence:
                tags_seq = viterbi(sentence)
                for word, tag in zip(sentence, tags_seq):
                    fout.write(f"{word} {tag}\n")
                fout.write("\n")
                sentence = []
        else:
            sentence.append(line)

    if sentence:  # last sentence (no trailing newline)
        tags_seq = viterbi(sentence)
        for word, tag in zip(sentence, tags_seq):
            fout.write(f"{word} {tag}\n")
