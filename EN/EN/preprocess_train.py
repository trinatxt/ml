from collections import defaultdict

UNK_THRESHOLD = 3
input_path = "train"
output_path = "train.unk"

# Step 1: Count word frequency
word_counts = defaultdict(int)
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            word = line.strip().split()[0]
            word_counts[word] += 1

# Step 2: Replace rare words with #UNK#
with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        if line.strip():
            word, tag = line.strip().split()
            if word_counts[word] < UNK_THRESHOLD:
                word = "#UNK#"
            fout.write(f"{word} {tag}\n")
        else:
            fout.write("\n")
