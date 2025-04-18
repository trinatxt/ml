from collections import defaultdict

# Special symbols
START = "<s>"
STOP = "</s>"
UNK = "#UNK#"

# File paths
train_file = "train.unk"
output_emission = "emission_probs.txt"
output_transition = "transition_probs.txt"

def load_sentences(filepath):
    sentences = []
    sentence = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                word, tag = line.split()
                sentence.append((word, tag))
        if sentence:
            sentences.append(sentence)
    return sentences

def estimate_emission_probs(sentences, unk_smoothing=1):
    emission_counts = defaultdict(lambda: defaultdict(int))
    tag_counts = defaultdict(int)

    for sentence in sentences:
        for word, tag in sentence:
            emission_counts[tag][word] += 1
            tag_counts[tag] += 1

    # Add UNK token if missing
    for tag in tag_counts:
        if UNK not in emission_counts[tag]:
            emission_counts[tag][UNK] = unk_smoothing
            tag_counts[tag] += unk_smoothing

    emission_probs = defaultdict(dict)
    for tag in emission_counts:
        total = tag_counts[tag]
        for word in emission_counts[tag]:
            emission_probs[tag][word] = emission_counts[tag][word] / total

    return emission_probs

def estimate_transition_probs(sentences):
    transition_counts = defaultdict(lambda: defaultdict(int))
    for sentence in sentences:
        prev_tag = START
        for _, tag in sentence:
            transition_counts[prev_tag][tag] += 1
            prev_tag = tag
        transition_counts[prev_tag][STOP] += 1

    transition_probs = defaultdict(dict)
    for prev_tag in transition_counts:
        total = sum(transition_counts[prev_tag].values())
        for curr_tag in transition_counts[prev_tag]:
            transition_probs[prev_tag][curr_tag] = transition_counts[prev_tag][curr_tag] / total

    return transition_probs

def write_emission_probs(emission_probs, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for tag in emission_probs:
            for word in emission_probs[tag]:
                prob = emission_probs[tag][word]
                f.write(f"{tag} {word} {prob}\n")

def write_transition_probs(transition_probs, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for prev_tag in transition_probs:
            for curr_tag in transition_probs[prev_tag]:
                prob = transition_probs[prev_tag][curr_tag]
                f.write(f"{prev_tag} {curr_tag} {prob}\n")

def main():
    sentences = load_sentences(train_file)
    emission_probs = estimate_emission_probs(sentences)
    transition_probs = estimate_transition_probs(sentences)
    write_emission_probs(emission_probs, output_emission)
    write_transition_probs(transition_probs, output_transition)

if __name__ == "__main__":
    main()
