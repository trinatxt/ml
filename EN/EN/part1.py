from collections import defaultdict
import sys

# ===== Step 1: Read and Preprocess Training Data =====
def read_labeled_data(filepath):
    """Read labeled data (word, tag) and split into sentences."""
    sentences = []
    current_sentence = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                parts = line.split()
                if len(parts) != 2:
                    continue  # Skip malformed lines
                word, tag = parts
                current_sentence.append((word, tag))
    if current_sentence:
        sentences.append(current_sentence)
    return sentences

def replace_rare_words(sentences, k=3):
    """Replace words appearing <k times with #UNK#."""
    word_counts = defaultdict(int)
    for sentence in sentences:
        for word, _ in sentence:
            word_counts[word] += 1

    modified_sentences = []
    for sentence in sentences:
        modified_sentence = []
        for word, tag in sentence:
            if word_counts[word] < k:
                modified_sentence.append(("#UNK#", tag))
            else:
                modified_sentence.append((word, tag))
        modified_sentences.append(modified_sentence)
    return modified_sentences

# ===== Step 2: Estimate Emission Parameters =====
def estimate_emissions(sentences):
    """Compute emission probabilities e(x|y) = Count(y→x)/Count(y)."""
    tag_counts = defaultdict(int)
    emission_counts = defaultdict(lambda: defaultdict(int))

    for sentence in sentences:
        for word, tag in sentence:
            tag_counts[tag] += 1
            emission_counts[tag][word] += 1

    emission_probs = defaultdict(dict)
    for tag in emission_counts:
        for word in emission_counts[tag]:
            emission_probs[tag][word] = emission_counts[tag][word] / tag_counts[tag]

    known_words = set(word for tag in emission_counts for word in emission_counts[tag])
    return emission_probs, known_words, set(tag_counts.keys())

# ===== Step 3: Tag Development Data =====
def read_unlabeled_data(filepath):
    """Read unlabeled data (words only)."""
    sentences = []
    current_sentence = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence.append(line)
    if current_sentence:
        sentences.append(current_sentence)
    return sentences

def predict_tags(sentence, emission_probs, known_words, possible_tags):
    """Predict tags using argmax_y e(x|y)."""
    tagged_sentence = []
    for word in sentence:
        if word not in known_words:
            word = "#UNK#"
        
        # Find tag with highest emission probability
        best_tag = None
        best_prob = -1
        
        for tag in possible_tags:
            prob = emission_probs[tag].get(word, 0)
            if prob > best_prob:
                best_prob = prob
                best_tag = tag
        
        # Default to "O" if no tag has a non-zero probability
        if best_tag is None:
            best_tag = "O"
            
        tagged_sentence.append((word, best_tag))
    return tagged_sentence

# ===== Step 4: Write Predictions =====
def write_output(sentences, filepath):
    """Write predictions to file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for word, tag in sentence:
                f.write(f"{word} {tag}\n")
            f.write("\n")

# ===== Step 5: Evaluation (Precision/Recall/F1) =====
def extract_chunks(tag_sequence):
    """Convert tag sequence to chunks (start, end, type)."""
    chunks = []
    current_chunk = None
    for i, (word, tag) in enumerate(tag_sequence):
        if tag.startswith("B-"):
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = (i, i+1, tag[2:])
        elif tag.startswith("I-"):
            chunk_type = tag[2:]
            if current_chunk and current_chunk[2] == chunk_type:
                current_chunk = (current_chunk[0], i+1, chunk_type)
            else:
                # Handle O -> I-X transition (treat I- as B- in this case)
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = (i, i+1, chunk_type)
        else:  # O tag
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = None
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def evaluate(gold_file, pred_file):
    """Compute precision, recall, and F1."""
    gold_sentences = read_labeled_data(gold_file)
    pred_sentences = read_labeled_data(pred_file)

    # Make sure we have same number of sentences
    if len(gold_sentences) != len(pred_sentences):
        print(f"Warning: Gold has {len(gold_sentences)} sentences but predictions have {len(pred_sentences)}")
        # Use the smaller number
        min_sentences = min(len(gold_sentences), len(pred_sentences))
        gold_sentences = gold_sentences[:min_sentences]
        pred_sentences = pred_sentences[:min_sentences]

    gold_chunks = []
    pred_chunks = []

    for gold_sent, pred_sent in zip(gold_sentences, pred_sentences):
        gold_chunks.extend(extract_chunks(gold_sent))
        pred_chunks.extend(extract_chunks(pred_sent))

    gold_set = set(gold_chunks)
    pred_set = set(pred_chunks)

    correct = len(gold_set & pred_set)
    total_pred = len(pred_set)
    total_gold = len(gold_set)

    precision = correct / total_pred if total_pred > 0 else 0
    recall = correct / total_gold if total_gold > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

# ===== Main =====
if __name__ == "__main__":
    # File paths (update as needed)
    TRAIN_FILE = "train"
    DEV_IN_FILE = "dev.in"
    DEV_OUT_FILE = "dev.out"
    OUTPUT_FILE = "dev.p1.out"

    # Step 1: Preprocess training data
    train_sentences = read_labeled_data(TRAIN_FILE)
    train_sentences = replace_rare_words(train_sentences, k=3)

    # Step 2: Estimate emission parameters
    emission_probs, known_words, possible_tags = estimate_emissions(train_sentences)

    # Step 3: Tag dev data
    dev_sentences = read_unlabeled_data(DEV_IN_FILE)
    tagged_sentences = [
        predict_tags(sentence, emission_probs, known_words, possible_tags)
        for sentence in dev_sentences
    ]

    # Step 4: Write predictions
    write_output(tagged_sentences, OUTPUT_FILE)

    # Step 5: Evaluate
    # precision, recall, f1 = evaluate(DEV_OUT_FILE, OUTPUT_FILE)
    # print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")