import math
from collections import defaultdict, Counter
import heapq

# Constants
K_UNK = 3
UNK_TOKEN = "#UNK#"

def read_labeled_data(file_path):
    """Read labeled data (word, tag) and split into sentences."""
    sentences = []
    sentence = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                parts = line.split()
                if len(parts) != 2:
                    continue  # Skip malformed lines
                word, tag = parts
                sentence.append((word, tag))
    if sentence:
        sentences.append(sentence)
    return sentences

def read_unlabeled_data(file_path):
    """Read unlabeled data (words only)."""
    sentences = []
    sentence = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                sentence.append(line)
    if sentence:
        sentences.append(sentence)
    return sentences

def replace_rare_words(sentences, k=K_UNK):
    """Replace words appearing <k times with #UNK#."""
    word_freq = Counter(word for sent in sentences for word, _ in sent)
    new_sentences = []
    for sent in sentences:
        new_sent = []
        for word, tag in sent:
            if word_freq[word] < k:
                new_sent.append((UNK_TOKEN, tag))
            else:
                new_sent.append((word, tag))
        new_sentences.append(new_sent)
    return new_sentences

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

def estimate_transitions(sentences):
    """Compute transition probabilities q(yi|yi-1) = Count(yi-1,yi)/Count(yi-1)"""
    transition_counts = defaultdict(lambda: defaultdict(int))
    tag_counts = defaultdict(int)
    
    # Add START and STOP counts
    tag_counts['START'] = 0
    
    for sentence in sentences:
        prev_tag = 'START'
        tag_counts[prev_tag] += 1
        
        for _, tag in sentence:
            transition_counts[prev_tag][tag] += 1
            tag_counts[tag] += 1
            prev_tag = tag
        
        # Transition to STOP
        transition_counts[prev_tag]['STOP'] += 1
    
    # Compute probabilities
    transition_probs = defaultdict(dict)
    for prev_tag in transition_counts:
        for tag in transition_counts[prev_tag]:
            transition_probs[prev_tag][tag] = transition_counts[prev_tag][tag] / tag_counts[prev_tag]
    
    return transition_probs

def viterbi_kbest(sentence, emission_probs, transition_probs, known_words, possible_tags, k=4):
    """Find the k-best sequences using modified Viterbi algorithm"""
    n = len(sentence)
    if n == 0:
        return []
    
    # Initialize data structures
    # pi[i][tag] will store top k (prob, path) tuples for tag at position i
    pi = [defaultdict(list) for _ in range(n+2)]
    pi[0]['START'] = [(0.0, ['START'])]
    
    # Forward pass
    for i in range(1, n+1):
        word = sentence[i-1]
        if word not in known_words:
            word = UNK_TOKEN
        
        for curr_tag in possible_tags:
            # Get emission probability (with smoothing for UNK)
            e = emission_probs[curr_tag].get(word, 0)
            if e == 0:
                e = 1e-10  # Smoothing for unknown words
            
            # Collect all possible transitions
            candidates = []
            for prev_tag in pi[i-1]:
                if curr_tag in transition_probs.get(prev_tag, {}):
                    q = transition_probs[prev_tag][curr_tag]
                    for (prev_prob, prev_path) in pi[i-1][prev_tag]:
                        new_prob = prev_prob + math.log(q) + math.log(e)
                        new_path = prev_path + [curr_tag]
                        candidates.append((new_prob, new_path))
            
            # Keep top k candidates for current tag
            if candidates:
                pi[i][curr_tag] = heapq.nlargest(k, candidates, key=lambda x: x[0])
    
    # Handle STOP state
    candidates = []
    for prev_tag in pi[n]:
        if 'STOP' in transition_probs.get(prev_tag, {}):
            q = transition_probs[prev_tag]['STOP']
            for (prev_prob, prev_path) in pi[n][prev_tag]:
                new_prob = prev_prob + math.log(q)
                new_path = prev_path + ['STOP']
                candidates.append((new_prob, new_path))
    
    # Get top k paths (excluding START and STOP tags)
    if not candidates:
        # Fallback for if no valid paths found
        return ['O'] * n
    
    top_k = heapq.nlargest(k, candidates, key=lambda x: x[0])
    
    # Get the 4th best sequence if available
    if len(top_k) >= 4:
        fourth_best = top_k[3][1][1:-1]  # Remove START and STOP tags
    elif top_k:
        # If fewer than 4 paths exist, return the last one
        fourth_best = top_k[-1][1][1:-1]
    else:
        # Shouldn't happen due to earlier check, but just in case
        fourth_best = ['O'] * n
    
    return fourth_best

def predict_4th_best_tags(sentences, emission_probs, transition_probs, known_words, possible_tags):
    """Predict tags using 4th-best Viterbi sequence"""
    tagged_sentences = []
    for sentence in sentences:
        tags = viterbi_kbest(sentence, emission_probs, transition_probs, known_words, possible_tags, k=4)
        tagged_sentence = list(zip(sentence, tags))
        tagged_sentences.append(tagged_sentence)
    return tagged_sentences

def write_output(sentences, filepath):
    """Write predictions to file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for word, tag in sentence:
                f.write(f"{word} {tag}\n")
            f.write("\n")

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

if __name__ == "__main__":
    # File paths
    TRAIN_FILE = "train"
    DEV_IN_FILE = "dev.in"
    DEV_OUT_FILE = "dev.out"
    OUTPUT_FILE = "dev.p3.out"
    
    # Step 1: Preprocess training data
    train_sentences = read_labeled_data(TRAIN_FILE)
    train_sentences = replace_rare_words(train_sentences, k=K_UNK)
    
    # Step 2: Estimate parameters
    emission_probs, known_words, possible_tags = estimate_emissions(train_sentences)
    transition_probs = estimate_transitions(train_sentences)
    
    # Step 3: Tag dev data with 4th best sequence
    dev_sentences = read_unlabeled_data(DEV_IN_FILE)
    tagged_sentences = predict_4th_best_tags(dev_sentences, emission_probs, transition_probs, known_words, possible_tags)
    
    # Step 4: Write predictions
    write_output(tagged_sentences, OUTPUT_FILE)
    
    # Step 5: Evaluate
    # precision, recall, f1 = evaluate(DEV_OUT_FILE, OUTPUT_FILE)
    # print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")