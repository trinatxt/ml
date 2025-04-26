import json
from collections import defaultdict

TRAIN_FILE = "train.unk"
WEIGHTS_FILE = "perceptron_weights.json"

TAGS = set()
START = "<s>"
STOP = "</s>"
UNK = "#UNK#"
EPOCHS = 15

# === Rich Feature Extraction ===
def extract_features(sentence, i, tag, prev_tag):
    word = sentence[i][0]
    prev_word = sentence[i-1][0] if i > 0 else START
    next_word = sentence[i+1][0] if i < len(sentence)-1 else STOP

    feats = [
        f"TAG:{tag}",
        f"TAG:{tag}_WORD:{word}",
        f"BIGRAM:{prev_tag}->{tag}",
        f"TAG:{tag}_PREV_WORD:{prev_word}",
        f"TAG:{tag}_NEXT_WORD:{next_word}"
    ]

    if word[0].isupper():
        feats.append(f"TAG:{tag}_CAPITALIZED")

    if len(word) >= 3:
        feats.append(f"TAG:{tag}_SUFFIX:{word[-3:]}")

    return feats

# === Weight Updates ===
def update_weights(weights, features, scale):
    for feat in features:
        weights[feat] += scale

# === Viterbi Decoding ===
def viterbi_decode(sentence, weights, possible_tags):
    n = len(sentence)
    v = [{} for _ in range(n)]
    bp = [{} for _ in range(n)]

    for tag in possible_tags:
        feats = extract_features(sentence, 0, tag, START)
        v[0][tag] = sum(weights.get(f, 0) for f in feats)
        bp[0][tag] = START

    for i in range(1, n):
        for tag in possible_tags:
            best_score = float("-inf")
            best_prev = None
            for prev_tag in v[i-1]:
                feats = extract_features(sentence, i, tag, prev_tag)
                score = v[i-1][prev_tag] + sum(weights.get(f, 0) for f in feats)
                if score > best_score:
                    best_score = score
                    best_prev = prev_tag
            if best_prev is not None:
                v[i][tag] = best_score
                bp[i][tag] = best_prev

    best_last_tag = max(v[n-1], key=v[n-1].get)
    tags = [best_last_tag]
    for i in range(n-1, 0, -1):
        tags.insert(0, bp[i][tags[0]])
    return tags

# === Train with Feature Averaging ===
def train():
    weights = defaultdict(int)
    total_weights = defaultdict(int)
    counter = 1

    sentences = []
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        sentence = []
        for line in f:
            if line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                word, tag = line.strip().split()
                TAGS.add(tag)
                sentence.append((word, tag))
        if sentence:
            sentences.append(sentence)

    for epoch in range(EPOCHS):
        correct = 0
        total = 0
        for sent in sentences:
            gold_tags = [tag for _, tag in sent]
            pred_tags = viterbi_decode(sent, weights, TAGS)

            if gold_tags != pred_tags:
                gold_feats = []
                pred_feats = []
                prev_tag_gold = START
                prev_tag_pred = START
                for i in range(len(sent)):
                    gold_feats.extend(extract_features(sent, i, gold_tags[i], prev_tag_gold))
                    pred_feats.extend(extract_features(sent, i, pred_tags[i], prev_tag_pred))
                    prev_tag_gold = gold_tags[i]
                    prev_tag_pred = pred_tags[i]
                update_weights(weights, gold_feats, +1)
                update_weights(weights, pred_feats, -1)
            else:
                correct += 1

            total += 1

            for feat in weights:
                total_weights[feat] += weights[feat]
            counter += 1

            if total % 500 == 0:
                print(f"  Processed {total}/{len(sentences)}")

        print(f"Epoch {epoch+1}: Accuracy = {correct}/{total} ({100.0 * correct / total:.2f}%)")

    avg_weights = {feat: total_weights[feat] / counter for feat in total_weights}

    with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(avg_weights, f)

if __name__ == "__main__":
    train()
