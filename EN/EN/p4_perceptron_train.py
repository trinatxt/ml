import json
from collections import defaultdict

TRAIN_FILE = "train.unk"
WEIGHTS_FILE = "perceptron_weights.json"
TAGS = set()

START = "<s>"
STOP = "</s>"
UNK = "#UNK#"
EPOCHS = 5

def extract_features(sentence, tags):
    features = []
    for i in range(len(sentence)):
        word = sentence[i][0]
        tag = tags[i]
        prev_tag = tags[i-1] if i > 0 else START

        feats = [
            f"TAG:{tag}",
            f"TAG:{tag}_WORD:{word}",
            f"BIGRAM:{prev_tag}->{tag}",
        ]
        features.extend(feats)
    return features

def update_weights(weights, features, scale):
    for feat in features:
        weights[feat] += scale

def score_sequence(weights, sentence, tags):
    score = 0
    for i in range(len(sentence)):
        word = sentence[i][0]
        tag = tags[i]
        prev_tag = tags[i-1] if i > 0 else START

        feats = [
            f"TAG:{tag}",
            f"TAG:{tag}_WORD:{word}",
            f"BIGRAM:{prev_tag}->{tag}",
        ]
        for feat in feats:
            score += weights.get(feat, 0)
    return score

def viterbi_decode(sentence, weights, possible_tags):
    n = len(sentence)
    v = [{} for _ in range(n)]
    bp = [{} for _ in range(n)]

    for tag in possible_tags:
        prev_tag = START
        feats = [
            f"TAG:{tag}",
            f"TAG:{tag}_WORD:{sentence[0][0]}",
            f"BIGRAM:{prev_tag}->{tag}",
        ]
        w_sum = sum(weights.get(feat, 0) for feat in feats)
        v[0][tag] = w_sum
        bp[0][tag] = prev_tag

    for i in range(1, n):
        word = sentence[i][0]
        for tag in possible_tags:
            best_score = float("-inf")
            best_prev = None
            for prev_tag in v[i-1]:  # only use reachable previous tags
                feats = [
                    f"TAG:{tag}",
                    f"TAG:{tag}_WORD:{word}",
                    f"BIGRAM:{prev_tag}->{tag}",
                ]
                w_sum = sum(weights.get(feat, 0) for feat in feats)
                score = v[i-1][prev_tag] + w_sum

                if score > best_score:
                    best_score = score
                    best_prev = prev_tag
            if best_prev is not None:
                v[i][tag] = best_score
                bp[i][tag] = best_prev

    best_final_tag = max(v[n-1], key=v[n-1].get)
    tags = [best_final_tag]
    for i in range(n-1, 0, -1):
        tags.insert(0, bp[i][tags[0]])
    return tags

def train():
    weights = defaultdict(int)
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
        print(f"\nEpoch {epoch+1}")
        for idx, sent in enumerate(sentences):
            gold_tags = [tag for _, tag in sent]
            pred_tags = viterbi_decode(sent, weights, TAGS)
            if gold_tags != pred_tags:
                gold_feats = extract_features(sent, gold_tags)
                pred_feats = extract_features(sent, pred_tags)
                update_weights(weights, gold_feats, +1)
                update_weights(weights, pred_feats, -1)
            else:
                correct += 1
            total += 1
            if idx % 500 == 0:
                print(f"  Processed {idx}/{len(sentences)}")

        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}: Accuracy = {correct}/{total} ({acc:.2f}%)")

    # Convert defaultdict to normal dict for JSON dumping
    weights_dict = dict(weights)
    with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(weights_dict, f)

if __name__ == "__main__":
    train()
