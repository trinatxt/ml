import json

DEV_FILE = "dev.in"
OUTPUT_FILE = "dev.p4.out"
WEIGHTS_FILE = "perceptron_weights.json"
START = "<s>"

def extract_features(word, tag, prev_tag):
    return [
        f"TAG:{tag}",
        f"TAG:{tag}_WORD:{word}",
        f"BIGRAM:{prev_tag}->{tag}",
    ]

def viterbi_decode(sentence, weights, tag_set):
    n = len(sentence)
    v = [{} for _ in range(n)]
    bp = [{} for _ in range(n)]

    for tag in tag_set:
        feats = extract_features(sentence[0], tag, START)
        v[0][tag] = sum(weights.get(feat, 0) for feat in feats)
        bp[0][tag] = START

    for i in range(1, n):
        word = sentence[i]
        for tag in tag_set:
            best_score = float("-inf")
            best_prev = None
            for prev_tag in v[i-1]:
                feats = extract_features(word, tag, prev_tag)
                score = v[i-1][prev_tag] + sum(weights.get(feat, 0) for feat in feats)
                if score > best_score:
                    best_score = score
                    best_prev = prev_tag
            if best_prev is not None:
                v[i][tag] = best_score
                bp[i][tag] = best_prev

    best_last_tag = max(v[-1], key=v[-1].get)
    tags = [best_last_tag]
    for i in range(n-1, 0, -1):
        tags.insert(0, bp[i][tags[0]])
    return tags

def main():
    with open(WEIGHTS_FILE, "r", encoding="utf-8") as f:
        weights = json.load(f)

    # Infer the tag set from weights
    tag_set = set()
    for feat in weights:
        if feat.startswith("TAG:") and "_WORD" not in feat:
            tag = feat.split(":")[1]
            tag_set.add(tag)

    with open(DEV_FILE, "r", encoding="utf-8") as f:
        sentences = []
        sentence = []
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

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for sentence in sentences:
            tags = viterbi_decode(sentence, weights, tag_set)
            for word, tag in zip(sentence, tags):
                fout.write(f"{word} {tag}\n")
            fout.write("\n")

if __name__ == "__main__":
    main()
