import json

DEV_FILE = "dev.in"   # or "test.in" if testing
OUTPUT_FILE = "dev.p4.out"  # or "v3.test.p4.out" if testing
WEIGHTS_FILE = "perceptron_weights.json"
START = "<s>"
STOP = "</s>"

# Full tag set manually defined
TAG_SET = {
    'B-NP', 'I-NP', 'B-VP', 'I-VP', 'B-ADJP', 'I-ADJP',
    'B-ADVP', 'I-ADVP', 'B-PP', 'I-PP', 'O'
}

def extract_features(sentence, i, tag, prev_tag):
    word = sentence[i]
    prev_word = sentence[i-1] if i > 0 else START
    next_word = sentence[i+1] if i < len(sentence)-1 else STOP

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

def viterbi_decode(sentence, weights, tag_set):
    n = len(sentence)
    v = [{} for _ in range(n)]
    bp = [{} for _ in range(n)]

    for tag in tag_set:
        feats = extract_features(sentence, 0, tag, START)
        v[0][tag] = sum(weights.get(f, 0) for f in feats)
        bp[0][tag] = START

    for i in range(1, n):
        for tag in tag_set:
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

def main():
    with open(WEIGHTS_FILE, "r", encoding="utf-8") as f:
        weights = json.load(f)

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
            tags = viterbi_decode(sentence, weights, TAG_SET)
            for word, tag in zip(sentence, tags):
                fout.write(f"{word} {tag}\n")
            fout.write("\n")

if __name__ == "__main__":
    main()
