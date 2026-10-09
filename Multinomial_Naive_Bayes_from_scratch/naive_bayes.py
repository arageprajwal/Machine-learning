import math

pos_label = 1
neg_label = 0

def train(pos_train, neg_train, vocab, alpha):
    # Calculate the prior probabilities
    vocab_size = len(vocab)
    n = len(pos_train) + len(neg_train)
    pr_pos = len(pos_train) / n
    pr_neg = len(neg_train) / n

    counts = {pos_label: {}, neg_label: {}}

    # Count the occurrences of each word in the positive and negative training instances
    for doc in pos_train:
        for token in doc:
            if token not in counts[pos_label]:
                counts[pos_label][token] = 0
            counts[pos_label][token] += 1
    for doc in neg_train:
        for token in doc:
            if token not in counts[neg_label]:
                counts[neg_label][token] = 0
            counts[neg_label][token] += 1

    class_total_counts = {pos_label: sum(counts[pos_label].values()), neg_label: sum(counts[neg_label].values())}

    # Calculate the conditional probabilities with Laplace smoothing
    pos_dict = {}
    neg_dict = {}
    for token in vocab:
        pos_dict[token] = (counts[pos_label].get(token, 0) + alpha) / (class_total_counts[pos_label] + alpha * vocab_size)
        neg_dict[token] = (counts[neg_label].get(token, 0) + alpha) / (class_total_counts[neg_label] + alpha * vocab_size)

    pos_log_probs = {token: math.log(prob) if prob > 0 else float('-inf') for token, prob in pos_dict.items()}
    neg_log_probs = {token: math.log(prob) if prob > 0 else float('-inf') for token, prob in neg_dict.items()}

    return { "prior": {pos_label: pr_pos, neg_label: pr_neg}, 
            "log_prior": {pos_label: math.log(pr_pos), neg_label: math.log(pr_neg)},
            "probs": {pos_label: pos_dict, neg_label: neg_dict},
            "log_probs": {pos_label: pos_log_probs, neg_label: neg_log_probs} }

def predict(model, doc, use_log = True):

    if use_log:
        pos_log_score = model["log_prior"][pos_label]
        neg_log_score = model["log_prior"][neg_label]

        for token in doc:
            if token in model["log_probs"][pos_label]:
                pos_log_score += model["log_probs"][pos_label][token]

            if token in model["log_probs"][neg_label]:
                neg_log_score += model["log_probs"][neg_label][token]

        return pos_label if pos_log_score > neg_log_score else neg_label
    
    else:
        pos_score = model["prior"][pos_label]
        neg_score = model["prior"][neg_label]

        for token in doc:
            if token in model["probs"][pos_label]:
                pos_score *= model["probs"][pos_label][token]

            if token in model["probs"][neg_label]:
                neg_score *= model["probs"][neg_label][token]

        return pos_label if pos_score > neg_score else neg_label

def compute_metrics(tp, fp, fn, tn):
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {"accuracy": accuracy, 
            "precision": precision, 
            "recall": recall, 
            "confusion_matrix": [[tp, fn], [fp, tn]]}

def evaluate(model, pos_test, neg_test, use_log = True):
    tp,fp,fn,tn = 0,0,0,0
    for doc in pos_test:
        pred = predict(model, doc, use_log)
        if pred == pos_label:
            tp += 1
        else:
            fn += 1
    for doc in neg_test:
        pred = predict(model, doc, use_log)
        if pred == neg_label:
            tn += 1
        else:
            fp += 1
    return compute_metrics(tp, fp, fn, tn)