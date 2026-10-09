import json
import os
import random

from utils import load_training_set, load_test_set
import naive_bayes
import matplotlib.pyplot as plt


def _saveResults(name, results):
    os.makedirs("results", exist_ok=True)
    with open("results/results_" + name, 'w') as f:
        json.dump(results, f)
    print(json.dumps(results, indent=2))

def q1():
    pos_train, neg_train, vocab = load_training_set(0.2, 0.2)
    pos_test, neg_test = load_test_set(0.2, 0.2)

    model = naive_bayes.train(pos_train, neg_train, vocab, alpha=0)
    results = naive_bayes.evaluate(model, pos_test, neg_test, use_log=False)

    _saveResults("q1.json", results)

def q2():
    pos_train, neg_train, vocab = load_training_set(0.2, 0.2)
    pos_test, neg_test = load_test_set(0.2, 0.2)

    # alpha = 1
    model = naive_bayes.train(pos_train, neg_train, vocab, alpha=1)
    results_alpha_1 = naive_bayes.evaluate(model, pos_test, neg_test)
    _saveResults("q2_alpha_1.json", results_alpha_1)

    #multiple alphas
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    accuracies = []
    for alpha in alphas:
        model = naive_bayes.train(pos_train, neg_train, vocab, alpha=alpha)
        results = naive_bayes.evaluate(model, pos_test, neg_test)
        accuracies.append(results["accuracy"])

    plt.plot(alphas, accuracies)
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Alpha')
    plt.savefig('results/q2_accuracy_vs_alpha.png')

    _saveResults(
        "q2_accuracies.json",
        {str(alphas[i]): accuracies[i] for i in range(len(alphas))},
    )

    best_alpha = alphas[accuracies.index(max(accuracies))]
    print(f"Best alpha: {best_alpha} with accuracy: {max(accuracies)}")
    return best_alpha

def q3(best_alpha, pos_test, neg_test):
    pos_train, neg_train, vocab = load_training_set(1.0, 1.0)
    #pos_test, neg_test = load_test_set(1.0, 1.0)

    model = naive_bayes.train(pos_train, neg_train, vocab, alpha=best_alpha)
    results = naive_bayes.evaluate(model, pos_test, neg_test)
    _saveResults("q3.json", results)

def q4(best_alpha, pos_test, neg_test):
    pos_train, neg_train, vocab = load_training_set(0.3, 0.3)

    model = naive_bayes.train(pos_train, neg_train, vocab, alpha=best_alpha)
    results = naive_bayes.evaluate(model, pos_test, neg_test)
    _saveResults("q4.json", results)

def q6(best_alpha, pos_test, neg_test):
    pos_train, neg_train, vocab = load_training_set(0.1, 0.5)
    #pos_test, neg_test = load_test_set(1.0, 1.0)

    model = naive_bayes.train(pos_train, neg_train, vocab, alpha=best_alpha)
    results = naive_bayes.evaluate(model, pos_test, neg_test)
    _saveResults("q6.json", results)

if __name__ == '__main__':
    random.seed(42)  # For reproducibility
    best_alpha = None
    q1()
    best_alpha = q2()
    if best_alpha is None:
        with open("results_q2_accuracies.json", "r") as f:
            accuracies = json.load(f)

        best_alpha = float(max(accuracies, key=accuracies.get))
    print(best_alpha)
    pos_test, neg_test = load_test_set(1.0, 1.0)

    q3(best_alpha, pos_test, neg_test)
    q4(best_alpha, pos_test, neg_test)
    q6(best_alpha, pos_test, neg_test)
