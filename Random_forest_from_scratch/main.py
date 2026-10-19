import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random_forest import RandomForest
 
K = 5
NTREES = [1, 5, 10, 20, 30, 40, 50]
SETTINGS = [(7,5)] 
 
def stratified_kfold(y, k):
  # k disjoint folds, each with (roughly) the same class proportions as y
  folds = [[] for _ in range(k)]
  for label in np.unique(y):
    idx = np.random.permutation(np.where(y == label)[0])
    for i, part in enumerate(np.array_split(idx, k)):
      folds[i].extend(part)
  return [np.array(f) for f in folds]
 
 
def compute_metrics(y_true, y_pred, positive):
  y_pred = np.array(y_pred, dtype=object)
  tp = np.sum((y_pred == positive) & (y_true == positive))
  fp = np.sum((y_pred == positive) & (y_true != positive))
  fn = np.sum((y_pred != positive) & (y_true == positive))
  accuracy = np.mean(y_pred == y_true)
  precision = tp / (tp + fp) if tp + fp > 0 else 0
  recall = tp / (tp + fn) if tp + fn > 0 else 0
  f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
  return [accuracy, precision, recall, f1]
 
 
def cross_validate(X, y, folds, ntree, numerical, positive, max_depth, min_size):
  scores = []
  for i in range(K):
    test_idx = folds[i]
    train_idx = np.concatenate([folds[j] for j in range(K) if j != i])
    forest = RandomForest(ntree=ntree, numerical_features=numerical, max_depth=max_depth,
                          min_size_for_split=min_size, seed=i)
    forest.fit(X[train_idx], y[train_idx])
    scores.append(compute_metrics(y[test_idx], forest.predict(X[test_idx]), positive))
  return np.mean(scores, axis=0)    # average over the K folds
 
 
if __name__ == "__main__":
  np.random.seed(0)
 
  # (name, csv file, positive class label, column index of the label)
  datasets = [("WDBC", "data/wdbc.csv", 0, -1),
              ("Loan", "data/loan.csv", 1, -1),
              ("Raisins", "data/raisin.csv", 1, -1),
              ("Titanic", "data/titanic.csv", 1, 0)]
 
  summary = []    # one row per (dataset, setting)
 
  for name, path, positive, label_col in datasets:
    df = pd.read_csv(path)
    y = df.iloc[:, label_col].values.astype(object)
    features = df.drop(columns=df.columns[label_col])
    X = features.values.astype(object)
    numerical = [i for i, c in enumerate(features.columns) if c.endswith("_num")]   # columns named attrN_num
    print(name, X.shape, "numerical columns:", numerical, "class counts:", pd.Series(y).value_counts().to_dict())
 
    folds = stratified_kfold(y, K)      # same folds for every ntree and every setting
 
    for max_depth, min_size in SETTINGS:
      print("\n", name, "max_depth =", max_depth, "min_size_for_split =", min_size)
      results = []
      for ntree in NTREES:
        scores = cross_validate(X, y, folds, ntree, numerical, positive, max_depth, min_size)
        results.append(scores)
        print(name, "ntree =", ntree, "acc/prec/rec/f1 =", scores.round(4))
      results = np.array(results)
 
      # summary: average over ntree >= 20 (where the curves have levelled off)
      mean_big = results[np.array(NTREES) >= 20].mean(axis=0)
      summary.append([name, max_depth, min_size, mean_big[0], mean_big[3]])
 
      for j, metric in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
        plt.figure()
        plt.plot(NTREES, results[:, j], marker="o")
        plt.xlabel("ntree")
        plt.ylabel(metric)
        plt.title(name + ": " + metric + " vs ntree (max_depth=" + str(max_depth) + ")")
        plt.savefig("results/" + name + "_" + metric + "_depth" + str(max_depth) + "_min" + str(min_size) + ".png")
        plt.close()
 
  table = pd.DataFrame(summary, columns=["dataset", "max_depth", "min_size", "mean_acc(ntree>=20)", "mean_f1(ntree>=20)"])
  print("\n", table.round(4).to_string(index=False))
  #table.to_csv("settings_summary.csv", index=False)
