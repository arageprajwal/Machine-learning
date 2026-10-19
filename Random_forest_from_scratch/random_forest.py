import numpy as np
from decision_tree import DecisionTree
from collections import Counter

class RandomForest:
  def __init__(self, ntree=10, numerical_features=None, m=None, max_depth=None,
               min_size_for_split=2, min_gain=0.0, use_gini=False, seed=0):
    self.ntree = ntree
    self.numerical_features = numerical_features
    self.m = m                                  # None => round(sqrt(#attributes))
    self.max_depth = max_depth
    self.min_size_for_split = min_size_for_split
    self.min_gain = min_gain
    self.use_gini = use_gini
    self.rng = np.random.default_rng(seed)
    self.trees = []
 
  def bootstrap(self, X, y):
    # Sample len(X) instances WITH replacement from the training set.
    idx = self.rng.integers(0, len(X), size=len(X))
    return X[idx], y[idx]
 
  def fit(self, X, y):
    X, y = np.asarray(X, dtype=object), np.asarray(y, dtype=object)
    m = self.m if self.m is not None else max(1, int(round(np.sqrt(X.shape[1]))))
    self.trees = []
    for _ in range(self.ntree):
      Xb, yb = self.bootstrap(X, y)
      tree = DecisionTree(self.numerical_features, m=m, max_depth=self.max_depth,
                          min_size_for_split=self.min_size_for_split, min_gain=self.min_gain,
                          use_gini=self.use_gini, rng=self.rng)
      self.trees.append(tree.fit(Xb, yb))
    return self
 
  def predict(self, X):
    # Majority vote over all trees (ties go to the label that reached the tie first).
    X = np.asarray(X, dtype=object)
    votes = np.array([t.predict(X) for t in self.trees], dtype=object)    # ntree x n_instances
    return [Counter(votes[:, j]).most_common(1)[0][0] for j in range(votes.shape[1])]
 
