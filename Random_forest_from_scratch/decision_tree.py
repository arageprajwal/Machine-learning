import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class Node:
  def __init__(self,
               information_gain = None,
               feature_index = None,
               majority_class=None,
               children = None,
               value = None,
               threshold = None):
    # conditional node
    self.information_gain = information_gain
    self.feature_index = feature_index
    self.majority_class = majority_class # for fallback -- if an instance with a new feature_value is found while testing which was not used to construct the decision tree
    self.children = children if children is not None else {}
    self.threshold = threshold  
    # leaf node
    self.value = value

class DecisionTree:
  def __init__(self, numerical_features=None, m=None, max_depth=None,
               min_size_for_split=2, min_gain=0.0, use_gini=False, rng=None):
    self.numerical_features = set(numerical_features) if numerical_features is not None else set()
    self.m = m
    self.max_depth = max_depth
    self.min_size_for_split = min_size_for_split
    self.min_gain = min_gain
    self.use_gini = use_gini
    self.rng = rng if rng is not None else np.random.default_rng()
    self.root = None
 
  # ---- training
  def fit(self, X, y):
    data = np.concatenate((np.asarray(X, dtype=object), np.asarray(y, dtype=object).reshape(-1, 1)), axis=1)
    self.root = self.build_decision_tree(data, list(range(data.shape[1] - 1)), depth=0)
    return self
 
  def build_decision_tree(self, data, available_features, depth):
    y = data[:, -1]
    majority_class = Counter(y).most_common(1)[0][0]
 
    stop = (len(available_features) == 0
            or len(np.unique(y)) == 1
            or len(y) < self.min_size_for_split                              # minimum_size_for_split
            or (self.max_depth is not None and depth >= self.max_depth))     # maximum_depth
    if stop:
      return Node(value=majority_class)
 
    split = self.get_ideal_split(data, available_features)
 
    if split["feature_index"] is None or split["information_gain"] <= self.min_gain:   # minimum_gain
      return Node(value=majority_class)
 
    f = split["feature_index"]
    # categorical features are used up after a split; numerical ones can be split again deeper
    remaining = list(available_features) if f in self.numerical_features \
                else [a for a in available_features if a != f]
 
    children = {key: self.build_decision_tree(sub, remaining, depth + 1)
                for key, sub in split["children"].items()}
 
    return Node(information_gain=split["information_gain"], feature_index=f,
                majority_class=majority_class, children=children, threshold=split["threshold"])
 
  def get_ideal_split(self, data, available_features):
    # random attribute selection: only m of the available attributes are candidates at this node
    if self.m is not None and self.m < len(available_features):
      candidates = self.rng.choice(available_features, size=self.m, replace=False)
    else:
      candidates = available_features
 
    best = {"information_gain": -1, "feature_index": None, "children": {}, "threshold": None}
    parent_y = data[:, -1]
 
    for f in candidates:
      f = int(f)
      if f in self.numerical_features:
        threshold, gain = self.find_best_threshold(data[:, f], parent_y)
        if threshold is None:
          continue
        children = self.split_data_numerical(data, f, threshold)
      else:
        threshold = None
        children = self.split_data(data, f)
        if len(children) <= 1:
          continue
        gain = self.compute_information_gain(parent_y, [c[:, -1] for c in children.values()])
 
      if gain > best["information_gain"]:
        best = {"information_gain": gain, "feature_index": f, "children": children, "threshold": threshold}
    return best
 
  def find_best_threshold(self, x, y):
    """Sort by value, score the midpoint between each pair of consecutive distinct values, keep the best."""
    x = x.astype(float)
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    n = len(ys)
 
    classes, y_idx = np.unique(ys, return_inverse=True)
    onehot = np.eye(len(classes))[y_idx]
    left_counts = np.cumsum(onehot, axis=0)[:-1]
    right_counts = onehot.sum(axis=0) - left_counts
    n_left = np.arange(1, n)
    n_right = n - n_left
 
    def impurity(counts, sizes):
      p = counts / sizes[:, None]
      if self.use_gini:
        return 1 - np.sum(p ** 2, axis=1)
      with np.errstate(divide="ignore", invalid="ignore"):
        return -np.sum(np.where(p > 0, p * np.log2(p), 0.0), axis=1)
 
    parent = self.get_gini_index(ys) if self.use_gini else self.get_entropy(ys)
    gains = parent - (n_left / n) * impurity(left_counts, n_left) - (n_right / n) * impurity(right_counts, n_right)
 
    valid = xs[1:] != xs[:-1]                 # can't split between identical values
    if not valid.any():
      return None, -1
    gains = np.where(valid, gains, -np.inf)
    best = int(np.argmax(gains))
    threshold = (xs[best] + xs[best + 1]) / 2
    if threshold >= xs[best + 1]:             # float round-off guard: keep right side non-empty
      threshold = xs[best]
    return threshold, gains[best]
 
  def split_data_numerical(self, data, f, threshold):
    values = data[:, f].astype(float)
    return {"<=": data[values <= threshold], ">": data[values > threshold]}
 
  def split_data(self, data, f):
    return {val: data[data[:, f] == val] for val in np.unique(data[:, f])}
 
  def compute_information_gain(self, parent_y, children_y):
    impurity = self.get_gini_index if self.use_gini else self.get_entropy
    weights = [len(c) / len(parent_y) for c in children_y]
    return impurity(parent_y) - sum(w * impurity(c) for w, c in zip(weights, children_y))
 
  def get_entropy(self, y):
    result = 0
    for label in np.unique(y):
      p = np.mean(y == label)
      result -= p * np.log2(p)
    return result
 
  def get_gini_index(self, y):
    return 1 - sum(np.mean(y == label) ** 2 for label in np.unique(y))
 
  # ---- prediction
  def predict(self, X):
    return [self._predict_row(row, self.root) for row in np.asarray(X, dtype=object)]
 
  def _predict_row(self, row, node):
    if node.value is not None:
      return node.value
    v = row[node.feature_index]
    if node.threshold is not None:
      return self._predict_row(row, node.children["<=" if float(v) <= node.threshold else ">"])
    child = node.children.get(v)
    return self._predict_row(row, child) if child is not None else node.majority_class
 