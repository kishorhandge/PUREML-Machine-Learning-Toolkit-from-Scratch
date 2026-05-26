"""
tree.py
=======
Decision Tree utilities for a mini ML library.

Functions
---------
gini_impurity       : Gini = 1 - Σ pi²
entropy             : Entropy = -Σ pi * log2(pi)
information_gain    : IG = parent_entropy - weighted_child_entropy
best_split          : Find the feature + threshold with highest IG
build_tree          : Recursively build a CART-style decision tree
predict_tree        : Traverse a built tree to classify one sample
majority_vote       : Return the most common label in a list

Helper (internal)
-----------------
_class_probabilities : Convert raw label list → probability list
_gini_from_labels    : Gini directly from labels
_entropy_from_labels : Entropy directly from labels
"""

import math
from collections import Counter


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _class_probabilities(y):
    """
    Convert a list of raw class labels into a list of class probabilities.

    Parameters
    ----------
    y : list — class labels  e.g. [0, 1, 1, 0, 1]

    Returns
    -------
    list of float — [p_class1, p_class2, ...]

    Raises
    ------
    ValueError : if y is empty
    """
    if len(y) == 0:
        raise ValueError("Label list y must not be empty.")

    counts = Counter(y)
    n = len(y)
    return [count / n for count in counts.values()]


def _gini_from_labels(y):
    """Convenience: compute Gini directly from raw labels."""
    return gini_impurity(_class_probabilities(y))


def _entropy_from_labels(y):
    """Convenience: compute entropy directly from raw labels."""
    return entropy(_class_probabilities(y))


# ---------------------------------------------------------------------------
# 1. Gini Impurity
#    Gini = 1 - Σ pi²
# ---------------------------------------------------------------------------

def gini_impurity(p):
    """
    Compute the Gini impurity from a list of class probabilities.

    Gini = 1 - Σ (pi)²

    A value of 0 means perfectly pure (all samples are one class).
    Maximum value is (1 - 1/n_classes).

    Parameters
    ----------
    p : list of float — class probabilities (each in [0, 1], should sum to 1)

    Returns
    -------
    float — Gini impurity in [0, 0.5] for binary, [0, 1) in general

    Raises
    ------
    TypeError  : if p is not a list or tuple
    ValueError : if p is empty or any value is not in [0, 1]
    """
    if not isinstance(p, (list, tuple)):
        raise TypeError("p must be a list or tuple of probabilities.")

    if len(p) == 0:
        raise ValueError("p must not be empty.")

    for pi in p:
        if not isinstance(pi, (int, float)):
            raise TypeError("All values in p must be numeric.")
        if pi < 0 or pi > 1:
            raise ValueError(f"Probability {pi} is out of range [0, 1].")

    return 1.0 - sum(pi ** 2 for pi in p)


# ---------------------------------------------------------------------------
# 2. Entropy
#    Entropy = -Σ pi * log2(pi)
# ---------------------------------------------------------------------------

def entropy(p):
    """
    Compute the Shannon entropy from a list of class probabilities.

    Entropy = -Σ pi * log2(pi)

    A value of 0 means perfectly pure. Maximum is log2(n_classes).
    Terms where pi == 0 are skipped (0 * log(0) is treated as 0).

    Parameters
    ----------
    p : list of float — class probabilities (each in [0, 1], should sum to 1)

    Returns
    -------
    float — entropy in [0, log2(n_classes)]

    Raises
    ------
    TypeError  : if p is not a list or tuple
    ValueError : if p is empty or any value is out of [0, 1]
    """
    if not isinstance(p, (list, tuple)):
        raise TypeError("p must be a list or tuple of probabilities.")

    if len(p) == 0:
        raise ValueError("p must not be empty.")

    for pi in p:
        if not isinstance(pi, (int, float)):
            raise TypeError("All values in p must be numeric.")
        if pi < 0 or pi > 1:
            raise ValueError(f"Probability {pi} is out of range [0, 1].")

    return -sum(pi * math.log2(pi) for pi in p if pi > 0)


# ---------------------------------------------------------------------------
# 3. Information Gain
#    IG = parent_entropy - Σ (child_size / parent_size) * child_entropy
# ---------------------------------------------------------------------------

def information_gain(parent_entropy, child_entropies, child_sizes, parent_size):
    """
    Compute the information gain of a split.

    IG = parent_entropy - Σ (child_size / parent_size) * child_entropy

    Parameters
    ----------
    parent_entropy  : float  — entropy of the node before splitting
    child_entropies : list   — entropy of each child node after splitting
    child_sizes     : list   — number of samples in each child node
    parent_size     : int    — total samples in the parent node

    Returns
    -------
    float — information gain (higher = better split)

    Raises
    ------
    TypeError  : on wrong input types
    ValueError : if lengths mismatch, parent_size == 0, or negative sizes
    """
    if not isinstance(child_entropies, (list, tuple)):
        raise TypeError("child_entropies must be a list or tuple.")

    if not isinstance(child_sizes, (list, tuple)):
        raise TypeError("child_sizes must be a list or tuple.")

    if len(child_entropies) != len(child_sizes):
        raise ValueError(
            "child_entropies and child_sizes must have the same length."
        )

    if parent_size <= 0:
        raise ValueError("parent_size must be a positive integer.")

    weighted = sum(
        (child_sizes[i] / parent_size) * child_entropies[i]
        for i in range(len(child_entropies))
    )

    return parent_entropy - weighted


# ---------------------------------------------------------------------------
# 4. Best Split
#    Scan every feature and every candidate threshold; pick the (feature,
#    threshold) pair that produces the highest information gain.
# ---------------------------------------------------------------------------

def best_split(X, y):
    """
    Find the best feature index and threshold to split on.

    For each feature, all unique midpoint thresholds between sorted values
    are tested. The split yielding the highest information gain is returned.

    Parameters
    ----------
    X : list of list — feature matrix  [[f0, f1, ...], ...]
    y : list         — class labels, same length as X

    Returns
    -------
    dict with keys:
        'feature'   : int   — index of best feature
        'threshold' : float — best split threshold
        'gain'      : float — information gain of this split

    Returns None if no split improves impurity (all gains are 0).

    Raises
    ------
    TypeError  : on wrong input types
    ValueError : if X / y are empty or length-mismatched
    """
    if not isinstance(X, (list, tuple)) or not isinstance(y, (list, tuple)):
        raise TypeError("X and y must be list or tuple.")

    if len(X) == 0 or len(y) == 0:
        raise ValueError("X and y must not be empty.")

    if len(X) != len(y):
        raise ValueError(
            f"X has {len(X)} samples but y has {len(y)} labels."
        )

    n_features = len(X[0])
    parent_entropy = _entropy_from_labels(y)
    parent_size = len(y)

    best_feature   = None
    best_threshold = None
    best_gain      = -1.0

    for feature_idx in range(n_features):

        # Collect all unique values for this feature, sorted
        col_values = sorted(set(row[feature_idx] for row in X))

        # Candidate thresholds = midpoints between consecutive unique values
        thresholds = [
            (col_values[i] + col_values[i + 1]) / 2.0
            for i in range(len(col_values) - 1)
        ]

        for threshold in thresholds:

            # Split samples into left (<) and right (>=) groups
            left_y  = [y[i] for i in range(parent_size)
                        if X[i][feature_idx] < threshold]
            right_y = [y[i] for i in range(parent_size)
                        if X[i][feature_idx] >= threshold]

            # Skip degenerate splits
            if len(left_y) == 0 or len(right_y) == 0:
                continue

            left_entropy  = _entropy_from_labels(left_y)
            right_entropy = _entropy_from_labels(right_y)

            gain = information_gain(
                parent_entropy,
                [left_entropy, right_entropy],
                [len(left_y),  len(right_y)],
                parent_size
            )

            if gain > best_gain:
                best_gain      = gain
                best_feature   = feature_idx
                best_threshold = threshold

    if best_feature is None:
        return None

    return {
        "feature":   best_feature,
        "threshold": best_threshold,
        "gain":      best_gain,
    }


# ---------------------------------------------------------------------------
# 5. Build Tree
#    Recursively build a CART-style classification tree.
# ---------------------------------------------------------------------------

def build_tree(X, y, depth=0, max_depth=5, min_samples_split=2):
    """
    Recursively build a decision tree using information gain (entropy).

    Each internal node is a dict:
        {
            "feature"   : int,    # which feature to split on
            "threshold" : float,  # split value  (left < threshold)
            "left"      : node,   # subtree for samples < threshold
            "right"     : node,   # subtree for samples >= threshold
        }

    Leaf nodes are plain class labels (not dicts).

    Stopping conditions
    -------------------
    - All labels are the same (pure node)
    - Maximum depth reached
    - Fewer than min_samples_split samples remain
    - No split improves impurity (best_split returns None)

    Parameters
    ----------
    X               : list of list — feature matrix
    y               : list         — class labels
    depth           : int          — current depth (start at 0)
    max_depth       : int          — max tree depth (default 5)
    min_samples_split : int        — min samples required to split (default 2)

    Returns
    -------
    dict (internal node) or class label (leaf node) or None (empty input)

    Raises
    ------
    TypeError  : on wrong input types
    ValueError : on length mismatch
    """
    if not isinstance(X, (list, tuple)) or not isinstance(y, (list, tuple)):
        raise TypeError("X and y must be list or tuple.")

    # Empty node
    if len(y) == 0:
        return None

    # --- Stopping conditions ---

    # 1. Pure node
    if len(set(y)) == 1:
        return y[0]

    # 2. Max depth reached
    if depth >= max_depth:
        return majority_vote(y)

    # 3. Too few samples to split
    if len(X) < min_samples_split:
        return majority_vote(y)

    # --- Find best split ---
    split = best_split(X, y)

    # 4. No beneficial split found
    if split is None or split["gain"] <= 0:
        return majority_vote(y)

    feature_idx = split["feature"]
    threshold   = split["threshold"]

    # --- Partition data ---
    left_indices  = [i for i in range(len(X)) if X[i][feature_idx] < threshold]
    right_indices = [i for i in range(len(X)) if X[i][feature_idx] >= threshold]

    left_X  = [X[i] for i in left_indices]
    left_y  = [y[i] for i in left_indices]
    right_X = [X[i] for i in right_indices]
    right_y = [y[i] for i in right_indices]

    # Safety: if one side is empty, return majority (shouldn't happen after
    # best_split skips degenerate splits, but guard anyway)
    if len(left_y) == 0 or len(right_y) == 0:
        return majority_vote(y)

    # --- Build subtrees ---
    return {
        "feature":   feature_idx,
        "threshold": threshold,
        "left":  build_tree(left_X,  left_y,  depth + 1, max_depth, min_samples_split),
        "right": build_tree(right_X, right_y, depth + 1, max_depth, min_samples_split),
    }


# ---------------------------------------------------------------------------
# 6. Predict Tree
#    Traverse a built tree for a single sample.
# ---------------------------------------------------------------------------

def predict_tree(tree, sample):
    """
    Traverse a built decision tree to classify one sample.

    Parameters
    ----------
    tree   : dict or class label — tree built by build_tree()
    sample : list or tuple       — feature vector for one sample

    Returns
    -------
    Predicted class label.

    Raises
    ------
    TypeError  : if sample is not a list or tuple
    ValueError : if sample is empty
    KeyError   : if tree dict is malformed (missing expected keys)
    """
    if not isinstance(sample, (list, tuple)):
        raise TypeError("sample must be a list or tuple.")

    if len(sample) == 0:
        raise ValueError("sample must not be empty.")

    # Leaf node — isinstance(tree, dict) is correct; avoids type(x) == dict
    # which would break for dict subclasses. Also handles None gracefully.
    if not isinstance(tree, dict):
        return tree

    feature   = tree["feature"]
    threshold = tree["threshold"]

    if sample[feature] < threshold:
        return predict_tree(tree["left"],  sample)
    else:
        return predict_tree(tree["right"], sample)


# ---------------------------------------------------------------------------
# 7. Majority Vote
#    Return the most frequent label (mode); nearest-first tie-breaking.
# ---------------------------------------------------------------------------

def majority_vote(y):
    """
    Return the most common class label in y.

    In case of a tie, the label that appears earliest in y wins
    (consistent, reproducible tie-breaking).

    Parameters
    ----------
    y : list — class labels

    Returns
    -------
    Most frequent class label.

    Raises
    ------
    TypeError  : if y is not a list or tuple
    ValueError : if y is empty
    """
    if not isinstance(y, (list, tuple)):
        raise TypeError("y must be a list or tuple.")

    if len(y) == 0:
        raise ValueError("y must not be empty.")

    freq = Counter(y)
    max_count = max(freq.values())

    # Pick the first label in y among those tied for the top count
    for label in y:
        if freq[label] == max_count:
            return label


# ---------------------------------------------------------------------------
# Quick self-test  (run: python tree.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=== gini_impurity ===")
    print(gini_impurity([0.5, 0.5]))          # 0.5   (max impurity, 2 classes)
    print(gini_impurity([1.0]))               # 0.0   (pure)
    print(gini_impurity([0.7, 0.3]))          # 0.42

    print("\n=== entropy ===")
    print(entropy([0.5, 0.5]))                # 1.0   (max entropy, 2 classes)
    print(entropy([1.0]))                     # 0.0   (pure)
    print(entropy([0.75, 0.25]))              # ~0.811

    print("\n=== information_gain ===")
    ig = information_gain(1.0, [0.0, 0.0], [3, 3], 6)
    print(ig)                                 # 1.0   (perfect split)
    ig2 = information_gain(1.0, [1.0, 1.0], [3, 3], 6)
    print(ig2)                                # 0.0   (no gain)

    print("\n=== majority_vote ===")
    print(majority_vote([1, 1, 0, 1]))        # 1
    print(majority_vote(["cat", "dog", "cat", "dog", "cat"]))  # cat

    print("\n=== build_tree + predict_tree ===")
    # Simple XOR-like dataset
    X_train = [
        [2.5], [1.0], [4.0], [3.5],
        [7.0], [6.5], [8.0], [9.0],
    ]
    y_train = ["low", "low", "low", "low",
               "high", "high", "high", "high"]

    tree = build_tree(X_train, y_train, max_depth=3)
    print("Tree structure:", tree)

    test_samples = [[1.5], [3.0], [7.5], [8.5]]
    for s in test_samples:
        pred = predict_tree(tree, s)
        print(f"  sample={s}  →  {pred}")
