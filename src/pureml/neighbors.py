"""
neighbors.py
============
K-Nearest Neighbours (KNN) utilities for a mini ML library.

Formulas implemented
--------------------
41. KNN Distance  : d(x, xi) = sqrt( sum( (xj - xij)^2 ) )
42. Prediction    : ŷ = mode of k nearest labels
43. Probability   : P(class | x) = count_of_class / k
"""

import math
from collections import Counter


# ---------------------------------------------------------------------------
# 41. KNN Distance
# ---------------------------------------------------------------------------

def knn_distance(x, y):
    """
    Compute the Euclidean distance between two feature vectors.

    Parameters
    ----------
    x : list or tuple of numbers
    y : list or tuple of numbers  (must be same length as x)

    Returns
    -------
    float  — Euclidean distance d(x, y)

    Raises
    ------
    TypeError   if x or y are not list/tuple, or contain non-numeric values
    ValueError  if x or y are empty, or have different lengths
    """
    if not isinstance(x, (list, tuple)) or not isinstance(y, (list, tuple)):
        raise TypeError("Both x and y must be list or tuple.")

    if len(x) == 0 or len(y) == 0:
        raise ValueError("Feature vectors must not be empty.")

    if len(x) != len(y):
        raise ValueError(
            f"Length mismatch: x has {len(x)} features, y has {len(y)} features."
        )

    total = 0.0
    for xi, yi in zip(x, y):
        if not isinstance(xi, (int, float)) or not isinstance(yi, (int, float)):
            raise TypeError("All feature values must be numeric (int or float).")
        total += (xi - yi) ** 2

    return math.sqrt(total)


# ---------------------------------------------------------------------------
# 42. Prediction  (mode of k nearest labels)
# ---------------------------------------------------------------------------

def knn_predict(labels):
    """
    Return the most frequent label (mode) from a list of k-nearest labels.

    In case of a tie, the label that appears first among the tied labels
    (i.e. closest neighbour wins) is returned — a common, reproducible
    tie-breaking strategy.

    Parameters
    ----------
    labels : list
        The labels of the k nearest neighbours, ordered by distance
        (nearest first).

    Returns
    -------
    The predicted class label.

    Raises
    ------
    TypeError  if labels is not a list or tuple
    ValueError if labels is empty
    """
    if not isinstance(labels, (list, tuple)):
        raise TypeError("labels must be a list or tuple.")

    if len(labels) == 0:
        raise ValueError("labels must not be empty.")

    # Count frequencies
    freq = Counter(labels)
    max_count = max(freq.values())

    # Among all labels with the max count, pick the one that appears
    # earliest in the original list (nearest-neighbour tie-breaking).
    for label in labels:
        if freq[label] == max_count:
            return label


# ---------------------------------------------------------------------------
# 43. Probability
# ---------------------------------------------------------------------------

def probability(labels, target_class):
    """
    Compute the KNN probability estimate for a given class.

    P(class | x) = count_of_class / k

    Parameters
    ----------
    labels       : list — labels of the k nearest neighbours
    target_class : the class whose probability is requested

    Returns
    -------
    float in [0.0, 1.0]

    Raises
    ------
    TypeError  if labels is not a list or tuple
    ValueError if labels is empty
    """
    if not isinstance(labels, (list, tuple)):
        raise TypeError("labels must be a list or tuple.")

    if len(labels) == 0:
        raise ValueError("labels must not be empty.")

    k = len(labels)
    count = sum(1 for label in labels if label == target_class)
    return count / k


# ---------------------------------------------------------------------------
# Bonus: full KNN classify helper
# ---------------------------------------------------------------------------

def knn_classify(query, X_train, y_train, k=3):
    """
    Full KNN classification pipeline.

    Given a query point, training data, and labels:
    1. Compute distance from query to every training point.
    2. Sort by distance and pick the k nearest neighbours.
    3. Return the predicted label and a probability dict for all classes.

    Parameters
    ----------
    query   : list/tuple — the feature vector to classify
    X_train : list of list/tuple — training feature vectors
    y_train : list — training labels (same length as X_train)
    k       : int — number of neighbours (default 3)

    Returns
    -------
    dict with keys:
        'prediction'  : predicted class label
        'probability' : dict mapping each neighbour class to its probability
        'neighbours'  : list of (distance, label) for the k nearest points

    Raises
    ------
    TypeError  if X_train / y_train are not lists/tuples
    ValueError if lengths are inconsistent, or k < 1 or k > len(X_train)
    """
    if not isinstance(X_train, (list, tuple)) or not isinstance(y_train, (list, tuple)):
        raise TypeError("X_train and y_train must be list or tuple.")

    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train has {len(X_train)} samples but y_train has {len(y_train)} labels."
        )

    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")

    if k > len(X_train):
        raise ValueError(
            f"k ({k}) cannot be greater than the number of training samples ({len(X_train)})."
        )

    # Step 1: compute distances
    distances = []
    for xi, yi in zip(X_train, y_train):
        d = knn_distance(query, xi)
        distances.append((d, yi))

    # Step 2: sort ascending by distance and take k nearest
    distances.sort(key=lambda pair: pair[0])
    k_nearest = distances[:k]

    # Step 3: extract neighbour labels
    neighbour_labels = [label for _, label in k_nearest]

    # Step 4: predict and compute class probabilities
    prediction = knn_predict(neighbour_labels)

    unique_classes = list(dict.fromkeys(neighbour_labels))   # preserve order
    prob_dict = {cls: probability(neighbour_labels, cls) for cls in unique_classes}

    return {
        "prediction": prediction,
        "probability": prob_dict,
        "neighbours": k_nearest,
    }


# ---------------------------------------------------------------------------
# Quick self-test  (run: python neighbors.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Distance
    print("=== knn_distance ===")
    print(knn_distance([1, 2], [4, 6]))          # expected: 5.0
    print(knn_distance([0, 0, 0], [1, 1, 1]))    # expected: ~1.732

    # Predict
    print("\n=== knn_predict ===")
    print(knn_predict(["cat", "dog", "cat"]))    # expected: cat
    print(knn_predict([1, 2, 2, 1, 2]))          # expected: 2

    # Probability
    print("\n=== probability ===")
    print(probability(["cat", "dog", "cat"], "cat"))   # expected: 0.667
    print(probability(["cat", "dog", "cat"], "dog"))   # expected: 0.333

    # Full classify
    print("\n=== knn_classify ===")
    X_train = [[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]]
    y_train = ["A",    "A",    "A",    "B",    "B",    "B"]
    result = knn_classify([5, 5], X_train, y_train, k=3)
    print("Prediction :", result["prediction"])
    print("Probability:", result["probability"])
    print("Neighbours :", result["neighbours"])
