"""
model_selection.py
------------------
Data splitting, shuffling, and cross-validation utilities.

Functions:
    train_test_split   -- simple sequential split
    shuffle_data       -- random in-place shuffle (supports random_state)
    batch_iterator     -- divide data into mini-batches
    k_fold_split       -- k-fold cross-validation splits
    stratified_split   -- train/test split preserving class ratios
                         (works with any number of classes, not just binary)
"""

import random

__all__ = [
    "train_test_split",
    "shuffle_data",
    "batch_iterator",
    "k_fold_split",
    "stratified_split",
]


def _validate_xy(X, y):
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length: {len(X)} vs {len(y)}"
        )
    if len(X) == 0:
        raise ValueError("X and y must not be empty")


# --------------------------------------------------
# Train-Test Split
# Sequential (no shuffling)
# --------------------------------------------------
def train_test_split(X, y, test_size=0.2):
    """
    Split arrays into train and test subsets (no shuffling).

    Parameters
    ----------
    test_size : float in (0, 1) -- proportion of data for testing

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    _validate_xy(X, y)
    if not (0 < test_size < 1):
        raise ValueError("test_size must be in (0, 1)")
    n = len(X)
    split = int(n * (1 - test_size))
    return X[:split], X[split:], y[:split], y[split:]


# --------------------------------------------------
# Shuffle Data
# Shuffles X and y in unison
# --------------------------------------------------
def shuffle_data(X, y, random_state=None):
    """
    Randomly shuffle X and y together (keeping pairs matched).

    Parameters
    ----------
    random_state : int or None -- seed for reproducibility

    Returns
    -------
    X_shuffled, y_shuffled
    """
    _validate_xy(X, y)
    if random_state is not None:
        random.seed(random_state)
    combined = list(zip(X, y))
    random.shuffle(combined)
    X_shuffled = [pair[0] for pair in combined]
    y_shuffled = [pair[1] for pair in combined]
    return X_shuffled, y_shuffled


# --------------------------------------------------
# Batch Iterator
# Yields successive batches of (X_batch, y_batch)
# --------------------------------------------------
def batch_iterator(X, y, batch_size):
    """
    Divide X and y into mini-batches of size `batch_size`.

    The last batch may be smaller than batch_size.

    Parameters
    ----------
    batch_size : int -- number of samples per batch

    Returns
    -------
    list of (X_batch, y_batch) tuples
    """
    _validate_xy(X, y)
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    n = len(X)
    batches = []
    for i in range(0, n, batch_size):
        batches.append((X[i:i + batch_size], y[i:i + batch_size]))
    return batches


# --------------------------------------------------
# K-Fold Split
# --------------------------------------------------
def k_fold_split(X, y, k=5):
    """
    Split data into k equally-sized folds for cross-validation.

    Parameters
    ----------
    k : int -- number of folds (must be >= 2)

    Returns
    -------
    list of k tuples (X_fold, y_fold)
        Each fold can be used as a test set; the others form the train set.
    """
    _validate_xy(X, y)
    if k < 2:
        raise ValueError("k must be >= 2")
    if k > len(X):
        raise ValueError(f"k={k} is larger than the dataset size={len(X)}")
    n = len(X)
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = n if i == k - 1 else start + fold_size
        folds.append((X[start:end], y[start:end]))
    return folds


# --------------------------------------------------
# Stratified Split
# Preserves class proportions in both splits.
# Works with any number of distinct classes.
# --------------------------------------------------
def stratified_split(X, y, test_size=0.2, random_state=None):
    """
    Train/test split that maintains the original class distribution.

    Works with binary and multi-class labels.

    Parameters
    ----------
    test_size    : float in (0, 1) -- proportion of each class for testing
    random_state : int or None     -- seed for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    _validate_xy(X, y)
    if not (0 < test_size < 1):
        raise ValueError("test_size must be in (0, 1)")

    if random_state is not None:
        random.seed(random_state)

    # Group indices by class label
    class_indices = {}
    for i, label in enumerate(y):
        class_indices.setdefault(label, []).append(i)

    train_idx = []
    test_idx = []

    for label, indices in class_indices.items():
        indices_copy = list(indices)
        random.shuffle(indices_copy)
        n_test = max(1, int(len(indices_copy) * test_size))
        test_idx.extend(indices_copy[:n_test])
        train_idx.extend(indices_copy[n_test:])

    # Rebuild X and y from index lists
    X_train = [X[i] for i in train_idx]
    X_test  = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test  = [y[i] for i in test_idx]

    return X_train, X_test, y_train, y_test