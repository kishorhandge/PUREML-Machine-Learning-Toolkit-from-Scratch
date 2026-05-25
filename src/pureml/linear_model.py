"""
linear_model.py
---------------
Regression and classification linear models built from scratch.

Simple (single-feature) models:
    linear_regression    -- OLS closed-form (1 feature)
    predict              -- apply slope + intercept to a list
    gradient_descent     -- 1-feature gradient descent trainer

Multi-feature models:
    multi_linear_regression   -- OLS with multiple features (normal equation)
    multi_predict             -- apply weight vector to a 2-D sample list
    multi_gradient_descent    -- multi-feature gradient descent trainer

Logistic:
    logistic_regression  -- sigmoid of a single dot product
    logistic_predict_proba -- vectorised logistic predictions
    logistic_train       -- train multi-feature logistic regression
"""

import math

__all__ = [
    "linear_regression",
    "predict",
    "gradient_descent",
    "multi_linear_regression",
    "multi_predict",
    "multi_gradient_descent",
    "logistic_regression",
    "logistic_predict_proba",
    "logistic_train",
]


# ──────────────────────────────────────────────────
# Simple (single-feature) linear regression
# ──────────────────────────────────────────────────

def linear_regression(X, y):
    """
    Ordinary Least Squares for a single feature.

    Uses the closed-form formula:
        slope     = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        intercept = (Σy - slope*Σx) / n

    Parameters
    ----------
    X : list of float -- feature values (1-D)
    y : list of float -- target values

    Returns
    -------
    (slope, intercept) : tuple of float
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if len(X) == 0:
        raise ValueError("X and y must not be empty")

    n = len(X)
    sum_x  = sum(X)
    sum_y  = sum(y)
    sum_xy = sum(X[i] * y[i] for i in range(n))
    sum_xx = sum(X[i] ** 2   for i in range(n))

    denom = n * sum_xx - sum_x ** 2
    if denom == 0:
        raise ValueError(
            "Cannot compute slope: all X values are identical (zero variance)"
        )

    slope     = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


def predict(X, slope, intercept):
    """
    Apply a linear model  ŷ = slope * x + intercept  to each x in X.

    Returns a list of predictions.
    """
    return [slope * x + intercept for x in X]


def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    """
    Train a single-feature linear model with batch gradient descent.

    Returns
    -------
    (slope, intercept) : tuple of float
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    n = len(X)
    slope = 0.0
    intercept = 0.0

    for _ in range(epochs):
        slope_grad = 0.0
        intercept_grad = 0.0

        for i in range(n):
            error = (slope * X[i] + intercept) - y[i]
            slope_grad     += error * X[i]
            intercept_grad += error

        slope     -= learning_rate * (2.0 / n) * slope_grad
        intercept -= learning_rate * (2.0 / n) * intercept_grad

    return slope, intercept


# ──────────────────────────────────────────────────
# Multi-feature linear regression
# ──────────────────────────────────────────────────

def _dot(a, b):
    """Dot product of two equal-length lists."""
    return sum(a[i] * b[i] for i in range(len(a)))


def multi_linear_regression(X, y):
    """
    OLS for multiple features using the normal equation.

    WARNING: Naive O(n * f²) Gram-matrix inversion — fine for educational
    use with small feature counts.

    Parameters
    ----------
    X : list of lists -- shape (n_samples, n_features)
    y : list of float -- target values

    Returns
    -------
    weights : list of float -- [intercept, w1, w2, ...]
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if len(X) == 0:
        raise ValueError("X and y must not be empty")

    n = len(X)
    f = len(X[0])

    # Add bias column (intercept) → X_b shape (n, f+1)
    X_b = [[1.0] + list(row) for row in X]
    m = f + 1  # number of columns including bias

    # Gram matrix G = Xᵀ X  (m × m)
    G = [[0.0] * m for _ in range(m)]
    for row in X_b:
        for i in range(m):
            for j in range(m):
                G[i][j] += row[i] * row[j]

    # Xᵀ y  (m,)
    Xty = [0.0] * m
    for idx in range(n):
        for i in range(m):
            Xty[i] += X_b[idx][i] * y[idx]

    # Solve G * w = Xty via Gaussian elimination with partial pivoting
    # Augmented matrix [G | Xty]
    aug = [G[i][:] + [Xty[i]] for i in range(m)]

    for col in range(m):
        # Find pivot
        max_row = col
        for row in range(col + 1, m):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular — features may be collinear")

        # Eliminate
        for row in range(m):
            if row == col:
                continue
            factor = aug[row][col] / pivot
            for j in range(col, m + 1):
                aug[row][j] -= factor * aug[col][j]

    weights = [aug[i][m] / aug[i][i] for i in range(m)]
    return weights


def multi_predict(X, weights):
    """
    Predict using a multi-feature linear model.

    Parameters
    ----------
    X       : list of lists -- shape (n_samples, n_features)
    weights : list of float -- [intercept, w1, w2, ...]

    Returns
    -------
    list of float -- predictions
    """
    preds = []
    for row in X:
        val = weights[0] + sum(weights[j + 1] * row[j] for j in range(len(row)))
        preds.append(val)
    return preds


def multi_gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    """
    Train a multi-feature linear model with batch gradient descent.

    Parameters
    ----------
    X            : list of lists -- shape (n_samples, n_features)
    y            : list of float
    learning_rate: float
    epochs       : int

    Returns
    -------
    weights : list of float -- [intercept, w1, w2, ...]
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    n = len(X)
    f = len(X[0])
    # weights[0] = intercept, weights[1..f] = feature weights
    weights = [0.0] * (f + 1)

    for _ in range(epochs):
        grads = [0.0] * (f + 1)
        for i in range(n):
            pred = weights[0] + sum(weights[j + 1] * X[i][j] for j in range(f))
            error = pred - y[i]
            grads[0] += error
            for j in range(f):
                grads[j + 1] += error * X[i][j]
        for j in range(f + 1):
            weights[j] -= learning_rate * (2.0 / n) * grads[j]

    return weights


# ──────────────────────────────────────────────────
# Logistic Regression
# ──────────────────────────────────────────────────

def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def logistic_regression(x, weights):
    """
    Compute P(y=1) for a single sample using the logistic function.

    Parameters
    ----------
    x       : list of float -- feature vector
    weights : list of float -- [intercept, w1, w2, ...]

    Returns
    -------
    float -- probability in (0, 1)
    """
    z = weights[0] + sum(weights[j + 1] * x[j] for j in range(len(x)))
    return _sigmoid(z)


def logistic_predict_proba(X, weights):
    """
    Return P(y=1) for every sample in X.

    Returns
    -------
    list of float
    """
    return [logistic_regression(row, weights) for row in X]


def logistic_train(X, y, learning_rate=0.01, epochs=1000):
    """
    Train a logistic regression model with gradient descent.

    Parameters
    ----------
    X            : list of lists -- shape (n_samples, n_features)
    y            : list of int   -- binary labels {0, 1}
    learning_rate: float
    epochs       : int

    Returns
    -------
    weights : list of float -- [intercept, w1, w2, ...]
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    n = len(X)
    f = len(X[0])
    weights = [0.0] * (f + 1)

    for _ in range(epochs):
        grads = [0.0] * (f + 1)
        for i in range(n):
            pred  = logistic_regression(X[i], weights)
            error = pred - y[i]
            grads[0] += error
            for j in range(f):
                grads[j + 1] += error * X[i][j]
        for j in range(f + 1):
            weights[j] -= learning_rate * (1.0 / n) * grads[j]

    return weights