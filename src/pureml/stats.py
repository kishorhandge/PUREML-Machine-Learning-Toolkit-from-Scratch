"""
stats.py
--------
Core statistical operations built from scratch.

Functions:
    mean        -- arithmetic average
    median      -- middle value
    mode        -- most frequent value(s)
    variance    -- population variance
    std_dev     -- population standard deviation
    covariance  -- joint variability of two variables
    correlation -- Pearson correlation coefficient
    percentile  -- value at given percentile rank
"""

import math

__all__ = [
    "mean",
    "median",
    "mode",
    "variance",
    "std_dev",
    "covariance",
    "correlation",
    "percentile",
]


def _check_nonempty(arr, name="arr"):
    if len(arr) == 0:
        raise ValueError(f"'{name}' must not be empty")


# --------------------------------------------------
# Mean
# μ = (1/n) * Σ xi
# --------------------------------------------------
def mean(arr):
    """Return the arithmetic mean of a list of numbers."""
    _check_nonempty(arr)
    return sum(arr) / len(arr)


# --------------------------------------------------
# Median
# Middle value of a sorted sequence
# --------------------------------------------------
def median(arr):
    """Return the median (middle value) of a list."""
    _check_nonempty(arr)
    sorted_arr = sorted(arr)
    n = len(sorted_arr)
    mid = n // 2
    if n % 2 == 1:
        return sorted_arr[mid]
    else:
        return (sorted_arr[mid - 1] + sorted_arr[mid]) / 2.0


# --------------------------------------------------
# Mode
# Most frequently occurring value(s)
# Returns a list (there may be multiple modes)
# --------------------------------------------------
def mode(arr):
    """Return the most frequent value(s) as a list."""
    _check_nonempty(arr)
    freq = {}
    for v in arr:
        freq[v] = freq.get(v, 0) + 1
    max_count = max(freq.values())
    return sorted([k for k, c in freq.items() if c == max_count])


# --------------------------------------------------
# Population Variance
# σ² = (1/n) * Σ (xi - μ)²
# --------------------------------------------------
def variance(arr):
    """Return the population variance of a list."""
    _check_nonempty(arr)
    m = mean(arr)
    return sum((v - m) ** 2 for v in arr) / len(arr)


# --------------------------------------------------
# Standard Deviation
# σ = sqrt(σ²)
# --------------------------------------------------
def std_dev(arr):
    """Return the population standard deviation."""
    return math.sqrt(variance(arr))


# --------------------------------------------------
# Covariance
# cov(X,Y) = (1/n) * Σ (xi - x̄)(yi - ȳ)
# --------------------------------------------------
def covariance(x, y):
    """
    Return the population covariance of two equal-length lists.

    Positive → both move together.
    Negative → they move in opposite directions.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    _check_nonempty(x, "x")
    x_bar = mean(x)
    y_bar = mean(y)
    n = len(x)
    return sum((x[i] - x_bar) * (y[i] - y_bar) for i in range(n)) / n


# --------------------------------------------------
# Pearson Correlation Coefficient
# r = cov(X,Y) / (σx * σy)
# Range: [-1, 1]
# --------------------------------------------------
def correlation(x, y):
    """
    Return the Pearson correlation coefficient.

    +1 = perfect positive,  0 = none,  -1 = perfect negative.
    Returns 0.0 if either list has zero standard deviation.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    _check_nonempty(x, "x")
    sx = std_dev(x)
    sy = std_dev(y)
    if sx == 0 or sy == 0:
        return 0.0
    return covariance(x, y) / (sx * sy)


# --------------------------------------------------
# Percentile
# Value below which a given percentage of data falls
# Uses linear interpolation (same as numpy's default)
# --------------------------------------------------
def percentile(arr, p):
    """
    Return the p-th percentile of arr.

    Parameters
    ----------
    arr : list of numbers
    p   : float in [0, 100]
    """
    _check_nonempty(arr)
    if not (0 <= p <= 100):
        raise ValueError("p must be in [0, 100]")
    sorted_arr = sorted(arr)
    n = len(sorted_arr)
    # Compute the rank index (0-based, with interpolation)
    index = (p / 100.0) * (n - 1)
    lower = int(index)
    upper = lower + 1
    fraction = index - lower
    if upper >= n:
        return float(sorted_arr[-1])
    return sorted_arr[lower] + fraction * (sorted_arr[upper] - sorted_arr[lower])