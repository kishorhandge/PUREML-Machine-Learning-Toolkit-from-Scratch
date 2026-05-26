"""
mak_mini_ml
===============
A mini machine-learning library built from scratch.

Modules
-------
metrics         → evaluation metrics (accuracy, precision, recall, …)                                   
model_selection → train/test splitting, cross-validation helpers                                   
distances       → distance functions used by KNN and clustering                                   
activations     → activation functions (sigmoid, relu, softmax, …)                                   
preprocessing   → feature scaling and encoding                                   
stats           → descriptive statistics                               
linear_model    → linear & logistic regression                                   
neighbors       → K-Nearest Neighbours (KNN)                                   
tree            → decision tree (CART-style)                                   

Usage
-----
    import mak_mini_ml as ml

    tree = ml.build_tree(X_train, y_train)
    pred = ml.predict_tree(tree, sample)

    # Or import specific modules:
    from mak_mini_ml import neighbors, tree
"""

_version_ = "0.1.3"

_author_ = "Aryan Kakade, Kishor Handge"

_license_ = "MIT"

import warnings


# ---------------------------------------------------------------------------
# Safe per-module imports
# ---------------------------------------------------------------------------

_import_errors = {}


def _safe_import(module_name):

    try:
        import importlib

        mod = importlib.import_module(
            f".{module_name}",
            package=_name_
        )

        return mod

    except Exception as exc:

        _import_errors[module_name] = exc

        warnings.warn(
            f"[mak_mini_ml] Could not import '{module_name}': {exc}. "
            f"Functions from this module will be unavailable.",
            ImportWarning,
            stacklevel=3,
        )

        return None


# ---------------------------------------------------------------------------
# Load every submodule
# ---------------------------------------------------------------------------

_metrics = _safe_import("metrics")

_model_selection = _safe_import("model_selection")

_distances = _safe_import("distances")

_activations = _safe_import("activations")

_preprocessing = _safe_import("preprocessing")

_stats = _safe_import("stats")

_linear_model = _safe_import("linear_model")

_neighbors = _safe_import("neighbors")

_tree = _safe_import("tree")


# ---------------------------------------------------------------------------
# Populate the package namespace carefully
# ---------------------------------------------------------------------------

def _load_public_names(mod, mod_name):

    if mod is None:
        return

    if hasattr(mod, "_all_"):
        names = mod._all_

    else:
        names = [n for n in dir(mod) if not n.startswith("_")]

    current_globals = globals()

    for name in names:

        if not hasattr(mod, name):
            continue

        if name in current_globals:

            existing_src = getattr(
                current_globals[name],
                "_module_",
                "unknown"
            )

            warnings.warn(
                f"[mak_mini_ml] Name collision: '{name}' defined in both "
                f"'{existing_src}' and '{mod._name_}'. "
                f"Keeping the version from '{existing_src}'. "
                f"Use 'mak_mini_ml.{mod_name}.{name}' to access both.",
                UserWarning,
                stacklevel=2,
            )

        else:

            current_globals[name] = getattr(mod, name)


_load_public_names(_metrics, "metrics")

_load_public_names(_model_selection, "model_selection")

_load_public_names(_distances, "distances")

_load_public_names(_activations, "activations")

_load_public_names(_preprocessing, "preprocessing")

_load_public_names(_stats, "stats")

_load_public_names(_linear_model, "linear_model")

_load_public_names(_neighbors, "neighbors")

_load_public_names(_tree, "tree")


# ---------------------------------------------------------------------------
# Expose submodules as attributes for dotted access
# ---------------------------------------------------------------------------

if _metrics is not None:
    metrics = _metrics

if _model_selection is not None:
    model_selection = _model_selection

if _distances is not None:
    distances = _distances

if _activations is not None:
    activations = _activations

if _preprocessing is not None:
    preprocessing = _preprocessing

if _stats is not None:
    stats = _stats

if _linear_model is not None:
    linear_model = _linear_model

if _neighbors is not None:
    neighbors = _neighbors

if _tree is not None:
    tree = _tree


# ---------------------------------------------------------------------------
# Utility: show which modules failed to load
# ---------------------------------------------------------------------------

def check_imports():

    all_modules = [

        "metrics",
        "model_selection",
        "distances",
        "activations",
        "preprocessing",
        "stats",
        "linear_model",
        "neighbors",
        "tree",
    ]

    print(f"mak_mini_ml v{_version_} — import status")

    print("-" * 45)

    for mod in all_modules:

        if mod in _import_errors:

            print(
                f"  ✗  {mod:<20} FAILED  →  {_import_errors[mod]}"
            )

        else:

            print(f"  ✓  {mod:<20} OK")

    print("-" * 45)

    if _import_errors:

        print(
            f"  {len(_import_errors)} module(s) failed to load."
        )

    else:

        print("  All modules loaded successfully.")


# ---------------------------------------------------------------------------
# Clean helper names
# ---------------------------------------------------------------------------

del warnings

del _safe_import

del _load_public_names

del _metrics

del _model_selection

del _distances

del _activations

del _preprocessing

del _stats

del _linear_model

del _neighbors

del _tree