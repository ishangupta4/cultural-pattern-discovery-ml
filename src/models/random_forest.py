# class_weight='balanced' vs 'balanced_subsample':
#   'balanced' computes class weights once from the full training set and applies
#   the same weights to every tree. 'balanced_subsample' recomputes weights from
#   the bootstrap sample used for each individual tree — so weights vary tree to
#   tree, matching the actual class distribution seen by that tree. For imbalanced
#   datasets, 'balanced_subsample' is generally preferred because it avoids
#   applying a global correction to a locally resampled view of the data.
#
# max_depth=None vs a fixed value:
#   None lets each tree grow until all leaves are pure (or contain fewer than
#   min_samples_leaf samples). This maximises variance — individual trees overfit
#   the training data, but the ensemble average cancels most of that noise. A
#   fixed max_depth (e.g. 30) caps tree complexity, which speeds up training and
#   reduces memory, and often generalises as well or better on wide sparse data
#   where very deep trees learn noise. Start with a fixed depth; loosen it if
#   validation metrics plateau.
#
# Why RF generalises better than a single decision tree:
#   A single tree is a high-variance estimator: small changes in training data
#   produce very different splits. RF trains N trees on independent bootstrap
#   samples (bagging) and additionally randomises the feature subset considered
#   at each split (feature randomness). Averaging N decorrelated trees reduces
#   variance by roughly 1/N without increasing bias, giving a much smoother
#   decision boundary than any one tree alone.

import argparse
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train(X_train, y_train, n_estimators=100, max_depth=30):
    """Fit a balanced random forest on sparse training data. Returns fitted model."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    model.fit(X_train, y_train)
    return model


def save_feature_importances(model, pipeline, top_n=30):
    """Plot and return a DataFrame of the top_n most important features.

    Falls back to generic feature indices if the pipeline's custom transformers
    do not implement get_feature_names_out().
    """
    importances = model.feature_importances_

    try:
        names = pipeline.get_feature_names_out()
    except Exception:
        names = np.array([f"feature_{i}" for i in range(len(importances))])

    df = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    top = df.head(top_n)

    out_dir = os.path.join(PROJECT_ROOT, "outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, top_n * 0.35 + 1))
    ax.barh(top["feature"][::-1], top["importance"][::-1])
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title(f"RF feature importances — top {top_n}")
    fig.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"rf_feature_importance_{ts}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved feature importance chart → {path}")

    return df


def save(model, path="models/rf_model.joblib"):
    """Save a fitted model to disk with joblib. Creates parent dirs if needed."""
    full_path = os.path.join(PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    joblib.dump(model, full_path)
    print(f"Saved model → {full_path}")


if __name__ == "__main__":
    import time

    from src.models.evaluate import load_data, plot_confusion_matrix, print_metrics, save_report

    parser = argparse.ArgumentParser(description="Train a Random Forest on Met Museum data.")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=30)
    args = parser.parse_args()

    data = load_data()

    t0 = time.time()
    model = train(data["X_train"], data["y_train"], args.n_estimators, args.max_depth)
    print(f"Training time: {time.time() - t0:.1f}s")

    y_pred = model.predict(data["X_test"])
    report = print_metrics(data["y_test"], y_pred, data["le"].classes_, "rf")
    save_report(report, "rf")
    plot_confusion_matrix(data["y_test"], y_pred, data["le"].classes_, "rf")
    save_feature_importances(model, data["pipeline"])
    save(model)
