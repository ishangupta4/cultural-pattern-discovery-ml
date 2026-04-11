# Gradient boosting vs Random Forest:
#   RF builds N trees in parallel, each on an independent bootstrap sample, and
#   averages their predictions. Variance is reduced by the ensemble but each tree
#   has no knowledge of the others' errors. Gradient boosting builds trees
#   sequentially: each new tree fits the *residual errors* of all previous trees,
#   so the ensemble is a sum of corrections rather than an average of independent
#   estimators. Boosting typically achieves lower bias than RF at the cost of being
#   slower to train (sequential) and more sensitive to overfitting.
#
# What learning_rate controls:
#   After each tree is fit, its contribution is scaled by learning_rate (also called
#   shrinkage). A smaller value (e.g. 0.05) means each tree corrects less of the
#   residual, so more trees are needed to converge — but the final model generalises
#   better because no single tree dominates. Typical rule: halve learning_rate and
#   double n_estimators for the same training loss with lower variance.
#
# Why compute_sample_weight instead of class_weight:
#   XGBoost's XGBClassifier does not accept a class_weight parameter the way sklearn
#   estimators do. Instead, it accepts sample_weight in .fit(). compute_sample_weight
#   ('balanced', y) maps each sample to its class weight
#   (n_samples / (n_classes * n_samples_in_class)), producing the exact same
#   per-sample correction that sklearn's class_weight='balanced' would apply.
#
# What device='cuda' does and why GPU helps:
#   XGBoost's hist tree method is fully GPU-accelerated on CUDA devices. Building
#   each tree involves scanning the full dataset to find optimal split points —
#   this is embarrassingly data-parallel and maps well onto GPU SIMD lanes.
#   On a dataset with 388k rows and 214 features, GPU training is typically
#   5–20× faster than CPU. On the HPC cluster, pass --device cuda to activate it.

import argparse
import os
import scipy.sparse as sp

import joblib
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train(X_train, y_train, n_estimators=100, max_depth=6, learning_rate=0.1, device="cpu"):
    """Fit an XGBoost classifier with balanced sample weights. Returns fitted model."""
    sample_weights = compute_sample_weight("balanced", y_train)
    X_csr = sp.csr_matrix(X_train)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        device=device,
        verbosity=1,
    )
    model.fit(X_csr, y_train, sample_weight=sample_weights)
    return model


def save(model, path="models/xgb_model.json"):
    """Save model in XGBoost native JSON format. Creates parent dirs if needed."""
    full_path = os.path.join(PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    model.save_model(full_path)
    print(f"Saved model → {full_path}")


def load(path="models/xgb_model.json"):
    """Load and return an XGBoost model from its native JSON format."""
    full_path = os.path.join(PROJECT_ROOT, path)
    model = XGBClassifier()
    model.load_model(full_path)
    return model


if __name__ == "__main__":
    import time

    from src.models.evaluate import load_data, plot_confusion_matrix, print_metrics, save_report

    parser = argparse.ArgumentParser(description="Train XGBoost on Met Museum data.")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    data = load_data()
    X_train_csr = sp.csr_matrix(data["X_train"])
    X_test_csr = sp.csr_matrix(data["X_test"])

    t0 = time.time()
    model = train(X_train_csr, data["y_train"], args.n_estimators, args.max_depth, args.learning_rate, args.device)
    print(f"Training time: {time.time() - t0:.1f}s")

    y_pred = model.predict(X_test_csr)
    report = print_metrics(data["y_test"], y_pred, data["le"].classes_, "xgb")
    save_report(report, "xgb")
    plot_confusion_matrix(data["y_test"], y_pred, data["le"].classes_, "xgb")
    save(model)
