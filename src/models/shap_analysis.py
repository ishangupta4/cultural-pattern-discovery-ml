# What a SHAP value's sign means:
#   A SHAP value for class C on sample i represents the contribution of one feature
#   to pushing the model's log-odds output *toward* class C relative to the average
#   prediction. Positive = this feature value pushes the model toward predicting C;
#   negative = pushes away from C. The magnitude indicates how much. For a 19-class
#   problem, each sample has a (n_features, n_classes) SHAP matrix — one column of
#   contributions per class.
#
# Why sample 2000 rows instead of the full test set:
#   TreeExplainer computes exact Shapley values by traversing every tree for every
#   sample. With XGBoost at n_estimators=100 and 19 classes, a single sample
#   requires ~1,900 tree traversals. On 97k test rows that is ~185M traversals —
#   several hours on CPU. 2,000 rows stratified by class gives ~105 samples per
#   department on average, which is sufficient for stable mean |SHAP| estimates
#   while keeping runtime under a few minutes.
#
# TreeExplainer vs KernelExplainer:
#   KernelExplainer is model-agnostic: it estimates SHAP values via weighted linear
#   regression over feature coalitions (exponential in n_features without sampling).
#   It is slow and approximate. TreeExplainer exploits the tree structure directly —
#   it computes exact conditional expectations by following every branch, making it
#   orders of magnitude faster and exact (no approximation error) for tree-based
#   models. Always use TreeExplainer for XGBoost, LightGBM, and sklearn forests.

import glob
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shap
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _get_feature_names(pipeline, n_features):
    """Return feature names from pipeline, falling back to generic indices."""
    try:
        return list(pipeline.get_feature_names_out())
    except Exception:
        return [f"feature_{i}" for i in range(n_features)]


def run_shap(model_path="models/xgb_model.json", sample_size=2000):
    """Load model and test data, compute SHAP values on a stratified sample.

    Returns (shap_values, X_shap_csr, le, feature_names).
    shap_values shape: (n_samples, n_features, n_classes).
    """
    from src.models.evaluate import load_data
    from src.models.xgboost_model import load as load_xgb

    data = load_data()
    model = load_xgb(model_path)

    # Stratified sample — SHAP on full test set is very slow
    _, X_shap, _, y_shap = train_test_split(
        data["X_test"], data["y_test"],
        test_size=sample_size / len(data["y_test"]),
        stratify=data["y_test"], random_state=42,
    )
    X_shap_csr = sp.csr_matrix(X_shap)
    n_features = X_shap_csr.shape[1]
    feature_names = _get_feature_names(data["pipeline"], n_features)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap_csr)
    # shap_values shape: (n_samples, n_features, n_classes)

    return shap_values, X_shap_csr, data["le"], feature_names


def plot_global_importance(shap_values, X_shap, feature_names):
    """Bar chart of mean |SHAP| across all classes. Saves to outputs/figures/."""
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)

    # shap_values: (n_samples, n_features, n_classes) — average across classes
    if shap_values.ndim == 3:
        sv_for_plot = shap_values.mean(axis=2)
    else:
        sv_for_plot = shap_values

    shap.summary_plot(
        sv_for_plot,
        X_shap,
        feature_names=feature_names,
        plot_type="bar",
        max_display=20,
        show=False,
    )
    path = os.path.join(out_dir, f"shap_global_importance_{_ts()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved global SHAP importance → {path}")


def plot_class_shap(shap_values, X_shap, feature_names, class_idx, class_name):
    """Per-class SHAP summary plot (beeswarm) for one department.

    Saves to outputs/figures/shap_{class_name}.png.
    """
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)

    # Slice to (n_samples, n_features) for this class
    if shap_values.ndim == 3:
        sv_class = shap_values[:, :, class_idx]
    else:
        sv_class = shap_values

    shap.summary_plot(
        sv_class,
        X_shap,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )
    plt.title(f"SHAP — {class_name}")
    safe_name = class_name.replace(" ", "_").replace(",", "").replace("&", "and")
    path = os.path.join(out_dir, f"shap_{safe_name}_{_ts()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved class SHAP plot ({class_name}) → {path}")


if __name__ == "__main__":
    shap_values, X_shap_csr, le, feature_names = run_shap()

    # Top 10 features by mean |SHAP| globally
    if shap_values.ndim == 3:
        mean_abs = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    top10 = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False).head(10)
    print("\nTop 10 features by mean |SHAP| (global):")
    print(top10.to_string())

    plot_global_importance(shap_values, X_shap_csr, feature_names)

    # Find best / worst / mid F1 departments from the most recent xgb report
    report_pattern = os.path.join(PROJECT_ROOT, "outputs", "metrics", "xgb_report_*.csv")
    report_files = sorted(glob.glob(report_pattern))
    if not report_files:
        raise FileNotFoundError("No xgb_report_*.csv found in outputs/metrics/")

    report_df = pd.read_csv(report_files[-1], index_col=0)
    class_names = le.classes_
    # Keep only rows that correspond to actual department labels
    f1_series = report_df.loc[
        report_df.index.isin(class_names), "f1-score"
    ].sort_values()

    worst_name  = f1_series.index[0]
    best_name   = f1_series.index[-1]
    mid_name    = f1_series.index[len(f1_series) // 2]

    for dept_name in [best_name, worst_name, mid_name]:
        class_idx = list(class_names).index(dept_name)
        plot_class_shap(shap_values, X_shap_csr, feature_names, class_idx, dept_name)
