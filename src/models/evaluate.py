import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix


def _ts():
    """Return a timestamp string for use in output filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data():
    """Load processed train/test splits, label encoder, and preprocessing pipeline.

    Returns a dict with keys:
      X_train, X_test  — scipy sparse matrices
      y_train, y_test  — numpy arrays of integer-encoded labels
      le               — fitted LabelEncoder
      pipeline         — fitted sklearn preprocessing pipeline
    """
    return {
        "X_train": scipy.sparse.load_npz(os.path.join(PROJECT_ROOT, "data", "processed", "X_train.npz")),
        "X_test": scipy.sparse.load_npz(os.path.join(PROJECT_ROOT, "data", "processed", "X_test.npz")),
        "y_train": np.load(os.path.join(PROJECT_ROOT, "data", "processed", "y_train.npy")),
        "y_test": np.load(os.path.join(PROJECT_ROOT, "data", "processed", "y_test.npy")),
        "le": joblib.load(os.path.join(PROJECT_ROOT, "models", "label_encoder.joblib")),
        "pipeline": joblib.load(os.path.join(PROJECT_ROOT, "models", "preprocessing_pipeline.joblib")),
    }


def print_metrics(y_test, y_pred, label_names, model_name):
    """Print macro F1 and full per-class classification report.

    Returns the report as a nested dict (sklearn output_dict=True format).
    """
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    macro_f1 = report["macro avg"]["f1-score"]
    print(f"[{model_name}] Macro F1: {macro_f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_names))
    return report


def save_report(report_dict, model_name):
    """Convert a classification report dict to a DataFrame and save as CSV.

    Saves to outputs/metrics/{model_name}_report.csv.
    """
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "metrics")
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(report_dict).T
    path = os.path.join(out_dir, f"{model_name}_report_{_ts()}.csv")
    df.to_csv(path)
    print(f"Saved report → {path}")


def plot_confusion_matrix(y_test, y_pred, label_names, model_name):
    """Plot and save a normalized confusion matrix.

    Normalized by true label (normalize='true'). Figure is 16×14 inches with
    x-axis tick labels rotated 90°. Saved to outputs/figures/.
    """
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(16, 14))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, colorbar=True, xticks_rotation=90)
    ax.set_title(f"{model_name} — Normalized Confusion Matrix")
    fig.tight_layout()

    path = os.path.join(out_dir, f"{model_name}_confusion_matrix_{_ts()}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix → {path}")


def plot_model_comparison(reports_dict):
    """Plot a grouped bar chart of per-class F1 for multiple models.

    Args:
        reports_dict: mapping of model_name → classification report DataFrame,
                      e.g. {'LR': df, 'RF': df, 'XGB': df}.

    Saves to outputs/figures/model_comparison.png.
    """
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)

    # Collect per-class F1 rows (exclude aggregate rows)
    f1_data = {}
    for model_name, df in reports_dict.items():
        aggregate_rows = {"accuracy", "macro avg", "weighted avg"}
        class_rows = [r for r in df.index if r not in aggregate_rows]
        f1_data[model_name] = df.loc[class_rows, "f1-score"]

    comparison = pd.DataFrame(f1_data)

    n_classes, n_models = comparison.shape
    x = np.arange(n_classes)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(14, n_classes * 0.6), 6))
    for i, model_name in enumerate(comparison.columns):
        ax.bar(x + i * width, comparison[model_name], width, label=model_name)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(comparison.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("F1-score")
    ax.set_title("Per-class F1 by Model")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(out_dir, f"model_comparison_{_ts()}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved model comparison → {path}")
