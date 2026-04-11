# Hierarchical (two-stage) classification
#
# What it is and when it helps:
#   Instead of asking one model to distinguish all 19 departments at once, we first
#   predict a coarse group (6 groups), then route each sample to a specialist model
#   that only needs to distinguish the departments within that group.  This works
#   well when the inter-group decision boundary is much cleaner than the fine-grained
#   intra-group one — the specialist can focus its capacity on the hard cases.
#   It also mirrors how a human curator might think: "this is clearly Asian, now
#   is it Chinese, Japanese, or Korean?"
#
# Why European is the hardest group:
#   The Cloisters and Medieval Art both cover medieval European material.
#   Robert Lehman Collection spans paintings, drawings, and decorative arts that
#   overlap heavily with European Paintings and European Sculpture and Decorative
#   Arts.  The metadata features (Medium, Culture, Period) do not cleanly separate
#   these sub-collections — a 14th-century French ivory might appear in Medieval Art
#   or The Cloisters depending on provenance, and the feature representation does
#   not encode that distinction.
#
# The error cascade tradeoff:
#   Stage 1 errors are fatal: if an artwork is mis-routed to the wrong group,
#   the stage 2 specialist will predict a department that cannot be correct.
#   There is no recovery mechanism.  Concretely, if stage 1 accuracy is 90%,
#   then 10% of samples are already wrong before stage 2 even runs — stage 2
#   accuracy is bounded above by stage 1 accuracy on a per-sample basis.
#   This means the pipeline only beats a flat classifier when the specialist
#   models gain more accuracy within their groups than stage 1 loses across groups.

import argparse
import os

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEPARTMENT_GROUPS = {
    'Ancient': [
        'Ancient Near Eastern Art',
        'Egyptian Art',
        'Greek and Roman Art',
    ],
    'Asian': [
        'Asian Art',
    ],
    'European': [
        'European Paintings',
        'European Sculpture and Decorative Arts',
        'Medieval Art',
        'The Cloisters',
        'Robert Lehman Collection',
    ],
    'American': [
        'The American Wing',
        'The Libraries',
    ],
    'Modern': [
        'Modern and Contemporary Art',
        'Photographs',
        'Costume Institute',
    ],
    'Specialized': [
        'Drawings and Prints',
        'Arms and Armor',
        'Musical Instruments',
        'Islamic Art',
        'Arts of Africa, Oceania, and the Americas',
    ],
}


def build_group_labels(y, le, department_groups):
    """Map integer department labels to integer group labels.

    Returns:
        y_group  — array of group integers (same length as y)
        group_le — LabelEncoder fitted on group names
    """
    dept_to_group = {}
    for group_name, depts in department_groups.items():
        for dept in depts:
            dept_to_group[dept] = group_name

    group_names = np.array([dept_to_group[le.classes_[label]] for label in y])
    group_le = LabelEncoder()
    y_group = group_le.fit_transform(group_names)
    return y_group, group_le


def _make_xgb(device='cpu'):
    return XGBClassifier(
        n_estimators=100,
        max_depth=6,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        device=device,
        verbosity=0,
    )


def train(X_train, y_train, le, department_groups, device='cpu'):
    """Train stage 1 group classifier and per-group specialist classifiers.

    Returns:
        stage1_model      — fitted group-level XGBClassifier
        specialist_models — dict mapping group_name → fitted XGBClassifier (or None
                            when the group has only one department)
        group_le          — LabelEncoder for group names
        department_groups — passed through unchanged
    """
    X_csr = sp.csr_matrix(X_train)
    y_group, group_le = build_group_labels(y_train, le, department_groups)

    # Stage 1: group classifier
    stage1_model = _make_xgb(device)
    stage1_model.fit(X_csr, y_group, sample_weight=compute_sample_weight('balanced', y_group))
    print("Stage 1 training complete.")

    # Stage 2: specialist per group
    specialist_models = {}
    for group_name, depts in department_groups.items():
        if len(depts) == 1:
            specialist_models[group_name] = None
            print(f"  [{group_name}] single department — no specialist needed")
            continue

        group_idx = group_le.transform([group_name])[0]
        mask = (y_group == group_idx)
        X_group = X_csr[mask]
        y_group_depts = y_train[mask]

        # Re-encode to contiguous [0, n_depts) — XGBoost requires this
        local_le = LabelEncoder()
        y_local = local_le.fit_transform(y_group_depts)

        specialist = _make_xgb(device)
        specialist.fit(
            X_group, y_local,
            sample_weight=compute_sample_weight('balanced', y_local),
        )
        # Store model + local encoder together so predict can decode back to global labels
        specialist_models[group_name] = (specialist, local_le)
        print(f"  [{group_name}] specialist trained on {mask.sum()} samples ({len(depts)} classes)")

    return stage1_model, specialist_models, group_le, department_groups


def predict(X, stage1_model, specialist_models, group_le, le, department_groups):
    """Two-stage prediction: route to group, then predict department within group.

    Processes samples in batches by predicted group for efficiency.
    Returns y_pred as an integer array of department labels.
    """
    X_csr = sp.csr_matrix(X)
    group_preds = stage1_model.predict(X_csr)
    y_pred = np.empty(X_csr.shape[0], dtype=int)

    # Build a lookup: group_name → single department int (for singleton groups)
    single_dept = {}
    for group_name, depts in department_groups.items():
        if len(depts) == 1:
            single_dept[group_name] = le.transform([depts[0]])[0]

    for group_idx in np.unique(group_preds):
        group_name = group_le.classes_[group_idx]
        mask = (group_preds == group_idx)

        if group_name in single_dept:
            y_pred[mask] = single_dept[group_name]
        else:
            specialist, local_le = specialist_models[group_name]
            local_preds = specialist.predict(X_csr[mask])
            y_pred[mask] = local_le.inverse_transform(local_preds)

    return y_pred


def save(stage1_model, specialist_models, group_le, path_prefix='models/hierarchical'):
    """Save stage 1 model, all specialist models, and the group LabelEncoder."""
    models_dir = os.path.join(PROJECT_ROOT, os.path.dirname(path_prefix))
    os.makedirs(models_dir, exist_ok=True)

    stage1_path = os.path.join(PROJECT_ROOT, f"{path_prefix}_stage1.json")
    stage1_model.save_model(stage1_path)
    print(f"Saved stage 1 → {stage1_path}")

    for group_name, entry in specialist_models.items():
        if entry is None:
            continue
        specialist, local_le = entry
        safe_name = group_name.replace(' ', '_').lower()
        model_path = os.path.join(PROJECT_ROOT, f"{path_prefix}_{safe_name}.json")
        specialist.save_model(model_path)
        print(f"Saved specialist [{group_name}] → {model_path}")
        le_path = os.path.join(PROJECT_ROOT, f"{path_prefix}_{safe_name}_le.joblib")
        joblib.dump(local_le, le_path)
        print(f"Saved specialist LabelEncoder [{group_name}] → {le_path}")

    le_path = os.path.join(PROJECT_ROOT, f"{path_prefix}_group_le.joblib")
    joblib.dump(group_le, le_path)
    print(f"Saved group LabelEncoder → {le_path}")


if __name__ == "__main__":
    import time

    from src.models.evaluate import load_data, plot_confusion_matrix, print_metrics, save_report

    parser = argparse.ArgumentParser(description="Train hierarchical two-stage classifier.")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    data = load_data()
    le = data["le"]
    X_train = sp.csr_matrix(data["X_train"])
    X_test = sp.csr_matrix(data["X_test"])
    y_train = data["y_train"]
    y_test = data["y_test"]

    t0 = time.time()
    stage1_model, specialist_models, group_le, department_groups = train(
        X_train, y_train, le, DEPARTMENT_GROUPS, device=args.device
    )
    print(f"Total training time: {time.time() - t0:.1f}s")

    # Stage 1 accuracy on test set
    y_group_test, _ = build_group_labels(y_test, le, DEPARTMENT_GROUPS)
    group_preds_test = stage1_model.predict(X_test)
    stage1_acc = accuracy_score(y_group_test, group_preds_test)
    print(f"\nStage 1 group accuracy: {stage1_acc:.4f}")

    # Per-group breakdown
    print("\nPer-group breakdown (test set):")
    print(f"  {'Group':<30} {'Samples':>8}  {'Stage1 Acc':>10}  {'Stage2 Macro F1':>16}")
    print("  " + "-" * 68)
    for group_name, depts in DEPARTMENT_GROUPS.items():
        group_idx = group_le.transform([group_name])[0]
        true_mask = (y_group_test == group_idx)
        n_samples = true_mask.sum()

        # Stage 1 accuracy for this group (among samples truly in this group)
        s1_acc = accuracy_score(
            y_group_test[true_mask],
            group_preds_test[true_mask],
        )

        # Stage 2 macro F1 within this group (using true group membership)
        if len(depts) == 1:
            stage2_f1 = 1.0
        else:
            X_group = X_test[true_mask]
            y_group_depts_true = y_test[true_mask]
            specialist, local_le = specialist_models[group_name]
            y_group_depts_pred = local_le.inverse_transform(specialist.predict(X_group))
            dept_labels = [le.transform([d])[0] for d in depts if d in le.classes_]
            report = classification_report(
                y_group_depts_true, y_group_depts_pred,
                labels=dept_labels,
                output_dict=True,
                zero_division=0,
            )
            stage2_f1 = report["macro avg"]["f1-score"]

        print(f"  {group_name:<30} {n_samples:>8}  {s1_acc:>10.4f}  {stage2_f1:>16.4f}")

    # Full end-to-end prediction
    y_pred = predict(X_test, stage1_model, specialist_models, group_le, le, DEPARTMENT_GROUPS)

    report = print_metrics(y_test, y_pred, le.classes_, "hierarchical")
    save_report(report, "hierarchical")
    plot_confusion_matrix(y_test, y_pred, le.classes_, "hierarchical")
    save(stage1_model, specialist_models, group_le)
