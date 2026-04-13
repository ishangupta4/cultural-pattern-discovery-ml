"""
Export pre-computed artifacts for the Streamlit demo app.

Outputs (written to demo/data/):
  umap_embeddings.csv  — 3-D UMAP projection of a 30k-row subsample of X_train
  model_metrics.json   — macro avg F1 / precision / recall for all five models

Run from the project root:
  python demo/export_demo_data.py
"""

import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from umap import UMAP

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "demo", "data")
PROC_DIR  = os.path.join(ROOT, "data", "processed")
MODELS    = os.path.join(ROOT, "models")
METRICS   = os.path.join(ROOT, "outputs", "metrics")
RAW_CSV   = os.path.join(ROOT, "data", "MetObjects.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  UMAP EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

print("── Step 1: UMAP embeddings ──────────────────────────────────────────")

# Load processed training data
print("  Loading X_train.npz …")
X_train = scipy.sparse.load_npz(os.path.join(PROC_DIR, "X_train.npz"))
y_train = np.load(os.path.join(PROC_DIR, "y_train.npy"))
le      = joblib.load(os.path.join(MODELS, "label_encoder.joblib"))
print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")

# Reconstruct which raw-CSV rows went into training.
# The pipeline did stratified 80/20 split on all 484,956 rows with random_state=42.
print("  Reconstructing train split indices …")
df_dept = pd.read_csv(RAW_CSV, usecols=["Department"], low_memory=False)
y_dept  = df_dept["Department"].values
all_idx = list(range(len(df_dept)))
train_idx, _ = train_test_split(
    all_idx, test_size=0.2, random_state=42, stratify=y_dept
)
train_idx = np.array(train_idx)   # shape: (387964,)
print(f"  train_idx length: {len(train_idx)}  (expected 387964)")

# Random subsample of 30,000 rows (same seed for reproducibility)
print("  Subsampling 30,000 rows …")
rng          = np.random.RandomState(42)
sub_pos      = rng.choice(X_train.shape[0], size=30_000, replace=False)
X_sub        = X_train[sub_pos]          # sparse (30000, 214)
y_sub        = y_train[sub_pos]
raw_sub_idx  = train_idx[sub_pos]        # row positions in the raw CSV

# TruncatedSVD → dense before UMAP (UMAP cannot accept sparse input directly)
print("  TruncatedSVD(50) …")
svd   = TruncatedSVD(n_components=50, random_state=42)
X_svd = svd.fit_transform(X_sub)         # (30000, 50)
print(f"  Explained variance ratio sum: {svd.explained_variance_ratio_.sum():.3f}")

# UMAP 3-D projection
print("  UMAP(n_components=3) — this may take a few minutes …")
reducer   = UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(X_svd)  # (30000, 3)
print(f"  Embedding shape: {embedding.shape}")

# Load metadata columns for the subsample rows
print("  Loading metadata from raw CSV …")
meta_cols = ["Culture", "Medium", "Object Name", "Object Begin Date"]
df_meta   = pd.read_csv(RAW_CSV, usecols=meta_cols, low_memory=False)
df_sub    = df_meta.iloc[raw_sub_idx].reset_index(drop=True)

# Build output DataFrame
dept_names = le.inverse_transform(y_sub)
df_emb = pd.DataFrame({
    "umap_x":            embedding[:, 0],
    "umap_y":            embedding[:, 1],
    "umap_z":            embedding[:, 2],
    "department":        dept_names,
    "culture":           df_sub["Culture"].fillna("Unknown").values,
    "medium":            df_sub["Medium"].fillna("Unknown").values,
    "object_name":       df_sub["Object Name"].fillna("Unknown").values,
    "object_begin_date": df_sub["Object Begin Date"].fillna("Unknown").values,
})

out_emb = os.path.join(DATA_DIR, "umap_embeddings.csv")
df_emb.to_csv(out_emb, index=False)
print(f"  Saved → {out_emb}  ({len(df_emb):,} rows)")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  MODEL COMPARISON METRICS
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Step 2: Model metrics JSON ───────────────────────────────────────")

# Model registry: (glob pattern, display name)
MODEL_REGISTRY = [
    ("lr_report.csv",               "Logistic Regression"),
    ("rf_report*.csv",              "Random Forest"),
    ("xgb_report*.csv",             "XGBoost"),
    ("hierarchical_report*.csv",    "Hierarchical XGBoost"),
    ("mlp_report*.csv",             "MLP"),
]

# Check whether model_comparison.csv exists (used for logging; individual
# reports are still loaded because they carry precision + recall)
comp_path = os.path.join(METRICS, "model_comparison.csv")
if os.path.exists(comp_path):
    print(f"  model_comparison.csv found — loading per-model precision/recall from individual reports")
else:
    print("  model_comparison.csv not found — loading all metrics from individual reports")

metrics_out = []
for pattern, display_name in MODEL_REGISTRY:
    matches = sorted(
        glob.glob(os.path.join(METRICS, pattern)),
        key=os.path.getmtime,
    )
    if not matches:
        print(f"  WARNING: no report found for pattern '{pattern}' — skipping")
        continue

    report_path = matches[-1]   # most recent if multiple timestamps
    df_report   = pd.read_csv(report_path, index_col=0)

    if "macro avg" not in df_report.index:
        print(f"  WARNING: 'macro avg' row missing in {report_path} — skipping")
        continue

    macro = df_report.loc["macro avg"]
    metrics_out.append({
        "model":     display_name,
        "macro_f1":  round(float(macro["f1-score"]),  4),
        "precision": round(float(macro["precision"]), 4),
        "recall":    round(float(macro["recall"]),    4),
    })
    print(f"  {display_name}: F1={metrics_out[-1]['macro_f1']:.4f}  "
          f"P={metrics_out[-1]['precision']:.4f}  R={metrics_out[-1]['recall']:.4f}")

out_metrics = os.path.join(DATA_DIR, "model_metrics.json")
with open(out_metrics, "w") as f:
    json.dump(metrics_out, f, indent=2)
print(f"  Saved → {out_metrics}  ({len(metrics_out)} models)")

# ══════════════════════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Complete ─────────────────────────────────────────────────────────")
print(f"  demo/data/umap_embeddings.csv  {os.path.getsize(out_emb):>12,} bytes")
print(f"  demo/data/model_metrics.json   {os.path.getsize(out_metrics):>12,} bytes")
