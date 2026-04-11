# Classification Notes — Epic 3

CS 6140 · Neural Beings · April 2026

---

## Models Trained

### Logistic Regression (`src/models/logistic_regression.py`)
| Hyperparameter | Value | Rationale |
|---|---|---|
| solver | saga | Variance-reduced SGD; stays sparse throughout — no dense Hessian |
| max_iter | 1000 | Converged before limit on this dataset |
| class_weight | balanced | `w_c = n_samples / (n_classes × n_c)` — corrects 323× imbalance |
| random_state | 42 | Reproducibility |

### Random Forest (`src/models/random_forest.py`)
| Hyperparameter | Value | Rationale |
|---|---|---|
| n_estimators | 100 (local) / 300 (HPC) | More trees → lower variance; HPC budget allows 300 |
| max_depth | 15 (local) / 30 (HPC) | Caps memory and prevents very deep noise-fitting |
| class_weight | balanced_subsample | Recomputes weights per bootstrap sample — better than global correction on an imbalanced set |
| n_jobs | -1 | Full parallelism (RF supports it, unlike LR in sklearn ≥1.8) |

### XGBoost (`src/models/xgboost_model.py`)
| Hyperparameter | Value | Rationale |
|---|---|---|
| n_estimators | 100 | Starting point; can be tuned upward |
| max_depth | 6 | Standard default; shallower than RF because boosting corrects incrementally |
| learning_rate | 0.1 | Shrinkage per tree; balance between speed and generalisation |
| subsample | 0.8 | Row subsampling — reduces overfitting |
| colsample_bytree | 0.8 | Feature subsampling per tree |
| sample_weight | compute_sample_weight('balanced') | XGBClassifier has no class_weight param; per-sample weights replicate the same effect |
| device | cpu (local) / cuda (HPC) | GPU hist builder is 5–20× faster on large datasets |

---

## Final Macro F1 Scores

| Model | Macro F1 | Weighted F1 | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.3998 | 0.5513 | 0.5331 |
| Random Forest | 0.8669 | 0.9317 | 0.9249 |
| XGBoost | **0.9637** | **0.9832** | **0.9828** |

XGBoost outperforms both baselines on every metric. The gap between LR and the tree methods is large, likely because:
- TF-IDF features are high-dimensional and sparse — trees handle this better than a linear model without feature selection
- LR requires many more iterations to converge on 19-class sparse problems; the SAGA run may not have fully converged

---

## Key SHAP Findings

*(Fill in after running `python -m src.models.shap_analysis`)*

- **Most globally important features (mean |SHAP|):** ___
- **Best department (The Libraries, F1 = 1.00):** Driven by ___
- **Worst department (The Cloisters, F1 = 0.85):** Confused with ___; key discriminating features are ___
- **Mid department (The American Wing, F1 = 0.98):** ___

---

## Key Decisions and Why

**Why `balanced_subsample` over `balanced` for RF:**
`balanced` applies the same global weight to every tree, ignoring the fact that each tree sees a different bootstrap sample. `balanced_subsample` recomputes weights from the actual sample each tree receives, so the correction matches the local class distribution.

**Why `compute_sample_weight` instead of `class_weight` for XGBoost:**
XGBClassifier does not accept a `class_weight` keyword. Passing per-sample weights to `.fit()` is the documented equivalent and produces identical loss weighting.

**Why SHAP samples 2000 rows instead of the full test set:**
TreeExplainer is exact but scales linearly with samples. At 97k rows × 100 trees × 19 classes, full-set SHAP would take hours. 2000 stratified rows gives stable mean |SHAP| estimates in minutes.

**Why XGBoost uses native JSON format (`.json`) for saving:**
The native format is version-portable and human-readable. `joblib` pickle is faster to write but breaks across XGBoost versions; `.json` is stable for sharing with teammates and the grader.

---

## Output Files

### Models
| File | Description |
|---|---|
| `models/lr_model.joblib` | Fitted LogisticRegression |
| `models/rf_model.joblib` | Fitted RandomForestClassifier |
| `models/xgb_model.json` | Fitted XGBClassifier (native JSON) |

### Metrics
| File | Description |
|---|---|
| `outputs/metrics/lr_report_<ts>.csv` | Per-class precision/recall/F1 — LR |
| `outputs/metrics/rf_report_<ts>.csv` | Per-class precision/recall/F1 — RF |
| `outputs/metrics/xgb_report_<ts>.csv` | Per-class precision/recall/F1 — XGBoost |
| `outputs/metrics/model_comparison.csv` | Side-by-side F1 for all three models |

### Figures
| File | Description |
|---|---|
| `outputs/figures/class_distribution.png` | Training set class balance |
| `outputs/figures/lr_confusion_matrix_<ts>.png` | Normalised confusion matrix — LR |
| `outputs/figures/rf_confusion_matrix_<ts>.png` | Normalised confusion matrix — RF |
| `outputs/figures/xgb_confusion_matrix_<ts>.png` | Normalised confusion matrix — XGBoost |
| `outputs/figures/rf_feature_importance_<ts>.png` | Top-30 features by mean decrease in impurity |
| `outputs/figures/shap_global_importance_<ts>.png` | Top-20 features by mean \|SHAP\| across all classes |
| `outputs/figures/shap_best_*_f1_*.png` | Per-class SHAP beeswarm — best F1 department |
| `outputs/figures/shap_worst_*_f1_*.png` | Per-class SHAP beeswarm — worst F1 department |
| `outputs/figures/shap_mid_*_f1_*.png` | Per-class SHAP beeswarm — mid F1 department |
