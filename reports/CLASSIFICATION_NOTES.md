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
| n_estimators | 100 | Starting point; more trees reduce variance at cost of runtime |
| max_depth | 30 | Caps memory; prevents very deep noise-fitting on sparse features |
| class_weight | balanced_subsample | Recomputes weights per bootstrap sample — better than global correction on imbalanced data |
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

### MLP (`src/models/mlp.py`)
| Hyperparameter | Value | Rationale |
|---|---|---|
| hidden_dims | [512, 256, 128] | Standard funnel — broad first layer, progressive compression |
| dropout | 0.3 | Regularisation; helps with the sparse input distribution |
| batch_size | 512 | Large batch for stable gradients on imbalanced data |
| epochs | 50 | Sufficient for convergence on this dataset size |
| optimizer | Adam, lr=0.001, weight_decay=1e-4 | Adaptive LR + L2; standard for tabular MLP |
| scheduler | ReduceLROnPlateau (patience=5, factor=0.5) | Halves LR when val loss plateaus; prevents stalling |
| criterion | CrossEntropyLoss with balanced class weights | Same imbalance correction as other models |
| device | auto (CUDA → MPS → CPU) | M-series Mac acceleration via MPS |

**Architecture:** Input(215) → Linear(512) → BN → ReLU → Dropout → Linear(256) → BN → ReLU → Dropout → Linear(128) → BN → ReLU → Dropout → Linear(21)

### Hierarchical XGBoost (`src/models/hierarchical.py`)
Two-stage classifier: stage 1 predicts one of 6 coarse groups; stage 2 routes to a per-group XGBoost specialist.

**Groups and departments:**
| Group | Departments |
|---|---|
| Ancient | Ancient Near Eastern Art, Egyptian Art, Greek and Roman Art |
| Asian | Asian Art (singleton — no specialist needed) |
| European | European Paintings, European Sculpture and Decorative Arts, Medieval Art, The Cloisters, Robert Lehman Collection |
| American | The American Wing, The Libraries |
| Modern | Modern and Contemporary Art, Photographs, Costume Institute |
| Specialized | Drawings and Prints, Arms and Armor, Musical Instruments, Islamic Art, Arts of Africa, Oceania, and the Americas |

Each specialist and the stage 1 classifier use: `n_estimators=100, max_depth=6, compute_sample_weight('balanced')`.

**Label encoding note:** Specialist models re-encode department labels to local `[0, n_depts)` integers because XGBoost requires contiguous class indices. A per-specialist `LabelEncoder` maps predictions back to global department integers.

---

## Final Results — All Five Models

| Model | Macro F1 | Weighted F1 | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.3998 | 0.5513 | 0.5331 |
| Random Forest | 0.8669 | 0.9317 | 0.9249 |
| XGBoost | 0.9637 | 0.9832 | 0.9828 |
| MLP | 0.5892 | 0.6326 | 0.5957 |
| **Hierarchical XGBoost** | **0.9764** | **0.9897** | **0.9896** |

**Model progression:** LR (0.400) → RF (0.867) → XGBoost (0.964) → MLP (0.589) → Hierarchical (0.976)

---

## Why MLP Underperformed

The MLP scored 0.589 macro F1 — worse than Random Forest (0.867). Several factors explain this:

1. **Sparse tabular data is gradient-boosting territory.** The feature matrix has ~60% zero entries (from TF-IDF on Tags coverage). Gradient boosting handles sparsity natively via the histogram builder; dense MLP layers compute on all zeros, wasting capacity and creating a poor gradient signal.

2. **Class imbalance is harder to fix in neural networks.** Even with balanced CrossEntropyLoss weights, the training loop sees many more majority-class batches. Gradient boosting's sequential error-correction naturally focuses on hard examples across all classes.

3. **Loss curve behaviour.** Validation loss stopped improving around epoch 15–20 (ReduceLROnPlateau triggered), but training loss continued to decrease — a sign of overfitting on the majority classes. The model learned the easy departments well but struggled with sparse, underrepresented ones.

4. **Feature scale.** TF-IDF features are already normalised, but one-hot encoded categoricals and raw booleans at different scales in the same dense layer create ill-conditioned gradients without careful per-feature normalisation.

**Conclusion:** Classical gradient boosting is the right tool for structured museum metadata with high sparsity and severe class imbalance. The MLP result confirms this.

---

## Hierarchical Model — Per-Group Results

| Group | Test Samples | Stage 1 Acc | Stage 2 Macro F1 |
|---|---|---|---|
| Ancient | 13,584 | 0.9990 | 0.9988 |
| Asian | 7,400 | 0.9954 | 1.0000 |
| European | 11,548 | 0.9939 | 0.9731 |
| American | 3,814 | 0.9921 | 1.0000 |
| Modern | 16,762 | 0.9921 | 0.9975 |
| Specialized | 43,884 | 0.9886 | 0.9923 |

**Why European is hardest:** The Cloisters and Medieval Art both cover medieval European material. Robert Lehman Collection spans paintings, drawings, and decorative arts that overlap with European Paintings and European Sculpture. The metadata features (Medium, Culture, Period) do not cleanly separate these sub-collections — a 14th-century French ivory might appear in either Medieval Art or The Cloisters depending on provenance, and no metadata feature encodes that distinction.

**What the two-stage architecture solved:** The Cloisters improved from F1 = 0.854 (flat XGBoost) to 0.906 (hierarchical). By first routing all European samples together, the European specialist learned a refined boundary between The Cloisters and Medieval Art using only within-group variation — a signal that was drowned out when training on all 19 departments simultaneously.

---

## Key SHAP Findings

- **Most globally important features (mean |SHAP|):** `Is Public Domain` and `artist_nationality` dominate over all text-based features. Era and provenance beat materials as the primary signal for departmental classification.
- **Best department (The Libraries, F1 = 1.00):** Uniquely driven by publication-type Medium tokens and Is Public Domain=False — the only department with modern printed text objects.
- **Worst department (The Cloisters, F1 = 0.85 → 0.91 after hierarchical):** Confused primarily with Medieval Art. Key discriminating features are object provenance tokens in Medium and specific Culture values for French/Spanish medieval work.
- **Mid department (The American Wing, F1 = 0.98):** Driven by American-origin Culture values and `artist_nationality` encoding US artists, plus date features reflecting 18th–19th century American objects.

**Why `Is Public Domain` dominates:** It is a simple boolean flag but acts as a precise era proxy. Ancient, Classical, and pre-20th-century European departments are almost entirely public domain; Modern/Contemporary and Photographs departments are not. This single bit cleanly separates two macro-regions of the classification space before any text features are consulted.

**Why `artist_nationality` beats Medium tokens:** Medium descriptions are noisy and inconsistently filled (e.g., "oil on canvas" appears across European Paintings, The American Wing, and Modern Art). Nationality encodes a strong geographic prior that the Met's curatorial structure is explicitly built around — Asian Art = Asian artists, The American Wing = American artists — making it more precise than material descriptions.

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

**Why the hierarchical model re-encodes labels per specialist:**
XGBoost requires class labels in `[0, n_classes)`. A group's rows retain global department integers (e.g., `{4, 6, 11, 15, 17}` for European), which exceeds the specialist's class count. A local `LabelEncoder` maps to `[0, 5)` for fitting and decodes predictions back to global integers for the final output.

---

## Recommended Model for Production

**Hierarchical XGBoost** — macro F1 0.976, accuracy 0.990.

Reasons:
1. Highest macro F1 across all models — best on the hardest, rarest departments.
2. The domain-knowledge grouping is interpretable and maintainable: adding a new sub-department to an existing group only requires retraining that group's specialist, not the full 19-class model.
3. Fast inference: two XGBoost forward passes per sample, no GPU required.
4. Graceful degradation: if stage 2 is missing for a new group, stage 1 output is still a useful coarse label.

Flat XGBoost (macro F1 0.964) is the recommended single-model baseline if simplicity is preferred over the last 1.2 points of F1.

---

## Output Files

### Models
| File | Description |
|---|---|
| `models/lr_model.joblib` | Fitted LogisticRegression |
| `models/rf_model.joblib` | Fitted RandomForestClassifier |
| `models/xgb_model.json` | Fitted XGBClassifier (native JSON) |
| `models/mlp_model.pt` | MLP state dict (PyTorch) |
| `models/hierarchical_stage1.json` | Stage 1 group classifier |
| `models/hierarchical_{group}.json` | Per-group specialist classifiers |
| `models/hierarchical_group_le.joblib` | Group LabelEncoder |
| `models/hierarchical_{group}_le.joblib` | Per-specialist local LabelEncoders |
| `models/label_encoder.joblib` | Global department LabelEncoder |
| `models/preprocessing_pipeline.joblib` | Fitted preprocessing pipeline |

### Metrics
| File | Description |
|---|---|
| `outputs/metrics/lr_report.csv` | Per-class precision/recall/F1 — LR |
| `outputs/metrics/rf_report_<ts>.csv` | Per-class precision/recall/F1 — RF |
| `outputs/metrics/xgb_report_<ts>.csv` | Per-class precision/recall/F1 — XGBoost |
| `outputs/metrics/mlp_report_<ts>.csv` | Per-class precision/recall/F1 — MLP |
| `outputs/metrics/hierarchical_report_<ts>.csv` | Per-class precision/recall/F1 — Hierarchical |
| `outputs/metrics/model_comparison.csv` | Side-by-side F1 for all five models |

### Figures
| File | Description |
|---|---|
| `outputs/figures/class_distribution.png` | Training set class balance |
| `outputs/figures/lr_confusion_matrix.png` | Normalised confusion matrix — LR |
| `outputs/figures/rf_confusion_matrix_<ts>.png` | Normalised confusion matrix — RF |
| `outputs/figures/xgb_confusion_matrix_<ts>.png` | Normalised confusion matrix — XGBoost |
| `outputs/figures/mlp_confusion_matrix_<ts>.png` | Normalised confusion matrix — MLP |
| `outputs/figures/hierarchical_confusion_matrix_<ts>.png` | Normalised confusion matrix — Hierarchical |
| `outputs/figures/rf_feature_importance_<ts>.png` | Top-30 features by mean decrease in impurity |
| `outputs/figures/shap_global_importance_<ts>.png` | Top-20 features by mean \|SHAP\| across all classes |
| `outputs/figures/shap_best_*_f1_*.png` | Per-class SHAP beeswarm — best F1 department |
| `outputs/figures/shap_worst_*_f1_*.png` | Per-class SHAP beeswarm — worst F1 department |
| `outputs/figures/shap_mid_*_f1_*.png` | Per-class SHAP beeswarm — mid F1 department |
