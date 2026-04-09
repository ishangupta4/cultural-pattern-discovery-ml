# Preprocessing Notes

**Project:** Neural Beings — Met Museum ML  
**Course:** CS 6140, Northeastern University, Spring 2026  
**Author:** Ishan Gupta

---

## 1. Pipeline Summary

The preprocessing pipeline (`src/preprocessing.py`) takes the raw `MetObjects.csv` — after dropping identifier, URL, and high-null columns — and produces a sparse feature matrix ready for both supervised classification and unsupervised clustering. It handles type coercion, date clipping, feature derivation, null imputation, frequency and ordinal encoding for categoricals, and TF-IDF vectorization for three text fields (`Medium`, `Tags`, `Period`). All fitting (vocabulary selection, frequency maps, scalers) happens on training data only, so there is no leakage into the test set. The pipeline is serialized as `models/preprocessing_pipeline.joblib` and can be applied to any new raw DataFrame with a single `.transform()` call.

---

## 2. Columns Dropped

### Identifiers — no feature signal
| Column | Reason |
|--------|--------|
| Object ID | Unique integer row identifier |
| Object Number | Unique accession string identifier |
| Constituent ID | Artist ID integer — identifier, not a feature |

### URLs — no feature signal
| Column | Reason |
|--------|--------|
| Link Resource | URL to Met webpage |
| Artist ULAN URL | External authority URL |
| Artist Wikidata URL | External authority URL |
| Object Wikidata URL | External authority URL (85.7% null) |
| Tags AAT URL | URL parallel to Tags column |
| Tags Wikidata URL | URL parallel to Tags column |

### Structurally unusable
| Column | Reason |
|--------|--------|
| Metadata Date | 100% null |
| Repository | Constant value across all 484,956 rows — zero variance |

### Geographic columns — >90% null, no recoverable signal
| Column | Null % |
|--------|--------|
| River | 99.6% |
| State | 99.5% |
| Locus | 98.4% |
| County | 98.2% |
| Locale | 96.8% |
| Excavation | 96.6% |
| Subregion | 95.4% |
| Region | 93.5% |
| City | 93.2% |

### Admin / operational fields
| Column | Reason |
|--------|--------|
| Rights and Reproduction | Legal admin text — 94.9% null, no feature signal |
| Portfolio | Grouping note for prints — 94.5% null |
| Gallery Number | Current gallery location (operational, changes over time) — 89.8% null |
| Credit Line | Acquisition/donation text — no feature signal |

### Redundant or low-signal artist fields
| Column | Reason |
|--------|--------|
| Artist Prefix | Honorifics (Mr., Dr.) — no signal |
| Artist Suffix | Suffixes (Jr., Sr.) — no signal |
| Artist Alpha Sort | Alphabetical sort key, redundant with Display Name |
| Artist Display Bio | Free-text biography — noisy, no consistent structure |
| Artist Display Name | High-cardinality; `has_artist` flag derived before dropping |

### Flagged columns — dropped for now
| Column | Null % | Reason |
|--------|--------|--------|
| Dynasty | 95.2% | Too sparse for classification; may revisit for clustering |
| Reign | 97.7% | Extremely sparse, domain-specific |
| Geography Type | 87.6% | Very sparse; marginal signal vs. cost |
| Country | 84.3% | Very sparse; marginal signal vs. cost |
| Artist Begin Date | 41.7% | Stored as string; complex parsing; revisit if model plateaus |
| Artist End Date | 41.7% | Stored as string; complex parsing; revisit if model plateaus |
| Dimensions | 15.5% | Complex free-text to parse; stretch goal |
| Title | 5.9% | Too object-specific to generalize; low TF-IDF signal |
| Object Date | 2.8% | Human-readable date string; redundant with Begin/End Date |

---

## 3. Imputation Strategy

| Column type | Strategy |
|-------------|----------|
| Numeric (AccessionYear, dates) | `pd.to_numeric(errors="coerce")` then fill with column **median** |
| Categorical (Culture, Period, etc.) | Fill with `"Unknown"` — treated as its own category by encoders |
| Text fed to TF-IDF (Medium, Tags, Period) | Fill with `""` — produces an all-zero TF-IDF row, no spurious token |
| Boolean flags | Fill with `False` → cast to `int` (0) |

**Why median over mean for numeric:** Median is robust to outliers. `Object Begin Date` has values as extreme as −400,000 before clipping; mean would be badly distorted by these placeholder values.

---

## 4. Categorical Encoding

| Column | Method | Why |
|--------|--------|-----|
| Culture | **Frequency encoding** — replace each value with its row count in training data | 2,000+ unique values; top-N would lose too much tail; frequency preserves popularity signal |
| Artist Nationality | **Frequency encoding** (same rationale as Culture) | ~7,000 unique values |
| Object Name | **Top-50 + OrdinalEncoder** — values outside top 50 → "Other" | ~400 unique values; top 50 covers the overwhelming majority |
| Classification | **Top-50 + OrdinalEncoder** | ~300 unique values; same rationale |
| Artist Role | **Top-15 + OrdinalEncoder** | Low cardinality (~15–20 unique); top-15 captures nearly all |
| Artist Gender | **Top-3 + OrdinalEncoder** | Very low cardinality; only a few meaningful values |

**Why OrdinalEncoder over OneHotEncoder:** After top-N grouping, cardinality is small enough (≤51 categories) that ordinal encoding doesn't introduce serious ordinality bias for tree-based models. OrdinalEncoder keeps the matrix dense and narrow; one-hot would add 50+ binary columns per feature. Revisit for linear models.

**Unseen categories at transform time:** `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` — unknown test categories get −1 rather than raising an error.

---

## 5. TF-IDF Parameters

| Field | `max_features` | `ngram_range` | `sublinear_tf` | `stop_words` | Preprocessing |
|-------|---------------|---------------|----------------|--------------|---------------|
| Medium | 100 | (1, 2) | True | `"english"` | Replace `,` with space; collapse whitespace |
| Tags | 50 | (1, 1) | True | None | Replace `\|` with space |
| Period | 50 | (1, 2) | True | `"english"` | Replace `,` with space; collapse whitespace |

**`sublinear_tf=True`:** Replaces raw term frequency with `1 + log(tf)`. Prevents a material repeated 10× in a description from scoring 10× higher than one mentioned once — both describe the same thing.

**Bigrams for Medium and Period (`ngram_range=(1,2)`):** Two-word phrases like `"oil canvas"`, `"black ink"`, `"new kingdom"` are more informative than the individual tokens alone. Tags are pipe-separated atomic terms with no meaningful bigrams.

---

## 6. Final Shapes

| Split | Rows | Columns | Format |
|-------|------|---------|--------|
| X_train | 387,964 | 213 | scipy sparse (CSR) |
| X_test | 96,992 | 213 | scipy sparse (CSR) |
| y_train | 387,964 | — | numpy int array |
| y_test | 96,992 | — | numpy int array |

**Column breakdown (213 total):**
- 5 numeric (Object Begin/End Date, AccessionYear, object_age, object_span)
- 1 Artist Nationality (frequency-encoded + scaled)
- 4 categorical, OrdinalEncoded (Object Name, Classification, Artist Role, Artist Gender)
- 3 boolean flags (Is Highlight, Is Public Domain, Is Timeline Work)
- 100 Medium TF-IDF features
- 50 Tags TF-IDF features
- 50 Period TF-IDF features

---

## 7. How to Load

```python
import joblib
import numpy as np
from scipy.sparse import load_npz

# ── Load train / test splits ──────────────────────────────────────────────────
X_train = load_npz("data/processed/X_train.npz")
X_test  = load_npz("data/processed/X_test.npz")
y_train = np.load("data/processed/y_train.npy")
y_test  = np.load("data/processed/y_test.npy")

# ── Convert integer predictions back to department names ──────────────────────
le = joblib.load("models/label_encoder.joblib")

# Example: model returns integer labels → decode to strings
# predictions = model.predict(X_test)          # array of ints, e.g. [3, 7, 0, ...]
# dept_names  = le.inverse_transform(predictions)

# See all 19 class labels:
print(le.classes_)

# ── Apply pipeline to new raw data ────────────────────────────────────────────
import sys
sys.path.insert(0, ".")          # run from repo root
from src import build_pipeline

pipeline = joblib.load("models/preprocessing_pipeline.joblib")

# df_new must be a pandas DataFrame with the original raw CSV columns
# (identifiers / URL / constant columns can be present — pipeline drops them)
# X_new = pipeline.transform(df_new)
```

> All paths above are relative to the **repo root**.  
> The pipeline was fit on training data only — always call `.transform()`, never `.fit_transform()`, on new data.
