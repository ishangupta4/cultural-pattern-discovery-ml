# Feature Engineering Plan


---

## Column Categorization

All 54 columns are assigned to one of four buckets:

- **DROP** — remove entirely before building the pipeline
- **IMPUTE_NUMERIC** — numeric column with some nulls; fill with column median
- **IMPUTE_CATEGORICAL** — categorical/text column with nulls; fill with `"Unknown"` (or `""` for TF-IDF inputs)
- **KEEP_AS_IS** — no nulls or already boolean; no imputation needed

---

### DROP

| Column | Null % | Reason |
|--------|--------|--------|
| Object ID | 0.0% | Unique row identifier — not a feature |
| Object Number | 0.0% | Unique accession identifier — not a feature |
| Constituent ID | 41.7% | Artist ID integer — identifier, not a feature |
| Link Resource | 0.0% | URL to Met webpage — no signal |
| Artist ULAN URL | 53.1% | External URL — no signal |
| Artist Wikidata URL | 53.8% | External URL — no signal |
| Object Wikidata URL | 85.7% | External URL — no signal |
| Tags AAT URL | 60.3% | URL parallel to Tags — no signal |
| Tags Wikidata URL | 60.3% | URL parallel to Tags — no signal |
| Metadata Date | 100.0% | Entirely empty — nothing to use |
| Repository | 0.0% | Constant value across all rows — zero variance |
| River | 99.6% | >95% null with no recoverable signal |
| State | 99.5% | >95% null with no recoverable signal |
| Locus | 98.4% | >95% null with no recoverable signal |
| County | 98.2% | >95% null with no recoverable signal |
| Locale | 96.8% | >95% null with no recoverable signal |
| Excavation | 96.6% | >95% null with no recoverable signal |
| Subregion | 95.4% | >95% null with no recoverable signal |
| Rights and Reproduction | 94.9% | Admin/legal field, >90% null, no feature signal |
| Portfolio | 94.5% | Grouping note for prints/drawings, >90% null |
| Region | 93.5% | Geographic — >90% null, no recoverable signal |
| City | 93.2% | Geographic — >90% null, no recoverable signal |
| Gallery Number | 89.8% | Operational (current gallery location) — not a property of the object itself; changes over time |
| Artist Prefix | 41.7% | Honorifics (Mr., Dr.) — no signal |
| Artist Suffix | 41.8% | Suffixes (Jr., Sr.) — no signal |
| Artist Alpha Sort | 41.7% | Alphabetical sort key, redundant with Artist Display Name |
| Artist Display Bio | 42.2% | Free-text biography — noisy, no consistent structure |
| Artist Display Name | 41.7% | High-cardinality string (~100k unique); `has_artist` flag derived before dropping |
| Object Date | 2.8% | Human-readable free-text date (e.g. "ca. 1870") — redundant with Object Begin/End Date |
| Credit Line | 0.1% | Acquisition/donation text — administrative, no feature signal |

---

### IMPUTE_NUMERIC

| Column | Null % | Dtype | Notes |
|--------|--------|-------|-------|
| AccessionYear | 0.8% | str (→ int) | Parse to integer first, then fill nulls with median year |

---

### IMPUTE_CATEGORICAL

Fill with `"Unknown"` unless noted. For columns fed into TF-IDF (Medium, Tags, Classification, Title, Object Name), fill with `""` (empty string) so the vectorizer produces an all-zero row rather than a spurious "unknown" token.

| Column | Null % | Encoding Plan |
|--------|--------|---------------|
| Medium | 1.5% | TF-IDF after comma-splitting; fill with `""` |
| Tags | 60.3% | Parse pipe-separated values → TF-IDF; fill with `""` |
| Classification | 16.2% | TF-IDF or top-N frequency encode; fill with `""` |
| Title | 5.9% | TF-IDF (low signal — see flags); fill with `""` |
| Object Name | 0.5% | Top-N frequency encode + "Other"; fill with `""` |
| Culture | 57.1% | Top-100 frequency encode + "Other" bucket; fill with `"Unknown"` |
| Period | 81.2% | Top-N frequency encode + "Other" bucket; fill with `"Unknown"` |
| Artist Nationality | 41.7% | Top-N frequency encode + "Other" bucket; fill with `"Unknown"` |
| Artist Role | 41.7% | Low-cardinality → label encode; fill with `"Unknown"` |
| Artist Gender | 78.0% | Binary flag (known female / other); fill with `"Unknown"` |
| Geography Type | 87.6% | Low-cardinality label encode; fill with `"Unknown"` ⚑ see flags |
| Country | 84.3% | Top-N frequency encode + "Other" bucket; fill with `"Unknown"` ⚑ see flags |

---

### KEEP_AS_IS

No nulls; no imputation needed. Some require transformation (clipping, casting) but not imputation.

| Column | Null % | Dtype | Notes |
|--------|--------|-------|-------|
| Department | 0.0% | str | **Classification target** — label-encode separately, not part of feature pipeline |
| Object Begin Date | 0.0% | int64 | Clip values < −4000; used to derive `object_age` and `object_span` |
| Object End Date | 0.0% | int64 | Clip values < −4000; used to derive `object_age` and `object_span` |
| Is Highlight | 0.0% | bool | Already binary (0/1) |
| Is Timeline Work | 0.0% | bool | Already binary (0/1) |
| Is Public Domain | 0.0% | bool | Already binary (0/1) |

---

## Derived Features

These features do not exist in the raw CSV — they are engineered during preprocessing.

| Feature | Derivation | Type | Notes |
|---------|-----------|------|-------|
| `has_artist` | `1 if Artist Display Name is not null else 0` | Binary | Captures the anonymous/attributed distinction; more robust than imputing each artist column separately |
| `object_age` | `2025 − Object End Date` (after clipping) | Numeric | Approximate age of the object in years |
| `object_span` | `Object End Date − Object Begin Date` (after clipping) | Numeric | Production period length; clip negative values to 0 |

---

## Flagged Columns

These columns need a team decision before the pipeline is finalized.

| Column | Null % | Issue                                                                                                                                                                                                                                                                                                |
|--------|--------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Dynasty** | 95.2% | EDA preprocessing plan lists it for top-N encoding, but 95% null means only ~24k rows have a value. Potentially useful for unsupervised clustering (Egyptian/Chinese dynasties) but very sparse for classification. **Thought Process: include for clustering track, exclude for supervised track.** |
| **Reign** | 97.7% | Similar to Dynasty — extremely sparse but culturally specific. **Thought Process: DROP unless clustering experiments show signal.**                                                                                                                                                                  |
| **Geography Type** | 87.6% | Qualifies geographic relationship (e.g. "Made in", "Found in"). Only ~60k rows have a value. Low cardinality (~10 categories). **Thought Process: include as optional feature; measure contribution vs. cost.**                                                                                      |
| **Country** | 84.3% | Only ~76k rows have a value. Potentially meaningful for clustering. **Thought Process: include for both tracks; use top-50 + "Other".**                                                                                                                                                              |
| **Artist Begin Date / Artist End Date** | 41.7% | Stored as strings, require parsing. Could derive artist lifespan or active century. **Thought Process: skip for now; revisit if model performance plateaus.**                                                                                                                                        |
| **Dimensions** | 15.5% | Free text (e.g. "H. 12 in. (30.5 cm)"). Could extract numeric height/width but parsing is fragile. **Thought Process: DROP for now; revisit as stretch goal.**                                                                                                                                       |
| **Title** | 5.9% | TF-IDF on titles is usually noisy (titles are too object-specific to generalize). **Thought Process: omit from initial pipeline; test marginal F1 contribution before including.**                                                                                                                   |

---

## Summary Counts

| Bucket                       | Count |
|------------------------------|-------|
| DROP                         | 30 |
| IMPUTE_NUMERIC               | 1 |
| IMPUTE_CATEGORICAL           | 12 |
| KEEP_AS_IS                   | 6 |
| FLAGS [Dropping all for now] | 6 |
| **Total**                    | **49** |

> Note: `Department` is in KEEP_AS_IS but is the classification target, not a pipeline feature.
