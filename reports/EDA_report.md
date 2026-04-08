# EDA Findings

**Project:** Discovering Cultural Periods from Museum Metadata with Classical Machine Learning  
**Course:** CS 6140 — Machine Learning, Northeastern University, Spring 2026  
**Team:** Neural Beings — Dishaa Bornare, Ishan Gupta, Kaushal Nair

---

## Overview

This document summarizes the key findings from our exploratory data analysis of the Metropolitan Museum of Art open-access dataset (`MetObjects.csv`). It is intended to inform all subsequent preprocessing and modeling decisions across both the supervised and unsupervised tracks.

The full EDA is in `met_eda.ipynb`.

---

## Dataset

| Property | Value |
|---|---|
| Records | 484,956 |
| Columns | 54 |
| File size | ~303 MB on disk, ~1.2 GB in memory |
| License | Creative Commons Zero (CC0) |
| Source | [github.com/metmuseum/openaccess](https://github.com/metmuseum/openaccess) |

The dataset is dominated by string columns (46 of 54), with three boolean flags (`Is Highlight`, `Is Timeline Work`, `Is Public Domain`) and three integer columns. A note on naming: the CSV uses spaced title-case column names (e.g. `Object Begin Date`) whereas the Met API documentation uses camelCase (e.g. `objectBeginDate`). This distinction matters when cross-referencing the official field descriptions.

---

## Missing Data

Missingness is the defining data quality challenge of this dataset. Twenty-two columns are missing in more than 75% of records.

| Column | Missing % |
|---|---|
| Metadata Date | 100.0% |
| River | 99.6% |
| State | 99.5% |
| Locus | 98.4% |
| County | 98.2% |
| Reign | 97.7% |
| Locale | 96.8% |
| Excavation | 96.6% |
| Subregion | 95.4% |
| Dynasty | 95.2% |
| Rights and Reproduction | 94.9% |
| Portfolio | 94.5% |
| Region | 93.5% |
| City | 93.2% |
| Gallery Number | 89.8% |
| Geography Type | 87.6% |
| Object Wikidata URL | 85.7% |
| Country | 84.3% |
| Period | 81.2% |
| Artist Gender | 78.0% |
| Tags | 60.3% |
| Culture | 57.1% |
| Artist ULAN / Wikidata URLs | ~53.1% |
| Artist Display Name, Role, Bio, etc. | ~41.7% |
| Classification | 16.2% |
| Dimensions | 15.5% |
| Title | 5.9% |
| Medium | 1.5% |
| Object Begin Date | 0.0% |
| Object End Date | 0.0% |
| Department | 0.0% |

Two structural patterns are worth noting. First, all artist-related columns (Artist Display Name, Artist Role, Artist Begin Date, Artist Nationality, etc.) are missing at the same 41.7% rate, indicating a block pattern: these fields are absent together for records that have no attributed artist, not missing independently. A single `has_artist` binary flag may be more informative than attempting to impute each column separately.

Second, the geographic columns (River, State, Locus, County, Locale, Excavation, Subregion) are all above 95% missing and are candidates for exclusion from the feature set entirely.

---

## Target Variable: Department

The classification target, `Department`, is fully populated across all 484,956 records and contains 19 classes.

| Department | Count | % of Dataset |
|---|---|---|
| Drawings and Prints | 172,630 | 35.6% |
| European Sculpture and Decorative Arts | 43,051 | 8.9% |
| Photographs | 37,459 | 7.7% |
| Asian Art | 37,000 | 7.6% |
| Greek and Roman Art | 33,726 | 7.0% |
| Costume Institute | 31,652 | 6.5% |
| Egyptian Art | 27,969 | 5.8% |
| The American Wing | 18,532 | 3.8% |
| Islamic Art | 15,573 | 3.2% |
| Modern and Contemporary Art | 14,696 | 3.0% |
| Arms and Armor | 13,623 | 2.8% |
| Arts of Africa, Oceania, and the Americas | 12,367 | 2.6% |
| Medieval Art | 7,142 | 1.5% |
| Ancient Near Eastern Art | 6,223 | 1.3% |
| Musical Instruments | 5,227 | 1.1% |
| European Paintings | 2,626 | 0.5% |
| Robert Lehman Collection | 2,586 | 0.5% |
| The Cloisters | 2,340 | 0.5% |
| The Libraries | 534 | 0.1% |

The imbalance ratio between the largest and smallest class is **323x**. Every department has at least 534 records, so no class needs to be dropped, but the imbalance has direct implications for modeling: accuracy will be a misleading metric, and macro-averaged F1 should be used as the primary evaluation criterion. Resampling strategies (e.g. class-weighted loss, SMOTE) warrant investigation for the minority classes.

---

## Key Feature Columns

| Column | Unique Values | Missing % | Type |
|---|---|---|---|
| Medium | 65,907 | 1.5% | Free text, comma-delimited |
| Tags | 44,171 | 60.3% | Free text, pipe-delimited multi-value |
| Object Name | 28,631 | 0.5% | Categorical |
| Culture | 7,313 | 57.1% | High-cardinality categorical |
| Artist Nationality | 6,945 | 41.7% | High-cardinality categorical |
| Period | 1,891 | 81.2% | Categorical |
| Classification | 1,244 | 16.2% | Categorical |
| Object Begin Date | 2,076 unique values | 0.0% | Numeric (supports BCE as negative integers) |
| Object End Date | 2,041 unique values | 0.0% | Numeric (supports BCE as negative integers) |

### High-Cardinality Categoricals

Despite their large number of unique values, Culture and Artist Nationality are practically manageable through frequency-based encoding. A coverage analysis of Culture shows:

| Top-N Values | Coverage of Non-Null Records |
|---|---|
| Top 10 | 58.7% |
| Top 25 | 69.2% |
| Top 50 | 75.9% |
| Top 100 | 81.5% |
| Top 200 | 86.5% |

Artist Nationality follows a similar pattern. A top-N frequency encoding with an "Other" bucket for the tail is a practical strategy for both columns.

---

## Text Features

### Medium

`Medium` is the richest and most reliable text feature in the dataset: 98.5% non-null and with vocabulary that varies clearly by department. The field is comma-delimited and should be split on commas before vectorization, not treated as a single token.

Representative top tokens by department:

| Department | Characteristic Medium Terms |
|---|---|
| Drawings and Prints | watercolor, gouache, pen and black ink, graphite |
| European Sculpture and Decorative Arts | enamel, ivory, gilt bronze, mother-of-pearl, silk |
| Greek and Roman Art | glazed clay, white marble, bronze, carnelian gold |
| Egyptian Art | paint limestone, paint wood, pottery and ink |
| Costume Institute | silk, cotton, leather, metal, plastic, synthetic |
| Photographs | black-and-white, color, video tape transfer terms |

The vocabulary difference across departments suggests TF-IDF on `Medium` will be a high-signal feature for the classification task.

### Tags

Only 39.7% of objects (192,455 records) have any tags. Among those that do, objects carry an average of 2.2 tags (range: 1–18), stored as a pipe-separated string (e.g. `"Portraits|Men|Costumes"`). Tags must be parsed before vectorization. Given the 60% missingness, tags are a useful supplementary feature but cannot be treated as a primary signal.

---

## Temporal Features

`Object Begin Date` and `Object End Date` are fully populated and support negative integers for BCE dates. The raw range observed in the data is −400,000 to +5,000, which includes values that are clearly data entry errors or administrative placeholders.

- Values below −4,000: 2,074 records
- Values above 2,025: 2 records

These extremes should be clipped before deriving any temporal features. Two useful engineered features from these columns are:

| Feature | Definition | Mean | Std |
|---|---|---|---|
| `object_age` | `2025 − Object End Date` | 610 years | 970 |
| `object_span` | `Object End Date − Object Begin Date` | 70 years | 179 |

205 records have a negative `object_span` (i.e. End Date < Begin Date), which are data errors and should be handled before using span as a feature.

When visualizing date distributions, a single axis is heavily distorted by the BCE tail. A two-panel layout works better: one panel covering the full historical range (clipped to −4,000 to 2,025) and one restricted to CE dates (0 to 2,025).

---

## Planned Preprocessing Strategy

| Column | Planned Encoding |
|---|---|
| Department | Label encode (classification target) |
| Medium | TF-IDF after comma-splitting |
| Tags | Parse pipe-separated values → TF-IDF |
| Classification | TF-IDF or top-N frequency encode |
| Culture | Top-N frequency encode + "Other" bucket |
| Period | Top-N frequency encode + "Other" bucket |
| Dynasty | Top-N frequency encode + "Other" bucket |
| Artist Nationality | Top-N frequency encode + "Other" bucket |
| Artist Role | Low-cardinality → label encode |
| Artist Gender | Binary flag |
| Geography Type | Low-cardinality → label encode |
| Country | Top-N frequency encode + "Other" bucket |
| Object Begin Date | Numeric — clip values below −4,000 |
| Object End Date | Numeric — clip values below −4,000 |
| AccessionYear | Numeric — parse to integer |
| Is Highlight / Is Timeline Work / Is Public Domain | Boolean flags, already binary |

### Columns to Exclude

The following columns are candidates for exclusion from the feature set prior to modeling:

- `Metadata Date` — entirely empty (100% missing)
- `River`, `State`, `Locus`, `County`, `Locale`, `Excavation`, `Subregion` — all above 95% missing with no recoverable signal
- `Repository` — constant value across all records
- `Object Number`, `Object ID` — identifiers, not features
- `Link Resource`, `Artist ULAN URL`, `Artist Wikidata URL`, `Object Wikidata URL`, `Tags AAT URL`, `Tags Wikidata URL` — URL identifiers with no feature signal

---

## Key Observations for Modeling

**Class imbalance is severe.** At 323x, the imbalance between Drawings and Prints and The Libraries is large enough that a naive classifier will perform poorly on minority classes. Macro F1 should be the agreed-upon primary metric across supervised modeling and any cluster purity evaluation.

**Medium is the strongest single feature.** Near-complete coverage and clearly department-discriminative vocabulary make it the highest-confidence input for classification. TF-IDF on Medium alone is a reasonable sanity-check baseline before adding other features.

**Culture and Period are valuable but sparse.** Both are highly informative for the clustering task — they encode art-historical context directly — but their missingness (57% and 81% respectively) requires careful handling in the supervised pipeline. Using them as optional features and measuring their marginal contribution is advisable.

**The artist block should be treated as a unit.** Because artist-related columns are missing together, a single `has_artist` indicator may be more robust than imputing individual fields. Where artist information is present, Artist Nationality is the most usable feature given its coverage and cardinality profile.

**Tags coverage limits their role.** With 60% of records having no tags, Tags are best treated as an optional enrichment feature rather than a core input. Models should be evaluated with and without them to quantify their actual contribution.

**Date outliers require clipping before any use.** A value of −400,000 in `Object Begin Date` will silently distort any age calculation or normalization if not addressed in the preprocessing step.