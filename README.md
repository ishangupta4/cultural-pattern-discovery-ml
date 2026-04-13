# Discovering Cultural Periods from Museum Metadata with Classical Machine Learning

![The Metropolitan Museum of Art](met_museum.jpg)

Machine learning analysis of the Metropolitan Museum of Art dataset to classify curatorial departments and discover latent cultural patterns using metadata features.

## About

This project explores the Metropolitan Museum of Art's open-access metadata to discover cultural periods and classify artworks using classical machine learning techniques, without relying on any image data.

The dataset contains 484,956 records described by 54 columns — including medium, culture, artist nationality, date ranges, and curatorial department — and presents nearly every challenge found in real-world tabular data: 22 columns with more than 75% null values, a 323:1 class imbalance between the largest and smallest department, high-cardinality categoricals (Culture with 7,000+ unique values, Artist Nationality with 6,000+), and free-text fields requiring vectorization.

The project has two parallel tracks: (a) supervised classification of curatorial department from metadata features, progressing from Logistic Regression through Random Forest, XGBoost, MLP, and a two-stage Hierarchical XGBoost informed by domain knowledge about the museum's structure; and (b) unsupervised clustering via K-Means and hierarchical agglomerative methods on SVD-reduced features to discover latent cultural groupings and compare them against established art-historical period labels.

## Dataset

The Met Open Access dataset is released under a CC0 license and is available here:
- **GitHub:** https://github.com/metmuseum/openaccess
- **Direct CSV:** https://github.com/metmuseum/openaccess/blob/master/MetObjects.csv

**Note:** The `notebooks/met_eda.ipynb` takes care of downloading the `MetObjects.csv` file and placing it at `data/MetObjects.csv`.

## Getting Started

### Prerequisites

- Python 3.14+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/ishangupta4/cultural-pattern-discovery-ml
cd cultural-pattern-discovery-ml
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset (or run the EDA notebook which does this in the first cell)

```bash
mkdir -p data
curl -L "https://media.githubusercontent.com/media/metmuseum/openaccess/master/MetObjects.csv" \
     -o data/MetObjects.csv
```

## Notebooks

| Notebook | Description |
|---|---|
| `notebooks/met_eda.ipynb` | Exploratory data analysis — downloads the dataset, profiles missingness, visualizes class imbalance, audits feature usability, and spot-checks text features per department |
| `notebooks/02_preprocessing.ipynb` | Preprocessing pipeline — null imputation, date clipping, feature derivation (`object_age`, `object_span`, `has_artist`), frequency and ordinal encoding, TF-IDF vectorization, train/test split, and artifact export |
| `notebooks/03_classification.ipynb` | Classification results viewer — loads saved metrics and plots for all five models (no retraining required) |
| `notebooks/unsv_learning.ipynb` | Unsupervised learning — standardization, TruncatedSVD, UMAP visualization, K-Means and hierarchical clustering, feature group ablation, art-historical period analysis, and test-set validation |

## Reproducing the Preprocessing Pipeline

The fitted pipeline and label encoder are already committed to `models/`. The processed train/test splits (`data/processed/`) are git-ignored due to file size. To regenerate them, run `notebooks/02_preprocessing.ipynb` top to bottom after downloading the dataset.

All fitting (vocabulary selection, frequency maps, scalers) is performed on training data only — there is no leakage into the test set. The pipeline is serialized as `models/preprocessing_pipeline.joblib` and can be applied to any new raw DataFrame with a single `.transform()` call.

### Loading Preprocessed Data in a New Notebook

```python
import numpy as np
import joblib
from scipy.sparse import load_npz

# Features and labels
X_train = load_npz('../data/processed/X_train.npz')
X_test  = load_npz('../data/processed/X_test.npz')
y_train = np.load('../data/processed/y_train.npy')
y_test  = np.load('../data/processed/y_test.npy')

# Convert integer predictions back to department names
le = joblib.load('../models/label_encoder.joblib')
print(le.classes_)   # array of department names in label order

# Apply pipeline to new raw data
pipeline = joblib.load('../models/preprocessing_pipeline.joblib')
X_new = pipeline.transform(new_df)
```

## Feature Matrix

30 columns are dropped before modeling (identifiers, URLs, constant-value fields, geographic columns above 90% null, and admin fields with no feature signal). The preprocessing pipeline transforms the remaining columns into a sparse feature matrix with 213 columns:

| Feature Group | Columns | Method |
|---|---|---|
| Numeric (Object Begin/End Date, AccessionYear, object_age, object_span) | 5 | Clip date outliers at −4,000, median imputation, StandardScaler |
| Boolean (Is Highlight, Is Public Domain, Is Timeline Work) | 3 | Cast to int |
| Artist Nationality | 1 | Frequency encoding (replace each value with its training-set count) |
| Categorical (Object Name, Classification, Artist Role, Artist Gender) | 4 | Top-N grouping + OrdinalEncoder; unseen categories mapped to −1 |
| Medium (free text) | 100 | TF-IDF, sublinear TF, bigrams, comma-split preprocessing |
| Tags (pipe-delimited) | 50 | TF-IDF, sublinear TF, unigrams, pipe-split preprocessing |
| Period (free text) | 50 | TF-IDF, sublinear TF, bigrams, comma-split preprocessing |

The final train/test split is 80/20 with stratification on the department label, producing 387,964 training rows and 96,992 test rows stored as scipy sparse CSR matrices.

## Documentation

Detailed notes for each project phase are in `reports/`:

| File | Covers |
|---|---|
| `reports/feature_engineering.md` | Column categorization (drop/impute/keep), derived features, flagged columns with team decisions |
| `reports/PREPROCESSING_NOTES.md` | Full pipeline walkthrough — imputation strategy, encoding choices, TF-IDF parameters, final matrix shape |
| `reports/CLASSIFICATION_NOTES.md` | All five models with hyperparameter tables and rationale, per-group hierarchical results, SHAP findings, MLP underperformance analysis, output file inventory |
| `reports/Unsupervised_notes.md` | SVD component interpretation, K-Means and hierarchical clustering results, feature group ablation, art-historical period analysis, test-set validation |
