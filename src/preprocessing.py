"""
Met Museum preprocessing pipeline.

Exports: build_pipeline()
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler


def _to_float(X):
    """Cast array to float64. Used to convert bool columns before SimpleImputer."""
    return np.asarray(X, dtype=float)


# ── Column constants ──────────────────────────────────────────────────────────

NUMERIC_COLS = [
    "Object Begin Date",
    "Object End Date",
    "AccessionYear",
    "object_age",       # derived by DateFeatureEngineer
    "object_span",      # derived by DateFeatureEngineer
]

CATEGORICAL_COLS = [
    "Object Name",
    "Classification",
    "Artist Role",
    "Artist Gender",
]

BOOL_COLS = [
    "Is Highlight",
    "Is Public Domain",
    "Is Timeline Work",
]


# ── Custom transformers ───────────────────────────────────────────────────────

class DateFeatureEngineer(BaseEstimator, TransformerMixin):
    """Coerces date columns to numeric, clips outliers, and derives
    object_age and object_span.  Must run before ColumnTransformer so
    the derived columns exist when the numeric branch looks for them.
    """

    DATE_LOW  = -7000
    DATE_HIGH =  2026
    REF_YEAR  =  2026

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # AccessionYear is stored as a string in the raw CSV
        X["AccessionYear"] = pd.to_numeric(X["AccessionYear"], errors="coerce")

        # Clip date outliers
        X["Object Begin Date"] = (
            pd.to_numeric(X["Object Begin Date"], errors="coerce")
            .clip(self.DATE_LOW, self.DATE_HIGH)
        )
        X["Object End Date"] = (
            pd.to_numeric(X["Object End Date"], errors="coerce")
            .clip(self.DATE_LOW, self.DATE_HIGH)
        )

        # Derived temporal features
        X["object_age"]  = self.REF_YEAR - X["Object End Date"]
        X["object_span"] = (
            X["Object End Date"] - X["Object Begin Date"]
        ).clip(lower=0)

        return X


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replaces each category with its frequency in the training set.

    Unseen categories at transform time are mapped to 0.
    Input:  1-D array-like (single column from ColumnTransformer)
    Output: (n_samples, 1) float64 array
    """

    def fit(self, X, y=None):
        s = pd.Series(np.asarray(X).ravel()).fillna("Unknown")
        self.freq_map_ = s.value_counts().to_dict()
        return self

    def transform(self, X):
        s = pd.Series(np.asarray(X).ravel()).fillna("Unknown")
        return s.map(self.freq_map_).fillna(0).values.reshape(-1, 1)


class TextPreprocessorTfidf(BaseEstimator, TransformerMixin):
    """Cleans a single text column and applies TF-IDF vectorization.

    Steps:
      1. Fill nulls with ""
      2. Replace `sep` with a space  (comma for Medium/Period, | for Tags)
      3. Collapse consecutive whitespace
      4. Strip leading/trailing whitespace
      5. TfidfVectorizer.fit_transform / transform

    Input:  1-D array-like (single column from ColumnTransformer)
    Output: sparse (n_samples, max_features) matrix
    """

    def __init__(self, sep=",", max_features=100, ngram_range=(1, 2),
                 sublinear_tf=True, stop_words=None):
        self.sep = sep
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.stop_words = stop_words

    def _clean(self, X):
        return (
            pd.Series(np.asarray(X).ravel())
            .fillna("")
            .str.replace(self.sep, " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .tolist()
        )

    def fit(self, X, y=None):
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            stop_words=self.stop_words,
        )
        self.vectorizer_.fit(self._clean(X))
        return self

    def transform(self, X):
        return self.vectorizer_.transform(self._clean(X))


# ── Pipeline factory ─────────────────────────────────────────────────────────

def build_pipeline():
    """Return a scikit-learn Pipeline that preprocesses raw Met Museum data.

    Expected input: a pandas DataFrame containing the raw CSV columns (after
    dropping identifier/URL/constant columns) but BEFORE any encoding.
    The pipeline handles imputation, scaling, encoding, and TF-IDF internally
    so that fit() only ever sees training data — no leakage.

    Output: a sparse matrix combining dense and TF-IDF features.

    Column count breakdown:
      5  numeric (Object Begin/End Date, AccessionYear, object_age, object_span)
      1  Artist Nationality (frequency-encoded + scaled)
      4  categorical (OrdinalEncoded)
      3  boolean flags
    100  Medium TF-IDF
     50  Tags TF-IDF
     50  Period TF-IDF
    ────
    213  total columns
    """

    numeric_branch = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        # with_mean=False: ColumnTransformer may convert output to sparse;
        # subtracting the mean would densify a sparse matrix.
        ("scaler",  StandardScaler(with_mean=False)),
    ])

    categorical_branch = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,           # unknown test categories → -1
        )),
    ])

    bool_branch = Pipeline([
        ("cast",    FunctionTransformer(_to_float)),   # bool dtype → float64
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ])

    ct = ColumnTransformer(
        transformers=[
            ("numeric", numeric_branch, NUMERIC_COLS),
            ("artist_nationality", Pipeline([
                ("freq",   FrequencyEncoder()),
                ("scaler", StandardScaler(with_mean=False)),
            ]), "Artist Nationality"),
            ("categorical", categorical_branch, CATEGORICAL_COLS),
            ("bool",        bool_branch,        BOOL_COLS),
            ("medium", TextPreprocessorTfidf(
                sep=",", max_features=100, ngram_range=(1, 2),
                sublinear_tf=True, stop_words="english",
            ), "Medium"),
            ("tags", TextPreprocessorTfidf(
                sep="|", max_features=50, ngram_range=(1, 1),
                sublinear_tf=True, stop_words=None,
            ), "Tags"),
            ("period", TextPreprocessorTfidf(
                sep=",", max_features=50, ngram_range=(1, 2),
                sublinear_tf=True, stop_words="english",
            ), "Period"),
        ],
        remainder="drop",   # every column not listed above is silently dropped
    )

    return Pipeline([
        ("date_engineer", DateFeatureEngineer()),   # must run before ColumnTransformer
        ("preprocessor",  ct),
    ])
