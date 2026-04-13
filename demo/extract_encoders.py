"""
Extract and save the encoding artifacts that match X_train.npz / trained models.

The preprocessing notebook (02_preprocessing.ipynb) fit its encoders on the FULL
dataset before the train/test split, and those encoder objects were never saved.
This script replicates those notebook cells exactly and saves everything to
models/encoders.joblib so that demo/predictor.py can reconstruct the 214-column
feature vector without re-running the notebook.

Column layout of X_train.npz (214 features):
  0   Is Highlight
  1   Is Timeline Work
  2   Is Public Domain
  3   AccessionYear
  4   Object Name   (LabelEncoded, top-50 + "Other")
  5   Culture       (frequency count)
  6   Artist Role   (LabelEncoded, top-15 + "Other")
  7   Artist Nationality (frequency count)
  8   Artist Gender (LabelEncoded, top-3 + "Other")
  9   Object Begin Date
 10   Object End Date
 11   Classification (LabelEncoded, top-50 + "Other")
 12   object_age
 13   object_span
 14-113  Medium TF-IDF  (100 features)
114-163  Tags TF-IDF    (50 features)
164-213  Period TF-IDF  (50 features)

Run from project root:
  python demo/extract_encoders.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV   = os.path.join(ROOT, "data", "MetObjects.csv")
MODELS    = os.path.join(ROOT, "models")

COLS_TO_DROP = [
    "Object ID", "Object Number", "Constituent ID",
    "Link Resource", "Artist ULAN URL", "Artist Wikidata URL",
    "Object Wikidata URL", "Tags AAT URL", "Tags Wikidata URL",
    "Metadata Date", "Repository",
    "River", "State", "Locus", "County", "Locale",
    "Excavation", "Subregion", "Region", "City",
    "Rights and Reproduction", "Portfolio", "Gallery Number", "Credit Line",
    "Artist Prefix", "Artist Suffix", "Artist Alpha Sort",
    "Artist Display Bio", "Artist Display Name",
    "Dynasty", "Reign", "Geography Type", "Country",
    "Artist Begin Date", "Artist End Date",
    "Dimensions", "Title", "Object Date",
    "Department",   # target — not a feature
]

COLS_CATEGORICAL = [
    "Medium", "Tags", "Classification", "Object Name",
    "Culture", "Period", "Artist Nationality", "Artist Role", "Artist Gender",
]
COLS_BOOL    = ["Is Highlight", "Is Public Domain", "Is Timeline Work"]
COLS_NUMERIC = ["AccessionYear"]

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading MetObjects.csv …")
df = pd.read_csv(RAW_CSV, low_memory=False)
print(f"  shape: {df.shape}")

# Drop columns not used as features
drop_existing = [c for c in COLS_TO_DROP if c in df.columns]
df.drop(columns=drop_existing, inplace=True)

# ── Imputation (replicates notebook cell 7) ───────────────────────────────────
for col in COLS_NUMERIC:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

for col in COLS_CATEGORICAL:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

for col in COLS_BOOL:
    if col in df.columns:
        df[col] = df[col].fillna(False)

# Period: re-fill to "" for TF-IDF (empty = no period info, not "Unknown")
df["Period"] = df["Period"].fillna("")

# ── Cast types (cell 11) ──────────────────────────────────────────────────────
for col in COLS_BOOL:
    df[col] = df[col].astype(int)
df["AccessionYear"] = df["AccessionYear"].astype(int)

# Save AccessionYear median for imputing unseen inputs
accession_median = int(df["AccessionYear"].median())

# ── Date features (cell 12) ───────────────────────────────────────────────────
df["Object Begin Date"] = df["Object Begin Date"].clip(lower=-7000, upper=2026)
df["Object End Date"]   = df["Object End Date"].clip(lower=-7000, upper=2026)
df["object_age"]  = 2026 - df["Object End Date"]
df["object_span"] = (df["Object End Date"] - df["Object Begin Date"]).clip(lower=0)

# ── Culture frequency map (cell 15) ───────────────────────────────────────────
print("Building Culture frequency map …")
culture_freq_map = df["Culture"].value_counts().to_dict()

# ── Artist Nationality frequency map (cell 18) ────────────────────────────────
print("Building Artist Nationality frequency map …")
nationality_freq_map = df["Artist Nationality"].value_counts().to_dict()

# ── Object Name LabelEncoder (cell 16) ────────────────────────────────────────
print("Fitting Object Name encoder …")
top_object_names = set(df["Object Name"].value_counts().head(50).index)
df["Object Name"] = df["Object Name"].apply(
    lambda x: x if x in top_object_names else "Other"
)
object_name_encoder = LabelEncoder()
object_name_encoder.fit(df["Object Name"])

# ── Classification LabelEncoder (cell 17) ─────────────────────────────────────
print("Fitting Classification encoder …")
top_classifications = set(df["Classification"].value_counts().head(50).index)
df["Classification"] = df["Classification"].apply(
    lambda x: x if x in top_classifications else "Other"
)
classification_encoder = LabelEncoder()
classification_encoder.fit(df["Classification"])

# ── Artist Role LabelEncoder (cell 18) ────────────────────────────────────────
print("Fitting Artist Role encoder …")
top_artist_roles = set(df["Artist Role"].value_counts().head(15).index)
df["Artist Role"] = df["Artist Role"].apply(
    lambda x: x if x in top_artist_roles else "Other"
)
artist_role_encoder = LabelEncoder()
artist_role_encoder.fit(df["Artist Role"])

# ── Artist Gender LabelEncoder (cell 18) ──────────────────────────────────────
print("Fitting Artist Gender encoder …")
top_artist_genders = set(df["Artist Gender"].value_counts().head(3).index)
df["Artist Gender"] = df["Artist Gender"].apply(
    lambda x: x if x in top_artist_genders else "Other"
)
artist_gender_encoder = LabelEncoder()
artist_gender_encoder.fit(df["Artist Gender"])

# ── Medium TF-IDF (cells 20-22) ───────────────────────────────────────────────
print("Fitting Medium TF-IDF (100 features) …")
df["Medium"] = (
    df["Medium"]
    .fillna("")
    .str.replace(",", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
medium_vectorizer = TfidfVectorizer(
    max_features=100, ngram_range=(1, 2), sublinear_tf=True, stop_words="english"
)
medium_vectorizer.fit(df["Medium"])

# ── Tags TF-IDF (cells 21-22) ─────────────────────────────────────────────────
print("Fitting Tags TF-IDF (50 features) …")
df["Tags"] = (
    df["Tags"]
    .fillna("")
    .str.replace("|", " ", regex=False)
    .str.strip()
)
tags_vectorizer = TfidfVectorizer(
    max_features=50, ngram_range=(1, 1), sublinear_tf=True,
)
tags_vectorizer.fit(df["Tags"])

# ── Period TF-IDF (cells 24) ──────────────────────────────────────────────────
print("Fitting Period TF-IDF (50 features) …")
df["Period"] = (
    df["Period"]
    .str.replace(",", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
period_vectorizer = TfidfVectorizer(
    max_features=50, ngram_range=(1, 2), sublinear_tf=True, stop_words="english"
)
period_vectorizer.fit(df["Period"])

# ── Bundle and save ───────────────────────────────────────────────────────────
encoders = {
    "culture_freq_map":      culture_freq_map,
    "nationality_freq_map":  nationality_freq_map,
    "top_object_names":      top_object_names,
    "object_name_encoder":   object_name_encoder,
    "top_classifications":   top_classifications,
    "classification_encoder": classification_encoder,
    "top_artist_roles":      top_artist_roles,
    "artist_role_encoder":   artist_role_encoder,
    "top_artist_genders":    top_artist_genders,
    "artist_gender_encoder": artist_gender_encoder,
    "medium_vectorizer":     medium_vectorizer,
    "tags_vectorizer":       tags_vectorizer,
    "period_vectorizer":     period_vectorizer,
    "accession_median":      accession_median,
}

out_path = os.path.join(MODELS, "encoders.joblib")
joblib.dump(encoders, out_path)
print(f"\nSaved → {out_path}")

# ── Quick sanity check ────────────────────────────────────────────────────────
print("\nSanity check — feature vector for first row:")
import scipy.sparse

row = df.iloc[[0]]

# Dense features in column order matching X_train.npz
dense_vals = [
    int(row["Is Highlight"].iloc[0]),
    int(row["Is Timeline Work"].iloc[0]),
    int(row["Is Public Domain"].iloc[0]),
    int(row["AccessionYear"].iloc[0]),
    int(object_name_encoder.transform(
        [row["Object Name"].iloc[0] if row["Object Name"].iloc[0] in top_object_names else "Other"]
    )[0]),
    int(culture_freq_map.get(
        df.iloc[0]["Culture"] if "Culture" in df.columns else "Unknown", 0
    )),
    int(artist_role_encoder.transform(
        [row["Artist Role"].iloc[0] if row["Artist Role"].iloc[0] in top_artist_roles else "Other"]
    )[0]),
    int(nationality_freq_map.get(row["Artist Nationality"].iloc[0], 0)),
    int(artist_gender_encoder.transform(
        [row["Artist Gender"].iloc[0] if row["Artist Gender"].iloc[0] in top_artist_genders else "Other"]
    )[0]),
    int(row["Object Begin Date"].iloc[0]),
    int(row["Object End Date"].iloc[0]),
    int(classification_encoder.transform(
        [row["Classification"].iloc[0] if row["Classification"].iloc[0] in top_classifications else "Other"]
    )[0]),
    int(row["object_age"].iloc[0]),
    int(row["object_span"].iloc[0]),
]
dense_sp = scipy.sparse.csr_matrix(np.array(dense_vals, dtype=float).reshape(1, -1))

med_sp    = medium_vectorizer.transform([row["Medium"].iloc[0]])
tags_sp   = tags_vectorizer.transform([row["Tags"].iloc[0]])
period_sp = period_vectorizer.transform([row["Period"].iloc[0]])

X_test = scipy.sparse.hstack([dense_sp, med_sp, tags_sp, period_sp])
print(f"  Feature vector shape: {X_test.shape}  (expected (1, 214))")
assert X_test.shape == (1, 214), f"Expected (1, 214), got {X_test.shape}"
print("  Shape assertion passed ✓")
