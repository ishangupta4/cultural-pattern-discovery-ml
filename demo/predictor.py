"""
Self-contained prediction module for the Streamlit demo.

Accepts free-text / categorical inputs from the UI and returns a department
prediction using the trained XGBoost model and the original encoder artifacts.

The XGBoost model was trained on X_train.npz (214-feature matrix).  This
module reconstructs that exact feature layout without touching the raw dataset.

Feature layout (214 columns):
  0   Is Highlight        (0/1)
  1   Is Timeline Work    (0/1)
  2   Is Public Domain    (0/1)
  3   AccessionYear       (int)
  4   Object Name         (LabelEncoder, top-50 + "Other")
  5   Culture             (frequency count)
  6   Artist Role         (LabelEncoder, top-15 + "Other")
  7   Artist Nationality  (frequency count)
  8   Artist Gender       (LabelEncoder, top-3 + "Other")
  9   Object Begin Date   (int, clipped [-7000, 2026])
 10   Object End Date     (int, clipped [-7000, 2026])
 11   Classification      (LabelEncoder, top-50 + "Other")
 12   object_age          (2026 - Object End Date)
 13   object_span         (Object End Date - Object Begin Date, ≥ 0)
 14–113   Medium TF-IDF  (100 features)
114–163   Tags TF-IDF    (50 features)
164–213   Period TF-IDF  (50 features)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


# ── Lazy loaders ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_booster():
    import xgboost as xgb
    b = xgb.Booster()
    b.load_model(str(MODELS / "xgb_model.json"))
    return b


@lru_cache(maxsize=1)
def _load_label_encoder():
    import joblib
    return joblib.load(str(MODELS / "label_encoder.joblib"))


@lru_cache(maxsize=1)
def _load_encoders() -> dict:
    import joblib
    return joblib.load(str(MODELS / "encoders.joblib"))


# ── Feature construction ──────────────────────────────────────────────────────

def _build_feature_vector(inputs: dict[str, Any]) -> "scipy.sparse.csr_matrix":
    """Convert a dict of raw field values into a (1, 214) sparse matrix."""
    enc = _load_encoders()

    # ── Numeric / date fields ────────────────────────────────────────────────
    try:
        begin = int(float(inputs.get("Object Begin Date") or 0))
    except (ValueError, TypeError):
        begin = 0
    try:
        end = int(float(inputs.get("Object End Date") or 2026))
    except (ValueError, TypeError):
        end = 2026
    try:
        acq = int(float(inputs.get("AccessionYear") or enc["accession_median"]))
    except (ValueError, TypeError):
        acq = enc["accession_median"]

    begin = max(-7000, min(2026, begin))
    end   = max(-7000, min(2026, end))
    age   = 2026 - end
    span  = max(0, end - begin)

    # ── Boolean flags ─────────────────────────────────────────────────────────
    is_highlight    = int(bool(inputs.get("Is Highlight", False)))
    is_timeline     = int(bool(inputs.get("Is Timeline Work", False)))
    is_public       = int(bool(inputs.get("Is Public Domain", False)))

    # ── Frequency-encoded fields ──────────────────────────────────────────────
    culture = str(inputs.get("Culture") or "Unknown")
    culture_freq = enc["culture_freq_map"].get(culture, 0)

    artist_nat = str(inputs.get("Artist Nationality") or "Unknown")
    nat_freq = enc["nationality_freq_map"].get(artist_nat, 0)

    # ── LabelEncoded fields (top-N + "Other" fallback) ────────────────────────
    obj_name_raw = str(inputs.get("Object Name") or "Unknown")
    obj_name = obj_name_raw if obj_name_raw in enc["top_object_names"] else "Other"
    obj_name_enc = int(enc["object_name_encoder"].transform([obj_name])[0])

    classif_raw = str(inputs.get("Classification") or "Unknown")
    classif = classif_raw if classif_raw in enc["top_classifications"] else "Other"
    classif_enc = int(enc["classification_encoder"].transform([classif])[0])

    role_raw = str(inputs.get("Artist Role") or "Unknown")
    role = role_raw if role_raw in enc["top_artist_roles"] else "Other"
    role_enc = int(enc["artist_role_encoder"].transform([role])[0])

    gender_raw = str(inputs.get("Artist Gender") or "Unknown")
    gender = gender_raw if gender_raw in enc["top_artist_genders"] else "Other"
    gender_enc = int(enc["artist_gender_encoder"].transform([gender])[0])

    # ── Assemble dense block (14 cols) ────────────────────────────────────────
    dense = np.array([[
        float(is_highlight),
        float(is_timeline),
        float(is_public),
        float(acq),
        float(obj_name_enc),
        float(culture_freq),
        float(role_enc),
        float(nat_freq),
        float(gender_enc),
        float(begin),
        float(end),
        float(classif_enc),
        float(age),
        float(span),
    ]])

    # ── TF-IDF blocks ─────────────────────────────────────────────────────────
    medium_raw = str(inputs.get("Medium") or "")
    medium_text = (
        medium_raw
        .replace(",", " ")
        .replace("  ", " ")
        .strip()
    )
    med_v = enc["medium_vectorizer"].transform([medium_text])

    tags_raw = str(inputs.get("Tags") or "")
    tags_text = tags_raw.replace("|", " ").strip()
    tag_v = enc["tags_vectorizer"].transform([tags_text])

    period_raw = str(inputs.get("Period") or "")
    period_text = (
        period_raw
        .replace(",", " ")
        .replace("  ", " ")
        .strip()
    )
    per_v = enc["period_vectorizer"].transform([period_text])

    # ── Horizontal stack → (1, 214) ───────────────────────────────────────────
    return scipy.sparse.hstack([
        scipy.sparse.csr_matrix(dense),
        med_v,
        tag_v,
        per_v,
    ], format="csr")


# ── Public API ────────────────────────────────────────────────────────────────

def predict(
    inputs: "dict[str, Any] | None" = None,
    *,
    medium: str = "",
    culture: str = "",
    tags: str = "",
    period: str = "",
    object_begin_date: "int | None" = None,
    object_end_date: "int | None" = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Predict the Met Museum department from metadata inputs.

    Accepts either a dict (used by the Streamlit app) or individual keyword
    arguments (used by the smoke-test CLI):

        predict({"Medium": "Oil on canvas", ...})          # dict form
        predict(medium="Oil on canvas", culture="French")  # kwarg form

    Parameters — dict keys (all optional):
      "Medium", "Tags", "Period", "Culture", "Classification",
      "Object Name", "Artist Role", "Artist Nationality",
      "Artist Gender", "Object Begin Date", "Object End Date",
      "AccessionYear", "Is Highlight", "Is Timeline Work", "Is Public Domain"

    Returns
    -------
    dict with keys:
      "department"          : str   — predicted department name
      "predicted_department": str   — alias for "department"
      "confidence"          : float — probability of top class (0–1)
      "probabilities"       : dict  — {department_name: probability} for all 19 classes
      "top5"                : list  — [(dept, prob), ...] top-5 sorted by probability desc
    """
    import xgboost as xgb

    # Normalise calling convention: kwargs → dict
    if inputs is None:
        inputs = {
            "Medium":            medium,
            "Culture":           culture,
            "Tags":              tags,
            "Period":            period,
        }
        if object_begin_date is not None:
            inputs["Object Begin Date"] = object_begin_date
        if object_end_date is not None:
            inputs["Object End Date"] = object_end_date
        inputs.update(kwargs)

    booster = _load_booster()
    le = _load_label_encoder()

    X = _build_feature_vector(inputs)
    proba = booster.predict(xgb.DMatrix(X))[0]   # shape (19,)

    idx = int(np.argmax(proba))
    dept = le.classes_[idx]
    prob_map = {cls: float(proba[i]) for i, cls in enumerate(le.classes_)}
    top5 = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "department":           dept,
        "predicted_department": dept,          # alias for smoke-test compatibility
        "confidence":           float(proba[idx]),
        "probabilities":        prob_map,
        "top5":                 top5,          # alias for smoke-test compatibility
    }


def get_sample_inputs() -> list[dict[str, Any]]:
    """Return a list of representative sample inputs for the demo UI."""
    return [
        {
            "label":              "Egyptian votive statuette (Late Period)",
            "Medium":             "Cupreous metal",
            "Tags":               "",
            "Period":             "Late Period (Saite)-Ptolemaic Period",
            "Culture":            "",
            "Classification":     "",
            "Object Name":        "Statuette",
            "Artist Role":        "",
            "Artist Nationality": "",
            "Artist Gender":      "",
            "Object Begin Date":  -664,
            "Object End Date":    -30,
            "AccessionYear":      1920,
            "Is Highlight":       False,
            "Is Timeline Work":   False,
            "Is Public Domain":   True,
        },
        {
            "label":              "Japanese Edo-period leatherwork",
            "Medium":             "Leather",
            "Tags":               "Animals|Leaves",
            "Period":             "Edo period (1615-1868)",
            "Culture":            "Japan",
            "Classification":     "Leatherwork",
            "Object Name":        "Piece",
            "Artist Role":        "",
            "Artist Nationality": "Japanese",
            "Artist Gender":      "",
            "Object Begin Date":  1700,
            "Object End Date":    1899,
            "AccessionYear":      1915,
            "Is Highlight":       False,
            "Is Timeline Work":   True,
            "Is Public Domain":   True,
        },
        {
            "label":              "19th-century albumen silver photograph",
            "Medium":             "Albumen silver print",
            "Tags":               "Insects",
            "Period":             "",
            "Culture":            "",
            "Classification":     "Photographs",
            "Object Name":        "Photograph",
            "Artist Role":        "Artist",
            "Artist Nationality": "British",
            "Artist Gender":      "",
            "Object Begin Date":  1851,
            "Object End Date":    1855,
            "AccessionYear":      1990,
            "Is Highlight":       False,
            "Is Timeline Work":   False,
            "Is Public Domain":   True,
        },
        {
            "label":              "American Revolutionary War gorget",
            "Medium":             "Brass, gold",
            "Tags":               "",
            "Period":             "",
            "Culture":            "Anglo-American",
            "Classification":     "Armor Parts-Gorgets",
            "Object Name":        "Gorget",
            "Artist Role":        "",
            "Artist Nationality": "",
            "Artist Gender":      "",
            "Object Begin Date":  1781,
            "Object End Date":    1781,
            "AccessionYear":      1935,
            "Is Highlight":       False,
            "Is Timeline Work":   True,
            "Is Public Domain":   True,
        },
        {
            "label":              "European oil painting miniature",
            "Medium":             "Oil on ivory",
            "Tags":               "Seas|Storms|Ships",
            "Period":             "",
            "Culture":            "",
            "Classification":     "Miniatures",
            "Object Name":        "Painting, miniature",
            "Artist Role":        "Artist",
            "Artist Nationality": "British",
            "Artist Gender":      "Male",
            "Object Begin Date":  1837,
            "Object End Date":    1900,
            "AccessionYear":      1950,
            "Is Highlight":       False,
            "Is Timeline Work":   False,
            "Is Public Domain":   True,
        },
    ]
