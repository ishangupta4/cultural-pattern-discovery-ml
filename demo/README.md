# Neural Beings — Streamlit Demo

Interactive demo for the Met Museum ML project.  
Two tabs: a 3D UMAP explorer and a department prediction interface.

## Setup

Install demo dependencies (from the project root):

```bash
pip install -r requirements_demo.txt
```

## Run

From the **project root** (not the `demo/` folder):

```bash
streamlit run demo/app.py
```

The app opens at `http://localhost:8501`.

## Tabs

| Tab | What it does |
|-----|-------------|
| 🌌 UMAP Explorer | Interactive 3-D scatter of 30k artworks coloured by department, top-15 culture, or object age. Filter by department and adjust sample size. |
| 🔮 Guess the Department | Enter artwork metadata (medium, culture, tags, dates) and predict the Met curatorial department using the trained XGBoost model. Includes 5 pre-loaded examples. |

## Notes

- The UMAP chart loads from `demo/data/umap_embeddings.csv` (pre-computed, 30k rows).
- Predictions use `models/xgb_model.json` + `models/encoders.joblib` (no raw data needed at runtime).
- Charts use Plotly CDN — an internet connection is required for the charts to render.
- To regenerate the UMAP embeddings: `python demo/export_demo_data.py`
- To regenerate the encoder artifacts: `python demo/extract_encoders.py`
