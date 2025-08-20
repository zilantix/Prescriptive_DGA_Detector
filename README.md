# Prescriptive DGA Detector

demo: Streamlit App https://dga-detect.streamlit.app/

End-to-end tool that combines **AutoML (H2O / sklearn fallback)**, **SHAP explainability**, and **Generative AI** to:

- Detect if a domain is algorithmically generated (DGA) or legitimate.
- Explain *why* the model made that prediction using SHAP.
- Automatically produce a **prescriptive incident response playbook** for analysts.

---

## Repo Layout
```
model/
├── DGA_Leader_sklearn.pkl   # Trained sklearn model (or H2O MOJO if using AutoML)
├── feature_schema.json      # Feature schema
└── background_features.csv  # Background SHAP data

data/
└── dga_dataset_train.csv    # Training dataset (domain, features, label)

1_train_and_export.py          # H2O AutoML + MOJO export
1_train_and_export_sklearn.py  # Sklearn training + pickle export (fallback)
2_analyze_domain.py            # CLI: detect → explain → prescribe
utils_dga.py                   # Shared utilities
app_streamlit.py               # Streamlit UI for interactive analysis

.github/workflows/lint.yml     # CI linting
requirements.txt               # Python dependencies
README.md                      # Project documentation
TESTING.md                     # Manual testing guide
```
## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
(Optional) For GenAI playbooks, set your Gemini key:
```bash
export GOOGLE_API_KEY=YOUR_KEY
```
## Train
Assumes CSV at `data/dga_dataset_train.csv` with columns `domain` and `label` (Legit/DGA or 0/1).
```bash
python 1_train_and_export.py --train_csv data/dga_dataset_train.csv --label_col label --time_limit 300 --max_models 20
```
Artifacts appear in `model/`: `DGA_Leader.zip`, `feature_schema.json`, `background_features.csv`.
## CLI: Analyze a Domain
```bash
python 2_analyze_domain.py --domain google.com   --model_zip model/DGA_Leader.zip   --feature_schema model/feature_schema.json   --background_csv model/background_features.csv
```
## Streamlit App
```bash
streamlit run app_streamlit.py
```
Then open the local URL shown by Streamlit. Enter a domain and click **Analyze**.
## CI
GitHub Actions lints on push via `ruff` and `flake8`.
