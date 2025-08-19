# Prescriptive DGA Detector
End-to-end tool that combines **H2O AutoML**, **SHAP**, and **Generative AI** to detect DGAs, explain model reasoning, and produce a prescriptive incident response playbook.
## Repo Layout
```
model/                 # MOJO + artifacts (created by training)
data/                  # put your training CSV here
1_train_and_export.py  # AutoML + MOJO export
2_analyze_domain.py    # detect -> explain -> prescribe (CLI)
utils_dga.py           # shared functions
app_streamlit.py       # Streamlit UI
.github/workflows/lint.yml
requirements.txt
README.md
TESTING.md
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
