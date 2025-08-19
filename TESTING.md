# TESTING
## Legit Domain
```
python 2_analyze_domain.py --domain google.com   --model_zip model/DGA_Leader.zip   --feature_schema model/feature_schema.json   --background_csv model/background_features.csv
```
Expected: low DGA probability (<5%); SHAP shows entropy/length lowering risk.
## DGA-Like Domain
```
python 2_analyze_domain.py --domain q9z1x7v3k2t0a4b8c1.net   --model_zip model/DGA_Leader.zip   --feature_schema model/feature_schema.json   --background_features model/background_features.csv
```
Expected: high DGA probability (>70%); SHAP shows entropy/length/digit_ratio increasing risk.
## Gemini (Optional)
```
export GOOGLE_API_KEY=***
python 2_analyze_domain.py --domain google.com
```
Expected: A numbered IR playbook is generated.
