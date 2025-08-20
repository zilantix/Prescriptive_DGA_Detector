import argparse
import joblib
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utils_dga import (
    featurize_single,
    load_feature_schema,
    shannon_entropy,
    digit_ratio,
    vowel_ratio,
    unique_char_ratio,
    consonant_run_mean,
)

def build_features(df: pd.DataFrame, label_col: str):
    """Convert domains into feature matrix + labels"""
    X, y = [], []
    for _, row in df.iterrows():
        feats = {
            "f_len": float(len(str(row["domain"]))),
            "f_entropy": shannon_entropy(row["domain"]),
            "f_digit_ratio": digit_ratio(row["domain"]),
            "f_vowel_ratio": vowel_ratio(row["domain"]),
            "f_unique_ratio": unique_char_ratio(row["domain"]),
            "f_cons_run_mean": consonant_run_mean(row["domain"]),
        }
        X.append(list(feats.values()))
        y.append(row[label_col])
    feature_order = list(feats.keys())
    return pd.DataFrame(X, columns=feature_order), np.array(y), feature_order

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, help="CSV file with domains + label column")
    ap.add_argument("--label_col", required=True, help="Name of the label column")
    ap.add_argument("--outdir", default="model", help="Directory to save model + schema")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load
    df = pd.read_csv(args.train_csv)
    if args.label_col not in df.columns:
        raise ValueError(f"Label column {args.label_col} not in dataset!")

    # featurize
    X, y, feature_order = build_features(df, args.label_col)

    # split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # train model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # eval
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    # save model
    bundle = {"model": model, "feature_order": feature_order}
    joblib.dump(bundle, outdir / "DGA_Leader_sklearn.pkl")
    print(f"✅ Saved sklearn model to {outdir / 'DGA_Leader_sklearn.pkl'}")

    # save schema
    schema = {"feature_order": feature_order}
    (outdir / "feature_schema.json").write_text(json.dumps(schema, indent=2))
    print(f"✅ Saved feature schema to {outdir / 'feature_schema.json'}")

    # save background features (for SHAP)
    X.head(50).to_csv(outdir / "background_features.csv", index=False)
    print(f"✅ Saved background sample to {outdir / 'background_features.csv'}")

if __name__ == "__main__":
    main()
