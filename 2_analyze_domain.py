#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import shap
from utils_dga import load_feature_schema, featurize_single, predict_proba_auto, summarize_shap, gen_playbook
def parse_args():
    p = argparse.ArgumentParser(description="Analyze a domain: detect (MOJO), explain (SHAP), prescribe (GenAI)")
    p.add_argument("--domain", required=True)
    p.add_argument("--model_zip", default="model/DGA_Leader.zip")
    p.add_argument("--feature_schema", default="model/feature_schema.json")
    p.add_argument("--background_csv", default="model/background_features.csv")
    p.add_argument("--positive_label", default="DGA")
    p.add_argument("--top_k", type=int, default=3)
    return p.parse_args()
def main():
    args = parse_args()
    feature_order = load_feature_schema(Path(args.feature_schema))
    x = featurize_single(args.domain, feature_order)
    proba = predict_proba_auto(
    x,
    mojo_zip=Path(args.model_zip),
    skl_pkl=Path("model/DGA_Leader_sklearn.pkl"),
    positive_label=args.positive_label
    )
    if args.background_csv and Path(args.background_csv).exists():
        bg = pd.read_csv(args.background_csv)
    else:
        bg = pd.DataFrame(np.tile(x.values, (50, 1)), columns=x.columns)
    def f_predict(Xin):
        df = pd.DataFrame(Xin, columns=x.columns)
        p = predict_proba_mojo(Path(args.model_zip), df, positive_label=args.positive_label)
        return np.array([p])
    explainer = shap.KernelExplainer(f_predict, bg)
    shap_vals = explainer.shap_values(x.values, nsamples="auto")
    top = summarize_shap(x.iloc[0], shap_vals, top_k=args.top_k)
    playbook = gen_playbook(args.domain, proba, top)
    print("\n=== Prescriptive DGA Detector ===")
    print(f"Domain: {args.domain}")
    print(f"DGA Probability: {proba:.2%}")
    print("Top signals (SHAP):")
    for i, s in enumerate(top, 1):
        print(f"  {i}. {s['feature']}: value={s['value']:.4f}, impact={s['shap']:+.4f}")
    print("\n--- Incident Response Playbook ---")
    print(playbook)
if __name__ == "__main__":
    main()
