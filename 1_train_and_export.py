#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path
import numpy as np, pandas as pd
import h2o
from h2o.automl import H2OAutoML
VOWELS = set("aeiou")
def shannon_entropy(s: str) -> float:
    import pandas as pd, numpy as np, re
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    if not s: return 0.0
    counts = np.array(list(pd.Series(list(s)).value_counts()), dtype=float)
    probs = counts / counts.sum()
    return float((-probs * np.log2(probs)).sum())
def digit_ratio(s: str) -> float:
    import re
    s2 = re.sub(r"[^a-z0-9]", "", s.lower())
    return 0.0 if not s2 else sum(ch.isdigit() for ch in s2) / len(s2)
def vowel_ratio(s: str) -> float:
    import re
    s2 = re.sub(r"[^a-z]", "", s.lower())
    return 0.0 if not s2 else sum(ch in VOWELS for ch in s2) / len(s2)
def unique_char_ratio(s: str) -> float:
    import re
    s2 = re.sub(r"[^a-z0-9]", "", s.lower())
    return 0.0 if not s2 else len(set(s2)) / len(s2)
def consonant_run_mean(s: str) -> float:
    import re, numpy as np
    s2 = re.sub(r"[^a-z]", "", s.lower())
    if not s2: return 0.0
    runs, run = [], 0
    for ch in s2:
        if ch not in VOWELS: run += 1
        else:
            if run: runs.append(run)
            run = 0
    if run: runs.append(run)
    return float(np.mean(runs)) if runs else 0.0
def featurize(df: pd.DataFrame, domain_col: str) -> pd.DataFrame:
    import re
    feats = pd.DataFrame()
    d = df[domain_col].astype(str)
    d_clean = d.str.lower()
    d_alnum = d_clean.str.replace(r"[^a-z0-9]", "", regex=True)
    feats["f_len"] = d_alnum.str.len().fillna(0).astype(float)
    feats["f_entropy"] = d_alnum.apply(shannon_entropy).astype(float)
    feats["f_digit_ratio"] = d_alnum.apply(digit_ratio).astype(float)
    feats["f_vowel_ratio"] = d_clean.apply(vowel_ratio).astype(float)
    feats["f_unique_ratio"] = d_alnum.apply(unique_char_ratio).astype(float)
    feats["f_cons_run_mean"] = d_clean.apply(consonant_run_mean).astype(float)
    return feats
def infer_domain_col(columns):
    candidates = [c for c in columns if c.lower() in {"domain","fqdn","hostname"}]
    return candidates[0] if candidates else columns[0]
def parse_args():
    p = argparse.ArgumentParser(description="Train DGA detector with H2O AutoML and export MOJO")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--label_col", default="label")
    p.add_argument("--domain_col", default=None)
    p.add_argument("--time_limit", type=int, default=300)
    p.add_argument("--max_models", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="model")
    return p.parse_args()
def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.train_csv)
    domain_col = infer_domain_col([c for c in df.columns if c != args.label_col]) if args.domain_col is None else args.domain_col
    if args.label_col not in df.columns: raise ValueError(f"Label column '{args.label_col}' not found")
    if domain_col not in df.columns: raise ValueError(f"Domain column '{domain_col}' not found")
    y = df[args.label_col]
    if y.dtype != object: df[args.label_col] = y.map({0: "Legit", 1: "DGA"}).astype(str)
    X = featurize(df[[domain_col]].copy(), domain_col)
    Xy = X.copy(); Xy[args.label_col] = df[args.label_col].astype(str)
    h2o.init()
    from h2o import H2OFrame
    hf = H2OFrame(Xy); hf[args.label_col] = hf[args.label_col].asfactor()
    x_cols = [c for c in hf.columns if c != args.label_col]
    aml = H2OAutoML(max_models=args.max_models, max_runtime_secs=args.time_limit, seed=args.seed,
                    balance_classes=True, sort_metric="AUC", verbosity="info")
    aml.train(x=x_cols, y=args.label_col, training_frame=hf)
    leader = aml.leader
    mojo_path = leader.download_mojo(path=str(out_dir), get_genmodel_jar=False)
    (out_dir / "feature_schema.json").write_text(json.dumps({"feature_order": x_cols}, indent=2))
    bg = X.sample(n=min(200, len(X)), random_state=args.seed)
    bg.to_csv(out_dir / "background_features.csv", index=False)
    (out_dir / "model_info.json").write_text(json.dumps({
        "leader_model_id": leader.model_id, "mojo_path": mojo_path,
        "label_col": args.label_col, "domain_col": domain_col,
        "metrics": leader.model_performance(hf).metrics_list()
    }, indent=2))
    print(f"Exported MOJO -> {mojo_path}")
    print(f"Saved feature schema -> {out_dir/'feature_schema.json'}")
    print(f"Saved background set -> {out_dir/'background_features.csv'}")
if __name__ == "__main__":
    main()
