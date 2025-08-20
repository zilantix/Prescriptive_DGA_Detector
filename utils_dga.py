from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

# Lazy/optional H2O MOJO runtime
try:
    from h2o.estimators.mojo_predict_pandas import MojoPredictor  # type: ignore
    H2O_OK = True
except Exception:
    H2O_OK = False

# -----------------------------
# Feature engineering
# -----------------------------

_VOWELS = set("aeiou")

def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def shannon_entropy(s: str) -> float:
    s = _alnum(s)
    if not s:
        return 0.0
    counts = np.array(list(pd.Series(list(s)).value_counts()), dtype=float)
    probs = counts / counts.sum()
    return float((-probs * np.log2(probs)).sum())

def digit_ratio(s: str) -> float:
    s2 = _alnum(s)
    return 0.0 if not s2 else sum(ch.isdigit() for ch in s2) / len(s2)

def vowel_ratio(s: str) -> float:
    s2 = re.sub(r"[^a-z]", "", s.lower())
    return 0.0 if not s2 else sum(ch in _VOWELS for ch in s2) / len(s2)

def unique_char_ratio(s: str) -> float:
    s2 = _alnum(s)
    return 0.0 if not s2 else len(set(s2)) / len(s2)

def consonant_run_mean(s: str) -> float:
    s2 = re.sub(r"[^a-z]", "", s.lower())
    if not s2:
        return 0.0
    runs, run = [], 0
    for ch in s2:
        if ch not in _VOWELS:
            run += 1
        else:
            if run:
                runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return float(np.mean(runs)) if runs else 0.0

def featurize_single(domain: str, feature_order: List[str]) -> pd.DataFrame:
    d = str(domain)
    feats = {
        "f_len": float(len(_alnum(d))),
        "f_entropy": float(shannon_entropy(d)),
        "f_digit_ratio": float(digit_ratio(d)),
        "f_vowel_ratio": float(vowel_ratio(d)),
        "f_unique_ratio": float(unique_char_ratio(d)),
        "f_cons_run_mean": float(consonant_run_mean(d)),
    }
    row = {k: feats[k] for k in feature_order}
    return pd.DataFrame([row])

def load_feature_schema(path: Path) -> List[str]:
    schema = json.loads(Path(path).read_text())
    return list(schema["feature_order"])

# -----------------------------
# Inference backends
# -----------------------------

def _auto_pick_positive_idx(classes: List[str]) -> int:
    """Pick index of the 'positive' class using common names."""
    lowered = [str(c).lower() for c in classes]
    for cand in ("dga", "malicious", "bad", "1", "true", "yes"):
        if cand in lowered:
            return lowered.index(cand)
    # fallback: if binary, assume first is positive; else 0
    return 0

def predict_proba_sklearn(
    pkl_path: Path,
    X: pd.DataFrame,
    positive_label: str | None = None,
) -> float:
    """Return probability of the positive class for a single-row DataFrame."""
    bundle = joblib.load(pkl_path)
    model = bundle["model"]
    feature_order = bundle.get("feature_order", list(X.columns))
    X2 = X[feature_order]

    probs = model.predict_proba(X2)[0]  # shape (n_classes,)
    classes = [str(c) for c in getattr(model, "classes_", [])]

    if positive_label is None:
        idx = _auto_pick_positive_idx(classes)
        return float(probs[idx])

    # case-insensitive match
    for i, c in enumerate(classes):
        if c.lower() == str(positive_label).lower():
            return float(probs[i])

    # fallback
    return float(probs[0])

def predict_proba_mojo(
    mojo_zip: Path,
    X: pd.DataFrame,
    positive_label: str = "DGA",
) -> float:
    if not H2O_OK:
        raise RuntimeError("H2O MOJO runtime not installed. Install h2o==3.44.0.2 or use sklearn .pkl.")
    mp = MojoPredictor(mojo_zip)
    out = mp.predict(X)
    if positive_label in out.columns:
        return float(out[positive_label].iloc[0])
    prob_cols = [c for c in out.columns if c != "predict"]
    if prob_cols:
        return float(out[prob_cols[0]].iloc[0])
    pred = str(out["predict"].iloc[0])
    return 1.0 if pred.lower() == positive_label.lower() else 0.0

def predict_proba_auto(
    X: pd.DataFrame,
    mojo_zip: Path = Path("model/DGA_Leader.zip"),
    skl_pkl: Path = Path("model/DGA_Leader_sklearn.pkl"),
    positive_label: str = "DGA",
) -> float:
    """Prefer sklearn .pkl; fall back to MOJO if needed."""
    if skl_pkl.exists():
        # auto-detect positive label inside the sklearn model
        return predict_proba_sklearn(skl_pkl, X, positive_label=None)
    if mojo_zip.exists():
        return predict_proba_mojo(mojo_zip, X, positive_label=positive_label)
    raise FileNotFoundError(
        f"No usable model found. Looked for sklearn PKL '{skl_pkl}' or H2O MOJO '{mojo_zip}'."
    )

# -----------------------------
# SHAP & Playbook
# -----------------------------

def summarize_shap(x_row: pd.Series, shap_values, top_k: int = 3):
    vals = np.array(shap_values).flatten()
    names = list(x_row.index)
    order = np.argsort(np.abs(vals))[::-1][:top_k]
    return [
        {"feature": names[i], "value": float(x_row.iloc[i]), "shap": float(vals[i])}
        for i in order
    ]

def gen_playbook(domain: str, proba_dga: float, top_signals):
    """Use Gemini if GOOGLE_API_KEY is present; otherwise local template."""
    api_key = os.getenv("GOOGLE_API_KEY")
    sys_prompt = (
        "You are a SOC incident response copilot. Given a domain verdict and the top signals, "
        "produce a step-by-step, actionable response playbook. Include triage, containment, "
        "detection, eradication, recovery, and lessons learned. Be concise but specific."
    )
    user_summary = {
        "domain": domain,
        "verdict": f"{proba_dga:.1%} probability of DGA",
        "top_signals": top_signals,
    }
    if api_key:
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"{sys_prompt}\n\nContext:\n{json.dumps(user_summary, indent=2)}\n\nOutput a numbered playbook."
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", "") or ""
            text = text.strip()
            if text:
                return text
        except Exception:
            pass

    # Fallback playbook
    bullets = [
        "1) **Triage**: Review DNS/passive DNS for volume, first-seen; correlate EDR/Proxy around lookup times.",
        "2) **Containment**: Block at resolver and egress proxy; quarantine endpoints with repeated lookups.",
        "3) **Detection**: Add rules for high-entropy and abnormal length (per signals); watch for NXDOMAIN spikes.",
        "4) **Eradication**: Trace process ancestry; remove persistence (services, tasks, Run keys).",
        "5) **Recovery**: Re-enable access after cleanup; rotate credentials on impacted hosts.",
        "6) **Lessons Learned**: Add IOCs/detections to SIEM; update DGA watchlist and runbook.",
    ]
    signal_txt = "\n".join(
        [f"- {s['feature']}: value={s['value']:.4f}, impact={s['shap']:+.4f}" for s in top_signals]
    )
    header = (
        f"Domain: `{domain}`\n"
        f"Verdict: **{proba_dga:.1%} DGA probability**\n\n"
        f"Top signals (from SHAP):\n{signal_txt}\n\n"
        f"Prescriptive playbook:\n"
    )
    return header + "\n".join(bullets)