from __future__ import annotations
import json, os, re
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

from h2o.estimators.mojo_predict_pandas import MojoPredictor
import shap

VOWELS = set("aeiou")
def shannon_entropy(s: str) -> float:
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    if not s: return 0.0
    counts = np.array(list(pd.Series(list(s)).value_counts()), dtype=float)
    probs = counts / counts.sum()
    return float((-probs * np.log2(probs)).sum())
def digit_ratio(s: str) -> float:
    s2 = re.sub(r"[^a-z0-9]", "", s.lower())
    return 0.0 if not s2 else sum(ch.isdigit() for ch in s2) / len(s2)
def vowel_ratio(s: str) -> float:
    s2 = re.sub(r"[^a-z]", "", s.lower())
    return 0.0 if not s2 else sum(ch in VOWELS for ch in s2) / len(s2)
def unique_char_ratio(s: str) -> float:
    s2 = re.sub(r"[^a-z0-9]", "", s.lower())
    return 0.0 if not s2 else len(set(s2)) / len(s2)
def consonant_run_mean(s: str) -> float:
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
def featurize_single(domain: str, feature_order: List[str]) -> pd.DataFrame:
    d = str(domain).lower()
    d_alnum = re.sub(r"[^a-z0-9]", "", d)
    feats = {"f_len": float(len(d_alnum)), "f_entropy": float(shannon_entropy(d_alnum)),
             "f_digit_ratio": float(digit_ratio(d_alnum)), "f_vowel_ratio": float(vowel_ratio(d)),
             "f_unique_ratio": float(unique_char_ratio(d_alnum)), "f_cons_run_mean": float(consonant_run_mean(d))}
    row = {k: feats[k] for k in feature_order}
    return pd.DataFrame([row])
def load_feature_schema(path: Path) -> List[str]:
    schema = json.loads(Path(path).read_text())
    return list(schema["feature_order"])
def predict_proba_mojo(mojo_zip: Path, X: pd.DataFrame, positive_label: str = "DGA") -> float:
    mp = MojoPredictor(mojo_zip)
    out = mp.predict(X)
    if positive_label in out.columns:
        return float(out[positive_label].iloc[0])
    prob_cols = [c for c in out.columns if c != "predict"]
    if prob_cols: return float(out[prob_cols[0]].iloc[0])
    pred = str(out["predict"].iloc[0])
    return 1.0 if pred.lower() == positive_label.lower() else 0.0
def summarize_shap(x_row: pd.Series, shap_values, top_k: int = 3):
    vals = np.array(shap_values).flatten()
    names = list(x_row.index)
    order = np.argsort(np.abs(vals))[::-1][:top_k]
    return [{"feature": names[i], "value": float(x_row.iloc[i]), "shap": float(vals[i])} for i in order]
def gen_playbook(domain: str, proba_dga: float, top_signals):
    api_key = os.getenv("GOOGLE_API_KEY")
    sys_prompt = ("You are a SOC incident response copilot. Given a domain verdict and the top signals, "
                  "produce a step-by-step, actionable response playbook. Include triage, containment, "
                  "detection, eradication, recovery, and lessons learned. Be concise but specific.")
    user_summary = {"domain": domain, "verdict": f"{proba_dga:.1%} probability of DGA", "top_signals": top_signals}
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"{sys_prompt}\n\nContext:\n{json.dumps(user_summary, indent=2)}\n\nOutput a numbered playbook."
            resp = model.generate_content(prompt)
            text = resp.text.strip() if hasattr(resp, "text") and resp.text else ""
            if text: return text
        except Exception as e:
            print(f"[WARN] GenAI failed: {e}. Falling back to local template.")
    bullets = []
    bullets.append("1) **Triage**: Check DNS logs and passive DNS for resolution volume and first-seen of the domain. Correlate with EDR/Proxy for connection attempts.")
    bullets.append("2) **Containment**: Block domain at DNS resolver and egress proxy; add to sinkhole if available. Quarantine hosts with repeated lookups.")
    bullets.append("3) **Detection**: Create rules for high-entropy domains and anomalous length (based on top SHAP signals). Monitor spikes in NXDOMAINs.")
    bullets.append("4) **Eradication**: Identify implants via process ancestry around lookup times. Remove persistence (Run keys, services, tasks).")
    bullets.append("5) **Recovery**: Re-enable network access post-clean. Rotate credentials found on affected hosts.")
    bullets.append("6) **Lessons Learned**: Add IOCs and detection logic to SIEM. Update DGA watchlist and user awareness notes.")
    signal_txt = "\n".join([f"- {s['feature']}: value={s['value']:.4f}, impact={s['shap']:+.4f}" for s in top_signals])
    header = f"Domain: `{domain}`\nVerdict: **{proba_dga:.1%} DGA probability**\n\nTop signals (from SHAP):\n{signal_txt}\n\nPrescriptive playbook:\n"
    return header + "\n".join(bullets)
