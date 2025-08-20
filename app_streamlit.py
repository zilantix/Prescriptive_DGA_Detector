from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import shap
import streamlit as st

from utils_dga import (
    load_feature_schema,
    featurize_single,
    predict_proba_auto,
    summarize_shap,
    gen_playbook,
)

st.set_page_config(page_title="Prescriptive DGA Detector", layout="centered")
st.title("🔎 Prescriptive DGA Detector")
st.caption("AutoML (H2O/sklearn) ➜ SHAP ➜ Prescriptive IR Playbook (Gemini or local)")

# Sidebar inputs
with st.sidebar:
    st.header("Model Artifacts")
    model_zip = st.text_input("MOJO zip (optional)", "model/DGA_Leader.zip")
    feat_schema = st.text_input("Feature schema JSON", "model/feature_schema.json")
    bg_csv = st.text_input("Background CSV (for SHAP)", "model/background_features.csv")
    top_k = st.slider("Top SHAP signals", 1, 6, 3)
    threshold = st.slider("Decision threshold (DGA if ≥ T)", 0.0, 1.0, 0.50, 0.01)

domain = st.text_input("Domain", "google.com")
run = st.button("Analyze")

if run:
    try:
        # 1) Feature engineering (match training exactly)
        feature_order = load_feature_schema(Path(feat_schema))
        x = featurize_single(domain, feature_order)

        # 2) Predict probability (sklearn .pkl preferred; H2O MOJO fallback)
        proba = predict_proba_auto(
            x,
            mojo_zip=Path(model_zip),
            skl_pkl=Path("model/DGA_Leader_sklearn.pkl"),
            positive_label="dga",  # explicit for MOJO; sklearn auto-detects inside utils
        )
        st.metric("DGA Probability", f"{proba:.2%}")

        verdict = "🚨 Likely DGA" if proba >= threshold else "✅ Likely Legit"
        st.subheader(f"Classification: {verdict}")

        # 3) Background features for SHAP
        if bg_csv and Path(bg_csv).exists():
            bg = pd.read_csv(bg_csv)
        else:
            # fallback: replicate the single row to build a small background
            bg = pd.DataFrame(np.tile(x.values, (50, 1)), columns=x.columns)

        # 4) Batch‑safe SHAP wrapper
        def f_predict(Xin: np.ndarray) -> np.ndarray:
            df = pd.DataFrame(Xin, columns=x.columns)
            probs = [
                predict_proba_auto(
                    df.iloc[[i]],
                    mojo_zip=Path(model_zip),
                    skl_pkl=Path("model/DGA_Leader_sklearn.pkl"),
                    positive_label="dga",
                )
                for i in range(len(df))
            ]
            return np.array(probs, dtype=float)

        explainer = shap.KernelExplainer(f_predict, bg)
        shap_vals = explainer.shap_values(x.values, nsamples="auto")

        # 5) Top signals (numeric list we can chart)
        top = summarize_shap(x.iloc[0], shap_vals, top_k=top_k)
        st.subheader("🔎 Top SHAP Feature Explanations")
        # Build a small DF for a Streamlit bar chart (no matplotlib dependency)
        top_df = (
            pd.DataFrame(top)
            .assign(abs_imp=lambda d: d["shap"].abs())
            .sort_values("abs_imp", ascending=False)
            .drop(columns=["abs_imp"])
            .set_index("feature")[["shap"]]
        )
        st.bar_chart(top_df)

        # 6) Prescriptive IR playbook
        st.subheader("Incident Response Playbook")
        playbook = gen_playbook(domain, proba, top)
        st.markdown(playbook)

    except Exception as e:
        st.error(f"Error: {e}")