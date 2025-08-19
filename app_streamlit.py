import streamlit as st
import pandas as pd, numpy as np
from pathlib import Path
from utils_dga import load_feature_schema, featurize_single, predict_proba_mojo, summarize_shap, gen_playbook
import shap
st.set_page_config(page_title="Prescriptive DGA Detector", layout="centered")
st.title("🔎 Prescriptive DGA Detector")
st.caption("AutoML (H2O) ➜ SHAP ➜ Prescriptive IR Playbook (Gemini or local)")
with st.sidebar:
    st.header("Model Artifacts")
    model_zip = st.text_input("MOJO zip", "model/DGA_Leader.zip")
    feat_schema = st.text_input("Feature schema JSON", "model/feature_schema.json")
    bg_csv = st.text_input("Background CSV (for SHAP)", "model/background_features.csv")
    top_k = st.slider("Top SHAP signals", 1, 6, 3)
domain = st.text_input("Domain", "google.com")
run = st.button("Analyze")
if run:
    try:
        feature_order = load_feature_schema(Path(feat_schema))
        x = featurize_single(domain, feature_order)
        proba = predict_proba_mojo(Path(model_zip), x, positive_label="DGA")
        st.metric("DGA Probability", f"{proba:.2%}")
        if bg_csv and Path(bg_csv).exists():
            bg = pd.read_csv(bg_csv)
        else:
            bg = pd.DataFrame(np.tile(x.values, (50, 1)), columns=x.columns)
        def f_predict(Xin):
            df = pd.DataFrame(Xin, columns=x.columns)
            p = predict_proba_mojo(Path(model_zip), df, positive_label="DGA")
            return np.array([p])
        explainer = shap.KernelExplainer(f_predict, bg)
        shap_vals = explainer.shap_values(x.values, nsamples="auto")
        top = summarize_shap(x.iloc[0], shap_vals, top_k=top_k)
        st.subheader("Top Signals (SHAP)")
        for s in top:
            st.write(f"• **{s['feature']}** — value={s['value']:.4f}, impact={s['shap']:+.4f}")
        st.subheader("Incident Response Playbook")
        playbook = gen_playbook(domain, proba, top)
        st.markdown(playbook)
    except Exception as e:
        st.error(f"Error: {e}")
