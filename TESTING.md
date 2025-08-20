## Manual verification

### Legit example
Input: google.com
Expected: Low DGA probability (≈0–5%), Classification: ✅ Likely Legit
Verify: SHAP shows low entropy/length impact.

### DGA-like example
Input: xj29qwe9z0asd.biz
Expected: High DGA probability (≈70–100%), Classification: 🚨 Likely DGA
Verify: SHAP top signals include high entropy, high digit ratio, longer length.

### Mixed
Input: microsoft.com -> low
Input: stanford.edu -> low
Input: cnn.com -> moderate (dataset/lexical effects)