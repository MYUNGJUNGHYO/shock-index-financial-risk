# Source reconciliation

| Source | Scope | Settings / outcomes |
|---|---|---|
| Korean Mathematical Society poster and `explanation(5).pdf` | Synthetic scenario selection and initial market interpretation | Rolling 20, threshold 3.70; F1 0.783 for absolute synthetic jump ≥ .03 and 0.759 for ≥ .05. |
| `Model_Eng`, `Model_Kor` | Proposed stochastic volatility jump process | `lambda_t = lambda_0 exp(gamma_q q_t)`; a model specification, not fitted parameters for this website. |
| Supplied `app.py` | Later exploratory Streamlit stock visualization | Default rolling 20 and SDI minimum 2.5, optional 4/6/8 percent visualization levels; not the poster's validated detector. |
| Supplied notebook | Mixed upward/downward jump simulation | Separate experiment and seed; do not combine its metrics with poster results. |

The site defaults to the poster's detection setting and exposes a labeled hybrid variant inspired by the later app. It does not call either an exchange API or Yahoo Finance. Market periods, event counts, precision, and recall require an actual licensed series and, for performance metrics, independently verified jump labels. Uploaded CSV rows stay in browser memory and disappear on reload.
