# Shock Index | Financial Risk Analytics

A static, responsive research portfolio with interactive jump detection. Open [the dashboard](index.html), upload a `date,close` CSV, and vary the rolling window, threshold, and detection mode. Processing happens in the browser; GitHub Pages needs no API key or server.

## Research story

1. The Korea Maritime and Ocean University team presented Shock Index based jump risk analysis at the Korean Mathematical Society. The poster selected **window 20 / threshold 3.70** over synthetic scenarios and reported F1 **0.783** for simulated jumps of absolute size ≥ 0.03, and **0.759** for ≥ 0.05.
2. An experimental Streamlit stock visualization was explored separately after the presentation, including optional percentage-move rules and cross-stock relationships.
3. This project reorganizes the preprocessing and detection logic into a transparent, deployable website. The website does **not** fit the full SDI–SVJ model or reproduce the poster's empirical figures without the original market data.

## Data and calculation

- Input: daily positive `close`, ISO `date`, one asset per file. Sort dates, remove invalid observations and duplicate dates (keep last).
- `r_t = ln(S_t / S_(t-1))`.
- `sigma_(t-1) = sample_std(r_(t-w), ..., r_(t-1))` with `ddof=1`; **shift by one day** prevents current return from influencing its own baseline.
- `q_t = abs(r_t) / sigma_(t-1)`. Missing or zero volatility produces no detection.
- Core detector: `q_t >= threshold`; positive returns are up, negative returns down.
- Experimental hybrid mode adds `abs(simple daily return) >= 4%` to the core rule. It should not be interpreted with the poster's validation metrics.
- The proposed model uses `lambda_t = lambda_0 exp(gamma_q q_t)` within a stochastic volatility jump framework. **This dashboard only computes q and threshold events; it does not estimate lambda_0, gamma_q, latent variance, jump probabilities, or forecasting accuracy.**

The default series is **synthetic**, generated reproducibly in JavaScript with seed 1219. It is not KOSPI or KRW/USD market data. The user supplied PDFs mention market applications, but their underlying daily CSVs were not attached. We therefore do not publish invented market curves, event counts, or live quotes.

## Run

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

Upload a CSV such as:

```csv
date,close
2023-01-02,100.25
2023-01-03,101.10
```

Provide more than 65 valid rows for the browser demo. For a reproducible Python preprocessing check and JSON export:

```bash
python3 -m pip install -r requirements.txt
python3 src/prepare_data.py my_prices.csv data/my_prices.json --name KOSPI
```

Only use data you are licensed to redistribute. Do not commit private or vendor restricted CSV files.

## Deploy to GitHub Pages

1. Create a repository, upload the contents of this folder, and commit.
2. In **Settings → Pages → Build and deployment**, choose **Deploy from a branch**, then `main` and `/ (root)`.
3. GitHub will provide a `https://USERNAME.github.io/REPOSITORY/` address. Add that URL to the repository description and your application portfolio.

No GitHub repository or public deployment is created by this package.

## Project map

- `index.html`, `assets/`: responsive dashboard and browser analysis.
- `src/shock_index.py`: matching Python calculations for local market CSVs.
- `src/prepare_data.py`: CSV preprocessing and JSON export.
- `docs/original-simulation.py`, `docs/original-streamlit-app.py`: supplied experimental source snapshots, preserved for provenance, not run by the static page. The original Streamlit source fetches Yahoo Finance prices; its ticker list and mixed detection rules are exploratory.
- `docs/research-notes.md`: source distinctions and limitations.

## Application summary

> 금융시장 급변동을 정량화하기 위해 과거 변동성 대비 당일 로그수익률의 크기를 나타내는 Shock Index를 분석했습니다. 대한수학회에서 팀 연구 결과를 발표한 뒤, 별도로 AI를 활용해 Streamlit 시각화 시안을 가상 구현하고 수정했습니다. 이후 전처리·수식·탐지 기준을 확인할 수 있는 웹 포트폴리오로 정리했습니다. 웹 대시보드의 예시 그래프는 합성 데이터이며 실제 시장 성과로 제시하지 않습니다.

## References and credits

- Supplied team poster: *Shock index-based analysis of asset-specific jump risk: An empirical assessment of the SDI-SVJ model* (Junghyo Myeong, Minseo Shin, Geongjin Jung, Minsoo Kim).
- Supplied `explanation(5).pdf`: multi-scenario synthetic validation and parameter selection.
- Supplied `Model_Eng` and `Model_Kor`: proposed model specification.
- Supplied `app.py` and `급락+급등 (1).ipynb`: later visualization and simulation experiments.

The uploaded journal PDFs informed the background but are not redistributed here. A rule-based threshold exceedance is an operational flag, not proof of a discontinuous stochastic jump or a trading signal.
