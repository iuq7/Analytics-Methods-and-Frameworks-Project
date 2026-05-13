# Analytics-Methods-and-Frameworks-Project

NovaBank — Predictive Retention at Scale. Final project for the Analytics Methods and Frameworks course.

Predictive framework that scores customer disengagement risk on the UCI Bank Marketing dataset and routes each customer to a tiered retention action via an RFM × Risk Action Matrix.

## Quick start

```bash
cd novabank
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
```

Outputs:
- `novabank/reports/exec_memo.pdf` — one-page executive memo
- `novabank/reports/slides_outline.md` — 10-slide deck spec
- `novabank/reports/figures/` — PR curve, calibration, feature importance
- `novabank/reports/*.csv` — metrics, Action Matrix volumes, sensitivity, fairness audits

## Layout

```
novabank/
├── src/             # pipeline modules (data, features, models, evaluate, decision, explain)
├── notebooks/       # exploration (placeholder)
├── data/            # raw cache + processed splits (gitignored)
├── reports/         # memo, slides outline, metrics, figures
├── requirements.txt
└── run_all.sh       # end-to-end build
```

## Approach

- Target: `engaged = (y == "yes")`. Risk = 1 − p_engaged.
- Features: native fields + RFM proxies + 7 documented synthetic signals.
- Models: Logistic Regression baseline + Random Forest with isotonic calibration.
- Decision: top-20% risk × RFM segment → Action Matrix (18 cells).
- Audits: per-group fairness (age, marital, job).



We are testing to push code on remote repositoy.

See `novabank/reports/data_dictionary.md` and `novabank/reports/exec_memo.md` for full detail.
