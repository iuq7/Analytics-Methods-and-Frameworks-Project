# NovaBank — Predictive Retention at Scale

Predictive framework for identifying customers at risk of disengagement using the UCI Bank Marketing dataset (id=222).

## Approach

Engagement Analysis scenario: `y == "no"` (rejected term deposit offer) used as a proxy for pre-churn disengagement risk. Models score risk, RFM segmentation drives action selection.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Or programmatically:

```bash
python -m src.data       # fetch + split
python -m src.features   # build feature matrix
python -m src.models     # train baselines + RF
python -m src.evaluate   # metrics + thresholds
```

## Layout

- `src/` — reusable pipeline modules
- `notebooks/` — exploration and reporting
- `data/` — raw cache + processed parquet
- `reports/` — memo, figures, results tables

## Deliverables

- `reports/exec_memo.pdf` — 1-page executive memo
- `reports/slides_link.txt` — link to 8-10 slide appendix
