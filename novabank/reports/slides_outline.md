# Slide Deck Outline (8–10 slides)

Render in Google Slides / Keynote. Drop final link in `reports/slides_link.txt`.

## Slide 1 — Problem & Decision
- Context: NovaBank margins under pressure, churn = profit erosion.
- Decision: who gets retention spend next 90 days.
- Reframe: predict offer rejection (engagement-failure proxy) — full churn label not in data, validated in pilot.
- Success measures: PR-AUC (rare class) + Precision@top-20%.

## Slide 2 — Data & Target Framing
- UCI Bank Marketing (id=222), 45,211 customers, 17 features.
- Target: `engaged = (y == yes)`, baseline 11.7%. `risk_score = 1 − p_engaged`.
- Dropped `duration` (post-call leakage).
- 70/30 stratified split, seed 42.
- 7 synthetic features documented (overdraft, products, velocity, etc.) — assumptions in data dictionary.

## Slide 3 — Baseline Models
- Logistic Regression: PR-AUC 0.418, ROC 0.773, Brier 0.185.
- RFM 5×5×5 → 6 segments collapsed (Champions / Loyal / At-Risk HV / New Potential / Mid / Hibernating).
- Risk drivers (LogReg coefficients): poutcome_success ↑ engaged, contact_unknown ↑ risk, balance ↑ engaged.
- Plain-language baseline: rejected-offer-before → still rejecting now.

## Slide 4 — Improved Model + Diagnostics
- Random Forest (400 trees, depth 12, balanced class weight) + isotonic calibration.
- PR-AUC 0.44 (+0.022), Brier 0.083 (calibration win, −0.075 vs raw RF).
- Show: PR curve, calibration curve, feature-importance bar chart.
- Top features: poutcome_success, contact_unknown, age, transaction_velocity, recency.

## Slide 5 — Explainability + Decision Rules
- Surrogate 4-depth tree extracts top rules:
  - `poutcome_success == 1` → low risk
  - `pdays > 200 AND balance < median` → high risk
  - `contact == unknown AND age > 45` → mid-high risk
- SHAP top-5 features (per-customer waterfall in appendix).
- Plain-English summary for non-tech stakeholders.

## Slide 6 — Action Matrix (RFM × Risk Tier)
- 6 segments × 3 risk tiers = 18 actions.
- Headline cells:
  - At-Risk High Value × High → TOP PRIORITY 24h manager call (802 customers in test).
  - At-Risk HV × Mid → Win-back offer (1,030).
  - Hibernating × any → low-cost reactivation or ignore.
- Volumes table + €-impact estimate per cell.

## Slide 7 — Scenario Sensitivity
- Retention rate ±15pp → EV swings 246k ↔ 640k.
- Offer cost +€30 → EV drops 443k → 362k (still positive).
- LTV −€200 → EV drops to 233k.
- Drop top feature (poutcome_success) → PR-AUC drops 0.44 → 0.39, model still ranks.

## Slide 8 — Fairness Audit
- Per-group precision: 95–98% across age, marital, job. **No precision gap.**
- **Coverage gaps (flag to compliance):**
  - Age 61+ selection rate 0.3% (base rate 55% — model correct, but offers under-allocated → revenue risk).
  - Blue-collar 32% vs Management 14%.
- Action: pilot adds quota floor per protected group; compliance review at day 30.

## Slide 9 — Pilot Plan
- Scope: top 20% risk × full Action Matrix routing.
- Treatment: 5,000 customers. Holdout: 1,000 random control.
- KPIs: 90-day retention rate, redemption %, retention cost per save.
- Timeline: 2w setup, 8w run, 2w analysis. Gate at day 60.
- Stop conditions: precision drift > 5pp, fairness flag, redemption < 8%.

## Slide 10 — AI Usage Disclosure
- Claude Code (Opus 4.7) used for: pipeline scaffolding, feature engineering, model selection rationale, memo drafting.
- All model code, metrics, and decisions reviewed by analyst (you). No autonomous prod deploys.
- Datasets, hyperparameters, seeds documented for reproducibility.

---

Render commands:
- Memo PDF: `pandoc reports/exec_memo.md -o reports/exec_memo.pdf`
- Combined: keep memo PDF separate; slide deck hosted (Google Slides), link in `reports/slides_link.txt`.
