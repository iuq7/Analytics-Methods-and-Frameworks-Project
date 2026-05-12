# Data Dictionary

## Source
UCI Bank Marketing dataset, id=222. 45,211 rows, 17 columns. Marketing campaign of a Portuguese bank.

## Native fields

| Field | Type | Description |
|-------|------|-------------|
| age | int | Customer age in years. |
| job | str | Job category (11 levels + unknown). |
| marital | str | Marital status: single, married, divorced. |
| education | str | Education level: primary, secondary, tertiary. |
| default | str | Has credit in default (yes/no). |
| balance | int | Avg yearly account balance, EUR. |
| housing | str | Has housing loan (yes/no). |
| loan | str | Has personal loan (yes/no). |
| contact | str | Contact channel: cellular, telephone. |
| day_of_week | int | Last contact day of month (1–31). |
| month | str | Last contact month (jan–dec). |
| duration | int | Last contact duration (s). **DROPPED** — leakage; known post-call. |
| campaign | int | Contacts performed during this campaign. |
| pdays | int | Days since last contacted in prior campaign (-1 = never). |
| previous | int | Number of contacts before this campaign. |
| poutcome | str | Outcome of previous campaign: success, failure, other, unknown. |
| y | str | Target. yes = subscribed to term deposit. |

## Engineered fields

| Field | Type | Definition |
|-------|------|-----------|
| engaged | binary | y == "yes". Modeled directly (rare class, baseline ~11.7%). |
| risk_score | float | 1 − p(engaged). Higher = higher disengagement / churn-proxy risk. |
| recency | int | pdays with -1 mapped to max + 1 (never-contacted = highest recency value). |
| frequency | int | previous + campaign (lifetime contacts). |
| monetary | int | balance clipped at 0. |
| r_score, f_score, m_score | int 1–5 | Quintile rank on R, F, M (5 = best). |
| rfm_score | int | r + f + m. |
| segment | str | RFM segment (Champions, Loyal, At-Risk High Value, New Potential, Hibernating, Mid-Value). |
| risk_tier | str | Low / Mid / High by global risk quantiles (0.5, 0.8). |

## Synthetic fields (Scenario 3 — document assumptions in memo)

| Field | Type | Generation rule |
|-------|------|-----------------|
| overdraft_count_6m | int | 4 × (balance < 0) + Poisson(0.3). Assumption: negative balances overdraw frequently. |
| product_count | int | 1 + housing + loan + default. Assumption: more credit products = more bank engagement. |
| digital_user | binary | P = 0.2 + 0.4·(contact==cellular) + 0.2·(age<40). |
| direct_deposit_active | binary | balance > median AND job ∈ {admin, technician, management, blue-collar, services}. |
| unresolved_complaint | binary | 15% of poutcome == "failure". |
| transaction_velocity | float | 1 − pdays/max(pdays), clipped, with Gaussian noise. |
| fee_ratio | float | 10 / |balance|, clipped. Assumption: low balances incur more fees. |

## Target framing

Doc proposes `y == "no"` → churn_risk = 1. We invert and model `engaged = (y == "yes")` (rare class) for cleaner metrics (PR-AUC meaningful), then define **risk_score = 1 − p_engaged**. Operationally equivalent, statistically cleaner.

## Splits

Stratified 70/30 on `engaged`. random_state=42. Train: 31,647. Test: 13,564.
