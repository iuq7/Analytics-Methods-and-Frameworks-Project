---
title: "NovaBank — Predictive Retention at Scale"
date: \today
geometry: margin=0.55in
fontsize: 9.5pt
---

**To:** Executive Committee   **From:** Retention Analytics   **Date:** \today
**Re:** Proactive churn-risk targeting — recommendation and 90-day pilot

---

**Recommendation.** Stand up a monthly customer risk-scoring program. Each cycle, route the **top 20% highest-risk, highest-value customers** to a tiered retention playbook owned by Retention Ops. Launch as a **90-day pilot** with a 1,000-customer control group before scaling bank-wide.

**Why now.** Margins are tight and today's retention motion is reactive — we engage after customers leave. A predictive program lets the front line intervene weeks earlier, with offers sized to customer value instead of blanket discounts.

**Expected impact.** A monthly cycle delivers **~€443K in retained customer value** on a 13,564-customer holdout (€500 LTV, €30 offer, 40% save rate) — **~€1.5M annual upside** at full scale before pilot refinements. Spend concentrates where it matters: **802** high-value customers flagged for 24-hour manager call, **1,030** for structured win-back offer, **149** loyal customers for advisor outreach.

**Confidence.** Top-of-list customers are identified with **~97% accuracy** at the 20% budget cap, approximately **10% better than random targeting**. Probabilities are calibrated, so spend can be planned against expected save rates rather than ranks alone. Per-customer accuracy is consistent across age, marital status, and job category.

**Trade-offs.** ~3% of flagged customers would have stayed without intervention — planned cost, well below save value. ~78% of disengaged customers fall outside the 20% cap; that's a budget choice, not a model limit (raising to 30% captures more with diminishing returns). The dataset lacks an account-closure timestamp, so disengagement is proxied by offer rejection — pilot validates this against true 90-day retention.

**Top risks (ranked).**

1. **Proxy validation.** Offer rejection ≠ confirmed churn. Day-60 gate must compare treatment vs. control on true 90-day active rate before any scale-out.
2. **Fairness — coverage spread.** Per-customer accuracy is uniform, but selection rates differ by segment (e.g., 21% for ages 31–45 vs. <1% for 61+). Add compliance review at day 30 and a minimum-allocation floor by protected group during the pilot.
3. **Operational readiness.** ~800 priority calls per cycle requires confirmed relationship-manager capacity and a 24-hour SLA.

**Pilot (90 days).** 5,000 treatment, 1,000 random control. KPIs: 90-day active rate, redemption rate, retention spend per save. Weekly Ops/Compliance readout. Day-60 go/no-go gate. Stop conditions: precision drift >5pp, redemption <8%, or fairness flag.

**Asks.** (1) Approval to begin a 2-week ramp. (2) Relationship Management staffing confirmation for ~800 priority calls/month. (3) Compliance partner assigned for fairness review.

---
*Methods, model evidence, scenario sensitivity, and AI-usage disclosure: slide appendix — `reports/slides_link.txt`.*
