# CP-26 GFS+6 — Owner disposition before CP-28

**Decision date:** 2026-10-19. **Authority:** the Owner accepted the recommendations in this
conversation and explicitly instructed the necessary protocol amendment under a scoped
Lockdown suspension; the verbatim grant is in the r16→r17 amendment sheet.

## Evidence and decision

CP-26 is complete at scientific candidate `6f57febe3659b1f902c17f3db31ba41e41136178`, with
the fresh independent Integration verdict at
`evidence/cp-26/cold-rerun-2026-10-15/integration-final-r16-2-PASS.md` and evidence tip
`252725e0431826b6d32f1fc8200cfa3baf3750f6`. The current rendered report is
`reports/rebaseline/cold-rerun-2026-10-15/presentation-v21-r16/report.md`.

On all 2,068 paired eligible dates, GFS+6 worsens nominal v5 MAE by about 14% (registered
95% ratio interval +10% to +16%) and WIS by about 15% (+12% to +18%). Both exceed §24.5's
2% fragility threshold. Stored point scores for v4 likewise worsen by about 16.1% MAE and
17.4% WIS. These v4/v5 point differences alone do not establish a statistically resolved
difference in robustness between the models.

**Owner ruling:** accept the result as a **material, disclosed operational risk for research
continuation** under §24.5. Its status remains **fragile**. Keep nominal v5, including its
load forecast, as CP-28's control. Retain the registered missing-weather handling and all
prior data-vintage, selection and timing-risk qualifications. This acceptance is not an
assertion of operational robustness, publication clearance or live-deployment approval.

Rationale: the stress result identifies a real weakness worth understanding but does not
invalidate the completed baseline. Switching to v4 does not remove the observed weather
dependence. CP-28 asks a previously registered recombination question, which can proceed
in research with the weakness disclosed rather than choosing new arms from these results.

## Bounded follow-up and remaining authority

The Owner approves the separate post-results descriptive supplement
`evidence/cp-26/cp28-gfs-postresults-addendum-v21-r17.md`, using existing artifacts only,
and its planning scenarios. No new training or scored policy is authorized. Report unsupported
questions honestly; no favorable finding is a continuation requirement. A simple causal
availability-based decision rule may be recommended, but not fitted, selected or deployed
under this approval. Real arrival-frequency measurement stays in CP-18's already planned trial.

CP-26 LAND remains Owner-authored by hand. CP-28 execution still needs the Owner's confirmation
**after** that LAND and its issued brief. This pre-LAND ruling does not substitute for that later
confirmation. No change to the seven arms, control, inputs, statistics or adoption thresholds is
authorized. No publication, remote write or fresh-data access follows from this ruling.
