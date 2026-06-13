# Momentum-Following Plan v1 — 3-Agent Review (2026-05-20) — KILLED PRE-COMPUTE

Verdicts: Methodology **FUNDAMENTALLY-FLAWED** · Profitability
**NEEDS-REFOCUS** · Red-team **DO-NOT-PROCEED + TERMINATE LOOP**.

The red-team did not argue — they **measured** the plan's own pre-registered
M0 gates on the panel and the plan FAILS them ~10× before any compute:

| pre-registered M0 BLOCKING gate | required | measured | result |
|---|---|---|---|
| Distinctness `|corr(r24, return_1d)|` ≤ 0.30 | ≤0.30 | **0.9901** | FAIL by 3× |
| Distinctness `|corr(r24, ema_slope_20_1h)|` ≤ 0.30 | ≤0.30 | **0.7584** | FAIL |
| Persistence τ(r24) at 24h ≥ 0.40 | ≥0.40 | **+0.043** | FAIL by 10× |
| Persistence τ(r24) at 48h ≥ 0.25 | ≥0.25 | **+0.087** | FAIL |
| Sign-survival P(sign holds 24h) | implied high | 0.31 | refuted |
| Sign-survival P(sign holds 72h) | implied high | 0.02 | refuted |

`r24` ≡ `return_1d` (which IS in WINNER_21 / inside the closed B★/R3c ceiling).
Persistence at the relevant lags is essentially 0 (daily non-overlapping
−0.035). R2b already monotone-decayed on a SUPERSET signal at the exact hold
grid (48h Δ −1.02, 72h Δ −1.33). The "cost-regime" relabel was empirically
refuted on the panel before any test ran.

## Decision
By the plan's own pre-registered M0 rules (no goalpost-moving), this iteration
is a **closed honest negative**. M1 is forbidden by the data. The autonomous
loop has now failed FOUR consecutive in-scope iterations (sell-convexity,
funding-carry, lifecycle-probe data-driven re-validation, momentum-following)
with three independent unanimous 3-agent reviews each time converging on the
same terminal state. Continuing to spin in-scope plans IS the re-derivation
theater the discipline forbids — and the user explicitly forbade ("don't
re-derive closed negatives"). The honest, aligned action is to seal the
loop's terminal state with the now-measured evidence and put the scope
decision to the user. Agent ids: meth `ae5fa71111e2663c5`, prof
`a3f2da51273f47978`, red `a7abac731e5db0797`.
