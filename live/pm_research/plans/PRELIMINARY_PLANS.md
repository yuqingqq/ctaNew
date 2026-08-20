# Preliminary module plans — P-2026-003

Orchestrator's baseline, written 2026-08-20 while the detailed module planners
run. **Preliminary**: each section is the skeleton a dedicated planner will
refine or refute. Where a planner disagrees with evidence, the planner wins.

## What the evidence已 settled (don't re-litigate)

| finding | status |
|---|---|
| Settlement = `Up iff S60(T) ≥ S60(t0)`, w=60s | **verified, 99.8%** on 1,465 windows |
| Full-300s-TWAP reading | refuted (86.9%) |
| Our forecast vs the book | book wins at **every** horizon, through 3 σ treatments |
| Book miscalibration (FLB) | **measured**: −6.6¢ @0.1–0.2, −7.6¢ @0.3–0.4, +3.4¢ @0.6–0.7 (drift-adj) |
| Sample up-drift | +4.5pp, all coins, both days — a 2-day rally |
| Fees | maker 0 + ~70bps rebate/fill; taker ~3.5% ATM |
| Power | ~1,641 resolved windows, 2 days, **one** walk-forward test day |

**The consequence that shapes every plan below:** we hold no forecasting edge,
so this is a spread-capture strategy. The candidate edge is the book's own
miscalibration, and the economics live in fill quality, not prediction.

---

## 1. DA-State — knowledge-time layer  *(build first)*

**Purpose.** Make every downstream number honest. Nothing else can be trusted
until reads are knowledge-truncated.

**Design.** `Known[V]{value, t_event, t_known, t_known_prov, t_known_err, …}`;
`view(now)` the only entry point; no event-time filter API exists.
Composition `t_known = MAX(inputs)`.

**The trap.** `t_known` is `recv_ns` — an observation that does NOT exist on
historical/REST/on-chain data. The natural fallback `t_known := t_event` makes
the invariant hold by construction and launders the 1.7s peek invisibly.
⇒ imputed knowledge times must be MARKED (`IMPUTED`/`ASSUMED` + `t_known_err`),
and replay must refuse when `r` falls inside the error.

**Validation.** Assert `t_known ≤ now` on every read in replay; a deliberate
event-time read must change results measurably (if it doesn't, the guard is
inert).

---

## 2. EV-Markout — the measurement that decides everything

**Purpose.** Every gate number. Methodology is mature and inherited; the
harness is not built.

**Estimands.** `es = q(p−m)/m`; `Λ(τ) = q(m_{t+τ}−m)/m`; `rs(τ) = es − Λ(τ)`.

**Non-negotiables** (each learned the hard way):
- parent-event collapse on `(timestamp, side)` before any statistic
- **notional-weighted, never equal-weighted** — equal-weighting flipped
  +2.44bps to −0.32bps on the sibling programme and reversed a conclusion
- day-clustered inference + block bootstrap; window-level independence is false
- dedup trades on `transaction_hash` (double-reported across the token pair)
- direction is `(side × asset_id)`; `side` alone is not aggressor direction
- population fill = the AVERAGE maker's ⇒ every number is an **upper bound**

**Venue specifics.** Book is 2–4 ticks wide on a $0.01 grid (mid is a coarse
summary); the book is unified across the token pair.

---

## 3. EV-Calibration — scoring the one measured edge

**Design.** Paired (model and book on the same windows), Brier primary,
log-loss secondary under a frozen clip, stratified by time-in-window,
day-clustered.

**The drift separation.** FLB is a **slope** (over-price low, under-price high);
drift is a **hump** via `φ(d)` (largest ATM, ~0 at extremes). A naive
recalibration bakes the rally in. Must be separated explicitly.

---

## 4. BE-Belief — what we actually quote around

**The decision:** own forecast / book as-is / **book recalibrated** / blend.
Evidence points at recalibration: our forecast loses, the book's bias is
measured and monotone.

**Self-defeat check.** A belief that tracks the book cannot profit from
disagreeing with it — so the recalibration IS the edge, or there is none.

**Preliminary design.** Monotone map on book price, fitted **walk-forward**,
per symbol where powered, conditioned on time-in-window if the bias varies
with `r` (it plausibly does — the book's Brier collapses to 0.017 at r=30s).
Open: mid vs microprice as input; isotonic vs parametric FLB curve.

---

## 5. BE-Uncertainty (σ) — **next focus**

**Purpose is the open question.** If BE-Belief takes the book's level, σ is no
longer needed for the LEVEL and is needed only for DYNAMICS: inventory risk
`q²p(1−p)`, the participation frontier `φ(d)√(3L/r)`, `λ_bin` (how fast the
binary moves), sizing and stand-down. **Different purpose ⇒ different
estimator and different loss function.** Decide before building.

**Settled constraints.** Estimate the 60s-TWAP quantity DIRECTLY (no ÷40 round
trip through a point-price σ); rolling multi-window; per-symbol for **every**
fitted parameter; walk-forward; knowledge time; no annualisation; no
double-counted variance.

**The trap.** Rolling increment variance `Var[X_{t+r} − X_t]` ≠ conditional
settlement variance `Var_t[X_T]` — the trailing window rolls off, overstating
by ~4.1× at r=10s. State which object is estimated and why.

**Preliminary lean.** Empirical settlement innovations `X_T − X_t` from
completed windows (model-free, targets the right object) for the SHAPE, with a
fast regime scaler for the LEVEL — but only if the diagnostics show the
empirical curve is stable enough at ~190 windows/symbol.

---

## 6. BE-FlowAndFills — the P&L

**Highest-value unknown: the SIGN of ζ.** Does resting into a sweep get paid
(transient impact, reverting) or run over (permanent, informed)? Determines
whether market making works here at all. Measurable on data already collected
via the response function `R(ℓ) = E[(p_{t+ℓ} − p_t)·ε_t]`.

**Fill model.** Queue position is never observable (no L3) ⇒ bracket
pessimistic/optimistic; a sign flip across the bracket is a failure, not an
average. Interface must be pair-aware (unified book) and take `size` (our
orders are tape-scale: min 5 vs median trade 9).

---

## Build sequence

1. **DA-State** — everything downstream depends on it
2. **EV-Markout + EV-Calibration** — measure before deciding
3. **ζ-sign test** (BE-FlowAndFills) — decides if MM is viable at all
4. **BE-Belief recalibration** — the candidate edge, walk-forward
5. **σ for dynamics** (BE-Uncertainty) — scoped by whatever (4) concludes
6. DE-Constraints / participation frontier — needs σ + measured latency

Steps 1–4 run on data already on disk. Nothing here needs the decision layer,
whose two open structural rows do not touch this path.
