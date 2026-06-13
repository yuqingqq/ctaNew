# Review — iter-019 (fix-round 1)

**Candidate:** Transaction-cost-aware NO-TRADE BAND on held-book net weights
(ALPHA track, cost-efficiency layer; alpha champion unchanged).
**Script:** `research/convexity_portable_2026-05-20/scripts/X130_notrade_band.py`
**Verdict:** **PASS** (correctness/PIT clean — code is trustworthy for Evaluation;
expected economic verdict is REJECT, and the evidence supporting that REJECT is sound).

Engine reuse is the validated X123 `build_universe` (X117/X123/X124 lineage); only the
band execution layer is new, so the audit focuses there.

---

## 1. Look-ahead / PIT — PASS

**Band decision is fully PIT (X130:111-131).** At cycle `t`, `heldbook_band` reads:
- `target = nets[t]` — the 6-sleeve average built in `net_targets` (X130:84-91) from
  `cyc_w[max(0,t-HOLD+1):t+1]`, i.e. only sleeves entered ≤ t. The underlying
  `pred`/`mom30`/`beta`/`b30` are all trailing or `.shift(1)` inside `build_universe`
  (X123:111 `mom30.shift(1)`, X123:113 `beta.shift(1)`, X123:103/119 trailing rolling). No
  current/future bar in the target.
- `held` = weights EXECUTED through `t−1` (realized, lagged). `held` is advanced
  (`held = exe`, X130:131) ONLY after `exe`/`pnl[t]` for cycle `t` are computed. No
  forward sleeve return or future bar enters the execution decision.

**Held-weight carry is correct (X130:115-123).** When a trade is suppressed
(`|target−held| < band`), `exe[s] = pv` (the prior executed weight) → the position
persists. PnL at line 125 uses `exe` (the carried book), and cost (line 129) accrues
only the executed `turn`. The held book correctly carries into the next cycle's
`target−held` comparison. Verified the carry semantics against the spec block in
research/handoff.md (lines 23-34) — exact match.

**Turnover/cost on EXECUTED changes only (X130:118-129).** `turn += |tg−pv|` accrues
only inside the `>= band` branch; held symbols add zero turnover. `pnl = g − turn*0.5*cost`.
Monotone turnover decrease with δ (handoff 800→341) is the expected signature. Correct.

**δ is the only parameter (G3).** `BAND` is the single knob (X130:57-59). No other tuned
quantity. Flagged for nested-OOS. ✓

---

## 2. Correctness

**Base reproduction (δ=0) reproduces X117 — VERIFIED EQUIVALENT.**
- `net_targets` slice `cyc_w[max(0,t-HOLD+1):t+1]` and weight `wt/HOLD` are byte-identical
  to X123 `heldbook` (X123:220-223). Slice indices checked t=0..7 — identical.
- With `band=0.0`, `abs(tg−pv) >= 0.0` is **always True** (incl. `0>=0`), so `exe == target`
  every cycle ⇒ executed book == X123 `net` book.
- `turn += |tg−pv|` over `set(target)|set(held)` == X123 `turn = Σ|net−prev|` (X123:225).
- Gross `g = Σ exe[s]·rl[s]` (finite-guarded, X130:125-127) == X123 `c` (X123:227-228).
  X130 prunes `exe` (|w|>1e-9) before the sum; X123 sums over unpruned `net`. A pruned
  near-zero symbol contributes ~0 to PnL either way, and `prev.get(s,0)`/`held.get(s,0)`
  return ~0 either way next cycle → turnover identical to <1e-9. No material divergence.
- ⇒ δ=0 == X117 (handoff: HL70 @4.5bps +1.93 / −5674 / Calmar +1.68 / +10472). The
  reproduction claim is structurally correct; Eval can rely on it.

**Gross PnL emitted & correct (X130:108-133).** `gross[t]` = pre-cost executed-book return.
Cost-only band ⇒ executed≈target ⇒ gross≈base; bet-changing band ⇒ executed holds stale book
⇒ gross drifts. Diagnostic logic is sound and correctly distinguishes δ≤0.02 (gross flat,
+12272→+12441) from δ=0.05 (gross jump +13605). See §3.

**δ-sweep (X130:286-305).** Grid {0,0.01,0.02,0.03,0.05,0.08} (trap probes 0.05/0.08 added;
spec 0.005 dropped as ≈0.01 — disclosed deviation, acceptable). Per universe × cost.
folds_positive computed per-fold on Calmar. Correct.

**G4 matched random-skip placebo (X130:138-214).** `count_band_skips` replays the BANDED
trajectory and counts sub-δ skips (correct skip count for the real band). Placebo skips that
many trades AT RANDOM from the BAND=0 candidate set; 200 seeds (≥100); RNG seeded. The
candidate set is built on the *unbanded* trajectory while the real band's skips are counted on
the *banded* trajectory — the two trajectories differ slightly so the populations aren't
identical, but this is the standard matched-COUNT placebo the spec asks for ("skip the same
NUMBER of trades at random"); it is defensible and does not invalidate the rank. Rank `<p95`
on all four rows ⇒ FAIL, correctly reported.

**G6 paired CI (X130:218-231).** Block-bootstrap by fold of per-cycle (banded−base) diff,
2000 boots, CI on mean diff. `clears = (lo>0) or (hi<0)`. Correct. CI crosses 0 on HL70/EXT
⇒ FAIL, correctly reported.

**G3 nested-OOS (X130:379-420) — PIT-CORRECT, the decisive evidence is trustworthy.**
- δ chosen on `past = folds[:i]` (strictly earlier folds, `folds` sorted chronologically from
  the upstream walk-forward `fold` column) by max PAST Calmar; applied to `fut = folds[i]`.
  **No future info enters the δ choice.** ✓ — this is the load-bearing claim and it holds.
- Trap δ=0.05 kept IN the candidate menu (`BAND_OOS_GRID`, X130:59) so the test must reject it
  — correct per spec.
- Forward result: HL70 lift +0.24 (marginal), EXT lift −0.03 (1/7 folds) ⇒ does not transport.
  Honest REJECT.
- *Minor realism note (non-blocking):* `pnl_by_band[b]` is precomputed on the full series and
  sliced by fold (X130:384, 396-397). The held book is path-dependent, so a forward fold's PnL
  reflects a book warmed under band `b` since cycle 0 (not a clean per-fold reset when the
  forward-chosen δ differs between folds). This slightly flatters consistency but introduces NO
  look-ahead (δ still past-only; no future returns used) and cannot manufacture a false
  positive — the verdict is REJECT regardless. Acceptable.

**NaN guards.** `metrics` drops non-finite (X130:65); gross sum finite-guard + c=0 fallback
(X130:125-128); weights <1e-9 pruned (X130:123). No silent NaN→0 hiding missing data. ✓

**Annualization / measurement.** `ANN=6*365`, Sharpe `mean/sd·√ANN`, Calmar `mean·ANN/|maxDD|`,
maxDD on cumsum equity — all match conventions.md / evaluation_contract.md. ✓

**RNG.** Seeded `default_rng(12345)`; deterministic. ✓

---

## 3. Gross-PnL distinguishes cost-only vs bet-changing — CONFIRMED CORRECT

The diagnostic mechanism is correctly implemented and the handoff's reading is right:
- δ≤0.02: gross PnL ≈ flat (+12272 → +12441), turnover ↓ slightly (800→790) ⇒ genuinely
  cost-only (trades less of the SAME bet). Honest, tiny saving.
- δ=0.05: Sharpe/Calmar look spectacular (+2.31/+2.48) BUT gross PnL JUMPS (+12272 → +13605)
  ⇒ the executed book has DRIFTED from target — it is no longer cost-only, it changes the bet
  (holds a stale rank-boundary book that wins in-sample). Non-reproduction on EXT/S44 confirms
  single-universe flavour. The gross signal is the correct tell.

---

## Minor (non-blocking) findings
- **X130:160** is a dead no-op (`... if False else cand`) leaving `cand` unchanged;
  `base_held_seq` (X130:148-150) is also unused. Harmless cruft — recommend deleting for
  clarity, not required.

---

## Verdict
**PASS to Evaluation.** Band logic is fully PIT; held-weight carry, turnover/cost on executed
changes, base=δ0 X117 reproduction, gross-PnL diagnostic, G4 matched placebo, G6 paired CI,
and the **decisive G3 nested-OOS (δ chosen from past folds only)** are all correctly
implemented. The evidence supporting the expected REJECT is trustworthy — in particular the
nested-OOS does not peek forward, and δ=0 provably equals X117.
