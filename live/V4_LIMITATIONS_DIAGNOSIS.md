# v4 limitations — AUDITED-HONEST re-evaluation (2026-07-11, AUTHORITATIVE)

Replaces the 2026-07-10 version, which was written on **pre-audit leaked / over-masked data** — its `+2.30`
headline and its full-stack per-regime Sharpes are **retracted**. A 6-finding external audit
(`RESEARCH_LOOP_20260707.md` addendum 61) was verified → remediated (addenda 62-63) → **independently re-verified
by 3 adversarial subagents** (addendum 64). Every number here is on the audited-honest artifacts: gap-clean panel
(`gap_guard_panel.py`), honest gate (`.shift(1)`, cache regenerated), grid-safe 30d regime, `_honest` walk-forward
books. Generators: `honest_limitations.py`, `book_level_honest.py`.

## The measurement discipline (why the numbers moved)

The audit's decisive lesson (repo **pitfall #4**): the FULL-STACK replay Sharpe is **path-coupled** — the DD-stop +
regime overlays amplify tiny prediction differences ~10-20×, so it swings ±0.3 and is **unreliable** for comparing
near-identical books (proven: gap-clean vs old books have **0.996-correlated** preds, yet the full-stack Sharpe
swings +2.09↔+2.45). So performance is quoted at **BOOK LEVEL** (rank-IC + 1L/2S selection spread,
path-independent). The full-stack Sharpe is a fragile **range**, not a performance number.

## Status

- **Deployed pipeline is audited-honest:** gap-clean panel (real-kline-gap-keyed, guards features AND labels), gate
  `.shift(1)` (no same-day-volume look-ahead), regime 30d **wall-clock** (was row-based), per-symbol RidgeCV
  walk-forward with `exit_time` label-purge + 1-day embargo.
- **Verified honest 3 ways** (addendum 64): fixes real + active (claim=code, un-shifted paths inert); numbers
  reproduce with **NO train/test/preproc leak** (every feature |IC|≤0.064 < 0.10; model IC < best-feature IC);
  no remaining look-ahead. rank-IC persistent across **41/42 folds** = a genuine *weak* alpha.
- **NOT live-proven:** no forward trading on the honest models; the forward ledger is the standing gap.
- **Open audit items:** #3 PIT Hyperliquid listing history (needs external data); #6 artifact commit-tracking
  (panel/books/cache gitignored → correctly-measured but not fresh-clone reproducible).

## Performance (audited-honest)

**RELIABLE metric — book level (path-independent):**

| frame | rank-IC | 1L/2S selection-spread Daily Sharpe (pre-overlays) |
|---|---|---|
| **recent 2025-10+** | **+0.030** | +2.59 |
| **OOS 2023-25** | **+0.024** | +0.92 |

rank-IC positive in recent **9/9 months** + OOS **32/33 folds** (mild YoY decay, all positive) — a genuine but
*small* alpha, no leak.

**Full-stack (deployed, with overlays) — PATH-COUPLED, quote as a RANGE, not performance:** recent **~+2.1–2.5**
daily, OOS **~+0.2** daily. (The old `+2.30` was a *cycle* Sharpe carrying a same-day gate look-ahead — retracted.)

**Per-regime, book-level (replaces the old path-coupled full-stack table):**

| regime | RECENT sel-spread / rank-IC | OOS sel-spread / rank-IC |
|---|---|---|
| side | +4.60 / +0.029 | +1.02 / +0.022 |
| bear | +0.10 / +0.030 | +3.80 / +0.035 |
| bull (mild → GATED flat) | +3.76 / +0.050 | −1.31 / +0.025 |
| deep-bull (mom1d overlay) | −3.75 / +0.016 | +0.81 / +0.024 |

**Long vs short leg (path-independent):** recent LONG hit 45% / Sharpe +0.49, SHORT hit **58% / +3.15**; OOS LONG
hit 47% / **+0.81**, SHORT hit 56% / **+0.33**. The short edge is **RECENT-ONLY** (OOS short +0.33 < OOS long +0.81).

## The strategy in one sentence

A ~beta-neutral cross-sectional book that ranks alts by residual-vs-BTC alpha and shorts the over-extended, whose
**selection alpha (rank-IC ~+0.03) is small but era-STABLE**, wrapped in regime overlays whose **deployed Sharpe is
fragile / path-coupled**.

## The definitive limitations (audited-honest)

### #1 — Era-VARIABILITY of the deployed Sharpe; the alpha itself is era-STABLE. [CORRECTED — was overstated]
- The old doc's "side and bear have OPPOSITE era signs" was substantially a **path-coupled artifact**. Book-level,
  the rank-IC is +0.02–0.05 across **every** regime and **both** eras, and side (+4.60/+1.02) + bear (+0.10/+3.80)
  are positive-to-flat in both — **not opposite**. The genuine sign-flips are confined to **bull** (+3.76 rec /
  −1.31 OOS) and **deep-bull** (−3.75 rec / +0.81 OOS) — exactly the regimes the strategy already GATES.
- What IS era-dependent: the **magnitude** (side +4.60 recent vs +1.02 OOS) and the overall deployed number (recent
  ~+2.1–2.5 vs OOS ~+0.2), driven by cross-sectional **dispersion** (which varies by era) + regime **mix** (OOS is
  heavier bull/deep-bull) + the path-coupled overlays. Rank quality is stable; PnL-per-rank is dispersion-dependent.
- Fixability: the alpha is more era-robust than previously stated; the deployed-Sharpe swing is part path-coupled
  overlay artifact (reducible), part genuine dispersion-dependence (data). Detect-the-era-and-switch FAILED the
  both-eras test (addendum 60: R²≈0.005 predictability; the observable-state→good/bad map itself flips by era). →
  **managed** (cap / monitor / kill-switch), not a modeling defect.

### #2 — Bull gate is a deliberate era-REFUSAL, not a defect. [holds]
- Mild-bull is book-level +3.76 recent but −1.31 OOS (genuinely opposite) → gated to flat (`BULL_GROSS_MULT=0`).
  Refusing the era bet is the correct conservative choice.

### #3 — Deep-bull: NOT "pure beta" (audit-corrected); a small directional lottery the beta-neutral book gates. [CORRECTED]
- Beta-neutral selection LOSES deep-bull recent (−3.75) but is +0.81 OOS; production runs a `mom1d_long` overlay.
  The AUDIT **retracted "pure beta"**: the deep-bull long PnL SURVIVES BTC-residualization (residual ≈ 0.33–0.59 of
  raw, not ≈0) — so it is *not* pure directional beta. The weaker "generic **non-selective** long exposure" holds
  (Q3 placebo p=0.215 — random long-alt picks do as well as return_1d picks). Small n (47 recent), era-mixed.

### #4 — Short-side squeeze tail, unhedged. [holds]
- The short leg GRINDS a steady edge (recent median +41.1, 58% hit) but its **mean sits below its median** (recent
  +24.0 < +41.1; OOS +1.9 ≪ +22.3) — dragged by occasional violent **squeeze** losses (crowded alts squeeze up).
  Unhedged: free price/funding can't tell which crowded alt squeezes (both-tailed). → **DATA1** (paid positioning)
  is the one signal-side lever.

### #5 — The event-concentration is the LONG-leg lottery, not the short edge. [CORRECTED]
- Book PnL LOOKS event-concentrated (recent ~90% from ~8 high-dispersion days), but the leg-split (addendum 59,
  holds — preds 0.996-corr) shows it's the **LONG** leg: LONG bleeds daily (−19.5 median, 45% hit), net-positive
  only via a fat right tail (221% of its total from 8 days) = a **beta-hedge lottery**. The **SHORT** leg (the real
  alpha) GRINDS (+41 median, 58% hit). So "event-concentrated edge" was a book-level artifact of the long hedge;
  the short alpha is steadier than it looked. (Concentration lives in the fragile *long* + the path-coupled overlays.)

### #6 — The regime label is a 30-day LAGGING classifier. [holds; now grid-safe]
- Keys off `btc_ret_30d` (now correctly **30d wall-clock** — audit #5 fixed the row-based `shift(180)`) + hysteresis.
  Still a lagging guess in fast transitions. A leading regime detector FAILED (addendum 60). Handled as
  conservatively as a lagging estimator allows.

## What can move these (honest map)

| limitation | binding? | lever with upside |
|---|---|---|
| #1 era-Sharpe-variability | partial (alpha stable; Sharpe fragile) | reduce overlay path-coupling; accept dispersion-dependence |
| #4 squeeze tail | YES | **DATA1** (paid positioning) |
| #5 long-leg lottery | minor | shrink/drop the long hedge? (short-only failed the CI, addendum) |
| #2 bull gate | no (deliberate) | keep gating |
| #3 deep-bull | no | keep/drop mom1d — small either way |
| #6 lagging regime | structural | none (leading regime detector failed) |

## The bottom line

v4 is a **small but GENUINE and era-stable cross-sectional alpha** — book-level rank-IC **~+0.03**, positive in
**41/42** independent folds, no leak, verified three ways. The prior "era-fragile, opposite-sign regimes" framing
was substantially a **path-coupled artifact**; the alpha is more era-robust than previously stated. What remains
genuinely fragile is the **deployed full-stack Sharpe** (~+2.1–2.5 recent / ~+0.2 OOS) — path-coupled by the
overlays and dispersion-dependent — which is why it is quoted as a **range**, not performance, and why the strategy
runs capped + kill-switched. The one signal-side lever with upside is **DATA1** (the short-side squeeze tail).
**Nothing here is overstated; the reliable statement is the rank-IC.** Live-forward is the standing gap.
