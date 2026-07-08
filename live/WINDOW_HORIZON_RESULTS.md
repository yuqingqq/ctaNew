# Window × horizon program — results (2026-07-08, FINAL)

Question (user): are feature windows useful for *some* holding sleeve even if useless at the
production 4h label? Pre-registration: `RESEARCH_LOOP_20260707.md` addenda 8 (design), 8b (13
design-review findings applied before first run), 8c (Phase A results + Phase B cells), 8d
(Phase B results, post-review). Machinery: `live/window_horizon_ic_surface.py` (screen),
`live/feature_variant_harness.py` cells B1/B2, `live/gen_sleeve72.py` (72h sleeve, both arms),
`live/score_variant_cell.py` (+ SCORE_FWD_CYCLES / SCORE_BLOCK_DAYS knobs).

**Answer: the surface contains real screen-level structure, but 0/3 book cells pass. No
promotable variant; no sleeve candidacy; V0_LEAN and the 4h stack stay frozen.**

## Phase A — IC surface (screen, 21 windows × 5 horizons × 2 eras)

Flag column = per-cycle XS rank-IC after residualization on the full 16-feature V0-span;
horizon-length blocks (1/1/1/2/3 days); h>24h flags require the marginal 24h→h label test.
Sanity passed (deployed features read absorbed, |t|≤0.9 vs raw t 8-14). 15 flags, 3 ridges:

| ridge | peak | t_v0 rec/oos | fate at book level |
|---|---|---|---|
| 12h-24h momentum × h4-h24 | ret_24h × h4 | +9.5 / +7.0 | B1: flag is the residual of a near-duplicate (see below) |
| dd_3d × h4-h12 | dd_3d × h12 | +4.3 / +3.9 | B2: noise-floor null |
| resid_ret_3d × h72 (passes marginal test +4.1/+2.2) | — | +4.4 / +3.1 | B3: REJECT (spread fails both eras) |

Surface: `live/IC_SURFACE_WINDOW_HORIZON.csv` (504 rows). Unflagged families (corr windows,
resid_rev 6c/12c extensions) CLOSED.

## Phase B — book cells (all numbers reproduce; review verified instruments)

| cell | recent Δrank-IC / spread | OOS Δrank-IC / spread | hits | verdict |
|---|---|---|---|---|
| B1 +ret_24h @ 4h | −0.0002 / −3.9 | −0.0003 [−.0007,+.0001] / −5.6 | 3/9, 13/33 | **NO ADDITION** |
| B2 +dd_3d @ 4h | +0.0002 / −19.3 | +0.0003 [−.0006,+.0012] / −2.1 | 6/9, 16/33 | **NO ADDITION** |
| B3 +resid_ret_3d @ 72h sleeve | −0.0020 / **−65.3 [−129.1,−6.7] entirely neg** | **+0.0027 [+.0002,+.0052] excl 0** / −2.9 | 3/9, 22/33 | **REJECT** |

- **B1 mechanism (corrected by results review — the headline lesson):** ret_24h is a
  near-duplicate of the deployed `return_1d` (per-symbol corr 0.995, XS rank corr 0.986). The
  screen flag (t +9.5) lives entirely in the ~2-3%-variance residual between them — the
  shift(1)-vs-same-bar freshness offset, i.e. a last-5m-bar reversal component, plausibly part
  bid-ask bounce. The incumbent pred carries none of it (corr −0.002): **not absorbed —
  model-inaccessible.** A ridge-regularized addition cell splits weight across a 0.99-collinear
  pair and shrinks exactly the low-variance direction where the flag lives. Do not re-mine; the
  isolated difference feature is 5m-close microstructure.
- **B3 grounds (era-independent kill first):** the selection-spread Δ is non-positive in BOTH
  windows — the rank-IC lift never converts to top/bot-K selection alpha even in the favorable
  era (recent entirely-negative call robust at 2-3× block lengths). Recent Δic also fails ≥0.
  The OOS lift is real but razor-thin at the CI lower bound and streak-concentrated (Jul-Sep
  2025 +0.013/+0.011/+0.010; negative months exist inside OOS); the apparent by-year "building
  trend" was the wrong read. Baseline-arm descriptives (V0_LEAN@h72: rank-IC +0.035 rec /
  +0.016 oos, spread +273/+31 bps/cyc) are NOT a validated strategy — no costs, 18× entry
  overlap, no gates.
- Deviations logged: B1/B2 matched-control trigger technically fired (ratios 0.9993/0.9961),
  controls not built — verdict-safe (population effect bounded ≤0.0005 by the C2/T1 controls,
  handicap direction anti-variant).

## Program lessons (append to the estimator law)

1. **Screen flags must be checked for near-duplicate parentage before spending a cell.** The
   strongest flag of the program (t=9.5) was the residual of a 0.995-correlated pair with a
   deployed feature. A one-line corr check against the deployed set would have re-aimed or
   killed the cell for free.
2. **Rank-IC lift without selection-spread conversion is not tradeable signal** (B3, both eras;
   C3 before it). The tips are the strategy; the middle of the book is not.
3. **The 8b screen discipline worked**: horizon-length blocks, marginal labels, V0-span
   orthogonalization, and the built-in sanity rows kept 15 correlated flags from becoming 15
   stories — three cells, three clean answers, budget spent, done.

## Standing state after this program

Feature layer: frozen (this program + the 6-cell program, `FEATURE_TUNING_RESULTS.md`).
Window/horizon axis: CLOSED at fixed model class and data. Remaining feature-side upside
requires new data classes or a model-class change — each a separate pre-registered question.
The v4 forward test (0.5× gross cap) remains the live workstream.
