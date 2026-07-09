# Window × horizon program — results (2026-07-09, post-completion FINAL)

Question: are feature windows useful for some holding sleeve even if useless at the production
4h label? Audit trail: `RESEARCH_LOOP_20260707.md` addenda 8/8b/8c/8d (original screen and books)
plus 17/17a/17b (separately committed sleeve-aligned completion and missing 4h matched controls).
Machinery: `window_horizon_ic_surface.py`, `feature_variant_harness.py`, `gen_sleeve72.py`,
`gen_sleeve_hk.py`, and `score_variant_cell.py`.

**Answer: sleeve alignment largely does not improve this system. Across the preregistered h12
and h72 ridge representatives, 0/4 pass the book-level bars: two h12 treatments have small,
positive but statistically unresolved estimates, while drawdown@h12 and residual momentum@h72
are rejected. No promotable variant; V0_LEAN and the 4h stack stay frozen. This closes this
defined grid under per-symbol Ridge, not all possible window tuning.**

## Phase A — IC surface

The screen covers 21 windows × 5 primary horizons × 2 eras, with marginal-label and raw-return
diagnostics in `IC_SURFACE_WINDOW_HORIZON.csv` (504 rows). The flag column is per-cycle XS
rank-IC after residualization on the 16-feature V0 span; blocks are 1/1/1/2/3 days and h>24h
requires the marginal 24h→h test. At h4 the deployed-feature sanity rows read absorbed
(|t|≤0.9 vs raw t 8-14). Drawdown uses the authoritative same-bar enter-at-close convention.
The screen produced 15 flags in three ridges:

| ridge | peak | t_v0 recent/OOS | book-level fate |
|---|---|---|---|
| 12h-24h momentum × h4-h24 | ret_24h × h4 | +9.5 / +7.0 | no reliable h4 or h12 improvement |
| drawdown × h4-h12 | dd_3d × h12 | +4.3 / +3.9 | h4 null; h12 harmful recent |
| resid_ret_3d × h72 | marginal +4.1 / +2.2 | +4.4 / +3.1 | h72 rejected |

Unflagged corr windows and resid_rev extensions are closed within this screen.

## Phase B — original cells and matched controls

| cell | recent Δrank-IC / spread | OOS Δrank-IC / spread | positive folds | verdict |
|---|---|---|---|---|
| B1 +ret_24h @ h4, matched control | −0.0002 / +7.4 | **−0.0006 [−.0010,−.0003] / −9.8 [−18.0,−1.8]** | 3/9, 14/33 | **NO ADDITION** |
| B2 +dd_3d @ h4, matched control | −0.0000 / −1.2 | +0.0003 / −0.2 | 4/9, 15/33 | **NO ADDITION** |
| B3 +resid_ret_3d @ h72 | −0.0020 / **−65.3 [−129.1,−6.7]** | +0.0027 / −2.9 | 3/9, 22/33 | **REJECT** |

The original B1/B2 matched-control deviation is closed in addendum 17a. B1 is mildly worse OOS
on both primary endpoints; B2 is a clean null. The ret_24h screen flag is the low-variance
shift(1)-versus-same-bar residual of deployed `return_1d` (per-symbol corr 0.995, XS rank corr
0.986). The addition cell does not monetize that direction; isolating it would test a separate
5m-close microstructure feature, outside this grid.

B3's OOS rank-IC lift does not convert to top/bottom-K spread, and its recent spread is entirely
negative and robust to 2-3× block lengths. The h72 baseline descriptives are not a validated
strategy: they exclude costs, gates, and the 18-entry overlap.

## Phase C — sleeve-aligned h12 completion

Each cell retrains V0_LEAN on the same h12 target and the same variant-defined population. The
h12 label is the sum of three consecutive 4h residual returns; purge exit is open_time+12h and
statistical blocks are one day.

| cell | recent Δrank-IC / spread | OOS Δrank-IC / spread | positive folds | verdict |
|---|---|---|---|---|
| Q1 +ret_24h @ h12 | +0.0006 / +1.4 | +0.0002 / +0.9 | 7/9, 19/33 | **KEEP — all CIs cross 0** |
| Q2 +resid_ret_24h @ h12 | +0.0012 / +3.7 | +0.0003 / −0.3 | 7/9, 18/33 | **KEEP — all CIs cross 0** |
| Q3 +dd_3d @ h12 | −0.0008 / **−24.3 [−50.0,−0.7]** | +0.0002 / +2.2 | 2/9, 16/33 | **REJECT** |

Q1 shows that sleeve alignment can change a treatment's direction: ret_24h is mildly negative
at h4 but positive at h12. The magnitude remains below the estimator's resolution, so this is
not evidence of improvement. Q2 is the strongest h12 candidate but also fails both CI
requirements. No strategy simulation was run because no cell passed the book-level bars.

## Post-review implementation note

The committed scorer still applies its evaluation grid guard only when k>6. Re-scoring Q1-Q3
with the guard applied for every k>1 removes two scored cycles per era and produces the h12
values above; no verdict changes. The attempted Phase-A residual-norm guard checks the design
and prediction norms rather than each residual column, so it does not yet eliminate the
numerical-residue diagnostic noted in review. Neither issue changes the matched-book conclusions.

## Program lessons

1. Screen flags must be checked for near-duplicate parentage before spending a cell.
2. Sleeve alignment can move an estimate without making it reliable: Q1/Q2 improve by only
   roughly 0.0002-0.0012 rank-IC and 1-4 bps of spread, with every CI crossing zero.
3. Rank-IC lift without selection-spread conversion is not tradeable signal for this top-K book.
4. Fixed representatives, matched controls, and dual-era book tests — not the screen — determine
   promotion.

## Standing state

No sleeve-aligned improvement was found among the preregistered momentum, residual-momentum,
and drawdown representatives at h12 or h72. This grid is closed for V0_LEAN + per-symbol Ridge.
This is not a claim that every feature window, holding sleeve, or model class is ineffective.
The v4 forward test at the 0.5× gross cap remains the live workstream.
