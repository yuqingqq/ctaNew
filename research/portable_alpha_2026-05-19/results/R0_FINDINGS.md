# R0 — Integrity Gate: FINDINGS (2026-05-19) — VERDICT: PASS

Pre-registered prediction: `target_A` matches (no target leak); ≥1 flagged
feature shows non-trivial Δ; the "PIT smell" is a bar-close convention, not
future leakage. **Outcome: confirmed.**

## Decisive results

| check | result | verdict |
|---|---|---|
| `basket_A_fwd` recompute | max\|Δ\|=6.0e-8 (3.6e-6·std) | PASS |
| `alpha_A` recompute | max\|Δ\|=0 (exact) | PASS |
| **`target_A` faithful recompute** | max\|Δ\|=1.4e-5 (**1.1e-5·std**), 5.83M rows | **PASS** |
| **prefix-causal truncation** (3 cuts) | max\|Δ\| = **exactly 0.0** at every cut | **PASS** |
| `dom_change_288b_vs_bk` (time-grid) | max\|Δ\|=9.5e-7 (**2.8e-5·std**), 5.83M rows | PASS |

## Interpretation

1. **The production target `target_A` is PIT-clean.** It is exactly the
   documented `add_targets('A')` recipe (per-open_time equal-weight basket of
   forward returns → residual → per-symbol expanding-mean/rolling-7d-std, both
   `.shift(48)`). The prefix-causal test is **identically zero** at all 3
   interior cuts — recomputing on a truncated panel yields bit-identical
   earlier values, i.e. **zero future data is used**. This empirically refutes
   the Round-2 methodology CRITICAL concern (a pooled-normalization
   out-of-time/out-of-universe leak): no such leak exists.

2. **`dom_change_288b_vs_bk` is faithful to its documented recipe** (matches to
   2.8e-5·std once recomputed on the continuous 5-min grid the builder uses).
   The initial R0(b) "FAIL" (66·std on 0.019% of rows) was a bug in the
   verification code — a positional `.shift(288)` crossing 4 tiny internal data
   gaps — **not** a panel defect. Fixed transparently in `R0c_dom_timegrid.py`
   (not a goalpost move: the pre-registered plan named "float32-aware" tolerance
   and the prefix-causal test as the real leak proof).

3. **Documented caveat (carried into R1).** At the 4 internal gaps the builder
   `ffill`s `dom_level_vs_bk` through the gap rather than emitting NaN,
   producing 576 ffill'd `dom_change` rows each on PUMP/STRK/VIRTUAL
   (0.019% total), all immediately post-listing. This is the bar-close/ffill
   *convention* (consistent across the whole validated panel), uses only past
   values (prefix-causal = 0), and is **not** future leakage. R1 will verify
   these rows are excluded by the production listing-eligibility / min-history
   filter (artifact `min_history_days=60`), or mask them, and report whether
   any ever enters a traded basket.

## Consequence for the plan

R0 PASS ⇒ all R1–R3 Sharpe/IC numbers are computed on a leak-free target and
faithful features. No column rebuild from `_full_pit` is required. The "PIT
smell" raised by the feature audit is resolved as a benign convention.
Scripts: `R0_integrity_gate.py`, `R0b_dom_diag.py`, `R0c_dom_timegrid.py`.
