# Cross-reference: corrected convexity v2 deploy preds (5/29 → 6/4)

Generated 2026-06-05 to settle the XLM cross-check (see `docs/recent_cycles_v2_README.md` + commit `90e117c`).

- `convexity_v2_preds_CORRECTED_0529_0604.{parquet,csv}` — deploy preds (frozen 5.29 models
  `convexity_v1_{long,short}_model.pkl`, md5 long `7d320599…`) **recomputed from the completed panel**,
  filtered to the low-vol book. Columns: `symbol, open_time, pred_long (resid_rev), pred_short (base)`.
  3,854 rows · 94 syms · 41 cycles.

## Use
Diff this against your current June preds, per symbol+cycle:
- **Match** → both boxes on fresh data; reconciled (XLM was the only material swap).
- **Differ** → real panel/feed delta to chase (not the stale-preds artifact, which is fixed by the
  trailing-recompute in `predict_twobook_incremental`).

## The swap that this corrects (5/29 00:00, long-ranker)
| symbol | stale (old run read) | corrected | Δ |
|---|---|---|---|
| XLM   | −0.1262 | +1.2771 | **+1.40** (rank 88 → #1) |
| SEI   | +0.0355 | +0.1770 | +0.14 |
| HYPE  | +0.5084 | +0.4598 | −0.05 |
| POLYX | +0.2260 | +0.1074 | −0.12 (dropped) |
| ZRO   | +0.1092 | +0.0370 | −0.07 (dropped) |
