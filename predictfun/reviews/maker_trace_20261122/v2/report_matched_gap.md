# MATCHED-MARKET-TURNOVER-GAP-v1 — results (review 8f2a2e753's next priority; USER go)

2026-11-23, session aea5bfb9.

**Provenance:**
- contract `46cd3cc56` + amendment 1 (`c0f237ffe`);
- the first run, `v2/matched_gap/`, is **kept as disclosed evidence** and is defective in the top-10 pooled carry view;
- fix `76921520a`; corrected run `mt_x_gap_v2_r1` (2.1 CPU s) → [`v2/matched_gap_r1/`](matched_gap_r1/).

**Checks:**
- our recorded value is reproduced bit-exactly per market;
- the identity holds against a direct recompute, including carried positions;
- the cross-view refusal is active.

**Scope:**
- **What is compared:** our accepted **CURRENT_ASYNC_FAST10** saved ledger vs the ex-ante top 3 / top 10, on the **same 19 BOOK markets** and **our interval [start, END]** (about 298 s), at **our END mark**, with one FIFO accounting.
- **Units:** USD totals; c = cents per share.
- **Limits:** 4 consumed days; no intervals. Descriptive: not an avoided-loss estimate or an imitation claim.

## Answer (primary carry view; `answer.json`)

| Party | Markets both active | Completed-close net: ours | Completed-close net: theirs | Residual: ours | Residual: theirs | Total: ours | Total: theirs | Per-market readings | Pooled reading |
|---|---|---|---|---|---|---|---|---|---|
| Rank 1 | 19 | −1.90 c | +0.75 c | +71 | −476 | −683 | −55 | turnover-loses 11, both-adverse 4, exposure 1, mixed 3 | `our_turnover_loses` |
| Rank 2 | 11 | −2.08 c | +2.40 c | +33 | −716 | −427 | −240 | 7 / 1 / 2 / 1 | `our_turnover_loses` |
| Rank 3 | 17 | −1.78 c | +3.39 c | +42 | −1,209 | −615 | −351 | 8 / 3 / 5 / 1 | `our_turnover_loses` |
| Top 3 pooled | 19 | −1.90 c | +1.71 c | +71 | −2,402 | −683 | −647 | 11 / 2 / 4 / 2 | `our_turnover_loses` |
| Top 10 | 19 | −1.90 c | +1.52 c | +71 | −2,972 (carry +22) | −683 | −492 | 12 / 0 / 5 / 2 | `our_turnover_loses` |

**Views:** ranks 1–3 had no fills before our start, so their carry and zero-start views coincide. The top-10 carry differs by +22 USD.

## What it says (computed readings only)

1. On the same markets and interval, **our completed round trips lose** (−1.8 to −2.1 c/share net) where the matched top makers' completed round trips **earn** (+0.75 to +3.39 c/share).
2. **Our residual exposure is smaller and better than theirs.** Their unclosed inventory loses 476–2,972 USD at our END mark; ours gains 33–71.
3. The observed gap is in **turnover quality**, not in inventory exposure. Fixing our exposure would not close it.

**Limits:** denominators and the market mix are in `per_market.csv` (`closes_by_role.csv` gives the roles). The total-net gap is smaller than the turnover gap, because the makers' residual losses offset part of their turnover edge.
