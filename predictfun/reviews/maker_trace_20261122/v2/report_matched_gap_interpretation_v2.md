# MATCHED-MARKET-TURNOVER-GAP-v1 — interpretation v2 (answers review 6ddad3533; supersedes wording in `report_matched_gap.md`; numbers unchanged)

2026-11-23, session aea5bfb9. Review [`20261123_matched_gap_review`](../../automatic/20261123_matched_gap_review/report.md). It accepts `matched_gap_r1`, reproduces it byte-for-byte, and an independent deque FIFO agrees on all 208 rows.

## Finding 1 (P2, holds) — the inventory prescription is withdrawn

**Withdrawn:** "Fixing our exposure would not close it" and "our residual exposure is smaller and better".

**Replacement:** on these saved paths, our completed closes lose while the benchmark's completed closes earn, and our residual mark contribution offsets some of the loss.
- **What is unmeasured:** which control decisions (posting, retention, taking, inventory or simulator mechanism) cause the turnover gap.
- **What is not tested:** FIFO completed vs residual is an accounting split of realized paths, not a test of an inventory intervention.

**Role cross-tab** (review; descriptive): all our completed closes are maker→maker, at −1.90 c/sh. The benchmark's maker→maker closes earn:

| Benchmark | Maker→maker net |
|---|---|
| Rank 1 | +4.49 c/sh |
| Rank 2 | +4.45 c/sh |
| Rank 3 | +4.06 c/sh |
| Top 10 | +2.68 c/sh |

So the role mix alone does not remove the gap. Sizes, prices, timing, selection and simulated fills still differ.

## Finding 2 (P2, holds) — qualifications that belong beside the answer

This is a **same-market, same-numerical-interval / same-mark, simulated-versus-recorded comparison**.

**Clocks:**
- Ours: the simulator's assumed fill time (trade anchor − 50 ms).
- Theirs: the UUID settlement upper bound.
- Boundary-sensitive top-10 rows: 69 have a lower spread anchor before start with settlement after it, and 20 span END.
- Physical-window membership and sub-second FIFO order are therefore not established.

**Fills:** our passive fills come from the replay's queue / public-flow assumptions, starting flat. Theirs are recorded maker and taker fills, with different sizes, participation and holdings.

**Fees:**
- Theirs are recorded and converted at the fill price; ours are the engine's zero maker fee. No rebate is credited.
- Revaluing the in-window fee-token position at END would change the totals by about −0.97 / +1.51 / +0.04 / +0.58 / +2.39 USD (rank 1 / rank 2 / rank 3 / top 3 / top 10).

**Not inferred from the table:** precise posting or cancel times, a causal queue advantage, or achievable profit.
