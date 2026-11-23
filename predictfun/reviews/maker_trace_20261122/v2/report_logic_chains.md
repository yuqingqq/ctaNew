# TOP-3 MAKER LOGIC CHAINS — results (USER plans 7d71ca40c / 72c5e5841 / 5cb2f0229 + USER focus; packet frozen 196af3323)

2026-11-23, session aea5bfb9.

**Runs:** all rc 0 — `mt_x_oe_v2`, `mt_x_take_v2` (28 CPU s), `mt_x_pull_v2` (49 s), `mt_x_pf_v2` (41 s).

**Outputs:** `v2/order_estimator/` and `v2/chains/{post_fill,pull,take}/`; every number below is from their `summary.csv`.

**Population:** the ex-ante top 3 on the 19 BOOK markets, Oct 24–27 (4 consumed days, point estimates). Descriptive only:
- filled orders are outcome-selected;
- cancel-possible is not "pulled";
- timing windows are bounds.

## POST and FILL (4,149 filled orders; `chains/post_fill`)

### How long they rested before the first fill

This is a proven upper bound from capacity: the posting window ends at the first fill.

| Resting time | Orders | Share |
|---|---|---|
| ≤ 1 s (identified late post) | 576 | 14% |
| ≤ 10 s (posting window ≤ 10 s) | 2,341 | 56% |
| > 10 s, early or late unidentified | 1,132 | 27% |
| unbounded | 100 | 2% |

So **at least 70% of filled orders were posted at most 10 s before their first fill**. Long-resting early queue posts are possible only for the 27% with wide windows, and public data can never prove one.

### Queue at the first fill

- mostly unidentified (2,944) or conflicting (889);
- front at acceptance 5;
- front reached after clearing 9;
- behind 276.

### Book position at posting

- mostly timing-sensitive (3,492);
- touch-joiner 359;
- behind the touch 295.

### BTC 5 s trend at posting

| Trend | Orders |
|---|---|
| flat | 2,431 |
| timing-sensitive | 1,353 |
| with | 204 |
| against | 153 |

### Net at resolution by posting class (c/sh)

| Posting class | Net | 1 s markout | Shares |
|---|---|---|---|
| ≤ 1 s late | −3.05 | −0.44 | 21.2 k |
| ≤ 10 s | +0.22 | +0.30 | 98.9 k |
| > 10 s possible | +6.67 | −0.32 | 43.0 k |

These are resolution values (noisy), not causal.

### What followed the fill

- a reprice: 2,962 (71%);
- a refill at the same price: 577 (14%);
- none: 610.

## PULL (S_FIFO, D = 250 ms, matched presence controls, ≥ 200 draws; `chains/pull`)

**Cancel-possible removals (2,741),** each pull against matched moments when the order kept resting:

| Context in the pull window | Pulls | Controls | Gap (pp) | p |
|---|---|---|---|---|
| BTC moved against the order within 1 s | 34.6% | 17.7% | +16.9 | 0.005 |
| BTC against over 5 s | 27.7% | 18.3% | +9.4 | 0.005 |
| Their inventory built on that side within 5 s | 87.6% | 71.0% | +16.6 | 0.005 |
| The order was behind the touch | 85.0% | 63.0% | +22 | 0.005 |
| The Predict mid moved against the order | 82.6% | 70.3% | +12.2 | 0.005 |

- **Timing:** pulls come later in the window (154 s left vs 173 s).
- **Reference:** exhausted (not pulled) orders show weaker enrichment: BTC against within 1 s +6.8 pp, inventory built +2.1 pp.
- **Re-quoting:** 1,573 of the 2,741 have a successor order within 5 s, i.e. a re-quote.

## TAKE (4,147 takes, rank 1: 3,620; `chains/take`)

**Classes:**

| Class | Takes |
|---|---|
| reduce_partial | 2,065 (50%) |
| add | 1,185 (29%) |
| timing_sensitive | 838 |
| no change | 38 |
| reversal / overshoot | 11 |
| from flat | 10 |

**Rank 1 against matched moments with similar |inventory| and no take** (p 0.002 each):

| Before the take | Takes | Controls |
|---|---|---|
| Own fills in the last 5 s | 5.26 | 3.53 |
| Offsetting shares filled in the last 5 s | 64 | 36 |
| BTC moving against its position over 5 s | 5.1% | 2.7% |

- **Ranks 2 and 3:** few takes (341 and 186), with no significant burst or BTC association.
- **Reducing vs adding takes** (pilot): offsetting shares are 35 vs 9 (p 0.002).

## Action census (pooled top 3; `chains/take` act_*)

| Action | Count |
|---|---|
| maker fills | 9,026 |
| keep-after-fill | 4,931 |
| reprice | 4,023 |
| two-sided quoting | 3,886 |
| posts at a new level | 2,601 |
| refill / posts at the same level | 1,548 |
| ladder overlaps | 1,007 |
| cancel-possible (S_FIFO) | 2,754 |
| exhausted-proven | 1,251 |
| takes: reduce | 2,065 |
| takes: add | 1,185 |

Mint-kind takes: 3,709.

## Detailed episodes (`order_estimator`; 88 deterministic cases → 62 connected episodes)

- **Sizes:** original sizes are mostly tied.
- **End of life:** exhausted vs pulled is unresolved without price priority.
- **Links:** repost vs ladder is at best sequential.
- **Own-order check:** 85 covered, 1 priority miss, 0 defects.

## The logic chain these numbers describe (descriptive; consumed days; not identified intent)

1. **Post:** quote both sides mostly **shortly before demand arrives** — at least 70% of filled orders rested ≤ 10 s — not as long-resting queue builders. Long-rest-possible orders earned more at resolution: +6.7 vs −3.1 c/sh for identified late posts.
2. **After a fill:** reprice (71%) or refill (14%).
3. **Pull** when, during the pull window, BTC moves against the order within ~1 s, their inventory has just built on that side, the order sits behind the touch and the mid moves against it. Most pulls are followed by a re-quote within 5 s.
4. **Take** (mainly rank 1) after a **burst of own fills** builds inventory, and more often when BTC moves against that inventory: an inventory unwind.

## Not identified by public data

- whether an order was posted early;
- exact original sizes;
- whether a removal was a pull or exhaustion;
- repost vs ladder;
- private decision times.

These are tied alternatives in the outputs.
