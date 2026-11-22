# ASYNC-EXECUTION-SENSITIVITY-v1 — superseding interpretation (v2)

2026-11-22, session aea5bfb9.
- **Supersedes:** the interpretation at `report.md:43, :72, :88` per review `3e84ca6d` P2.
- **Unchanged:** the frozen receipt (`85127b3d`), its numbers and its computed reading `assumptions_do_not_rescue`. `report.md` stays as provenance.
- **New measurement:** none. The identity below is computed by `../ladder_cancel_headroom_20261122/tools/async_identity.py` from this receipt's committed arms copy. It agrees with the review's independently rebuilt `end_decomposition.csv` on all 76 market × arm rows (0.01 c), and its falsifier passed.

## 1. END versus +1 s (replaces `report.md:43`)

`END − eligible +1 s` is not "the movement from +1 s to END". It also holds the END value of the fills that have no eligible +1 s mark (late or unsupported). The identity, in c:

| Arm | Eligible +1 s | Subsequent movement on that cohort | Late / unsupported +1 s fills at END | END |
|---|---:|---:|---:|---:|
| CURRENT_BASELINE | −85,164.28 | −1,416.84 | +471.51 | −86,109.61 |
| FRONT_BASELINE | −61,812.39 | −5,175.38 | +1,050.68 | −65,937.09 |
| CURRENT_ASYNC_FAST10 | −78,485.90 | +9,177.89 | +988.06 | −68,319.95 |
| FRONT_ASYNC_FAST10 | −61,051.63 | −26,694.24 | +785.26 | −86,960.60 |

The caution about the END contrasts stands: the subsequent movement is as large as the contrasts themselves. Neither this identity nor the END class flip identifies an inventory-management defect; no inventory-policy ablation was run.

## 2. The timing sentence (replaces `report.md:72`)

The 27 ms (live cancellation) and 48 ms (activation) gains apply **under this combined execution profile**, conditional on a fixed submission / request time. Complete feedback changes the request times and states too:
- a normally accepted post first becomes reviewable at +50 ms under ASYNC, versus +100 ms under BASELINE, on this grid;
- concurrency changes as well.

There is no paired same-action estimate of a pure 27–48 ms benefit.

## 3. The conclusion (replaces `report.md:88`)

"This fixed pressure ladder leaves large adverse through-price exposure even with the tested faster execution and hypothetical front priority. It remains unknown how much of that exposure can be removed by timely, selective decisions without sacrificing the good fills and subsequent queue value."

The experiment tested the **fixed Q5 pressure rule's use** of the decision-time information, not every predictive use of those inputs. The qualifications at `report.md:74–77` stand.

**What the next receipt found:** FEASIBLE-CANCEL-HEADROOM-v1 (`../ladder_cancel_headroom_20261122/report.md`) bounds the first part of that question under a hindsight oracle on this same path. One timely permanent cancel per order would remove ~89 % of the gross +1 s bad value while sacrificing ~0.2k c of good value; the cut nearly always falls at the first review. The same cut is worth far less at END. Whether any causal signal can make that discrimination, and what it is worth over a longer continuation, is unanswered.
