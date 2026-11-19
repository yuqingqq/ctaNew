# R1 (plan v2): the causal branch engine and `paired_continuation_v2` — 2026-11-19

**USER:** "Check my latest commit, next, can you ask codex research session for next priority. continue the implementing and next
priority loop".

**Review 92d8e52f's next priority:** "Develop R1: an executable 50 ms WAIT/re-entry branch engine with common END. The first reviewable
deliverable is code plus synthetic event/ledger traces — not a new model fit."

**What this deliverable is.**
- [`pf_branch_engine.py`](../../research/pf_branch_engine.py), with its [mutation harness](../../research/pf_branch_engine_mutants.py).
- Synthetic controls, example traces, and a bounded slice of one consumed day.

**What it is not.** No model is fitted or read. No label build is run. The supported pi0-50 teacher, labels and features for new states
are R2.

## What the engine does

| Part | Rule |
|---|---|
| **Engine** | `pf_flow_control_kernel.cpp` ported line for line, so every fill, request and removal has its own timestamp. Shared public volume is applied once to the portfolio in price / activation priority. The queue ahead is set at activation and only shrinks to later shown sizes (rule M). Post-only refusal at activation. The structural rule (no book, more than 3 ticks behind, crossing) runs on the 50 ms grid, with cancel latency 37 ms and the pending clock of 95 ms before activation. **One departure:** the structural rule applies to a generation only after its submission tick. The kernel checks every listed order at every grid, which for a later submission acts before the order exists; for orders submitted at 0 the two agree |
| **Client** | Decides on the 50 ms grid from client-known state only. A post's acceptance or refusal is known at activation (+58 ms). A fill is known 100 ms later. **A removal is acknowledged 100 ms after its effect (a declared assumption).** At most one open generation per side, with states absent, post_pending, live, cancel_pending and terminal. Reservations follow plan v2: a bid post needs I + B + 30 ≤ 30, an ask post −I + A + 30 ≤ 30, and an independent check holds the engine position inside [I − A, I + B] at every tick. JOIN at the best of the last received frame. Sides alternate by episode + tick, and the second side's view is rebuilt after the first side's command. Fallbacks and constraints (cap, near_end, no_book) are counted with their reasons |
| **Contract** | `paired_continuation_v2`. Y_post = R_T(POST now, then the policy) − R_T(WAIT this tick, then the policy). Y_keep = R_T(KEEP) − R_T(PULL now), each continued the same way. R_T = cash + inventory × the END mark over every generation and both sides. The common pre-fork fills are checked identical. Open exposure at END is disclosed and is never cancelled or liquidated for free. A missing mark censors the pair |

## Controls (synthetic tapes; each has a valid case and a known-bad case)

| Control | Result |
|---|---|
| **P1** kernel parity | The port in kernel mode equals the C library bit for bit: per-order shares, cash and position on 40 random tapes / order lists / actions. A deterministic case also covers own-order price priority (bids at 50 and 49, one sell through both: 30 then 5) |
| **P2** incremental = batch | Adaptive runs equal their replays from the final order list. A deterministic case also covers the submission guard |
| **F1** actual WAIT | WAIT posts at 50 ms, at 51 instead of 50, with a different history. POST at 0 ms: +60 c. WAIT, then post at 50 ms: +30 c. Y_post = +30 c ([traces](traces.json)) |
| **F2** re-entry | A PULL at 100 ms stays exposed: a 10-share fill at 120 ms comes before the effect at 137 ms. An ask fill offsets it, the acknowledgement arrives at 237 ms, and a new bid generation goes up at 250 ms at the new best, 49 |
| **F3** queue | A partial KEEP keeps its priority, so later arrivals stay behind. A replacement goes behind the queue that was there before it |
| **F4** economic identity | The fill ledger equals cash + inventory × mark in every arm. Y = the arm difference. A common fill counted twice is rejected |
| **F5** continuation | On seeded random tapes the engine produces cases in both directions: hold > 0 > chase, and hold < 0 < chase |
| **F6** causality | Events after a tick cannot change its view or action. A known frame changes the view |
| **F7** boundaries | A match just before the cancel effect fills and one at the effect does not. A match at activation fills. The notice is delayed 100 ms. A cap breach is refused. Other cases: post-only refusal, near-END post, missing mark (censored), unavailable book (a counted fallback), and a released reservation with an unacknowledged fill (rejected by the position check) |
| **F8** identity / support | Each of these is refused: a wrong horizon, wrong END, wrong timing scenario, an action invalid for its state, a stale generation, an undeclared pair kind |
| **F9** shared book | A bid fill then an ask fill reconcile to position 0 and +60 c. The ask waits while the bid is post-pending. The sides alternate |
| **Mutation suite** | **33 / 33** planted errors caught by a check ([log](logs/pbe_mutants3.log)) |

The first suite had 4 unexpected outcomes ([log](logs/pbe_mutants1_4_unexpected.log)):
- reversed own-order priority escaped, because random tapes never produced the case; the deterministic case above now catches it;
- 3 mutants tripped the engine's position check as an uncaught exception; the selftest now reports an engine invariant refusing a valid
  synthetic case as a control failure.

The first F2 tape was wrong: its 10-share cancel-pending fill correctly blocked the bid re-entry under the cap.

## The bounded slice on a consumed day

Two degenerate policies on the order's side only:
- post at 0, then KEEP;
- post at 0, then PULL at the first review.

| Slice | Orders | With fills | Result |
|---|---:|---:|---|
| v1: the first 200 eligible opportunities of Oct 24 ([json](slice_20261024.json), module b7e7b5b9) | 200 | 94 | **FAIL as declared.** Fills differ row by row on 47 (keep) + 1 (pull) orders, while END values, JOIN prices, marks and C-kernel parity agree on all 200 |
| v1 diagnosis ([script](diag_slice_v1.py), [log](logs/pbe_diag_slice_v1.log)) | 48 | | **All 48 are equal once summed per trade time.** The committed keep / pull run records one fill row per distinct trade time; the engine records one per public trade |
| v2: the first 1,000 (comparison per trade time, `fills_equal`; [json](slice_20261024_v2.json), module 7d9470f0) | 1,000 | 439 | **PASS.** Keep fills, pull fills (before the 137 ms landing), END values, JOIN prices, marks and C-kernel parity all exact on every order |

The v1 output is kept as provenance. The v2 comparison has its own tests (equal per time, plus a quantity, time and missing-fill difference
that must each fail) and two mutants.

## The R1 gate (plan v2 section 4)

| Gate requirement | Status |
|---|---|
| All controls pass before data fitting | Met (synthetic) |
| An executable WAIT branch | F1 |
| A versioned `paired_continuation_v2` contract | Met |
| Traceable lineage | Met: sources hashed, receipts |
| No unexplained accounting or causality mismatch | Met: F4, F6, P1 / P2, slice v2 |
| Synthetic first, then a bounded consumed slice | Met |

**Open, sent to the research thread as plan decisions (request queued 17:29 UTC):**
- (a) the acknowledgement timing;
- (b) the stand-in continuation policies until R2;
- (c) the structural-rule departure;
- (d) the cap: with size 30 and cap 30, any inventory blocks the side that would add to it;
- (e) whether the slice suffices.

## Limits

- **Costs are not in R_T** (fees, rebates). Plan v2 requires the Predict.fun cost convention to be copied and hashed before R4.
- **Stand-in policies only.** No `pi0-50` yet.
- **Timing scenario B only** (M50, 58 / 37 / 95 / 100 ms).
- **One instrument per episode, single 30-share JOIN generations.**
- **Synthetic controls are implementation evidence, not profitability evidence.**

## Files

- **Sources:** the engine, its mutation harness, and the [slice v1 diagnostic](diag_slice_v1.py).
- **Outputs:** slice v1 / v2 JSON and CSV; [traces.json](traces.json).
- **Logs:** [logs/](logs/). Hashes: [sources.sha256](sources.sha256).
- **Data:** `data/predict_fun/research/branch_engine_20261119`.
