# R1 v2: the branch engine's causal client boundary and engine-gate receipt (review 043ca403 answered) — 2026-11-19

**USER:** "Check my latest commit, next, can you ask codex research session for next priority. continue the implementing and next
priority loop".

**Review 043ca403's next priority:** "Finish R1's causal client/event boundary and publish its engine-gate receipt. First eliminate the
unacknowledged-fill structural-state leak; adopt the five decisions above, add the common-state continuation and boundary controls, then run
the fixed bounded regression/adaptive slice and commit the engine, contract, tests and receipt for review."

**Supersession.** This receipt supersedes [R1](../branch_engine_r1_20261119/report.md) (a9303c1f) in-band on everything below. R1 stays as
provenance. The module is [`pf_branch_engine.py`](../../research/pf_branch_engine.py), sha256 13239a24...

## The review's findings and decisions, at the code

| Item | Holds? | Handled |
|---|---|---|
| **P1:** the client's structural state depended on an unacknowledged fill | Holds. The rule lived in `Engine.advance` and skipped orders with engine-truth `rem <= EPS`. `Branch.status` exposed its request as `cancel_pending`. It was also in a9303c1f | The rule now has **two paths**. Without a client (batch, C-compatibility), it is the kernel's form. With a client, `Branch.structural` issues it on **client-known** state: every open generation without a request, after its submission tick, including an order the engine has filled but the client has not heard of (no execution effect). Single-generation fills and values are unchanged (slice v3 (1)). **F10:** zero, partial and full unnoticed fills, on both sides, with a pending-sensitive other side, leave every view, structural request and reservation identical up to the notice at 180 ms. After it, the known position differs legitimately. Mutant: the old engine-remainder skip, caught by F10 |
| **(a)** acknowledgements | Accepted as assumptions | Named fields, recorded separately in `SETTINGS`: activation 58, POST_NOTICE 0, FILL_NOTICE 100, cancel effect 37 (pre-activation max(c + 37, d + 95)), CANCEL_ACK 100. A guard refuses CANCEL_ACK < FILL_NOTICE (this contract has no cumulative-fill watermark). Explicit tie order at a tick: trades up to the tick, notices due, invariants, client structural requests, then decisions in the alternating order, with invariants after every command |
| **(b)** stand-in continuations; one fork state | Accepted / holds | F5 now forks **one saved state** and varies only the continuation, checked by equal state fingerprints. Both sign directions are found on seeded tapes (hold > 0 > chase, and hold < 0 < chase) |
| **(c)** submission guard; parity wording | Accepted / holds | The C-compatibility path is named as such. P1 is now **numerical aggregate parity** within declared tolerances (shares 1e-9, cash 1e-7 c, position 1e-9, **END value 1e-7 c**). It is not event parity; fill times are checked against the archive in the slice |
| **(d)** cap 30, size 30 | Intended | Both worst-case inequalities (I + B ≤ 30, −I + A ≤ 30) and the engine-position bounds are checked after every notice and every command. Falsifier: a planted I = 10 beside an open 30-share bid. Counters per side: absent feasible, absent cap-blocked, absent near-END, post_pending, cancel_pending, live KEEP / PULL, and fallbacks |
| **(e)** bounded slice | Done as specified | Slice v3 below |

## Checks

- **Selftest:** pass. P1 (with own-order priority), P2, F1–F10 ([log](logs/pbe2_selftest5.log)).
- **Mutation suite:** **40 / 40** caught by a check ([log](logs/pbe2_mutants4.log)). New mutants:
  - the engine-remainder skip in the client rule;
  - the client rule on the engine frame;
  - the kernel-path and client-path rules disabled;
  - a post known before activation;
  - the reservation inequality unchecked;
  - a saved fork state of another tick accepted;
  - cap blocks counted as feasible.
- **Kept as provenance:**
  - one harness replacement that broke indentation ([log](logs/pbe2_mutants1_harness_indent.log));
  - one F10 check that read the closed generation's `known_filled` instead of the known position ([log](logs/pbe2_selftest2.log)).

## Slice v3 (review 043ca403 (e); the selection was frozen before any result)

**Selection:** Oct 24 (consumed), the first 128 eligible archived opportunities per side by oid: 256 orders, all with a path, on 138
deduplicated (market, start) tapes ([json](slice_v3_20261024.json)).

| Check | Result |
|---|---|
| (1) Single generation (POST at 0 then KEEP; POST at 0 then PULL at the first review) | **PASS on all 256.** JOIN price, activation 58 ms, the pull's request at 100 / effect at 137 ms (or an earlier structural request), fills per trade time, END mark, cash, inventory and **both** END values all equal the archive (0 false in 13 predicates) ([orders](slice_v3_20261024_orders.csv)) |
| (2) C-compatibility aggregate parity | **PASS** (shares, cash, position, END value) |
| (3) Adaptive invariants: POST_KEEP and CHASE on both sides jointly; POST / WAIT at 0 ms and KEEP / PULL at 200 ms per side | **PASS.** 276 base runs and 1,066 forks with status ok. 0 invariant refusals. 0 incremental / batch fill-tuple mismatches (base runs and every arm). 0 pre-fork differences. 0 value-identity failures. 0 causality failures: views, reservations, requests and decisions up to 200 ms are equal on the tape truncated after 200 ms ([forks](slice_v3_20261024_forks.csv)) |

**(4) Coverage on real tape:**
- **Entry forks:** ok on all 138 tapes, per policy and side. Their WAIT arm posts later on every tape. With these always-post stand-ins, that later post is the next tick, 50 ms later; it is not a model-chosen delay.
- **Retention forks:** ok on 119 / 127 (chase, bid / ask) and 132 / 136 (post_keep). The remaining anchors keep their status `not_keep_state`, where the order was not live at 200 ms.
- **Re-entries after a PULL** occurred on 116 / 124 and 127 / 132 of those forks.

Over the 276 base runs:

| Base-run counter (side-ticks) | chase | post_keep |
|---|---:|---:|
| bid / ask absent, cap-blocked | 3,817 / 3,496 | 4,037 / 3,044 |
| bid / ask absent, feasible, POST | 457 / 462 | 264 / 245 |
| bid / ask cancel_pending | 725 / 700 | 398 / 302 |
| bid / ask live, PULL | 220 / 257 | 0 / 0 |
| bid / ask absent, near-END | 3 / 3 | 0 / 2 |

The cap with full-size posts makes a large share of absent time infeasible, as declared in review (d). This is an environment property, not
a prediction failure. The fork values are invariant-check by-products of stand-in policies, not policy results. Branch histories (fills,
commands, requests, END exposure) are in [slice_v3_20261024_histories.json.gz](slice_v3_20261024_histories.json.gz).

## The R1 gate (plan v2 section 4, with review 043ca403's additions)

| Requirement | Status |
|---|---|
| All synthetic controls pass, including the unnoticed-fill refusal and the common-fork continuation test | Met |
| Executable WAIT and re-entry | Met, on synthetic and real tapes |
| Versioned `paired_continuation_v2` with named timing fields | Met |
| Traceable lineage | Met |
| No unexplained accounting or causality mismatch | Met |
| The fixed bounded regression / adaptive slice | Met: (1), (2) and (3) PASS, (4) published |

**Next, per review 043ca403:** R2, the supported remaining-horizon pi0-50 teacher and target construction. It starts once this receipt has
been reviewed.

## Limits

- **Costs are not in R_T** (fees, rebates). They must be added before R4.
- **Stand-in policies only.** Timing scenario B only. One instrument per episode. 30-share JOIN generations.
- **Real-tape coverage comes from always-post stand-ins:** later posts are one tick late, and re-entries follow every PULL's acknowledgement.
- **Synthetic and slice results are implementation evidence, not profitability evidence.**

## Files

- **Module and mutation harness:** hashes in [sources.sha256](sources.sha256).
- **Slice v3 outputs:** the JSON, the orders and forks CSVs, and the gzipped histories (the original's hash is in sources.sha256).
- **Logs:** [logs/](logs/).
- **Data:** `data/predict_fun/research/branch_engine_20261119`.
