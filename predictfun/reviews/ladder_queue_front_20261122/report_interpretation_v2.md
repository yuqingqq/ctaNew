# QUEUE-PRIORITY-SENSITIVITY-v1 — superseding interpretation (v2)

2026-11-22, session aea5bfb9. This supersedes the causal wording of `report.md` (target `c7404f21`), per review `b8ab077a` finding 1.
- The numbers, checks and computed readings are unchanged and accepted: **economic MIXED, reading `mixed`, every arm negative on every day.**
- `report.md` stays as provenance; this file governs its interpretation.

## What is superseded

| `report.md` | Was | Now |
|---|---|---|
| line 54 | "the improvement is mostly that front orders fill out at the same price before sweeps reach them" | The volume / value split is an **arithmetic decomposition at CURRENT's per-share rates**, not an identified causal component. It cannot separate earlier depletion from changed sizes, prices, later generations or inventory restrictions. |
| line 56 | the 1 / 5 / 30 s class averages as a trajectory | They are **native-horizon averages** on different cohorts. FRONT same-price eligible shares are 85,088 / 84,257 / 78,789 at 1 / 5 / 30 s (review `class_horizon_denominators.csv`), not the path of one cohort. |
| line 58 | the END class flip | Numerically correct, but **not a diagnosis of inventory management**. Each class's END marks its fills to one terminal mark, and opposite inventory from the other class offsets it. At policy level the identity below holds, and most of the deficit is already present at +1 s. No inventory ablation was run. |
| line 76 | "not supported on this evidence" | **"Not supported by this test of the fixed policy."** |
| lines 80 and 90 | "queue priority only changes same-price fills"; "priority does not address either" (sweeps / inventory) | Only the *direct queue deduction in one match* is restricted to same-price prints (`pf_branch_engine.py:167-173`). Through the feedback loop, priority also changes later sweep and inventory exposure: eligible sweep shares fall from 41,632 to 31,905, and sweep +1 s value improves from −86,932 to −82,694 c. |

The END identity at policy level:

| Arm | Eligible +1 s | Movement to END | Late / unsupported +1 s | END |
|---|---:|---:|---:|---:|
| CURRENT_QUEUE | −85,164.28 | −1,416.84 | +471.51 | −86,109.61 |
| FRONT_AT_ACTIVATION | −61,812.39 | −5,175.38 | +1,050.68 | −65,937.09 |

## The governing interpretation

Front priority improves the pooled fill mix but does not make this policy profitable at the tested execution settings, and the effects are mixed by day. It directly changes same-price allocation and indirectly changes later sweep and inventory exposure. The remaining 1-second loss is concentrated in through-price fills. This test does not establish whether that residual is preventable by faster actions, earlier information, pricing or inventory decisions.

What follows from this:
- No tuned queue variant follows this mixed result.
- The test is not a monetary upper bound.
- It does not reject queue acquisition in general.
- The pooled 23.4 % deficit reduction is not an estimate that queue explains 23.4 % of all causes of loss. It is a complete-policy counterfactual under fixed external flow and the existing execution, sizing and inventory assumptions.

## Deferred (review finding 2, nonblocking; bounded workaround done by the review)

`fq_checks_v2.py` does not compare:
- the 5 / 30 s class values and censoring;
- the lost / added economic sums (it trusts the stored identity gap);
- the per-day sign and participation / inventory readings.

Family F checks the classes against saved trade metadata, not a separately loaded tape. The review checked all those values on the actual exports; they are correct.

When this checker is next reused, extend it with the review's three known-bad controls:
- +99,999 c on a class's `mo30s`;
- +99,999 c on `added_end`;
- FRONT's Oct 24 absolute sign set nonnegative.

## Process disclosure

The USER's two timing amendments landed on the local branch while this round was being implemented, before its freeze, and were not read before it:
- `9f7fe703` (03:40Z): the lower-latency FAST10 four-arm proposal;
- `c3fe1d4a` (03:49:01Z): the ASYNC_FAST10 concurrent-execution contract.

The freeze `b4ecbc34` (03:49:18Z) implemented exactly the two-arm design the USER approved through the session's question, with its 0.25 / 2.25 caps. Review `b8ab077a` confirms the FAST10 / async cases were neither implemented nor tested here. They are its ONE next priority, which needs the USER's resource approval before launch.
