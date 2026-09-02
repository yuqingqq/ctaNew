# Iteration 011 — Phase-2 winner: the USER's adjudication of record

**STATUS: RULED BY USER 2026-09-02 (R-424). This document records a ruling; it
computes nothing and it edits nothing frozen.** The preregistration
(`ITER011_CONDITIONAL_VALUE_PREREGISTRATION.md`, frozen R-232, commit `3b71d3e`)
is not touched — rule 13; this is the in-band record §9.2/§9.3 call for.

## The ruling, verbatim

> "Proceed according to your recommendation" — the USER, 2026-09-02 ~11:49Z,
> in the coordinator's session, on the pending USER decisions carried in
> `workspace/HANDOFF.md` ("PENDING USER DECISIONS — six") with the coordinator's
> recommendation beside each. For R-408(2) the recommendation adopted was:

> *do not advance the composed candidate (§9.2 names this case); record Q1 as
> the surviving component of record; no race admission (§9.3); next population
> under the frozen prospective 2,000-draw declaration (A2). Arm of record if
> any: LGBM.*

## What is ruled

| item | ruling |
|---|---|
| the composed iteration-011 candidate (both arms) | **DOES NOT ADVANCE.** Prereg §9.2: Q4, the decision metric, fails; a candidate does not advance on the strength of its hazard head |
| surviving component of record | **Q1_arrival** — the hazard head. Recorded as a component, not a candidate |
| forward-race admission for this family | **NONE** (§9.3). Multiplicity of the race is unchanged by this ruling |
| the next iteration-011 population | runs under amendment **A2 as frozen (R-397)**: matched-random resolution at **2,000 draws**, one-sided; this family stays adjudicated at 500 with its floor disclosure |
| arm of record, if one is ever named | **composed_lgbm** |
| Phase-4 grids (DE, frozen protocol R-397) | **stay gated** — there is no Phase-2 winner to key them on |

## The artifact this ruling reads (verified at the artifact, 2026-09-02T11:53Z)

`data/pm_5min/derived/iter011_conditional_value_v1__coin_btc.json` —
188,119 B, `as_of` 2026-09-02T05:21:34Z, sha256 `ca311c8f24e37564…`,
`preregistration_commit` `3b71d3e`, `development_evidence.is_a_validation`
**false**.

| head | cells | status | Holm p (one-sided, 24-cell family) |
|---|---|---|---|
| Q1_arrival | 6 | OK | 0.0479 all six — survives the joint reading on both arms |
| Q2_sign | 6 | NO_INCUMBENT_COUNTERPART | — (its gate names an incumbent term that does not exist) |
| Q3_magnitudes | 6 | OK | 0.0479 — passes its OWN gate, which carries no incumbent term |
| **Q4_combined_ev** | 6 | GATE_PARTIALLY_EVALUATED | **0.1199 / 0.2499 / 0.3598 / 0.3598 / 0.3598 / 0.4463 — fails all six** |

`cells_by_status` = 12 OK + 6 NO_INCUMBENT_COUNTERPART + 6
GATE_PARTIALLY_EVALUATED, denominator 24. Q1 at the action unit (R-400):
lgbm 0.790 / 0.864 / 0.876 by collapse rule against row-level 0.830; linear
0.735 / 0.798 / 0.814 vs 0.773; the candidate beats the incumbent hazard head
(0.7139) under every unit and rule on both arms. Every surviving p sits at the
1/501 permutation floor, disclosed in the cells as a bound, not a measurement.

## What this ruling does not do

- It starts no run, no grid and no race. Phase-4 stays armed and gated (R-397 4).
- It re-scores nothing; the artifact above is development evidence (prereg 4).
- It does not rule the freeze disposition of `harmful_reduced_fine_candidate_v1`
  (R-421 §3) — a separate USER decision, still open.
