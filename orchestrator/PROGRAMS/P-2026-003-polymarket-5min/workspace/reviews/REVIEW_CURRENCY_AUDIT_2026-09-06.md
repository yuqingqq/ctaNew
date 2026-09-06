# CURRENCY AUDIT — 40 flags drawn at seed 20260906: 3 STALE, 0 UNSUPPORTED, 1 not settled — **and 27 of 40 cite nothing at all**

**Filed** 2026-09-06T04:52Z · reviewer seat (pm-codex) · tip `3cf48c9` · no data
touched · nothing sealed opened · **STATUS.yml not edited** — this list is routed
to MEM as corrections.

## THE DRAW, DECLARED BEFORE IT WAS TAKEN

```
population : every key in STATUS.yml:flags, EXCLUDING NOTHING
N          : 619
order      : sorted(flags.keys())            -- deterministic, independent of YAML order
seed       : 20260906
rule       : random.Random(20260906).sample(sorted(flags.keys()), 40)
n          : 40, without replacement
```

Reproducible from the seed alone against the tip's STATUS.yml.

## THE THREE COUNTS

| verdict | n |
|---|---|
| **CURRENT** — the citation resolves and supports the claim as written | **36** |
| **STALE** — true when written, overtaken by a later artifact I name | **3** |
| **UNSUPPORTED** — a named artifact does not resolve | **0** |
| *(not settled — needs a run I did not do)* | *1* |

**And the number that matters more than all three: 27 of 40 (67.5%) carry NO
citation of any kind.** Their entire value is a verdict string — `LANDED`,
`LIVE-RISK`, `two`, `20`, `ARMED-TRIPLE-PRE-REGISTERED`, `active-enabled`. **They
are CURRENT only in the sense that nothing contradicts them, because there is
nothing to check them against.**

## MECHANICAL RESOLUTION (computed over all 40)

```
artifacts cited     3   ABSENT 1   <- and the absence IS the claim
R-entries cited    11   not in register 0
file:line cited     2   past EOF 0
commits resolved    9   NOT ancestor 3   <- and the non-ancestry IS the claim
```

**Every unresolved token in the sample belongs to a flag whose claim is precisely
that the thing is missing or not an ancestor** (`arms53.json` gone, `b43a9ce` not
on the branch). So the resolution failures are all correct reporting, not rot.

## THE THREE STALE

**1. `CODOMAIN_PREDICATE_REACHED_INDEPENDENTLY_BY_DA_round_35`** — claims *"I RAN
da_race_withdrawals --selftest MYSELF AND ALL 52 CHECKS PASSED (EXPECTED_CHECKS 52
at :59)"*. **`da_race_withdrawals.py:59` now reads `EXPECTED_CHECKS = 66`.** The
line number is still right and the value is not. Overtaken by DA's own growth of
that battery (52 → 57 → 59 → 64 → 66 across rounds 35–57). **The finding stands;
the verification is of a battery that no longer exists at that size.**

**2. `era_ruling_does_not_produce_g3`** — closes with *"G REMAINS 2 OF 5 AND THE
EARLIEST G=5 IS UNCHANGED AT 2026-09-06."* **G is 5.** 09-05 accrued and the race
reached G = 5; overtaken by R-531/R-532 and by BE 44's sealed 09-05 receipt. The
flag's substance (the era ruling did not grow the race *at the time*) was true;
its closing sentence is now false as written.

**3. `de39_r1_open`** — cites *"evaluate_predicates :770-798, verified at the
blob"*. **`evaluate_predicates` is at `de_phase4_diag_runner.py:1085`; lines
770–798 are now an unrelated comment about DE37-C2's known-bad.** The claim may
still hold at the new location — **but the citation no longer reaches the code it
is about**, and "verified at the blob" is exactly the phrase that makes a moved
line invisible: the blob it was verified against is not the file a reader opens.

## THE ONE NOT SETTLED

**`section_8_1_arm_state_4_of_7_measured_by_mem`** — I confirmed `arm_runnability`
exists (in `de_lane4_real_parity.py`, `de_lane4_results_doc.py`,
`da_arm_replay_verify.py`), so the instrument resolves. **I did not re-derive the
4-of-7 count**, and the arms have been restructured twice since 09-04 (cascade
ruling v2, per-arm baselines). **I will not call it stale without naming the
artifact that overtook it**, so I report it unsettled rather than guess.

## SPOT-CHECKS THAT CAME BACK CURRENT

`phase4_run_when`'s *"declared OUTDIR only, and it still does not exist"* — I
resolved `OUTDIR` and it does not exist ✓. `lgbm_freeze_receipt: LANDED` — three
matching receipts at the ledger ✓. `v41_collector_live` — `collect_pm.py` is
running ✓. `evaluation_timer: active-enabled` — timers present ✓.
`USER_RULING_the_day_set_is_the_UNTOUCHED_set: G-IS-6-FIXED-09-03-TO-09-08` —
matches the params file I verified two rounds ago ✓. And four flags I had already
driven myself in earlier rounds (`de73b…FIXTURE_RUN_IS_PROVEN_DATA_FREE`,
`ruled_run_result_invariant_and_immaterial`, `de68_design_no_arm_can_clear_holm_at_G5`,
`de66_vN_plus_1_verified_at_the_blob`) are current against my own drives.

## THE FAILURE CLASSES

**The dominant class is not staleness — it is unauditability by construction.**
Two thirds of the sample records a *verdict* where an *address* belongs:
`LANDED`, `LIVE-RISK`, `two`, `20`. A flag of that shape can never be found wrong,
which is the same property as never being found right; MEM's "455 of 619 never
audited" is not a backlog so much as a description of what most flags are. The
second class is **line-and-count drift in citations that were genuinely driven**:
all three STALE entries were *verified by execution when written* and were overtaken
by the instrument growing — `EXPECTED_CHECKS 52 → 66`, `:770-798 → :1085` — which
is the strongest possible argument for citing a **digest or a symbol name** rather
than a line number, because the honest, driven flags are exactly the ones that rot.
The third class is smaller and healthier than I expected: **relayed-not-read is
largely absent from this sample** — the long flags are conspicuously first-person
(*"MEM DROVE the predicate INSTEAD OF READING the date"*, *"verified by ME at the
artifact"*, *"reproduced AS A DEFECT by an out-of-module replay"*), and several
name their own instrument's gap (*"mem_flag_provenance.py CHECKS THAT A CHECKED
FLAG'S ARTIFACT EXISTS AND NEVER THAT IT IS THE RIGHT ONE"*). **The 09-05
"not-yet-verified-at-both-copies" pattern does not appear in this draw at all.**
What the audit actually exposes is an asymmetry: **the flags that carry the most
evidence are the ones that go stale, and the flags that carry none cannot.**

## ROUTED TO MEM

1. `CODOMAIN_PREDICATE_REACHED_INDEPENDENTLY_BY_DA_round_35` — 52 → 66 at `:59`.
2. `era_ruling_does_not_produce_g3` — the closing "G REMAINS 2 OF 5" is now false.
3. `de39_r1_open` — `:770-798` no longer reaches `evaluate_predicates` (`:1085`).
4. `section_8_1_arm_state_4_of_7_measured_by_mem` — re-derive or mark unverified.
5. **Structural, and the one worth a rule:** a flag whose value is a verdict string
   with no address cannot be audited. **If the currency audit is to mean anything,
   new flags should carry an artifact, a symbol, or an R-entry — and line numbers
   should be replaced by symbol names, because the three that rotted rotted on
   line drift alone.**

---

## CONTEXT

Far below the 80% reset threshold.

---

## APPENDIX — the 40 drawn, in draw order

  1. `lgbm_freeze_receipt`
  2. `evaluation_timer`
  3. `repo_constant_four_symptoms`
  4. `v2_freeze_was_never_an_open_user_item`
  5. `USER_RULING_the_day_set_is_the_UNTOUCHED_set`
  6. `MEM_provenance_note_the_arm_numbers_are_not_reproducible_yet`
  7. `v41_collector_live`
  8. `section_8_1_arm_state_4_of_7_measured_by_mem`
  9. `de66_vN_plus_1_verified_at_the_blob`
 10. `waiters_do_not_survive_the_turn_that_arms_them`
 11. `v5_heartbeat_deploy`
 12. `CODOMAIN_PREDICATE_REACHED_INDEPENDENTLY_BY_DA_round_35`
 13. `PROVENANCE_POINTERS_REPOINTED_BY_IDENTITY_and_the_instrument_gap_it_exposes`
 14. `da_independent_verifier_and_its_two_findings`
 15. `de68_design_no_arm_can_clear_holm_at_G5`
 16. `era_ruling_does_not_produce_g3`
 17. `btc_gap_degradation_ongoing`
 18. `ask5_per_answer_cost`
 19. `arm_output_NOT_USABLE_and_the_reason_changed_TWICE`
 20. `PROVENANCE_DEFECT_the_economic_artifact_names_a_RED_WIP_COMMIT`
 21. `de34_r7_ruling`
 22. `de38_c1_ruled_four_parts`
 23. `phase4_run_when`
 24. `loss_mechanisms`
 25. `co_13_open_be_r8`
 26. `TERMINAL_IN_A_GAP_IS_NOT_AVAILABLE_coordinator_ruled`
 27. `hostload_regression_control_reproduces_the_old_defect`
 28. `race_next_decision_point`
 29. `STANDING_RULE_no_number_reaches_the_user_before_review`
 30. `ruled_locator_is_load_bearing_now`
 31. `de73b_THE_20_HOUR_BLOCKER_IS_DEFUSED_AND_THE_FIXTURE_RUN_IS_PROVEN_DATA_FREE`
 32. `Q4_READ_THE_CONJUNCTS_NOT_THE_DETAIL_STRING_when_matched_random_lands`
 33. `tape5_gate_original`
 34. `ruled_run_result_invariant_and_immaterial`
 35. `r542e_ruled_on_a_relayed_description_of_my_own_file`
 36. `da18_r1_closed_at_the_rc`
 37. `governing_todo`
 38. `de39_r1_open`
 39. `contracts_version`
 40. `the_unseal_is_held_until_all_five_days_are_sealed`
