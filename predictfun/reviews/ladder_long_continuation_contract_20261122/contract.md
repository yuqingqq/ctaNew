# LONG-CONTINUATION-PAIRED-VALUE-v1 — contract (design only; no run, no budget)

2026-11-22, session aea5bfb9.
- **Governed by:** the USER's directions `90e82f8e` (whole-life queue retention), `c590b083` (longer-continuation value; `reviews/automatic/20261122_long_horizon_value_contract/plan.md`) and `60768c83` (two-sided turnover / inventory addendum).
- **Context:** written after FEASIBLE-CANCEL-HEADROOM-v1, whose oracle located the +1 s loss in a minority of generations at their first review. The same cut is worth far less at END.

**What this file is:** the contract, its controls and the pilot cost. Implementation, any replay and the pilot's resources need a frozen builder and a **USER resource decision** first. Nothing here is launched.

## 1. Question and estimand

At a recorded decision state *s* = (market, tick *k*, side, generation or slot), compare two actions on the **same pre-action state**, both followed to the **same supported market END**. For action *a*:

    Q_a(s) = (net_cash_END^a + I_END^a · mark_END) − (net_cash_k + I_k · mark_k)

**Paired value:** Y(s) = Q_retain − Q_alternative. This is the marked full-book ledger:
- both sides and every order;
- every partial fill;
- re-entries on their own identities;
- the same terminal mark in both arms.

It is not settlement profit and carries **no separate queue bonus**: queue value enters only through the arms' different fills.

**Costs:** net cash already includes any declared execution cost, counted once. The engine has no fees today, so the convention is recorded with value 0.

## 2. Decision populations (sampling manifest frozen before any arm runs)

| Kind | State | Retain arm | Alternative arm |
|---|---|---|---|
| **REVIEW** | A recorded live review of a CURRENT_ASYNC_FAST10 generation (the policy returned KEEP), including its **first** review (40 ms after activation, where the oracle cut 90 % of its headroom) | KEEP: the generation stays resting under the declared continuation (§3); it keeps its priority through partial fills | PULL at *k* (effect +10 ms); the ladder re-enters by its own slot rule. The replacement is a NEW generation with its real activation queue, timing, size and reservation |
| **ADMISSION** | A recorded slot decision where the policy POSTed | POST as recorded, then the same resting continuation | WAIT at *k*; the slot re-enters by the ladder's own rule at a later tick |

**Sample:**
- Per market, side and kind, the declared number of states with the smallest sha256(arm|day|m|kind|side|t|gid): outcome-independent, never chosen by hindsight class.
- Every sampled state keeps a status (`ok`, `not_live`, `post_not_feasible`, `missing_mark`, `incomplete`); none are dropped.
- The FEASIBLE-CANCEL-HEADROOM-v1 class of each sampled REVIEW state may be attached afterwards as a descriptor only. It is never used for selection or as a label.

**Days and markets:** consumed Oct 24–27 only, the pilot's 8 markets (first two complete per date). Any extension needs a separate decision. Untouched validation needs the USER.

## 3. Continuation contract (frozen before labels)

**Base policy:** both arms run the frozen CURRENT_ASYNC_FAST10 client: the ASYNC_FAST10 profile, the desired-levels slot model, LADDER_REST_INVENTORY_v1 sizing and inventory rules, the 200-share cap and hysteresis, structural removals, post-only refusals and notices. Other orders follow that policy in each arm and respond to their own arm's fills, inventory and reservations.

**The one intervention:**
- In the **retain** arm, the focal generation is exempt from `pressure_pull` for the rest of its life. It still ends by a structural removal, an inventory pull, a full fill or END, with the termination cause recorded.
- Without that exemption, KEEP would be a 50 ms deviation handed back to a rule that pulls it (the `c590b083` warning).
- In the **alternative** arm nothing is exempt.

**Sensitivity:** a second continuation, CONT-REST (the guard disabled for every order in both arms), is declared but runs only if the pilot cost allows.

**No other rule changes:** no holding timer, cooldown or forced re-entry. A serious new risk still acts through the inventory and structural rules.

## 4. Engine work (new builder; parity first)

**Forking:**
- Roll in with `pf_branch_ladder.run_episode(B, pol, until=k)` on a `ProfileBranch` + `PressureGuard` + `QueueObs`.
- Deep-copy the tuple (B, policy, observer) together.
- At tick *k*, apply the forced action to the focal generation or slot (a `Forced` wrapper consulted only at *k*, plus the exemption flag in the retain arm).
- Continue each copy with `run_episode(k0 = k)` to END, then `finish`.

This mirrors `pf_branch_engine.pair` / `fork_state` (paired_continuation_v2), extended from one generation per side and a 5 s END to the ladder and the market END.

**Required before any label** (each with a falsifier):
1. **No-op parity:** KEEP-vs-KEEP and POST-vs-POST pairs give bit-identical ledgers, and a fork with no intervention reproduces the recorded ASYNC record from *k* (fingerprint, fills, orders, requests, guard log, command array).
2. **Engaged-but-never-triggered exemption:** an exempt arm on a generation the guard never pulls equals the plain recorded path (the off-parity lesson of 2026-10-15).
3. **Pre-fork identity:** identical fills, notices and state up to *k* in both arms.
4. **Queue reset:** the alternative arm's re-entry is a new gid with its own activation ahead. A planted defect that copies the retain arm's later fills, or its priority, into the alternative must be refused.
5. **Ledger:** cash + I · mark = Σ per-gid fill values per arm; the per-gid decomposition sums to Y; position / reservation / desired-levels invariants hold every tick in both arms.
6. **Causality:** decisions read only client-known state. Moving a future arrival time must leave everything before it unchanged (the arrival-time falsifier).

## 5. Outputs per pair, per day / side and pooled (counts and statuses always)

1. **Value:** Y, and Q of each arm, decomposed exactly into execution edge (Σ signed q · (mid_fill − price)) and inventory movement (I_k · ΔM + Σ signed q · (M_END − mid_fill)). The focal generation's own fills and those of every other order are reported apart.
2. **Two-sided turnover:**
   - buy / sell filled shares and FIFO-matched volume, with starting inventory and remaining lots stated, reconciled to the ledger;
   - time-weighted |I|, peak and terminal I, lot holding times, time constrained by the cap.
3. **Queue and churn:** activation ahead of the focal generation and of any re-entry, re-entry delay and price, priority retained or lost, cancels / reposts, order lifetimes, post-to-pull reversals with their inputs.
4. **Horizons:** the same pair's value at 5 / 30 / 60 s where supported, on declared common cohorts, as diagnostics only. The END value is primary.

**USER categories, conditional on the declared alternative** (never "necessary" in general):

| Category | Definition |
|---|---|
| compensated queue continuity | Y > tol, and the retain arm's focal generation took early adverse fills before its later good ones |
| avoidable early loss | Y < −tol, and the alternative avoided the focal early loss while its re-entry kept enough later value |
| uncompensated exposure | Y < −tol otherwise, split into "later benefit too small" and "no later benefit" |
| unidentified / censored | \|Y\| ≤ tol, a missing mark, an incomplete continuation, or a difference not attributable to the focal order |

Fewer than 5 complete days: point estimates, no interval.

## 6. Pilot cost projection (measured inputs; nothing launched)

**Measured input:** a full-market ASYNC ladder arm costs 10.5–14.9 CPU s (ASYNC receipt logs: pilot 81.4–89.1 s per 6 arm-markets; rest 81.6–104.6 s per 6–10). Peak about 1.2 GiB per job, under the 1,536 MiB checkpoint.

**Cost formula:** one roll-in per market and kind (shared by its sorted forks), plus per pair 2 × (remaining fraction) × T_full. With the mean remaining fraction about 0.5, that is about T_full ≈ 13 s per pair.

| Pilot (8 markets, REVIEW + ADMISSION) | Pairs | Projected CPU | + 25 % margin |
|---|---:|---:|---:|
| 2 states per market × side × kind | 64 | 64 × 13 + 16 × 13 s ≈ 0.29 h | 0.36 h |
| 4 per cell | 128 | ≈ 0.52 h | 0.65 h |
| 8 per cell | 256 | ≈ 0.98 h | 1.23 h |

Add parity / falsifier tests (synthetic, ≈ 0.01 h) and a 0.03 h review reserve.

**The line does not cover it.** It stands at about 2.42 h after FEASIBLE-CANCEL-HEADROOM-v1, so 2.50 leaves about 0.06 h. **Any pilot needs a USER resource decision:** a new cap for the paired-continuation line and its size. The smallest informative pilot is 2 per cell, about 0.36 h with margin. Memory stays within the existing caps (one job per slot, at most 4 slots).

## 7. Gates after the pilot (not authorized here)

- The pilot measures label variation and engine validity, not model skill.
- A predictor may follow only if paired labels show both signs with material value. It must use client-known inputs, chronological evaluation, decision-value ranking against hold / re-entry baselines on matched states and separate sides, and must price the costs of false cancels.
- Good first-fill or +1 s forecasts do not meet this gate.
- POSITIVE-PRESSURE-GUARD-v1 stays stopped.
