# LONG-CONTINUATION-PAIRED-VALUE-v1 — declaration (frozen before any pair)

2026-11-22, session aea5bfb9.
- **Specified by:** review `32d102bd`'s ONE next priority, built to [contract_v2](../ladder_long_continuation_contract_20261122/contract_v2.md).
- **Approval:** the USER approved **≤ 0.40 additional core-hours** (controls and the 0.03 h review reserve included) and a **ladder line ceiling of 2.85**.
- **Builder:** `predictfun/research/pf_ladder_long_continuation.py`. Its docstring is the full rule.

**What this is:** a paired diagnostic of keeping versus rebuilding a queue under one declared continuation. It is not a strategy, a validation, a skill claim or a profitability claim. Consumed Oct 24–27 pilot markets only: no new day, fit or untouched validation. POSITIVE-PRESSURE-GUARD-v1 stays stopped.

## Population

CURRENT_ASYNC_FAST10 on the 8 pilot markets (the first two complete per date), from ASYNC-EXECUTION-SENSITIVITY-v1's hash-bound pilot records.

| Kind | State |
|---|---|
| REVIEW | per generation, its first **policy-consulted** recorded live KEEP (reason `keep`; a fallback KEEP never consulted the policy, so no action could replace it) |
| ADMISSION | every recorded POST |

**Selection:** per market × side × kind, the **2** states with the smallest sha256(arm|day|m|kind|side|gid), where ADMISSION adds |t. That gives up to 64 pairs.
- The full census (`census.csv`, its sha256 in `manifest.json`) records every eligible state, its key, its cell size and its draw probability.
- No sampling by any outcome or headroom class. The kinds stay separate; a generation in both is declared.

## Fork and continuation (contract_v2 §1, §3)

**Roll-in:** one per market, with the frozen ASYNC client (`ProfileBranch` + the frozen `PressureGuard` through `LineageGuard`, inert until armed — selftest group 1).

**Fork:** the tick-start state is copied (branch, policy and observer together). The ONE forced action is applied at the identified callback, after that tick's trades, notices, structural requests and any earlier side or order. The context at the callback is fingerprinted. Both arms continue to the same market END.

**Holding rule, the same in BOTH arms after the fork:**
- Every live order of the focal lineage is exempt from `pressure_pull`. The lineage is (side, birth rung): the focal gid, plus every gid submitted into that slot at or after the fork.
- Pressure-withholding, structural / inventory removals, the cap, sizes, notices and price locks stay active.
- Other slots follow the pinned policy with their own arm's feedback.

**Pairs:** REVIEW KEEP vs PULL (effective +10 ms); ADMISSION POST vs WAIT.

## Value and labels (contract_v2 §1, §4)

**Value:** Q = Σ post-fork s·q·(M_END − p) + I_fork·(M_END − M_fork), reconciled per arm to value_END − cash_fork − I_fork·M_fork. Y = Q_retain − Q_alternative.

**Economic axis:** `retain_better` (Y > 1e-6 c) / `alternative_better` (Y < −1e-6 c) / `tie`. Invalid statuses (`callback_not_reached`, `gate_failed`) are reported apart.

**Sequence axis (each arm):** END-signed. "Early" is the focal generation's first post-fork fill group; "later" is the lineage fills after it. Classes: `no_focal_fill` / `early_loss_later_benefit` / `early_loss_no_later_benefit` / `no_early_loss_later_benefit` / `no_early_loss_no_later_benefit`.

**Mechanism:** `retention_beneficial_with_early_loss` / `retention_beneficial_without_early_loss` / `alternative_better` / `tie`. **Queue dependence: `unidentified`** — the pilot does not separate queue loss from absence, price, size or other-order feedback.

**Decomposition:**
- focal lineage vs other orders;
- execution edge vs inventory movement;
- turnover: buy / sell shares, FIFO matched volume and holding time;
- inventory: time-weighted, peak and terminal |I|;
- participation;
- requests, reposts, pressure pulls, suppressed pulls, post-to-pull reversals within 100 ms;
- replacement delay and activation queue;
- 1 / 5 / 30 / 60 s markouts (native and common cohort; diagnostics only).

## Gates (computed; a failure is a status, and the reading becomes `not_reliable`)

1. **Base parity:** each market's roll-in, completed to END, equals the saved record (fingerprint, fills, orders, requests, value, guard log, command array).
2. **Unused exemption:** every retain arm whose exemption suppressed nothing equals the saved record.
3. **Exactly once:** the forced action is applied once per arm, at contexts equal in both arms.
4. **Pre-fork identity:** pre-fork fills are identical.
5. **Reconciliation:** each arm's value reconciles.
6. **Queue reset:** every lineage order's activation queue equals the shown size recomputed from its activation frame.
7. **No lineage pull:** no `pressure_pull` hits a lineage order after the fork, in either arm.
8. **Invariants:** the inherited per-command invariants hold (a violation raises).

## Reading (declared)

- **`reliable`** = every pair `ok` and every market's base parity holds.
- **`varied`** = among ok pairs, both `retain_better` and `alternative_better` reach ≥ 10 % of pairs, and Σ|Y| ≥ 1,000 c.

| Reading | Condition | What it permits |
|---|---|---|
| `reliable_and_varied` | reliable and varied | a later causal-discrimination design (no fit, no promotion) |
| `reliable_not_varied` | reliable, not varied | no tuned variant |
| `not_reliable` | otherwise | no tuned variant |

Point estimates only: 4 days, under 5 complete validation days.

## Run and resources

**Parts:**
1. **subset:** Oct 24's two pilot markets, 16 pairs.
2. **Write-once projection:** this batch's spent + 3 × the subset's CPU for the 6 remaining markets + a 60 s tail + the 0.03 reserve must be ≤ 0.40. The line + those must be ≤ 2.85, and the peak < 1,536 MiB.
3. **rest:** Oct 25 / 26 / 27, two markets each, run in parallel on `claude2..claude4`, only if the projection fits.

If it does not fit, stop. Never shrink the sample or change a rule after seeing economics.

**Ledger:** the line includes the headroom receipt's logs, plus its declared 30 s allowance, and the review `32d102bd`'s logs.

## Verification before the freeze

- **Selftest:** 6 groups:
  - the inert guard;
  - run_market gates on a synthetic market;
  - known-bad controls:
    - an exemption missing in one arm;
    - inherited replacement priority;
    - a reused arm state;
    - a wrong command cursor;
    - double-counted carried inventory;
    - a retain arm unequal to the saved record;
  - future-only perturbation;
  - the pure parts;
  - the guards.
- **Mutation:** 35 / 35 planted errors caught (`logs/mutants_try2`).
- **Checker:** `lc_checks.py`, families A / B / Y / G / P / S / I, recomputing from the saved ledgers and the ASYNC pilot records; a 12-case falsifier.
- **Rehearsal:** `tools/rehearse_lc.py` ran the builder's census, manifest, run_part and measure, then every checker family, on a synthetic day. All families passed except `A_committed`, and all 12 cases were refused.

## Multiplicity

There is no candidate. It is one diagnostic with two kinds (REVIEW, ADMISSION), reported separately and pooled.
