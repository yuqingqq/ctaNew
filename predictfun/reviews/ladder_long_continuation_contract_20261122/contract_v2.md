# LONG-CONTINUATION-PAIRED-VALUE-v1 — contract v2 (design; supersedes contract.md)

2026-11-22, session aea5bfb9.
- **What changed:** review `32d102bd` findings 1–2 and its preparation / acceptance contract are folded in. `contract.md` stays as v1 provenance.
- **Status:** nothing is implemented, frozen or run. **It needs the USER's resource decision first:** ≤ 0.40 additional core-hours including controls and a 0.03 h review reserve, and the ladder line ceiling raised from 2.50 to 2.85.

## 1. Estimand

At a recorded decision state on CURRENT_ASYNC_FAST10, fork the **exact pre-command state** of the identified callback. That state comes after the tick's trades, notices and any earlier side's commands, with outstanding requests and unnotified fills preserved. Apply ONE forced action and continue both arms to the **same supported absolute market END**:

    Q_a = (cash_END^a − cash_fork) + I_END^a · M_END − I_fork · M_fork,    Y = Q_retain − Q_alternative

- The inventory carried at the fork enters both arms identically and cancels in Y. The incremental decomposition counts **post-fork fills only**.
- Engine inventory is used for accounting; client-known inventory for every policy input.
- No queue bonus; no fees today (convention recorded as 0).

## 2. Population and sampling (frozen census before any paired outcome)

**Arm and markets:** CURRENT_ASYNC_FAST10 on the 8 consumed pilot markets (the first two complete per date, Oct 24–27).

**REVIEW:**
- For each generation, its **first eligible recorded live KEEP**.
- Per market × side, the **two distinct gids** with the smallest sha256(arm|day|m|REVIEW|side|gid).

**ADMISSION:**
- Per market × side, the **two distinct recorded POST decisions** with the smallest sha256(arm|day|m|ADMISSION|side|t|gid).

**Sample:** up to **64 pairs** (8 markets × 2 sides × 2 kinds × 2).
- Short cells are enumerated, and every status kept: `ok`, `not_live`, `post_not_feasible`, `missing_mark`, `incomplete`.
- Before any pair runs, the freeze records the full eligible census, the keys, the draw probabilities and their hashes.

**Sampling rules:**
- No sampling by an oracle or headroom class. No tick-weighted pool, so long-lived generations do not repeat.
- The two kinds stay separate. A generation sampled by both is declared, not treated as independent.

## 3. Continuation: one rule in both arms (finding 1)

**Focal lineage:** the forked order's (side, birth rung) slot.
- It is defined causally.
- Every generation submitted into that slot after the fork gets its own new gid at submission. Priority and fills never transfer between gids.

**After the single forced action, in BOTH arms:**
- Every live order of the focal lineage is exempt from `pressure_pull`.
- Pressure-**withholding** at admission stays the pinned rule.
- Structural and inventory removals, the 200-share cap, sizes, notices, acknowledgements and price locks stay active.
- A replacement gets the same holding rule and its actual new queue.
- Every other slot follows the pinned pressure policy with its own arm's fills, inventory and reservations.

**What this tests:** the marginal value of retaining versus rebuilding a queue under this declared continuation. It is an explicit diagnostic continuation, not an approved strategy, and not evidence about globally optimal quoting. The optional CONT-REST arm is dropped.

**The pairs:**
- **REVIEW:** KEEP vs PULL at the callback; the PULL takes effect at +10 ms, and the slot re-enters under the common rule.
- **ADMISSION:** POST vs WAIT; after WAIT, the slot's own later entry rule applies.

## 4. Labels: two axes (finding 2)

**Economic axis (exhaustive):** `retain_better` (Y > 1e-6 c) / `alternative_better` (Y < −1e-6 c) / `tie` (otherwise). Invalid statuses are reported apart. A tie is a valid outcome, not censored data.

**Sequence / mechanism axis** (every pair, including no early loss, no later benefit and no fill):
- Signed per-fill contributions **at the common END**, s·q·(M_END − p), in time order, for the focal lineage in each arm. The early / later split is at the first post-fork fill time of the retain arm's focal generation, declared in the freeze.
- Fill-time and +1 s markouts are kept as separately named adverse-selection diagnostics. They are never used as long-horizon labels.

**Decomposition:**
- Y splits into focal-lineage and other-order contributions.
- Per arm: replacement delays, prices, queue ahead at activation, inventory-constraint time.

**Wording:**
- "Retention beneficial with an early loss" applies only where the retain arm's focal lineage shows an early END-negative fill and Y > tol.
- A claim of **queue-dependent** compensation needs evidence separating queue loss from absence, changed prices, sizes and other-order feedback. Otherwise the mechanism is `unidentified`, and the valid Y is kept.

## 5. Gates before any label (each with a known-bad control)

1. **No-op factual parity:** the fork without intervention reproduces the recorded ASYNC record from the callback. Same-action pair parity is bit-identical.
2. **Unused-exemption control:** an engaged exemption on a lineage the guard never pulls equals the plain path.
3. **Replacement-queue reset:** the replacement's queue ahead is set at its own activation.
4. **Clean slate per pair:** future-only perturbation; exact command application at the callback (a second-side / same-tick notice-and-command fixture); per-event reservation checks; full cash / inventory reconciliation per arm.
5. **Known-bad controls, each refused:**
   - an alternative-only pressure pull;
   - inherited replacement priority;
   - a reused arm state;
   - a wrong command cursor;
   - double-counted pre-fork inventory.

## 6. Outputs

**Primary:** paired full-book END Y and both absolute Q, per pair, per day / side and pooled, with counts and statuses.

**Explained by:**
- focal lineage vs other orders;
- execution edge vs inventory movement;
- matched FIFO turnover and lot holding times;
- time-weighted, peak and terminal inventory;
- queue reset;
- participation and churn.

**Diagnostics:** the same pairs at 5 / 30 / 60 s where supported, with common and native counts. Exhaustive economic classes are published even where the mechanism is unidentified. Four days: point estimates.

## 7. Cost and stop rules

**Cost run:** time a predeclared small subset first: the first REVIEW and ADMISSION pair of each of the 8 markets, inside the allocation. Then a write-once projection of the remaining pairs from the measured roll-in and continuation cost, plus tests and review.
- If it cannot fit, stop. Never shrink the sample or change a rule after seeing economics.
- The earlier 0.5-remaining-lifetime figure is only a planning assumption.

**Allocation:** v1's estimate is about 0.36 h for the paired pilot plus about 0.01 h of controls. The review proposes **≤ 0.40 h including controls and a 0.03 h review reserve**, with the line ceiling raised **2.50 → 2.85**. The accepted line is 2.4189 h (2.4389 h with the last review's reserve).

**What the pilot decides:** only whether this target is reliable and economically varied enough to justify later discrimination work. It shows no skill, starts no model fit and starts no tuned variant if thin. POSITIVE-PRESSURE-GUARD-v1 stays stopped.
