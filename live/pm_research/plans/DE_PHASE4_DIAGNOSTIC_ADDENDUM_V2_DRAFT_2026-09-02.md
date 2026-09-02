# Phase-4 diagnostic — addendum v2, **DRAFT PROPOSAL** (not frozen, not in force)

**STATUS: PROPOSAL. Nothing here is frozen and nothing in the code may cite
it as authority.** It is written for the USER to rule on (rule 14: a seat
drafts; the USER decides). Addendum v1
(`DE_PHASE4_DIAGNOSTIC_ADDENDUM_2026-09-02.md`) and the frozen protocol
(`DE_PHASE4_PROTOCOL_DRAFT.md`) are **NOT edited** by this document — a
correction is a new dated document, never an edit of a frozen one (rule 13).

**Why it exists.** The estimand review (`REVIEW_DE_ESTIMAND_2026-09-02.md`)
settled four questions from the frozen lines and left three that the frozen
text does not answer. Two of them are **numbers I chose while building the
runner**, which is exactly what a seat may not settle for itself; the third
is a property of the control the frozen text is silent on. They are put here
together so the USER rules once.

---

## 1. The horizon the cell's number actually has (EST-R2)

**Proposed declaration:** *every Phase-4 diagnostic cell's value is computed
over `[t + L, end of the generation's hold]`, where `t` is the decision row's
time and `L` the cell's latency rung.*

**Why this and not the 1-second cap.** `DRAFT:68` prescribes
**generation-level tranche tables** as the feed and attaches the 1 s
declaration to a cell built on the **per-row latency labels** — the other
feed. The runner's number comes from the prescribed feed, so the 1 s claim
in v1's receipt named an estimand the number does not have. The code no
longer declares it: the binding field now reads `value_horizon = "[t + L,
end of hold]"` and the 1 s figure travels only beside the per-row table it
belongs to. **No new number is introduced by this declaration** — it names
what the frozen feed already produces.

## 2. `theta_repost` — a number I chose (EST-R3)

**What it is:** the score below which a cancelled generation may be
reposted. `harmful_stateful_policy.validate_params` REFUSES
`theta_repost >= theta_cancel`, so *some* value must exist for the policy to
load; `STATEFUL_HARMFUL_CANCEL_TODO.md:381-382` fixes the inequality and
requires a **declared** dwell, and no source of record fixes the value.

**What the runner does today:** `theta_repost = theta_cancel / 2`. That is
mine, and it is in the code only because the policy will not load without a
value.

**Proposed instead — a declared sensitivity pair, selecting neither:**

| rung | value | why |
|---|---|---|
| tight | `theta_cancel − ε` (ε = 1e-9) | repost as soon as the score falls at all: the least hysteresis the policy admits |
| loose | `0.5 × theta_cancel` | the runner's current value, kept as the other end rather than as the answer |

Both are reported for the PRIMARY cell; **neither is selected**, and the
ladder rule applies (nobody picks by looking at the results).

## 3. `REPOST_DWELL_S` — the second number I chose (EST-R3)

**What it is:** how long a repost waits after the cancel becomes effective.
The TODO requires a declared dwell and does not fix one; the runner uses
**2.0 s**, which is mine.

**Proposed instead:** the same shape — **0.5 s and 2.0 s**, both reported for
the PRIMARY cell, neither selected. The lower rung is the smallest dwell that
is still longer than the largest latency rung (250 ms), so the two are not
measuring the same thing twice.

## 4. `max_cancels_per_minute` (EST-R4)

**Proposed declaration:** `inf`, **per cell**, with the frozen reporting
identity carried per arm: `cancels_requested = cancels_rate_passed +
cancels_suppressed_rate_limited`.

`DRAFT:71` names the rate limit and asks for a per-cell declaration; `inf` is
a declared value rather than an absent one, and with the identity reported a
reader can see that no cancellation was suppressed rather than take it on
trust. The runner now carries all three counters per arm and evaluates the
identity in code.

## 5. Repost parity for the matched control (EST-R5)

**Proposed declaration:** *the acting control reposts on the same hysteresis
as the treated arm* — its score stream carries, for each drawn generation, a
crossing at that generation's own decision time and a below-threshold event
one dwell later, exactly as the treated arm's stream would.

The frozen text requires the control to be matched on action count, side and
hour (`DRAFT:147-156`) and is **silent on reposting**. Silence is not
permission to differ: a control that cancels and never reposts is a different
policy from the one under test, and the difference flatters or punishes the
treatment depending on the sign of the repost's value. This is a property,
not a number.

---

## What ruling this document asks for

1. Adopt the horizon declaration in §1 (no new number).
2. Rule the sensitivity pairs in §2 and §3 — or fix single values, in which
   case they are the USER's numbers and not mine.
3. Adopt `inf` with the identity in §4.
4. Adopt repost parity in §5.

**Until that ruling, the runner keeps the values it has and this document is
cited by nothing.** The code's own comments say the same: they name these as
proposals with their reasons, and no receipt field references this file.
