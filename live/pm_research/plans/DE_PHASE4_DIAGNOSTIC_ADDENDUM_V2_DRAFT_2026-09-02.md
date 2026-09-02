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

### 1a. What the null actually costs (EST-R2's other half, corrected)

Addendum v1 §d said 200 draws at one cell is "of order six hours", from
LANE4's 1,339.6 s single-arm replay. **That was wrong by construction**: a
null draw is not one replay. Each draw runs `arm_result`, which is the
PRIMARY conjunction — **2 protection modes × 2 repost-fill models = 4
replays** — so 200 draws is **800 replays**, not 200.

**Measured, not estimated** (471 synthetic windows, this runner, both
launchers cleared):

| quantity | measured |
|---|---|
| one `arm_result` (4 legs, 471 windows) | **0.03 s** → **0.007 s per replay** |
| one null draw | 4 replays |
| 200 draws at one cell | 800 replays ≈ **6 s** |
| one cell without a null (5 arms × 4 legs = 20 replays) | ≈ **0.15 s** |
| the whole grid (54 cells + 2 null cells) | **≈ 40 s of replay** |

**So the replay is not the cost — the FEED is** (~28.6 min once, measured in
round 33). The six-hour figure understated the per-draw work by 4× and
overstated the total by three orders of magnitude, because it priced the
replay as if it were LANE4's end-to-end pass. This paragraph is the
correction; v1 is not edited.

## 2. `theta_repost` — a number I chose (EST-R3, re-read under §5)

**What it is:** the score below which a cancelled generation may be
reposted. `harmful_stateful_policy.validate_params` REFUSES
`theta_repost >= theta_cancel`, so *some* value must exist for the policy to
load; `STATEFUL_HARMFUL_CANCEL_TODO.md:381-382` fixes the inequality and
requires a **declared** dwell, and no source of record fixes the value.

**What the runner does today:** `theta_repost = theta_cancel / 2`. That is
mine, and it is in the code only because the policy will not load without a
value.

**Proposed — a declared sensitivity pair, selecting neither:**

| rung | value | why |
|---|---|---|
| tight | `theta_cancel − ε` (ε = 1e-9) | repost as soon as the score falls at all: the least hysteresis the policy admits |
| loose | `0.5 × theta_cancel` | the runner's current value, kept as the other end rather than as the answer |

**DE35-R5 — does the tight rung move the control now?** Under §5 as revised,
**yes, and for both arms together.** The control's stream *is* the treated
arm's stream permuted, so every below-threshold event the head produced
exists in both arms; a `theta_repost` at `theta_cancel − ε` therefore admits
the same reposts on both sides and the pair is readable as a comparison.
Under round 35's shape it was not: the control's only below-threshold event
was an invented literal `0.0`, which sits under *every* candidate rung, so
the tight rung moved the treated arm and left the control unchanged and the
pair compared two different policies. **The pair is proposed only because
§5 makes it readable; if the USER declines §5, this pair should be declined
with it.**

## 3. `REPOST_DWELL_S` — the second number I chose (EST-R3, DE35-R4)

**What it is:** how long a repost waits after the cancel becomes effective.
The TODO requires a declared dwell and does not fix one; the runner uses
**2.0 s**, which is mine.

**DE35-R4, answered plainly: there is no rule that yields 0.5 s.** Round
35's draft said "the smallest dwell longer than 250 ms", which names no
number — every value above 0.25 s satisfies it. Two honest options, and I
propose the first:

- **`2 × the largest latency rung` = 0.5 s** — a *stated rule* rather than a
  taste: a repost that waits less than one round trip after the cancel is
  racing the cancel it follows. If the USER adopts this, 0.5 s is derived
  and not chosen.
- Otherwise **0.5 s is CHOSEN**, and should be labelled so in whatever the
  USER rules.

**Proposed pair, selecting neither:** **0.5 s** (by the rule above) and
**2.0 s** (the runner's current value).

## 4. `max_cancels_per_minute` (EST-R4)

**Proposed declaration:** `inf`, **per cell**, with the frozen reporting
identity carried per arm: `cancels_requested = cancels_rate_passed +
cancels_suppressed_rate_limited`.

`DRAFT:71` names the rate limit and asks for a per-cell declaration; `inf` is
a declared value rather than an absent one, and with the identity reported a
reader can see that no cancellation was suppressed rather than take it on
trust. The runner now carries all three counters per arm and evaluates the
identity in code.

## 5. The matched control's score stream (EST-R5, DE35-C1)

**Proposed declaration, in the reviewer's words:**

> The acting control's score stream **is** the treated arm's stream with the
> above-threshold assignments **permuted within `(side, hour)` strata**, so
> that the control cancels exactly the drawn generations and **no event
> exists in one arm that does not exist in the other**. Repost behaviour is
> therefore identical in kind to the treated arm's and is never
> manufactured; the control introduces no score value the head did not
> produce.

**What this replaces.** Round 35 emitted, per drawn generation, a literal
`1.0` at the generation's `t0` and a literal `0.0` one dwell later. That
manufactures a policy rather than permuting one: the `0.0` is a score the
head never produced, it sits below every candidate `theta_repost` (which is
why §2's pair was unreadable — DE35-R5), and it made the control's repost
behaviour a function of the DRAW where the treated arm's is a function of
the HEAD.

**This is a property, not a number**, and the runner already implements it:
the above-threshold score VALUES are reassigned to the drawn generations and
every below-threshold event is carried through untouched.

**Reported with it (DE31-R2):** `n_strata`, `strata_with_room`,
`n_distinct_draws`, and — where a stratum has no room — an explicit
**POINT MASS** declaration, because a forced draw contributes a constant
rather than a sample.

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
