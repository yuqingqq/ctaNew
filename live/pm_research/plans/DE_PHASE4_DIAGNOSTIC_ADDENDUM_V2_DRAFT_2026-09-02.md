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

### 1a. What the null costs — one measured number, one floor

Addendum v1 §d said 200 draws at one cell is "of order six hours". That
priced a replay as LANE4's end-to-end pass, and it also missed that a draw
is not one replay: each runs `arm_result`, the PRIMARY conjunction, so
**200 draws is 800 replays** — with the (γ) rule, **plus the rejected
attempts**, which the receipt now counts.

| quantity | status |
|---|---|
| the FEED for the §3 population, both coins | **MEASURED: ~28.6 min, once** (round 33, on the real population: selection 192.1 s btc / 187.5 s eth, then 4.34 s and 1.35 s per window) |
| one `arm_result` (4 legs) | **UNMEASURED on real data.** The synthetic figure is **0.007 s per replay** on a fixture of **20 slugs × one generation × one tranche × one side** |
| 200 accepted draws at one cell | ≈ 6 s **by that projection**, plus rejected attempts |
| the FEATURE ASSEMBLY, once per run | **UNMEASURED, and it is not small**: a tape index over `phase2_state_tape_v5.json` (3,170,987,711 B) and `_feature_pass` over `harmful_exposure_rows_v3_eraB.json` (1,241,115,096 B, 1,135,943 rows). This is the fit's own per-run cost, paid again here |

**The synthetic figure is a LOWER BOUND, not an estimate.** The fixture has
one generation and one tranche per slug; the real population has many of
both, and the replay's work scales with **generations × tranches**, not with
slugs. Until a replay is measured on the real feed, the only defensible
statement is: *the feed is measured at ~28.6 min and the replay is unmeasured
with a floor of seconds*. The runner now publishes the feed's own
`n_generations` and `rows_per_generation` per cell so that the first real run
prices the replay instead of projecting it.

**A third cost was missing from both the v1 estimate and the round-36
version of this section: the run cannot score anything until the feature
assembly has run.** The heads take vectors, not timestamps, and those
vectors come from `phase2_arms._feature_pass` against the fit's tape — the
same 3.2 GB index and 1.14M-row pass the fit paid for. It is wired as far as
it can be without executing it (composition, per-generation statistic, and
every cheap precondition, all falsified), and the runner REFUSES before the
feed rather than discover it at the first cell.

**And it raises a question this seat cannot answer (rule 4).** The §3
population is 08-24/08-25, which spans **both** of the fit's splits — the
tape carries 1,125,289 `train` rows and 638,917 `score` rows across those two
days. Scoring the diagnostic's population therefore means scoring generations
the heads were **fitted on**. For a diagnostic of the policy's *mechanics*
that may be exactly right, and §3 already declares the population CONSUMED;
for anything read as evidence about the heads it is not. **This addendum
does not choose.** The run declares which splits it consumed, per cell, in
the receipt, and the choice is put to the USER with the two §5 numbers.

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

**DE35-R5 under the (γ) stream — re-read.** Yes, and now for a stronger
reason than in round 36's draft. Under (γ) the control's stream is a
permutation of the treated arm's **whole** stream: every below-threshold
event the head produced exists in both arms, at the same generation's `t0`,
and the above values are the head's own. A `theta_repost` at
`theta_cancel − ε` therefore admits the same reposts on both sides, and the
pair is readable as a comparison of policies rather than of constructions.
Under round 35's shape it was not readable at all — the control's only
below-threshold event was an invented literal `0.0`, which sits under every
candidate rung. **The pair is proposed only because §5 makes it readable; if
the USER declines §5, decline this pair with it.**

**And it is readable only because the below values stay put (§5, DE37-C3).**
Repost eligibility is a *dwell* condition on the below-threshold path, so if
the control's below values were permuted too, the tight rung would be
comparing two different below-value paths and the pair would measure the
construction as much as the policy. With them fixed at their own
generations, the only difference between the arms is the one (γ) is about.

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

## 5. The matched control's score stream (EST-R5, DE35-C1, ruled (γ))

**Proposed declaration, in the reviewer's ruling words:**

> The acting control's stream is the treated arm's stream with the
> above-threshold score values **permuted within `(side, hour)` strata**:
> one event per generation in both arms, the same score multiset per
> stratum, and the drawn generations carrying the above-threshold values.
> Because the policy is stateful, a permutation does not fix WHICH
> generations act: the control is therefore matched on the frozen decision
> variable — the **per-stratum realised action count** — with draws that
> fail the match **rejected and redrawn**, and the attempts, acceptances
> and rejections reported. No score value the head did not produce ever
> enters either stream.

**Why the earlier wording was false of any buildable stream.** The previous
draft said the control "cancels exactly the drawn generations". Measured,
that cannot be arranged: a HELD side suppresses later crossings, so a
non-acting above event acts the moment the generation holding it stops
cancelling, and the realised set is neither the treated set nor the drawn
one. The frozen text asks for matching on **action count** (`DRAFT:147-156`),
not on identity.

**The below-threshold values stay at their own generations.** Only the swap
moves: a generation that is not drawn and carried a below-threshold value
keeps **its own**, and the below values displaced by drawn generations go to
the non-drawn generations that carried above values — there are exactly as
many of each. Moving them would be a **second** difference between the arms,
which is what (γ) exists to remove, and it is not inert: repost eligibility
is "score < `theta_repost` continuously for `REPOST_DWELL_S`", so a moved
below value changes **when** a held side becomes repost-eligible — §2's
number meeting §5's stream. The assignment is by sorted key, so it carries
no value order.

**BUILT, not merely declared — and here is what says so.** Round 37's runner
demanded the draw on the treated arm's **action** count, so in any stratum
holding a non-acting above event the permutation could not be honoured; the
stream was built anyway and replayed. It now demands over **all
above-threshold events**, so `|drawn| == |above|` per stratum by
construction, and **P1–P3 are computed per draw before the replay**: a draw
whose stream fails one is REJECTED under that predicate's own name and
redrawn.

**The IDENTITY draw is ADMITTED, COUNTED, and it is what makes a cell
degenerate.** Under (γ) a draw is a choice of which |above| generations
carry the above values, so the draw naming exactly the above-carrying
generations — the identity — is one of the C(N, k) permutations, and a null
that excludes it is not the permutation distribution. The identity guard,
written for a control matched on *actions*, is therefore **retired for (γ)**
rather than re-pointed: measured over 200 seeds on the proving fixture, it
fired 0 times as written and would have fired 65 times if handed the demand
— refusing exactly the sample points the null must contain.

**What that exposed, and what this addendum now asks the USER to read.** On
the fixture that proves (γ), *every accepted draw was the identity*: one
generation per stratum leaves the identity as the only draw satisfying the
match, so the "null" was the treated arm and the difference against its
median was 0.0 by construction. The remedy is reporting, not a change to the
frozen matching rule (`DRAFT:147-156` fixes matching on action count, side
and hour and is silent on degeneracy — silence there is this addendum's to
fill):

- the statistics that describe the null are computed on the **accepted**
  set, not the attempted one: `n_distinct_accepted` beside
  `n_distinct_attempted`, each labelled with its population;
- **per stratum, before any §3 number**: the accepted set's size, its
  distinct count, `n_accepted_identity`, and whether it **collapsed**;
- an accepted set of **one** distinct draw publishes
  **`null: DEGENERATE`** with its reason and **no `null_quantiles` and no
  `net_diff_vs_null_median_cents`** — rule 6 declares ≥ 200 *draws*, and 200
  copies of one draw is one draw. The cell falls back to the **labelled
  point estimate** the addendum already declares for cells without an
  interval; rho and retention for that cell are unaffected.

**Reported with it (DE31-R2, DE37-C1, DE38-C1):** `n_strata`, `strata_with_room`,
`n_distinct_accepted`, `n_distinct_attempted`,
`n_accepted_identity_whole_draw`, `accepted_by_stratum`,
`n_draws_attempted`, `n_draws_accepted`,
`n_rejected_by_stratum`, **`n_rejected_by_reason` (`PERM_NOT_OK`, `P1`,
`P2`, `P3`, `P4`)**, **`first_rejection`**, **`predicates_per_draw`**, and an
explicit **POINT MASS** declaration where a stratum has no room. A reader
who wants to know whether (γ) held on this run reads
`n_rejected_by_reason`: stream defects and decision-variable mismatches are
counted apart, so a null whose rejections are all `P4` is a matched null,
and one with any `P1`–`P3` is a construction that failed. Exhausting the
attempt budget **refuses**: a null built from the draws that happened to
match is matched on acceptance, not on the decision variable.

## 6. The pin's own claim (DE37-C2/R1) — stated here because the package rests on it

Not a question for the USER; a statement of what the pin now asserts, so
that a reader of this document knows what "the fit code is the fit's" means
here. Three functions in `harmful_exposure_rows.py` differ from the
fit-commit bytes and are **declared additive** with their reasons. Each
declaration now carries **two literal AST shas** — at the fit commit and at
the tip that declared it (`cb8aab5`) — written into the source and compared
against the file the run finds.

Each entry also **names the commit that changed the function**
(`851edaf`), and that claim is checked at both sides of it: at that commit
the three functions carry the declared tip shas, and at its parent they
carry the declared fit shas. Prose cannot be pinned; the fact the prose is
about can be, so a re-declaration that moves the shas while keeping the old
justification fails. Two limits stay open by nature and are stated rather
than worked around: the literals live in the file that reads them, so a seat
editing code and declaration in one commit re-seals silently — that is
closed by review, not by code — and the tip half pins the **declaration**,
so every future edit to a declared function BLOCKS until re-declared, which
is the intended cost.

Round 37 computed the declaring-tip sha at import **from the file it was
checking**, so the comparison was true by construction: driven twice, an
edited `select_v2_era` still read ADDITIVE_DECLARED and the run PROCEEDED,
while an edited undeclared function BLOCKED. The three declared functions
were a permanent exemption. With literals, an edit to any of them re-opens
the file to **BLOCKING** — driven on an edited function body, not on a
tampered dict.

---

## What ruling this document asks for

1. Adopt the horizon declaration in §1 (no new number).
2. Rule the sensitivity pairs in §2 and §3 — or fix single values, in which
   case they are the USER's numbers and not mine.
3. Adopt `inf` with the identity in §4.
4. Adopt repost parity in §5.
5. **Rule the split question in §1a, with 2 and 4 rather than after them.**
   The §3 population spans **both** of the fit's splits, so every cell
   scores generations the heads were fitted on. That is not a footnote to
   the numbers above: §2's and §3's rungs, §5's null and the split are read
   against the same stream, and a cell's meaning changes with the answer.
   Either the diagnostic is declared a MECHANICS diagnostic on a consumed
   population (in which case the split is admissible and the receipt says
   so per cell), or the run is restricted to the `score` split (in which
   case the population is smaller and §3's counts change). **This seat does
   not choose between them** (rule 14); the run declares the splits it
   consumed either way.

**Until that ruling, the runner keeps the values it has and this document is
cited by nothing.** The code's own comments say the same: they name these as
proposals with their reasons, and no receipt field references this file.
