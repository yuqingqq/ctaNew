# REVIEW 244 — validation day one is 09-12, which is a cancellation-lane day

REV round 207. Filed 2026-09-11T21:03:31Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled. Fetched at 20:59Z. Both
executing refs `9ec30e9`.

## 0. The freeze has not landed

Counted at both fetched refs, matching `fair_value.*(freeze|frozen)` under
`live/pm_research/declarations/` and the programme directory:

    origin/de-freeze-chain-v2   0
    origin/be-build-runner      0

Item (1) is armed. §2 below is the answer that cannot be given afterwards, so
it comes first.

---

## 1. THE DAY, AND WHY IT IS A PROBLEM

### 1.1 The day

§8: *"Validation starts with the first complete UTC day strictly after the full
pipeline freeze."* A day is complete only if it both begins after the freeze and
ends. So:

| freeze lands | day one | 14-day band |
|---|---|---|
| any time on 2026-09-11 UTC | **2026-09-12** | 09-12 … 09-25 |
| any time on 2026-09-12 UTC | 2026-09-13 | 09-13 … 09-26 |
| any time on 2026-09-13 UTC | **2026-09-14** | 09-14 … 09-27 |

At filing time it is 2026-09-11T21:03Z, so on today's landing **day one is
2026-09-12** and the band is 09-12..09-25, within which the first ten evaluable
BTC+ETH days are the population.

### 1.2 Nothing in 09-07..09-13 "can be mistaken for it" — two of them ARE it

Read at the artifact, `da_forward_test_declaration_v27.json` (DA 261, landed
2026-09-11T12:14:17Z), field `population_unchanged`:

    N=7, 09-07..09-13

**So 2026-09-12 and 2026-09-13 are days six and seven of the cancellation
lane's live forward-test population, and under a freeze today they are also
fair-value validation days one and two.** They are not confusable — they are the
same two calendar days, being measured by two lanes at once. The same
declaration lists `still_to_come` including `20260913`, so those days are ahead
of the cancellation lane, not behind it.

The plan does not disclose this. §6 names *"the 08-24..27 settlement split and
09-03..09-09 P-003 populations"* as explicitly consumed, and says nothing about
09-10..09-13. A reader of the plan alone would not learn that the fair-value
clock is about to start inside another lane's live population.

### 1.3 Three concrete surfaces, measured

**(a) Namespace.** Both lanes key artifacts by `YYYYMMDD` in the same
`data/pm_5min/derived/` tree under the same `p003_` / `da_` / `be_` prefixes —
`p003_de_forward_value_20260907.json`, `da_dayverdict_<DAY>.json`,
`p003_de_point_estimate_day_<DAY>_L250ms__<stamp>.json`. There is at least one
live wildcard with no lane filter and no day filter:

    be_fragment_diagnostic.py:1389
        cands = sorted(Path(DERIVED).glob("da_*verdict*.json"))

A fair-value day verdict named anything matching `da_*verdict*` is swept into
the cancellation lane's diagnostic. `be_gate1_state_tape.py:81`
(`phase2_state_tape*.json`) and `be_gate1_fragment.py:107`
(`{PINNED_PREFIX}*.json`) are the same shape.

**(b) Scheduled outcome-bearing work, on a timer nobody dispatches.** This is
the check my own standing note says to run before accepting any untouched
verdict, and it is answerable now:

    pm-measurement-pipeline.timer   next 2026-09-11T21:21:50Z (hourly)
    pm-evaluation-pipeline.timer    next 2026-09-11T21:50:10Z
    da-midnight-verify.timer        next 2026-09-12T00:06:00Z

Both pipelines run `--catch-up --since 2026-08-20 --max-days 1 --scheduled`, and
both currently report `{"as_of_day": "2026-09-11", "status": "IDLE"}` — caught
up, one day per run. **They will therefore process 2026-09-12 automatically once
that day closes, with nobody dispatching them.**

The separating question is whether they reach an outcome or only
coverage/quality. `pm-evaluation-pipeline` is the "full-batch Tier-2 evaluation
pipeline"; it writes `data/pm_5min/tier2/{calib_panel, markout_events, runs}`
and computes `maker_edge_per_share` / `maker_edge_cents`
(`evaluation_pipeline.py:399-400`). **A maker-edge number and a calibration
panel are outcomes.** `da-midnight-verify` is day-quality verification only,
which is candidate-blind and is exactly what §8 wants as an eligibility input —
that one is fine and should keep running.

To be fair about what this does and does not mean: C1, C2, the sigma contract,
the quote mapping and m=2 are frozen by the freeze commit, so nobody can tune a
candidate on what the panel shows. The exposure is not tuning; it is that after
the fact nobody can demonstrate the 09-12 panel was not read, and §8's whole
design is that the validation days are untouched at the moment the clock starts.
That has to be declared before the clock starts or it cannot be declared at all.

**(c) Contention.** §6 requires that consumed-day smoke runs *"must not starve
an active heavy replay"*. The cancellation lane's heavy valuations for 09-12 and
09-13 are still to come and take the same `data/.heavy_run.lock` the two timers
take. Two lanes needing the heavy lock on the same two days is a scheduling
fact, not a methodology one, but it is the kind that costs a day when it bites.

### 1.4 My recommendation, once

**Land the freeze on 2026-09-13 rather than today.** Day one becomes 2026-09-14
and the band 09-14..09-27 contains no cancellation-lane day; the collision
disappears rather than being managed. The cost is two days of clock.

If the freeze must land today, then the declaration has to carry, as fields:
the two shared days named explicitly; the statement that the tier-2 timer will
reach them and what it computes; and a lane discriminator in every fair-value
artifact name that no existing glob matches. Managing the overlap is possible;
discovering it in the receipt is not.

---

## 2. The checklist I will apply when the declaration lands

Published now so it can be written against rather than audited after. From §7
unless noted. A field the plan requires and the declaration omits is a blocker,
and I will name it as one.

**The frozen chain, in order** (§7, one commit):
`immutable inputs → labels/statuses → actions → sigma → FairPrice → fallback →
score → quote mapping → replay → P&L` — all ten links present and in that order.

**The recorded fields** (§7): 1 all file hashes · 2 commit ref · 3 candidate
count · 4 action key · 5 epsilon · 6 status grammar · 7 source manifests ·
8 initial inventory · 9 tick rounding · 10 latency · 11 fee rule · 12 quote
parameters · 13 null predicate · 14 success predicate.

**m = 2 forever** (§4): the forward multiplicity remains 2 *even if one
candidate fails a development or availability gate* — a declaration that makes
m contingent on C1 or C2 surviving is wrong, not merely incomplete.

**Placement latency** (§7): `placement_latency_ms = 250`, for every new
generation in **both** legs; described as a **simulation assumption, never as a
measured live end-to-end latency**; the count of every fill removed before
`generation_start + 250 ms`; no challenger receives an instantaneous first
placement; harmful-flow cancellation disabled; ordinary quote replacement on one
shared, separately declared lifecycle in both legs.

**The quote mapping** (§7): UP uses `p`, DOWN uses `1-p`; bid rounds **down** and
ask rounds **up** to the legal tick; prices bounded to the legal binary range; a
would-cross quote emits `PLACE_WITHHELD` with reason `MARKETABLE_CROSS` and is
**never silently clamped**; a candidate-induced change follows the same
cancel/replace lifecycle with no zero-latency privilege; only the Identity
anchor is replaced, and on non-OK status Identity is used.

**R-924's abstention constraint** (`da_step6_abstention_constraint_v1.json`,
DA 276): the constraint reproduced **verbatim** beside the primary number — not
paraphrased, not referenced by declaration name — and the native-OK intersection
reported **alongside** the primary with the **gap computed in the receipt**.

**And the §7 clause that governs all of it:** *"Every verdict is computed from
artifact fields; no prose-only pass is permitted."*

On landing I will also verify the declaration as a count at both fetched refs
and re-drive the six gate rows against the frozen blobs — a freeze that pins
superseded bytes is the failure this programme has already had six times.

---

## 3. The sweep: what a cold reader would find hardest to defend

I chose the declarations, because they are what an outside reader opens first
and because the finding is mechanical rather than aesthetic.

**The count.** 214 declaration files at the chain ref for one seven-day
experiment:

| family | versions |
|---|---|
| `de_multiday_gate1_params` | 33 |
| `da_forward_test_declaration` | 27 |
| `de_arm_freeze` (+amendments) | 23 |
| `da_population_freeze` | 18 |
| `da_code_freeze_declaration` | 11 |
| `da_fair_value_progress_ledger` | 7 |

Rule 13 makes supersession the norm and I ruled at REVIEW 219 that
version-ordered supersession is correct, so the *mechanism* is right. But 27
versions of one forward-test declaration in five days is not supersession
working, it is a document being edited through a version counter, and every
reader and every resolver pays for the chain.

**The measurement that makes it more than taste.** Across those 214 files there
are **50,686 field names, 5,156 distinct**. Of the distinct names, **1,859
(36.1%) appear as a literal anywhere in the lane's Python** — the other 64% are
read by nobody. Narrow to field names that are ALL-CAPS sentences of 45
characters or more, and there are **232 distinct, of which 7 are read by any
code**. Examples, verbatim keys:

    VALUATION_SCOPE_AGAINST_THE_OLD_PIN_THE_PRE_AMENDMENT_STATE
    I_WITHDRAW_MY_OWN_MASK_HYPOTHESIS_IT_IS_MEASURED_FALSE
    A_TRANSIENT_I_AM_NOT_REPORTING_AS_A_HAZARD
    THE_RULE_11_TIMELINE_AS_TWO_FIELDS_BECAUSE_ONLY_ONE_HALF_DISCHARGES_IT

§7 says no prose-only pass is permitted and rule 13 says automated readers
resolve receipt **fields**. These declarations satisfy the letter by making the
prose *be* the field name: 225 sentence-length keys that no reader resolves, and
that change wording between versions so nothing could resolve them stably even
if it wanted to. A cold reader would reasonably ask whether the rigour is in the
instruments or in the prose about the instruments — and the honest answer is
that in the code it is genuinely in the instruments, which is exactly why it is
worth not looking otherwise.

**The cheap fix, and I am not asking for a rewrite of what exists.** For new
declarations: stable snake_case keys a resolver can name, with the argument in a
`note` value rather than in the key. The seven sentence-keys code actually reads
are the ones to keep and normalise.

**One smaller thing in the same sweep.** `systemctl --user list-units` shows
failed leftovers from finished work — `da83book0903c`, `da83book0904`,
`da83book0904b`, `p003fwd0907`, `p003fwd0907c`, `p003fwd0908` all sitting in
`failed`. None affects a result; all of them make the machine look like
something is broken when nothing is.

## 4. Owed

- Item (1) on landing: the count at both refs, the six rows re-driven against
  the frozen blobs, and the §7 checklist above applied field by field.
- Standing: one review per day for the 09-09..09-13 records (REVIEW 228) — which
  now overlaps the fair-value band and is a second reason to settle §1 first;
  the round-boundary landing sweep as counts at fetched refs.
