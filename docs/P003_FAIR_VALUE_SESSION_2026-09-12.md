# P-2026-003 — cancellation forward test closed; fair-value lane built and blocked

**Session 2026-09-11T23:22Z → 2026-09-12T03:15Z. Five seats (BE/DA/DE/REV/MEM).**
Full audit trail: `orchestrator/PROGRAMS/P-2026-003-polymarket-5min/workspace/COORDINATION.md`
**R-928..R-940**; rules in `workspace/COORDINATOR_RUNBOOK.md` **§7k–§7m**;
reviews `workspace/reviews/REVIEW_246..277`. Canonical ref: `origin/mm-research`.

---

## 1. Cancellation forward test — CLOSED, verdict unchanged

Four days valued. Both arms futile; `STOP_ADVICE: STOP_FOR_FUTILITY` computed in
the artifact, not printed beside it.

| day | CONDVALUE D (cents) | HAZARD D (cents) |
|---|---|---|
| 09-07 | −14,645 | +4,925 |
| 09-08 | −49,304 | −23,978 |
| 09-09 | +10,296 | +10,836 |
| 09-10 | +1,741 | −6,575 |

No p below 0.43 on either arm on any day.

**Verified, not asserted:** across **all 30 record versions** of these four days
(09-07 ×10, 09-08 ×8, 09-09 ×8, 09-10 ×4) each arm has exactly **one** distinct
`observed_D_cents`. Checked with a positive control. Nothing the scheduled jobs
wrote changed a published number — established *by reach*: the day-valuation
module holds zero references to the tier data those jobs write, while the same
query finds 42 in the evaluation pipeline, so the zero is a real absence.

**The design carried one degree of freedom (REVIEW 247).** The test had exactly
one passing configuration — 7 of 7 — with no rung between `p = 0.015625` and
`0.125`. A single non-positive day ended an arm irrecoverably: CONDVALUE died
09-07, HAZARD 09-08. **Both were already unrecoverable when the day-two ruling
was made.** It could distinguish flawless from everything else and nothing finer.

---

## 2. Fair-value lane — built, rehearsed, and blocked

Six build gates green, verified by two instruments agreeing row by row. The whole
validation path was rehearsed **before** consuming any day, first on synthetic
inputs and then on real books. The rehearsal is what produced everything below.

### 2.1 The comparator is sound
`Identity` (PM executable-book midpoint) was suspected of being an artifact at
wide books — a 0.01/0.99 book has midpoint 0.50. **Refuted by the reviewer
against its own hypothesis:** artifact share **0.17% BTC / 0.58% ETH over
9,464k `price_change` states**, spread one tick at p50 *and* p90, zero one-sided,
zero crossed.

### 2.2 C1 is a half-tick perturbation of the baseline
`|C1 − Identity| ≤ spread/2`, **exact and by construction** — C1 is a convex
combination of the same two prices `Identity` is the midpoint of. Measured
**0.005 at p90**. Perfect-foresight ceiling on per-day `delta_LL`:

| | states | mean | max |
|---|---|---|---|
| BTC | 2,544,510 | 0.0823 | 0.6931 |
| ETH | 808,372 | 0.0584 | 0.6931 |
| pooled | 4,973,558 | **0.0745** | 0.6931 |

Structurally capped at `ln 2`. Controls run *before* the measurement: artifact
book 0.683, tight book 0.00995.

### 2.3 **THE TEST HAS NO MAGNITUDE FLOOR** — the most consequential finding
The exact sign test has **no magnitude resolution at all.** A candidate positive
on ten of ten days by `1e-9` nats yields `p = 0.001953125` and passes, exactly as
one positive by 0.08 nats would. None of §8's four adoption conditions carries an
effect floor. **So C1 can pass on an effect of any size above zero and be carried
into the economic clock on something economically indistinguishable from nothing.**
The plan declares a minimum *sample* and no minimum *effect*.

### 2.4 C2 is structurally dead
Bound to a 60-second model; most generations live under a second.

| | |
|---|---|
| days reaching the 95% gate | **0 of 8** |
| coverage range | 0.177 – 0.271 |
| best day's shortfall | 0.679 |
| multiple required | **4.06×** |

The conclusion rests on the **swing** being an order of magnitude too small to
close, not merely on the level being low.

### 2.5 The population does not exist yet
No ETH day book existed for any day. ETH is **buildable and the data is healthy**
— `coin` is already a parameter reaching every path; only the CLI is BTC-only —
and DA confirmed inputs at parity (eth 2,303 = btc 2,303, BTC as positive
control). A coin-specific fallback is **closed by §2 of the plan**: per-coin
tables cannot authorise a coin-specific decision without a new family and new
multiplicity.

---

## 3. Feasibility of the 14-night programme

### 3.1 Serial by memory — parallelism closed by arithmetic
First ETH day measured (09-05, consumed):

| stage | ETH wall | BTC wall | time | ETH bytes | size | peak |
|---|---|---|---|---|---|---|
| fragment | 3m20s | 12.2m | 0.27× | 364,601,364 | 0.67× | — |
| tape | 11m22s | 25.5m | 0.45× | 588,868,676 | 0.66× | **0.97×** |
| book | — | 22.1m | — | — | — | NOT MEASURED |

**Three quantities, three different ratios — the stable one (size) does not
predict runtime.** ETH tape + BTC tape = 11.92 GiB. `research.slice MemoryHigh`
is **12 GiB exactly** (12,884,901,888 B), `MemoryMax` 14 GiB, per-unit 8 GiB.
So it *fits* the soft limit by **0.7%** — measurement noise on single
observations, and a soft limit **throttles rather than kills**, so "fits" means
"runs under sustained reclaim". **Do not plan to overlap.** Nightly cost is the
sum, not the max.

**A mechanism was proposed and withheld.** BE declined to promote *"memory is set
by the per-day index, not data volume"* to a rule: it rests on two stages of one
day of one coin pair and the mechanism is inferred from a count, not from reading
the allocator. **One prediction is not evidence.** A second coin or second day
settles it. The decision rests on the *measurement*, not the mechanism.

### 3.2 The band is expected to fall short
**`P(fewer than 10)` is the margin, not `E[evaluable] − 10`** — they diverge
exactly where the rate is uncertain.

| basis | rate | outcome |
|---|---|---|
| recent 8 at the gate | 0.875 | 12.25 expected, P(fail) 0.023 |
| recent 8 × ETH 95% | — | P(fail) ≈ 0.072 |
| joint 0.736 | — | fails ~3 times in 10 |
| **all 11 days, union of criteria** | **0.636** | **under 9 expected vs 10 required** |

**No exclusion of the early days is licensed.** The reviewer searched the
instrument, era and supply for a condition and found none, naming both its unread
residuals and *the direction its own bias would push*. So **0.636 is the planning
rate and 0.875 the optimistic bound**, in that order.

---

## 4. Decisions that are the user's alone

1. **`minimum_meaningful_delta_LL`** — a required, currently-unset field; the
   freeze computes `False` on it alone. Neither the reviewer nor the coordinator
   may choose it, having seen the ceiling. Choosing it after seeing would void it.
2. **Whether to commit to ~14 nights** of serial two-coin builds against a band
   expected to fall short.
3. **Whether C2 remains in the frozen family** given it cannot reach its gate.

**A sequencing guard is in place so (1) cannot backfire:** `two_coin_production_
ready` is a second blocking term (false until an ETH book is produced by the new
launcher *and* its byte-identical-BTC control passes). Without it, setting the
floor would start the 14-day band against days that are all `COINS_INCOMPLETE`,
and §8 forbids extending — a one-shot unrecoverable loss triggered by answering
the question we asked.

Validation start is ruled as a **rule, not a date**: the first complete UTC day
strictly after the freeze becomes effective, **and not before 2026-09-14**,
whichever is later — because 09-13 sits inside the cancellation test's declared
population and *already decided is not untouched*.

---

## 5. Open and blocking

**The day-verdict producer is failed.** `da-midnight-verify.service`,
`ExecMainStatus=7`, since 2026-09-12T00:06:00Z; next fire 2026-09-13T00:06:00Z.
`da_dayverdict_20260911` is placeholder-only; `20260912` has zero files. §8
resolves eligibility from candidate-blind inputs and **the day verdict is one of
them**, so days cannot be marked evaluable going into the band. Each further day
adds one.

**The fix is ten minutes and the diagnosis was corrected at the last moment
(REVIEW 277):** the deploy refuses on a "dirty" `live/pm_research`, but of the 70
files **63 are tracked at `origin/de-freeze-chain-v2`** and only 7 are new — they
are the lane's own declarations and modules. **The tree reads dirty because the
shared working tree sits on a diverged local branch (305 behind, 275 ahead of
`origin/mm-research`), so files tracked at the canonical and executing refs are
absent from *its* HEAD.** **DO NOT CLEAN THE TREE** — the "debris" is the lane's
content. The fix is to put the tree on the right ref. Regenerability is already
built: `days_needing_verdict` fires on the next successful run with a late
`as_of` and a catch-up reason, so 09-11 and 09-12 regenerate rather than staying
placeholders — to be verified by driving it, not by trusting the comment.

**Also open:** the ETH book (two named refusals in a chain of missing producer
receipts — tape's score-split receipt now fixed and verified, fragment's builder
pin outstanding, third step priced at 3m20s); 09-11 ruled *shown but not counted*
and never built; ~15 rules the session produced that are still unwritten.

**All coordinator authorisations were withdrawn at stand-down.** Nothing heavy is
running; the heavy lock is free. Six systemd timers remain active by decision —
DA ruled `n_consumers = 0`, and the finding is retrospective: killing a timer that
has already run saves nothing and may destroy the day-quality record.

---

## 6. Method notes worth keeping

- **A guard is only trustworthy once it has fired on its author.** Every guard
  that failed tonight had never been pointed at whoever built it — an
  anti-amendment guard green at 37/37 with **no call site at all**, a pin that
  was a copy of the thing it pinned, a fence beside an open gate.
- **Bind every input to an artifact, never an argument.** Ask who supplies a
  guard's inputs; if the answer is "whoever it constrains", it is not a guard.
- **A claim of action names the act.** Three times the coordinator treated saying
  as doing. The fix is to cite the dispatch/entry/verified count, not to grep
  one's own prose afterwards — that instrument was built, over-flagged, and was
  retired in place with its reasoning.
- **An exception log is silent exactly when things are healthy** — liveness must
  come from an unconditional heartbeat.
- **Presence is three states, not two.** A 1.2 KB file where the day's median is
  8,621 KB *exists* and carries nothing.
- **A pattern that explains every observation explains none.** The session's
  unifying frame was narrowed after testing against an unprimed population where
  it fit **zero** cases.
