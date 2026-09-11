# REVIEW 169 — the three timer lanes are outcome-blind on the arms, one of them computes a P&L, and here is the per-day producer-family table

**REV 127, 2026-09-11T02:47:46Z** (clock read separately, before composing).
Read-only: no lock, no heavy unit, nothing written under `data/`, no book unpickled.
Tip when the work started `1cb7dbe`; tip at write time `0ec3d32` — **the programme moved
under me twice during this review** (DA 215/216 landed `da_forward_admissibility.py`; MEM 350
moved the population to **N=7, 09-07..09-13**). The table below therefore covers 09-13.

---

## 0. THE COORDINATOR'S REASONING IS RIGHT IN CONCLUSION AND WRONG IN FORM — SAID FIRST, AS ASKED

> "The lanes have ALREADY run on 09-07..09-09, so stopping saves nothing — the question is
> retrospective and binary."

**It is not binary.** The disjunction has a third branch. The lanes march **one day per night**:
the measurement batch commits day D−2 at ~00:26Z and the evaluation batch follows at ~01:2xZ
(measured, §4). So on the *outcome-reaching* horn, 09-07..09-09 are gone **and 09-10..09-13 are
not yet touched** — stopping would have saved four of seven days, not nothing. The retrospective
framing is only available *after* the answer is known.

**With the answer measured, the conclusion survives anyway** (§5): nothing in the three lanes
reaches an arm valuation or a score, so the remaining runs are as harmless as the ones already
made, and **I agree: do not stop the timers.** But one live choice remains open and belongs to
the coordinator, not to me: the **evaluation lane computes a realized maker P&L against
settlement** on every day it processes (§5.2). It is arm-agnostic and no committed code reads
it — but a seat has mined exactly that field at scale before (Q-DA-58, 1.68 M btc fills,
`maker_gross_cash`, routed into the STOP-MM-VIABLE dossier that reached the user). On the
current cadence it will produce that population for 09-10 (~09-12T01:2xZ), 09-11, 09-12 and
09-13 before the look. **Stopping `pm-evaluation-pipeline.timer` alone would prevent that and
would cost nothing the day-quality record depends on** — the quality record comes from the
measurement lane and the midnight verify, which are separate units. I am not recommending it;
I am naming it as a decision that is still live, because "stopping saves nothing" would have
closed it by assertion.

---

## 1. THE INSTRUMENT, AND ITS FALSIFIER DRIVEN IN BOTH DIRECTIONS (rule 15)

Three instruments, all scratch-only, all driven, none read:

| instrument | absolute path (scratch) | what it decides |
|---|---|---|
| day census | `/tmp/claude-1001/-home-yuqing-ctaNew/b69bbd0b-0be2-433c-9ac6-2be291c57ac9/scratchpad/day_census.py` | which artifact families are keyed to a day |
| producer roll-up | `…/scratchpad/rollup.py` | family → producer, and the outcome class |
| runtime read-set | `…/scratchpad/trace_open.py` (`sys.addaudithook`) | every path a producer actually opens |
| import closure | `…/scratchpad/import_closure.py` | every `live/pm_research` module reachable from an entry |

**Membership is the OPERATION, not a spelling** (REVIEW 122's correction, applied to myself): a
file belongs to day D if its **path** carries D in a target-day position, or its **content**
names D. A date token immediately followed by `T<hhmmss>` is a **production stamp**, never a
target day — that distinction is what separates "written on 09-09" from "about 09-09", and
without it every artifact emitted on a population day would have read as consuming it.

**Controls, each driven, each in both directions:**

| # | control | required | observed |
|---|---|---|---|
| C1 | fixture file `fake_scores_20260907_btc.json` | FLAGGED for 09-07 | flagged `1p/0c` |
| C2 | fixture `fake_sweep__20260907T113355Z.json` (**stamp only**) | NOT attributed to 09-07 | absent from the table |
| C3 | fixture `fake_both_2026-09-07_SEALED__20260909T010203Z.json` | 09-07 **yes**, 09-09 **no** | exactly that |
| C4 | prose-only mention of 09-08 | flagged as **content**, not path | `0p/1c` |
| C5 | a day with nothing (`2026-07-01`) | zero | zero |
| C6 | **known-bad: root that does not exist** | REFUSE | `CensusRefusal: ROOT_NOT_A_DIRECTORY` |
| C7 | **known-bad: `2026-09-32`** | REFUSE | `CensusRefusal: DAY_NOT_A_REAL_DATE` |
| C8 | **positive control on the REAL tree: 09-03** | must light up | **19 producer families, 11 of them ARM-OUTCOME** |
| C9 | audit hook vs a script that opens `be_daybook_20260903_btc.pkl` + its receipt | both logged | both logged |
| C10 | import closure of `de_point_estimate_day` (reaches the arms) | non-empty arm set | **79 modules, 49 arm/policy** |
| C11 | **known-bad: import closure of a module that does not exist** | REFUSE, never "0 modules" | `REFUSED ENTRY_MODULE_NOT_FOUND` |

C6, C7 and C11 were **added because the first cut of each instrument returned a silent zero on
a wrong input** — a zero from an instrument pointed at the wrong root would have read as
"untouched", which is the precise failure this review exists to prevent.

C8 is the falsifier the brief demanded: **the instrument fires on 09-03 before its silence
anywhere else means anything.** Every family the census found matched exactly one producer rule;
the roll-up refuses on an unmatched family and printed no refusal.

---

## 2. THE PER-DAY PRODUCER-FAMILY TABLE (the deliverable)

Path-keyed families only, rolled up to the producer. `X` = at least one artifact keyed to that
day. Class: **ARM OUTCOME** = a value of the thing under test; **P&L (non-arm)** = an outcome
quantity not conditioned on our arms; **LABEL** = the settlement winner as market metadata;
**QUALITY / QUALITY-JUDGEMENT** = a per-day or per-coin-day status; **INFRA** = tape, coverage,
price, resource.

| PRODUCER FAMILY | CLASS | 09-03 | 09-07 | 09-08 | 09-09 | 09-10 | 09-11 | 09-12 | 09-13 |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| BE daybook builder | ARM OUTCOME | X | **X** | . | . | . | . | . | . |
| BE gate1 fragment / exposure rows | ARM OUTCOME | X | **X** | . | . | . | . | . | . |
| DE decision ledger | ARM OUTCOME | X | **X** | . | . | . | . | . | . |
| DE gate1 day runner (SEALED) | ARM OUTCOME | X | **X** | . | . | . | . | . | . |
| DE point estimate | ARM OUTCOME | X | **X** | . | . | . | . | . | . |
| BE forward-day scorer | ARM OUTCOME | X | . | . | . | . | . | . | . |
| BE race read | ARM OUTCOME | X | . | . | . | . | . | . | . |
| DA book tier / pre-read / rehearsal | ARM OUTCOME | X | . | . | . | . | . | . | . |
| DE early read | ARM OUTCOME | X | . | . | . | . | . | . | . |
| DE settlement aggregate | ARM OUTCOME | X | . | . | . | . | . | . | . |
| DE asymmetry null | ARM OUTCOME | X | . | . | . | . | . | . | . |
| **evaluation lane — tier2 markout_events** | **P&L (non-arm)** | . | **X** | **X** | **X** | . | . | . | . |
| evaluation lane — tier2 calib_panel | LABEL | . | X | X | X | . | . | . | . |
| measurement lane — tier1 windows | LABEL | . | X | X | X | . | . | . | . |
| measurement lane — tier1 canary | QUALITY-JUDGEMENT | . | X | X | X | . | . | . | . |
| da-midnight-verify (day verdict) | QUALITY | X | X | X | X | X | X | . | . |
| da-midnight-verify (blackout mask) | QUALITY | X | X | X | X | X | . | . | . |
| BE gate1 state tape | INFRA | X | X | . | . | . | . | . | . |
| BE heavy-run recorder | INFRA | X | X | . | . | . | . | . | . |
| DE pe logs | INFRA | X | . | . | . | . | . | . | . |
| collect_pm (CLOB tape) | INFRA | X | X | X | X | X | X | . | . |
| collect_pm_prices | INFRA | X | X | X | X | X | X | . | . |
| measurement lane — tier1 quotes/trades/twap/coverage/health/runs/batches | INFRA | twap only | X | X | X | twap only | . | . | . |
| evaluation lane — tier2 runs | INFRA | . | X | X | X | . | . | . | . |

**COUNTS PER DAY**

| day | producer families | ARM-OUTCOME | reaching any P&L |
|---|---:|---:|---:|
| 2026-09-03 *(control, known consumed)* | 19 | **11** | 11 |
| **2026-09-07** | 22 | **5** | 6 |
| **2026-09-08** | 15 | **0** | 1 |
| **2026-09-09** | 15 | **0** | 1 |
| **2026-09-10** | 5 | 0 | 0 |
| **2026-09-11** | 3 | 0 | 0 |
| **2026-09-12** | 0 | 0 | 0 |
| **2026-09-13** | 0 | 0 | 0 |

*(09-11 is the open day; 09-12/09-13 have not started. `data/pm_5min` walked: 51,850 files,
3,571 content-read, 33 too large for the content arm — those 33 are carried by the path arm and
are listed in the census output.)*

---

## 3. BREAKING THE "8 OF 13" DOWN — AND WHAT I CANNOT REPRODUCE

**The 8 reproduces; the 13 does not, and I will not pretend otherwise.** The earlier figure was
spoken, never landed, and its grouping rule is unrecorded. Under the one grouping that yields 8
— **producers writing into `derived/`, counting the midnight verify's verdict-and-mask as one
producer** — 09-07's set is exactly these eight:

| # | family on 09-07 | artifact | reaches an OUTCOME? | what decides it |
|---|---|---|---|---|
| 1 | **BE daybook builder** | `be_daybook_20260907_btc.pkl` (314 MB) + `__L250ms` + receipts | **YES — arm** | the scored book; `asm["by_arm"]` is what every score is read from |
| 2 | **BE gate1 fragment / exposure rows** | `harmful_exposure_rows_v3_gate1_20260907_btc.json` (660 MB) + `.EMPTY_SCORE.json` + fragment receipt | **YES — arm** | decision-time exposure rows, the unit the arms score |
| 3 | **DE gate1 day runner** | `p003_de_gate1_day_run_20260907_SEALED__20260908T030451Z.json` | **YES — arm** | a SEALED day run: `per_day_sealed_artifacts`, `decision_populations`, `n_scored_rows` |
| 4 | **DE point estimate** | `p003_de_point_estimate_day_20260907_L250ms__20260908T122617Z.json` | **YES — arm** | `absolute.reconciliation.arm_total_minus_baseline_total`, `D_E0`, both arms' `trades_cash_flow_cents` — **finite numbers, `sealed: false`** |
| 5 | **DE decision ledger** | `p003_de_decision_ledger_20260907__*.jsonl.gz` (×2) | **YES — arm** | per-decision rows of the policy replay |
| 6 | BE gate1 state tape | `phase2_state_tape_gate1_20260907_btc.json` (1.08 GB) + receipt | **no — infrastructure** | the replay's input tape; carries no score |
| 7 | BE heavy-run recorder | `be_heavy_run_record_be104L250_20260907.jsonl` + stdout | **no — infrastructure** | rule-20 resource record (wall, RSS, five fields) |
| 8 | da-midnight-verify | `da_dayverdict_20260907.json` (+ superseded) and `da_blackout_mask_20260907.json` | **no — quality** | 318 keys censused; **zero outcome-class fields** (§5.3) |

**So of the eight, five reach an arm outcome and three do not.** Add the automated lanes and the
collectors and 09-07's full set is the 22 of §2 — the eight above plus tier1 (7 datasets),
tier2 (3), the CLOB tape and the price collector.

The number I cannot reproduce is the denominator: my universe of `derived/` producers that ever
key an artifact to a day is **15–16 depending on whether the verdict and the mask are one
producer or two**, not 13. Naming the gap rather than rounding to the remembered figure.

---

## 4. WHAT THE AUTOMATED LANES HAVE ACTUALLY DONE — FROM ARTIFACTS, NOT THE JOURNAL

**The journal is not the record and could not have answered this** (R-641, re-measured twice):
at 02:31Z the journal's oldest entry was `2026-09-10T19:48:01Z`; at 02:47Z it was
`2026-09-10T20:03:02Z` — **the window start advanced 15 minutes in 16 minutes.** Nothing before
last evening is in it. Every number below is from a file's own content and mtime.

| lane | day it committed | when (file mtime, UTC) |
|---|---|---|
| measurement (`tier1/batches/.../batch.json`, lanes `measurement` + `full`) | 2026-09-06 | 2026-09-08 00:26 / 01:11 |
| | 2026-09-07 | 2026-09-09 00:26 / 01:14 |
| | 2026-09-08 | 2026-09-10 00:26 / 01:22 |
| | **2026-09-09** | **2026-09-11 00:26 / 01:20** |
| evaluation (`tier2/runs/day=<D>/…/run.json`, `source_batch_lane: full`) | 2026-09-06 | 2026-09-08 01:14 |
| | 2026-09-07 | 2026-09-09 01:16 |
| | 2026-09-08 | 2026-09-10 01:27 |
| | **2026-09-09** | **2026-09-11 01:23** |
| da-midnight-verify (`derived/da_dayverdict_<D>.json`) | 09-06, **09-07, 09-08, 09-09, 09-10, 09-11** | nightly 00:06:2x–00:06:4x |

**Two corrections to the brief's list, both small and both in the same direction:** the midnight
verify has written a verdict for **09-07** as well (2026-09-08T00:06:26Z) and for **09-11**
(2026-09-11T00:06:41Z) — so **six of the seven population days already carry a day verdict**,
not four. The cadence is one day per night with a two-day lag; projected mechanically, 09-10
commits ~2026-09-12T00:26Z, 09-11 ~09-13, 09-12 ~09-14, 09-13 ~09-15.

**The scheduler enumeration is complete, not sampled** (the lesson that produced this review):
**six** systemd user timers, not three — `pm-measurement-pipeline` (hourly),
`pm-evaluation-pipeline` (chained `OnSuccess=` from measurement, not a wall clock),
`da-midnight-verify` (00:06Z), plus **`pm-lane-health`** (15 min) and **`pm-research-guard`**
(1 min), and `launchpadlib-cache-clean`. Two collector services run continuously
(`pm-collector-clob`, `pm-collector-prices`). The user crontab holds three jobs, **none** of
which touches `data/pm_5min` (convexity monthly retrain; two okxSolver scans). No system cron
unit and no `at` queue touches this programme. `pm-lane-health` and `pm-research-guard` were
checked and read **no** `derived/` path and **no** parquet value — lane-health counts partition
directories and reads the batch receipt's `r7_canary_amendment` for reporting.

---

## 5. AT THE CODE: DOES ANY OF IT REACH AN ARM VALUATION, A SCORE, OR A P&L?

### 5.1 `pm-measurement-pipeline` → tier1 — **NO**

Import closure by AST: **6 modules** — `measurement_batch`, `daily_pipeline`, `tier1_pipeline`,
`replay_canary`, `coverage_ledger`, `da_state`. **Zero** arm, policy, book, settlement or
scoring modules (the same predicate returns 49 for `de_point_estimate_day`, C10). Its outputs,
read at the parquet schema: `quotes` (top of book), `trades` (prints), `twap` (reference price),
`coverage` (admissibility metadata), `health`, `runs`/`batches`, `canary`.

One qualification, stated rather than buried: **`tier1/windows` carries `winner_up`** — the
settlement label, per market window, 288 rows per coin-day. That is the *label*, market-level and
arm-independent; it is the same fact the resolution feed publishes. It is not a valuation of
anything we do, and it has been present for every day since 08-20.

### 5.2 `pm-evaluation-pipeline` → tier2 — **NO arm valuation, NO score, but YES a P&L**

Import closure: **7 modules**, the same six plus itself. **Zero** arm/policy/book modules.

But `markout_events` is not a coverage dataset. Its schema, read at
`data/pm_5min/tier2/markout_events/day=2026-09-07/coin=btc/part-0.parquet`
(**77,469 rows, ×7 coins, for each of 09-07, 09-08, 09-09**), carries:

```
winner_up   outcome_up   maker_edge_per_share   maker_edge_cents   maker_gross_cash
```

— **one realized maker P&L per venue parent trade, valued against the settled outcome.**
`calib_panel` likewise carries `winner_up` beside the book row at each `r_s`.

What bounds it:
- it is the **venue's** parent trades, not our simulated maker fills — arm-agnostic,
  policy-agnostic, identical under every arm;
- the run record carries **no aggregate at all** — hashes, `status: COMPLETE`,
  `claim_status: DESCRIPTIVE_ARTIFACTS_ONLY`;
- **no committed code reads `markout_events` or `calib_panel`** — verified repo-wide across
  `.py`/`.sh`: `evaluation_pipeline.py` is the only file that names either, and it writes them.
  **Zero of the 673 `derived/` json/jsonl artifacts under 32 MB (subdirectories included)
  reference `maker_gross_cash` or `markout_events`.**

What does not bound it: **a seat has mined exactly this field before.** Q-DA-58 computed
drift concentration over `tier2/markout_events` `maker_gross_cash`, n = 1,683,558 btc fills,
days 08-20/21/22, and routed the result into the STOP-MM-VIABLE dossier that reached the user.
The reader was a seat, not a module, so "no consumer in the repo" understates the exposure.
**I found no such read on any population day** — the register's markout analyses name 08-20..22
only.

### 5.3 `da-midnight-verify` → the day verdict — **NO, and this one I drove**

I ran the production verifier on **09-03** under a `sys.addaudithook` recording every `open`,
writing to scratch (`--outdir`/`--out` under the scratchpad; `rc=0`, nothing written under
`data/`). **32,416 distinct paths opened.** The complete non-stdlib read-set:

```
data/pm_5min/raw/2026{0819..0903}/…        the collector's own tape (a day prefix)
data/pm_5min/markets.jsonl
data/pm_5min/collector_gaps.jsonl
data/pm_5min/collector_runs.jsonl
data/pm_5min/collector.log
data/pm_5min/derived/tape_density_v1.json
orchestrator/…/workspace/COORDINATION.md
its own two source files
```

**Zero books. Zero tier1. Zero tier2. Zero score, settlement or ledger artifacts.** The hook
demonstrably fires on such a read (C9). And the verdict it produced was key-censused: **318
keys, none of the outcome class** — coverage, gap series, tape density, era admission, content
liveness, race withdrawal.

**The finding attached to this one:** the day verifier's **import closure is 32 modules and 19 of
them are arm/policy modules** — `harmful_forward_scorer`, `policy_optimizer` and its three
variants, the six `adverse_move_*`, `placement_skew`, `skew_bound`, `edge_layer1`, `layer2_v1`,
`inventory_walk`, the two `flow_*`. Run to ground, both direct imports are innocent:
`harmful_forward_scorer` is imported by `da_governed_verdict_preflight` **only** for
`load_blackout_mask` (schema reuse, deliberately not re-implemented), and
`policy_optimizer_queue_realistic` is imported inside `verify_day` **only** as the holder of the
feature index — `fi.gaps_by_slug`, `fi.covered_slugs`, both coverage accessors.

So the outcome-blindness of the nightly quality tool is a **runtime property established by
drive, not a structural guarantee.** Contrast `da_forward_admissibility.py`, landed at `ba28469`
**during this review**, which does it structurally: an `ALLOWED` allowlist of two paths, a
`FORBIDDEN` regex that refuses anything matching `daybook|ledger|point_estimate|settle|asym|arm|
result|verdict|null|score|fill|tranche|p003_de_`, and falsifier cells that drive four known-bad
paths. **That is the right shape and the nightly verifier does not have it.**

### 5.4 The r7 canary amendment — a per-coin-day judgement, and the brief had half of it

Read at the batch receipts:

| day | reclassified | retained |
|---|---|---|
| 2026-09-07 | **`2026-09-07/eth`** | btc, sol, xrp, doge, bnb, hype |
| 2026-09-08 | *(none)* | all seven |
| 2026-09-09 | **`2026-09-09/doge`** | btc, eth, sol, xrp, bnb, hype |

**There are two reclassified population coin-days, not one** — the brief had 09-09/doge and not
09-07/eth. Establishing the relevance rather than assuming it:

1. **The test is btc-only** (`be_daybook_*_btc`, `phase2_state_tape_gate1_*_btc`), and **btc is
   RETAINED on all three processed days.** Both reclassifications are non-btc.
2. **The status is not an outcome.** `classify()` in `replay_canary.py` is the leak canary's own
   rule: it replays the day through a knowledge-time-truncated view and a deliberately leaky
   event-time twin and compares them. `r7_reclassified` is set when the two views **disagree on
   zero occasions with a measured zero delta** — an argument about *ordering*, about whether the
   truncation guard is wired. No arm, no policy, no valuation enters it.
3. **It gates nothing.** The only reader of `r7_canary_amendment` in the repo is
   `ops/pm_lane_health.py`, which reports it. Empirically: 09-07's batch is `status: COMPLETE`
   *with* eth reclassified, so a reclassification does not block a day, a coin or a lane.
4. **What it is, honestly:** a **per-coin-day quality judgement made automatically on population
   days**, carried in the batch receipt, pooled into a `drift_check` computed over every coin-day
   ≤ the target. If a day-quality decision were ever wired to read it, that decision would be
   reading a *replay* — one of the three input classes the declaration's own predicate forbids.
   Nothing reads it today. That is the whole of the exposure.

---

## 6. FINDINGS TO ROUTE

1. **The declaration's refusal name is a pin to nothing.**
   `da_forward_test_declaration_v2.json` → `POPULATION.a_failed_quality_day.refusal_if_violated`
   = `"QUALITY_DECISION_SAW_AN_OUTCOME"`. **That string appears in exactly three files in the
   repo (grep, whole tree, `.git` excluded): `da_forward_test_declaration_v1.json`,
   `..._v2.json`, and `da_forward_test.py` — the word-checker that reads them.** No code raises it.
   The instrument that *does* enforce the property, `da_forward_admissibility.py`, refuses under
   `ADMISSIBILITY_READ_A_NON_METADATA_SOURCE`. A reader resolving the declaration's field finds
   no producer. **Route: DA** — either the instrument raises the declared name, or a superseding
   declaration names the refusal the instrument actually raises. This is rule 16 in the
   declaration itself: the vocabulary matches and the identity does not.

2. **"No `arm_total_cents` anywhere" on 09-07 is true as a field name and misleading as a claim.**
   Verified at the artifact: `economic_settlement.status = NOT_VALUED_DAY_NOT_ADMISSIBLE`,
   `null_mean`/`null_sd`/`p_location`/`null_draws_summary.n` all
   `NULL_NOT_DRAWN_POINT_ESTIMATE_RUN` — **the declaration's estimand claim holds.** But the same
   artifact carries, `sealed: false`, for **both arms**: `absolute.arm.trades_cash_flow_cents`,
   `absolute.zero_cancel_baseline.trades_cash_flow_cents`,
   `absolute.reconciliation.arm_total_minus_baseline_total` and `reconciliation.D_E0` — all
   finite, all from "the same replay that computes D(E0)", present by USER ruling R-782 ("plz
   record the absolute number as well for reference"). **The settlement *leg* was not valued; the
   trades leg and its arm-minus-baseline difference were, and they are unsealed.** The
   declaration's own correlation argument — "the markout is CORRELATED with the settlement P&L,
   so 'never valued' is not 'unseen'" — applies at least as strongly to an arm-minus-baseline
   cash total. **Route: DA** — restate 09-07's status in those words in the superseding receipt.
   Nothing about the USER's inclusion ruling changes; the wording of what was conceded does.

3. **The nightly quality tool is outcome-blind by behaviour and not by construction.** 19
   arm/policy modules in a 32-module closure, against a sibling instrument landed the same night
   that restricts its inputs structurally. **Route: DA** — give `da_forward_day_verify` the
   `da_forward_admissibility` treatment (allowlist + forbidden-path guard + a falsifier that
   drives a book path through it), or at minimum an artifact-level guard that refuses to emit a
   verdict carrying a field outside the coverage vocabulary. Today the guarantee is my drive of
   one day, and a drive is not a guard.

4. **The evaluation lane's P&L is a standing decision, not a closed question** (§0, §5.2).
   **Route: coordinator.**

---

## 7. SCOPE, AND WHAT THIS DOES NOT CLOSE

- **Closed over:** every file under `data/pm_5min` (51,850 walked); both spellings of every day;
  the path arm and the content arm; the complete scheduler surface (systemd user + system timers,
  user crontab, system cron dirs, `at`); the import closure of all four relevant entry points;
  the runtime read-set of the midnight verifier on one real day.
- **Not closed over:** 33 text artifacts above 32 MB are carried by the **path arm only** — all
  33 are the multi-GB state tapes and exposure-row files, each of which already names its target
  day in its filename, so a content-only dependency inside one of them would be missed. Named
  rather than waved at.
- **Not closed over:** other data roots (`data/mm_hf`, `data/ml`) — out of this programme's tree.
- **The permanently unclosable residual stands unchanged** and this review does not touch it: a
  seat, or a lane, that **read** a day and **wrote nothing** consumed it invisibly. My audit-hook
  drive can establish the read-set of a process I run; it cannot establish the read-set of a
  process that ran last week. Every count above is a count of **writes**, and a pass on this
  population remains **attested, not guaranteed**.
