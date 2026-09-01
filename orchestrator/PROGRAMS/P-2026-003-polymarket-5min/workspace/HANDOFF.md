# HANDOFF — P-2026-003 Polymarket Crypto 5-min

Updated: 2026-09-01T14:05Z — the pending CLAUDE.md amendment is drafted for the
USER (`workspace/DRAFT_CLAUDE_MD_AMENDMENT.md`), and the TODO sweep is at
47/113. Prior line: R-374..R-381 swept in — reviewer seat is a Claude session,
the resource rule is a mechanism, coordination is batched both directions, the
DE seat is staffed, and the 011 re-run was killed mid-fit rather than emit
under a superseded predicate.

## READ FIRST — current project handoff

The governing project TODO is
`live/pm_research/plans/HARMFUL_FILL_HAZARD_TOXICITY_PLAN.md` §10. The
stateful cancel x skew TODO is a subordinate implementation worksheet; its
39/113 checkbox count is not total project completion.

### Current model state

- Dataset/PRED_STATE_V1 is complete and reproduced. The Phase-2 development
  receipt exists; the separate BTC hazard seed is frozen but marked
  unvalidated.
- Iteration 011 conditional signed value is user-released. The earlier halt on
  alleged gap-contaminated queue state is withdrawn: the modeled replay resets
  and re-anchors after gaps, and the existing row exclusions fire. **The
  memory-sliced BTC run has since COMPLETED and written its declared artifact**
  — `data/pm_5min/derived/iter011_conditional_value_v1__coin_btc.json`, 96,707
  bytes, 11:23:34Z, all 24 declared cells present (`54f899d` → `e326782` →
  `0b1f6bb`). It is **development evidence**: the artifact computes
  `is_a_validation=false`, G=0 complete UTC days, clustered on window where the
  ruled unit is the UTC day. Q1 survives Holm at 0.0479; **Q4, the decision
  metric, is UNADJUDICATED** because the incumbent it declares comparable did
  not load, so no economic result exists; Q3's p is withheld pending
  implementation of R-306's already-ruled conjunction. The 09:43Z statement that
  the attempt "wrote no result artifact" described the stopped 09:34Z attempt
  and is superseded by the 12:56Z entry below; it stays as provenance.
  **As of 13:53Z the two blockers are implemented and the artifact is
  mid-rebuild.** `20d3c3a` wired Q4's incumbent and implemented R-306 for Q3;
  `4438961` folded in Q-DA-197's F2/F1/F5/F6 and **killed the re-run mid-fit
  rather than let it emit** under the superseded survivor predicate. So the
  declared path still holds `0b1f6bb`'s artifact, `__as_verified_by_Q-DA-197`
  preserves it byte-identically, a `__readjudicated_v2` exists from the
  intermediate state, and **no artifact under the closed predicate exists yet —
  Q4 still has no economic result.**
- Typed fair-price Identity is built; the challenger protocol is not
  freeze-ready and no challenger has been scored.
- `QR_SKEW_ONLY` semantics are user-frozen. Seven-arm work remains contracts,
  parity stubs and inert trajectories; bit-identical parity against a real
  seven-arm replay, lifecycle economics and the integrated candidate freeze
  remain open.

### Seats (as of 2026-09-01T13:53Z; R-373 reset, R-379 DE, R-381 no-idle)

All seat contexts were cleared at ~12:52Z on USER order and re-loaded from
files; anything assumed from pre-12:52Z conversation is void. **Six seats now,
and two of them are new today.**

| seat | round | working on |
|---|---|---|
| **BE** (pm-be) | **open** | Q4's incumbent and Q3's ruling are implemented (`20d3c3a`); the batch now also carries Q-DA-197's F2/F1/F5/F6 (`4438961`). Fits re-running in-slice; no artifact under the closed predicate exists yet |
| **DA** (pm-da) | **re-opened** | round 1 closed 3/3 (`c473b0e`). Round 2: the content-liveness rule as DRAFT-FOR-USER-FREEZE, the breadth-statistic reconciliation, and the real 09-01 verdict after 00:06Z — a HEALTHY fail is a recorded result, not a problem to fix |
| **DE** (pm-de) | **open, first day** | staffed by the USER 2026-09-01 (R-379). Owns `harmful_stateful_policy.py`, `de_actionspace.py`, `de_constraints.py` per R-165's parking clause. First batch: real-data seven-arm parity, the EV-Replay registry-closure draft, the Phase-4 grid protocol as DRAFT-FOR-USER-FREEZE |
| **Reviewer** (pm-codex) | **prep only** | now a CLAUDE session (R-375). Files nothing and holds nothing this round; baselining instruments at HEAD so round diffs are attributable. Its round opens on BE's pinned tip |
| **coordinator** (pm-co) | — | 0b and 0d closed at 13:02Z (R-374); the item-2 re-review still held until BE's batch commits, so one round covers `1aaac18`'s claims and today's batch at a single tip (rule 5) |
| **MEM** (pm-memory) | closing | this sweep of R-374..R-381 |

**BOTH observations below are now CLOSED; they stay as provenance.** The 0d
mirror gap closed at `8b47dff` — `pm-evaluation-pipeline` and
`pm-measurement-pipeline` are both byte-identical to their repo copies, verified
by `diff` at 13:53Z, so the guard is reproducible from git and not merely live.
The `n_actions` mislabel was independently found by DA as **Q-DA-197 F1** (the
cell carries the arrival n in 12 of 24 cells — a 22× overstatement of the
population behind the statistic) and rides BE's open batch; it was a disclosure
defect, not a lost measurement, since the head-level n was preserved all along.

**A SECOND observation, on 0d, verified at the two files and filed for the
coordinator.** R-374's closure is true of the LIVE unit — `systemctl show` reads
`Slice=research.slice`, `MemoryMax=16G`, and the 13:02Z positive control ran in
the `research.slice` cgroup. **But the edit exists only in the installed unit.**
`~/.config/systemd/user/pm-evaluation-pipeline.service` carries the `Slice=` line
and its eight-line comment; `live/pm_research/ops/pm-evaluation-pipeline.service`
in git does not, and the working tree is clean. That is R-361 LIVE-3's shape one
unit over: a reinstall from the repo silently drops this lane back out of the
aggregate guard, and nothing would say so. The guard is live; it is not
reproducible from the repository.

**One observation from the artifact sweep, filed for BE/the coordinator and NOT
adjudicated here.** The 011 artifact reports the action unit **correctly at head
level and inconsistently at arm level**: every head carries `n_actions` beside
`n_rows` (Q1 177,674 / 311,640; Q2 17,604 / 33,622; Q3_m_harm 7,988 / 15,912),
and the cells carry `n_actions` 177,674 — but
`results.btc.<arm>.n_actions` reads **311,640**, which is the ROW count from
`populations.btc.eval.n_rows`. The estimand looks right and one summary field
looks mislabelled, but rule 2 is the rule this sits on, so it should be read at
the code rather than accepted from this note.

**TODO true-up in the same sweep** (`STATEFUL_HARMFUL_CANCEL_TODO.md`, 39 → 42
ticked): §5.1 separately-observable heads (`0b1f6bb`, the artifact's own heads
block), §5.1 the three-head composition (`0b1f6bb`, with A1.1's amendment noted
because the code deliberately differs from the formula as printed), and §5.3 the
`QR_SKEW_ONLY` freeze (`908e8f0`). Nothing else ticked: Q4's comparison, the
BTC-and-ETH-independently box and the three-specification comparison all still
fail their own text, and the reporting box asks for PR/lift, rank correlation,
tail value captured and favourable-fill sacrifice, which the artifact does not
carry. **`HARMFUL_FILL_HAZARD_TOXICITY_PLAN.md` §10 has no checkboxes at all** —
it is a numbered implementation order, and its state lives in STATUS/HANDOFF and
the register, so there was nothing to tick there.

### Current data and forward state

- `clob_v4_1` has run since the ruled 2026-08-31T22:00Z boundary. At this
  update the unit is active/running, PID 1108125, `NRestarts=0`.
- 2026-08-29 is era-inadmissible. 2026-08-30 does not accrue. 2026-08-31 is
  mixed `clob_v4`/`clob_v4_1` and BTC failed P1 at 298.52 s/hr.
- 2026-09-01 is the first era-pure, admissible v4.1 day, but it is incomplete.
  At 09:29:39Z all 113 elapsed BTC/ETH windows were present and both coins
  passed the governing P1/P2/P3 bars. BTC had 572.2 s accumulated loss, a
  pace-adjusted projection of about 60.3 s/hr against 120, P2=0 material
  windows and P3=185.2 s against 900. Forward reach remains `G=0/5`: judge the
  day only after the closed-day verifier runs.

### 2026-09-01 ~14:05Z (MEM) — THE CLAUDE.md AMENDMENT IS DRAFTED, AND ONE
### PREVIOUSLY-SUGGESTED FIX WOULD HAVE INSTALLED A SECOND CONTESTED CLAIM

**`workspace/DRAFT_CLAUDE_MD_AMENDMENT.md`** — two hunks, each with current text
verbatim, replacement verbatim, and its register citation. `CLAUDE.md` is
USER-owned: nothing is applied, and the USER applies it by their own hand
(R-274).

**Hunk A closes a collision that has been live since 2026-08-28.**
`SEAT_PROTOCOL` rule 6 concedes that a fresh seat following `CLAUDE.md` writes
MEM's two state files and is *behaving correctly* — which is why the
instruction, not the seat, has to change. The USER already ruled the substance
(R-274, option (a) on Q-DA-138); the text was drafted then and never landed.

**Hunk B removes rule 9's false parenthetical — and does NOT use the wording
R-253 suggested for it.** That suggestion (*"Chainlink TWAP-vs-open"*) predates
Q-DA-142's amendment A2 by hours, and A2 — confirmed by Q-DA-146 on a fresh
n=8,022 population — puts the settlement statistic at **60-second endpoints**
(99.85%, passing its pre-registered gate) with the **full-window mean refuted**
(85.2%). Meanwhile the venue's own prose says full-range TWAP, in all 26,099
records. **Pasting R-253's suggestion would have replaced one contested claim
with another**, in the file whose purpose is rules that hold. The draft fixes
the venue — Chainlink, never Binance — and says nothing about the statistic.

**Re-verified rather than cited:** `markets.jsonl` now holds **26,099 records,
26,099 Chainlink, 0 Binance**. R-253 measured 17,727/17,727; the population is
47% larger and the ratio is unchanged. The count asserts it read a non-empty
population, so the zero cannot be a vacuous parse.

**Both hunks anchor exactly**, checked in both directions: a corrupted copy of
hunk B is not found, and both replacements are confirmed absent — the check
proves the draft applies AND that nothing has been applied yet.

**TODO sweep: 42 → 47 of 113**, every tick behind a commit or an artifact —
`V_cancel` valued at each tranche's own time/level/shares with a 5 s markout
(`c4b235f`), the typed `FairPrice` record and `Identity` as mandatory baseline
(`d97c23e`, with the fallback half named as protocol-not-yet-exercised), the
freeze stamp `b3f7f9f` = epoch 1787897340 pinned by its own selftest, and
`ACCRUAL_RULE` (`ab2f984`).

**Three sections gained an OWNER line and no ticks** — §5.3, §6 and §7 moved to
the DE seat under R-379. Those boxes changed owner, not state, and saying so is
the point: a reader scanning for progress must not read a transfer as advance.

### 2026-09-01 ~13:53Z (MEM) — R-374..R-381: THE REVIEWER IS A CLAUDE SESSION,
### THE RESOURCE RULE IS A MECHANISM, AND THE 011 RE-RUN WAS KILLED ON PURPOSE

Round-3 sweep. Each claim verified at its artifact by execution — the register
entries, the commits, the units, the registry and the JSON — never from the
dispatch that ordered it.

**The reviewer seat changed hands, and what it cost is recorded** (R-375,
`8b47dff`). Codex quota exhausted; the USER restarted `pm-codex` as a **Claude**
session. Surface unchanged: filings under `workspace/reviews/`, holds and an
explicit HOLD RELEASED, never fixes code, never touches state files. **The
independence profile did not survive.** The reviewer is now the same model
family as every seat it reviews — R-348's correlated-blind-spot finding turned
on the review seat itself, and no prompt-side "be adversarial" removes it. The
mitigation is **ground, not prompt**: committed artifacts at the pinned tip
only, execution over reading, and agreement treated as consistency rather than
confirmation. Filings go `REVIEW_*` from here; the `CODEX_*` files are the
Codex-era record and are never edited, because the KIND of document has to stay
identifiable.

**The resource rule stopped being a discipline** (R-376, `8b47dff`). Verified
live: `pm-research-guard.timer` on a 60-second cadence, last run 19 s before
this sweep, and the guard runs **outside** `research.slice` on purpose — a
memory-saturated slice must not stall its own guard.

| class | rule | verified |
|---|---|---|
| `IN_SLICE` | report only; the kernel enforces | `classify()` at `pm_research_guard.py:95` |
| `COLLECTOR` | **never touched, in either home** | matches `pm-collector*` **and** `collectors.slice` (`:100`) — the exemption was corrected by measurement after the first draft missed the P-2026-002 collectors |
| `FLAG` | ≥ 2G outside the slice → alert | `FLAG_GB_DEFAULT = 2.0` |
| `KILL` | ≥ 8G outside the slice → SIGTERM→SIGKILL | `KILL_GB_DEFAULT = 8.0`; 8G is measured territory — Q-BE-111 polled 8.8G one to two minutes before the box died |

CPU is deliberately not killed on: weights already price contention, and on a
swapless box memory is the failure that destroys. **`pm-measurement-pipeline`
was the SECOND unguarded 16G unit**, found by looking for the class rather than
the instance; both pipelines now read `Slice=research.slice` **and their repo
mirrors are byte-identical to the installed units**, which closes my round-2
finding that the guard was live but not reproducible from git. Residual, named
in the entry itself: the guard cannot see a heavy job in its first ~60 s, and
FLAG depends on `pm-alert@` being read — it bounds damage; the launch pattern
is still the rule.

**Coordination is batched in both directions.** R-377 (USER) makes the review
cycle batch-complete: one filing per round, all fixes landed and pushed
together, the reviewer notified exactly once at a pinned tip. R-378 (USER)
applies the same law to the coordinator's own dispatch loop as `SEAT_PROTOCOL`
rule 18 — a seat gets its complete batch in one dispatch and nothing further
while it is in flight, stop-the-line excepted. R-381 (USER) adds the clause that
**batching is about completeness, not idleness**: a closed round is followed
promptly by the next complete batch, and a seat waiting between rounds is a
coordination miss rather than a discipline.

**The DE seat is staffed** (R-379, `d929031`), executing R-165's parking clause:
`harmful_stateful_policy.py`, `de_actionspace.py` and `de_constraints.py`
transfer to DE. **The module audit, verified by me at
`live/pm_research/contracts/contracts.yaml` (version 24, 28 modules):**

| module | registry state | verdict |
|---|---|---|
| **EV-Replay** | exact-case `Replay`: **zero hits** — no module, no type. The lowercase word appears twice, in prose bodies only (`:90`, `:210`) | **the one gap on the critical path.** A grep hit on vocabulary is not a reference (rule 16) |
| DE-ActionSpace | TYPE at `:1201`, referenced at `:386`, code exists; module list registers only DE-Constraints / DE-Actuator / DE-Allocator | registry inconsistency, reconciled in DE's draft |
| OP-LatencyBudget | zero hits | deferred-with-trigger — to be NAMED, not silently absent |

**On the 011 lane, the artifact state is not what a reader would assume.** DA's
Q-DA-197 ran an independent reader (`da_iter011_contract_verify.py`, no shared
code with `phase2_iter011*`, R-235). **I reproduced its verdict by running it:**
`14/23 contract checks hold; 9 FAIL` on `24 cells; 296 typed field reads`. The
nine are disclosure and predicate defects, not a moved number — F1 the cell
carries the arrival n in 12 of 24 cells (22× overstatement); **F2 the survivor
predicate was Holm ALONE**, so `NO_INCUMBENT_COUNTERPART` cells published as
surviving; F3 the declaration's handling never reaches Q3; F4 rule 10's fourth
instance, found independently by BE and DA in the same hours (which is what
R-235 exists for); F5 `fit_code_ref` null; F6 no `as_of`.

**BE killed the re-run mid-fit rather than let it emit** (`4438961`), because F2
changes a *published verdict field* and the artifact should be **born** under
the closed predicate instead of superseded afterwards. State on disk at 13:53Z:

| file | bytes | note |
|---|---|---|
| `…__coin_btc.json` | 96,707 | `0b1f6bb`'s original, untouched — the killed run never reached its write |
| `…__as_verified_by_Q-DA-197.json` | 96,707 | byte-identical (sha256 `7d8437e6523ed32d` both) |
| `…__readjudicated_v2.json` | 101,789 | the intermediate state DA ran its supplementary check on |

`iter011-fit-batch.service` is active/running. **No artifact under the closed
predicate exists yet, and Q4 still has no economic result.**

**DA's other two shipped** (`c473b0e`): the 0h breadth disclosure as
`REPORTED_NOT_GOVERNING` carrying both denominators with a refusal behind it —
its own fixture-mirror mutant survived first and was killed — and tonight's
00:06Z path proven **by execution** on closed 08-29, reproducing the HANDOFF row
by a separate run.

**Nothing about the forward race moved today.** 09-01 is still the first
possible forward day; reach is still G=0/5.

### 2026-09-01 ~12:56Z (MEM) — 011 FITTED: A REAL 24-CELL FAMILY, AND THE
### DECISION METRIC IS NOT IN IT

Doc true-up on USER order. Every claim below was verified at the artifact —
`git show` on each commit, plus the JSON on disk — never from the dispatch that
ordered the sweep.

**The three commits, and what each one actually fixed.**

| commit | time | what it did |
|---|---|---|
| `54f899d` | 10:20Z | fitted inside the **unraised** 12G cap by PACKING the design matrix |
| `e326782` | 10:48Z | first real fit; then the **`any_fill_ahead` valuation gate** restored in the runner |
| `0b1f6bb` | 11:27Z | first real 24-cell family; mode-aware output declaration fixed |

`compact_design` packs PM+FN+ST into one float64 array and **releases** the
lists-of-lists — 7.11 GB → 0.45 GB for the same 578,917 rows — so the topup pass
allocates into space already held instead of growing past the cap. The cap was
never raised. Two of its own defects were caught by guards on the way: a
`--coin` slice applied to the **tape index** starved eth to 0 of 520,033 rows
and the absorption bound refused it, and the source regression guard written to
prevent its reintroduction matched its own string literal.

**The gate defect was only ever reachable once 011 was actually fitted.**
`phase2_arms._feature_pass` projects kept rows to a fixed field list that omits
`any_fill_ahead`; frozen `phase2_iter011.validate_row` requires it and refuses
`MISSING_GATE`. Two frozen documents, each correct alone, that had never met.
Fixed **in the runner**, which declares itself outside the lattice, because
`phase2_arms.py` is in `CODE_IDENTITY_FILES` and the frozen candidate binds its
hash. The restoration **calls** `harmful_exposure_rows.any_fill_ahead` rather
than reimplementing the predicate, and stored-vs-derived agree on **1,125,289
fragment and 638,917 topup rows, zero disagreements**.

**The artifact.** `data/pm_5min/derived/iter011_conditional_value_v1__coin_btc.json`,
96,707 bytes, written 11:23:34Z. All **24 declared cells present** — OK 6 /
`NO_INCUMBENT_COUNTERPART` 12 / `AGGREGATION_UNDECLARED` 6 — with the Holm
denominator held at 24, declared and not evaluated.

| head | statistic | p | holm | status |
|---|---|---|---|---|
| Q1_arrival | auc 0.8303 lgbm / 0.7733 linear | 0.001996 | **0.0479** | **OK, survives** |
| Q2_sign | auc 0.6003 / 0.5824 | 0.001996 | 0.0479 | `NO_INCUMBENT_COUNTERPART` **by design** |
| Q3_magnitudes | slope 0.6888 (placeholder) | withheld | — | `AGGREGATION_UNDECLARED` |
| Q4_combined_ev | 6,962.4 – 14,477.0 net cents | withheld | — | `NO_INCUMBENT_COUNTERPART` |

**Q2's status is designed, Q4's is a defect, and the artifact distinguishes
them.** `incumbent_null_applicability.comparable` reads Q1 true, Q2 false, Q3
false, **Q4 true** — the incumbent has no sign or magnitude head, so for Q2 no
incremental null exists and the p it carries is the **matched-random** null (the
cell names both so neither is read as the other). For Q4 the incumbent *should*
have applied and **did not load**, so the +12,333.5c headline is the candidate's
own value and not an increment. **There is no economic result**, and it is a
defect to chase rather than a design limit.

**Q3 waits on a ruling that already exists.** R-306 (USER, 2026-08-29T04:40Z)
ruled Q3's two slope gates compose as **conjunction + worse side** — recorded in
the frozen `plans/ITER011_PREREG_AMENDMENT_A1.md` and never implemented in code.
The cell withholds p rather than guessing and preserves the per-coin evidence
(both `matched_random_p` 0.001996) so the ruling applies without re-running.

**Two limits govern how any of this may be read, and both are in the artifact.**

1. **Resolution.** Every surviving p is 0.001996 = **1/501, the floor at 500
   permutations**, and holm is 24 × 0.001996 = 0.0479. The family only just
   clears 0.05; at 26 cells no cell could survive whatever the effect.
2. **Status.** The artifact **computes** `is_a_validation=false` rather than
   asserting it: eval is the `da_development_topup` population, 08-25 alone,
   311,640 btc rows / 177,674 actions, **G=0** complete UTC days, intervals not
   claimable, clustered on **window** where the ruled unit is the **UTC day** —
   so the p-values are optimistic. Development evidence (prereg 4: it selects,
   it never validates).

**Scope and identity.** No full both-coin artifact exists: the slice records
`iter011_conditional_value_v1.json` as NOT WRITTEN, and eth is reported-only
under btc-only adjudication (R-306). The artifact carries fit_code
`ad535550d366347d` — the post-09:00Z **declared** drift, not the freeze's
`3d0b6c8c6dfe9466` — tape `c7ab02ebcf27d2fc`, fragment `19a50195c34d0af2`, topup
`e75d0e210590e2a8`, and `standalone.is_in_identity_lattice=false`. Item 0f's
disclosure obligation is unchanged and now has a concrete instance.

**My own check shipped a falsifier** (R-289, my own lesson): it refuses an
emptied family *and* a status-field-stripped copy, and admits the real artifact
— both directions, because a count assertion alone would have caught only one of
those two vacuum shapes.

**Nothing else in the 09:43Z entry moves.** Collector v4.1 live, 09-01 the first
era-pure admissible day but incomplete, G=0/5, era as an interlock, breadth
reported and not gated.

### 2026-09-01 ~09:43Z (Codex) — CRITERIA/LIVE-STATUS CORRECTION

The 09:30Z criteria clarification is sound in its operative form: collector
era is an interlock, not a quality grade; `clob_v4_1` satisfies it, and 09-01
is judged by its own frozen `day_bar_v2` regime. The change is prospective and
does not resurrect the seen 08-29 day. `da_forward_day_verify` passed 181/181
selftests; `harmful_exposure_rows` passed 48/48.

Two statements in the 09:45Z entry below are superseded:

1. **A short gap does not poison the rest of a modeled window.** At a gap,
   `policy_optimizer_queue_realistic` clears state and pending state, resets
   both modeled positions, then resynchronizes and reposts from the next quote.
   The real queue rank is unrecoverable, so breadth is a useful disclosure, but
   stale modeled state does not persist to window end.
2. **09-01 does not pass both bar families.** Its governing v2 family passes.
   The superseded v1 count predicate fails because hours 7 and 8 each exceed 15
   gaps, even though BTC's average is 10.95 gaps/hr. This has no effect on the
   v2 verdict; it corrects the diagnostic claim only.

At the 09:29:39Z check, 52/113 elapsed BTC windows had any coin-level gap
overlap. The open-day artifact also prints 52/288 = 18.1%, because that report
field always uses the complete-day denominator. Read `52/113` for live breadth
and `52/288` only as progress toward the final complete-day denominator. A gap
overlap is not a claim that the whole window or all of its rows are corrupt.
No new breadth threshold is introduced after seeing 09-01; carry the count
beside P1/P2/P3 in the closing receipt.

### 2026-09-01 ~04:20Z (coordinator) — TWO BLOCKERS THE DOC TRUE-UP DID NOT SEE

The 03:49Z true-up above is accurate about what it describes and **wrong about
what is blocking**. It reads as "waiting for calendar days plus a clean btc."
Both statements below were established by execution, not by reading.

**B1 — the accrual predicate is wired to a DEAD ERA. A perfect 09-01 cannot
accrue.** `race_accrual_eligible = day_quality_pass AND post_freeze_pass AND
era_admissible`. `post_freeze_pass` is the `entirely_post_freeze` predicate,
which asks `warning_window.select_holdout()`, which iterates
`flow_intensity.covered_slugs(fi.ERA)` — and **`fi.ERA` is the module constant
`"clob_v3_1"`, an era that closed 2026-08-30T05:30:01Z.** Any day after it is
absent from the selector, so the predicate fails *by construction*, whatever
the feed did. Measured:

| `fi.ERA` | selector days |
|---|---|
| `clob_v3_1` (current) | 08-28, 08-29, 08-30 |
| `clob_v4_1` (correct) | 08-31, **09-01** |

Falsification run on the real day: `verify_day('20260901')` gives
`entirely_post_freeze=False` ("absent from the selector") at `clob_v3_1` and
**`True`, all seven coins 49/49**, at `clob_v4_1`. The pin is the whole cause.

**Why it was invisible.** It went stale at the 08-30 boundary, but 08-30 and
08-31 are mixed-era and fail `era_admissible` anyway — so the pin has been
*masked* by other refusals for two days. **09-01 is the first day where it
binds**, and its failure message ("absent from the selector") reads like a data
problem, not a config one.

Same pin also makes two REPORTED fields vacuous — `era_covered_windows` is `0`
and `gap_affected_PER_SLUG` is `0`/`None` for 08-31 and 09-01, on days holding
288 windows. **Neither governs:** `COIN_LEVEL` is the bar's input (R-191) and is
era-correct (08-31 btc 214/288 = 74.3%). Misleading to a reader, not deciding.

This is `_discover_days()`'s defect one identifier over. `DAYS` was converted
from a literal to a derived value because it went stale four times in three
days; **`ERA` sitting beside it was never converted.** Fixing the constant to
`clob_v4_1` restores accrual now but goes stale again at the next boundary — so
the fix is to derive it, or to make the verifier take an explicit era and refuse
a stale one, exactly as it already refuses a missing `--freeze-epoch`.
**Changing a gate mid-race is a ruling, not a repair: not done, awaiting USER.**

**B2 — tier1:full and tier2 have been HARD-BLOCKED for 146 h by three trades.**
`pm-evaluation-pipeline` crashes every run at `tier1_pipeline.py:1149`
(`normalize_clob`): `ValueError: trade price must be strictly inside (0, 1)`.
Cause located by a scan carrying its own falsifier (769,372 `last_trade_price`
events parsed, so the population was genuinely read): **three SELL trades at
price exactly `1.0`** (sizes 10, 0.1, 58.63) in `btc-updown-5m-1787635800`
(2026-08-25T05:30:00Z). The lane retries hourly, dies on the same day, and can
never advance past 08-25 — `newest_committed` has read 2026-08-24 for six days
while `tier1:measurement` (a different path) is caught up to 08-30.

A strict `0 < p < 1` on venue data is the wrong shape: a degenerate price is a
STATUS, never a crash (rule 4). **Does not block the accrual clock** — that
reads the raw tape — but it blocks all Tier-2 scoring, so the two blockers are
independent and both must clear.

**My first scan of this was VACUOUS and I nearly reported it.** The raw tape is
`recv_ns \t payload`, not bare JSON; my parse failed on every line into an
`except: continue`, and printed "0 out-of-range trades" — a clean bill of health
from a check that read nothing. It was caught only by adding the parsed-event
count. Same shape as R-289's matched pair, third instance in this programme.

**Neither blocker appears anywhere in STATUS.yml, HANDOFF or the register.**

### 2026-09-01 ~05:10Z (coordinator) — BOTH BLOCKERS FIXED ON USER RULING

USER ruled both: B1 "point to 9.1 and use v4_1"; B2 "include both 0 and 1".

**B1 — the era is now DERIVED PER DAY, not set to a new literal.** A better
literal would go stale at the next boundary exactly as `clob_v3_1` did, so
`verify_day` takes the day's OWN era from `day_era_admission` — already computed
ten lines above for `era_admission` — and passes it to
`select_holdout(freeze_epoch, era=...)`. For 09-01 that resolves to
`clob_v4_1`, which is the ruling, and it cannot go stale. `fi.ERA` is
**deliberately left at `clob_v3_1`**: six other modules read it, including
`harmful_exposure_rows` (the harmful-fill dataset on the consumed 08-20..25
days), and flipping a global era constant would silently re-populate frozen
analyses. Mixed-era days are now refused **by name** ("spans eras [...]")
rather than by absence. `selector_era` is recorded in every verdict.

Measured, no patching: 09-01 `selector_era=clob_v4_1`, `post_freeze=True`,
7 coins 58/58 → **`race_accrual_eligible=True`**. 08-31 correctly `None` /
refused. 08-29 unchanged at `clob_v3_1`, still ineligible on era admission —
so the change moves no historical verdict.

Also fixed by the same derivation: `era_covered_windows` and
`gap_affected_PER_SLUG`, which read 0/None on days holding 288 windows. Those
are REPORTED, not governing (`COIN_LEVEL` is the bar's input, R-191), so this
corrected two misleading numbers, not a verdict.

**B2 — the bound is now closed `[0, 1]`, and the print is COUNTED.** New
`ParseStats.settled_price_trades`. `evaluation_pipeline` already accepted
`0 <= price_up <= 1` **inclusive**, so the two layers disagreed and the
stricter, cruder one won by crashing — Tier-2 was always written for closed
-interval prices. Verified safe at the bounds: `edge` is pure subtraction, and
both `logit` implementations clamp to `1e-6` and read book mid, not trade price.

**Both fixes ship falsifiers, and all mutants die.** B2: positive control
(0.0/1.0 admitted, counted, price carried), negative control (interior-price
fixture must count 0, so an always-on counter fails), known-bad refusal (1.5 and
-0.1 still raise). B1: the era must REACH both loaders, the bare-call fallback
still works for historical consumers, plus source guards. Six mutations run —
revert-to-open-interval, counter-never-increments, raise-removed,
verify_day-reverts-to-bare-call, gap-affected-reverts-to-literal,
select_holdout-ignores-its-argument — **6/6 KILLED**, files byte-identical after.

`tier1_pipeline --selftest` **added to `v5_deploy_gates`**: it held the check
that blocked two lanes for 146 h and the suite never invoked it. **ALL 15 GATES
PASS**; falsifier fires (exit 0, canary only).

**Open, and it is a RULING not a repair — `race_accrual_eligible` does not
require the day to be CLOSED.** With B1 fixed, 09-01 reads eligible at 4 h
elapsed, because `complete_tape` compares against windows *elapsed so far*.
The nightly verdicts the just-OPENED day too, so at 00:06Z on 09-02 it would
write `race_accrual_eligible=true` for a six-minute-old day. Before B1 that
read false only because the era pin was failing it — a wrong answer masking a
second one. Nothing consumes the field yet (the clock is unbuilt), so no count
is wrong today, and `day_closed_calendar` sits beside it in the same artifact.
**Recommended: make `day_closed_calendar` a conjunct** — rule 8 already says
accrual counts *complete* UTC days, so this enforces the stated policy rather
than making new one. Not done: changing an eligibility gate is the policy
layer's (rule 14).

**Watch:** `v4 behaviour (git-extracted)` failed once under back-to-back suite
runs and passes 3/3 standalone. Cause not established; it fails in the SAFE
direction (false red, never false green). Pre-existing, unrelated to these fixes.

### 2026-09-01 ~05:40Z (coordinator) — THE ACCRUAL RULE, STATED ONCE

USER: *"it looks a bit confusing, we can just make the rules simple and clean."*
The confusion was that **FINISHED was implicit**. It is now explicit, and the
whole rule lives in one named constant, `ACCRUAL_RULE`:

> **A day accrues iff FINISHED (closed UTC day) AND AFTER (post freeze commit)
> AND ADMISSIBLE (one ruled era) AND HEALTHY (quality bars).**

Four conjuncts, four different questions, no redundancy. Any one false means
the day does not count — and **not** that the day was bad: AFTER and
ADMISSIBLE are properties of the clock and the collector, never of the feed.
Every verdict now carries `day_closed` and the rule text beside the boolean.

`split_verdict` **refuses** a missing `day_closed`, the same precedent as
`era_admissible`: eligibility must not be obtainable by not asking.

Reads cleanly on real days:

| day | FINISHED | AFTER | ADMISSIBLE | HEALTHY | accrues |
|---|---|---|---|---|---|
| 09-01 | no | yes | yes | — | **no** |
| 08-31 | yes | no | no | no | no |
| 08-29 | yes | yes | no | yes | no |

**Why FINISHED is not implied by HEALTHY**, which is the trap it closes:
`complete_tape` measures against the windows elapsed SO FAR, so it passes
mid-day. 09-01 read *eligible* at 4 h; twenty minutes later it read *unhealthy*
because the in-flight 5-minute window made it 65 of 66. **Partial-day quality
flaps pass/fail with where in the window you look** — judging accrual on it was
never meaningful, and the nightly verdicts the just-OPENED day too, so at
00:06Z it would have written eligible for a six-minute-old day.

**Mutation-tested, and one survived first.** M4 (conjunct dropped) and M5
(refusal disabled) died immediately. **M6 — `verify_day` hardcoding
`day_closed=True` — SURVIVED**, because every `split_verdict` test supplies its
own value: I had enforced the rule in the FUNCTION and left the CALL PATH
open, the identical shape as the era literal. Added a path guard; M6 and M7
(per-coin) now both die. 181 checks.

**Known-flaky gate, pre-existing and unrelated to these changes.**
`v4 behaviour (git-extracted)` fails **2 in 5 suite runs at 9/10**, and
**0 in 28 standalone runs** — its checks carry wall-clock margins (a 0.4 s
subscribe bound, real backoff sleeps) that miss under the load the suite's two
mutation audits create. It fails in the SAFE direction (false red, never false
green). Not fixed; recorded so a red there is not read as a code regression —
and so it is not learned as a red to ignore.

### 2026-09-01 ~07:15Z (coordinator) — 011 RELEASED, SKEW FROZEN, AND THE
### RESOURCE GUARD WAS NOT BEING APPLIED

USER: *"release and do skew freeze, proceed"*, then *"we have the memory/cpu
limit right? do we apply that now"*.

**The resource question first, because the answer was NO.** `research.slice`
exists with `MemoryMax=60%`, `CPUQuota=1200%`, `CPUWeight=50`, `MemorySwapMax=0`
and a documented launch pattern — written after the **2026-08-26 03:55Z box
death from AGGREGATE memory exhaustion** (R-148/R-150), whose whole lesson is
that a single-job cap cannot bound the sum. It was reading **`Tasks: 0`.**
Nothing was in it. Every heavy job today ran in `app.slice` instead: my ad-hoc
scans (4.5 GB gz), the mutation loops, the 011 dry run — **and
`pm-evaluation-pipeline`, which is a heavy research job carrying only a
per-unit `MemoryMax=16G` and sitting outside the aggregate guard entirely.**
The guard was built, committed, documented, and then not used. This is Codex
**COL-R2** — filed, downgraded from a release gate, and still open.

The 011 fit is now launched through the documented pattern:
`systemd-run --user --slice=research.slice -p MemoryMax=10G -p MemorySwapMax=0
-p OOMScoreAdjust=1000 -p CPUWeight=50`. It yields to the collectors (weight
500) by construction, so a fit can no longer starve the tape.
**NOT changed: `pm-evaluation-pipeline`'s slice** — that needs a unit edit and
a restart, and restarting it now would abort the 08-26 rebuild mid-flight. Do
it when the backlog clears; until then the aggregate guard is still not whole.

**Iteration 011 RELEASED.** §A1.8 step 6 was the only outstanding gate: steps
2–5 are closed and prove it by execution — the run selftest is GREEN (0
failing), and `main()`'s own path is exercised by `--dry-run` on synthetic
populations (both arms fitted, applied, 24-cell family assembled and
adjudicated, output guard enforced), which exists because a component suite
cannot see an unwired `main()`. `main()` REFUSES to run on a red selftest, so
the numbers cannot come from an instrument that has not shown it can fire.
Fitting started 07:11Z; tape identity `c7ab02eb` / fragment `19a50195` / topup
`e75d0e21` pinned and printed before any fit.

**Skew lane FROZEN**, recording committed semantics only — no code change, no
new number. **§7 Q1–Q3 were not in the user's ruling**, and are resolved by the
coordinator under its authorization and labelled as such, because attributing
them to the user would be inventing a ruling. Each takes the same conservative
form — *record what code contains, invent nothing* — and each is CORRECTABLE:

| Q | resolution |
|---|---|
| Q1 desired exposure | **NAME THE PAIR** (`size`=5.0 + front/back intent); no invented scalar |
| Q2 marginal inventory-risk value | **NOT-PRESENT**; the "~9.4x" stays prose, never a frozen quantity |
| Q3 `charge_reset_cost_at_generation_start` | **OUT OF SCOPE**, cited not resolved — policy-layer priced decision (rule 14), a LIVE obligation before any lifecycle-economics number |

**Still owed on the skew lane:** bit-identical queue-reference parity has been
ASSERTED by `da_replay_parity_battery` but never RUN against a seven-arm replay.
The freeze fixes semantics; it does not evidence parity. Do not read it as both.

### 2026-09-01 ~07:50Z (coordinator) — 011 HALTED: A FLAG THAT NEVER FIRES,
### AND TWO OOMs

USER asked why we use 08-26+ data given the quality. **We do not** — and the
question found something a day earlier instead.

**The 011 tape holds TWO days.** `phase2_state_tape_v5.json` (built 08-27):
train = 08-24 (582,773) + 08-25 (542,516); score = 08-25 (638,917). Nothing
from 08-26 on is in it. The 08-26..08-30 lane work is DERIVATION (raw tape →
Tier-1 partitions), not model input, and those days cannot accrue anyway.

**`queue_ahead_missing` NEVER FIRES.** On the 311,640 clean btc score rows of
08-25 it is False on **every single row**, while **56.4% (175,793)** have a
data gap between window start and their decision cutoff — the exact interval
over which queue position is rebuilt by replaying adds/cancels/fills. So the
model is handed a confident `queue_ahead_of_level` flagged as KNOWN on 175,793
rows whose book replay crossed a hole in the tape. **A flag that cannot fire is
not a flag**, and queue position is the central feature of a queue-harm model.

The exclusion that does exist is ENDPOINT-ONLY: `GAP_AT_CUTOFF` marks the 192
rows (0.06%) whose cutoff INSTANT sits in a gap. Same shape as the convexity-v4
audit's "endpoint-only fix". Short-lookback rate features (50/250/1000 ms) are
genuinely fine — only 1.3% have a gap that close.

| btc score rows, 08-25 | share |
|---|---|
| window contains a gap | 79.2% |
| **gap BEFORE cutoff (queue-position interval)** | **56.4%** |
| gap within 1 s of cutoff (rate features) | 1.3% |
| cutoff instant inside a gap (the only class excluded) | 0.08% |

**This costs NO clock.** The defect is data-independent — found by intersecting
gap intervals with row timestamps, not by looking at a result — and no 011
number has been adjudicated. Fixing it now voids nothing, exactly the A1 class.

**Recommended (not ruled):** (1) make `queue_ahead_missing` fire when the
reconstruction crosses a gap, so the field stops being dead weight; AND (2)
exclude gap-crossed rows from adjudication rather than trusting the model to
learn around them — that still leaves 135,847 btc score rows. Rebuilding on
cleaner days is the third option (08-24 is 16.3% window-affected vs 08-25's
80.6%; eth is 0.7% but is reported-only under btc-only adjudication, R-306).

**ALSO OPEN: the prereg never declares its development population.** §4 lists
08-20..08-25 as CONSUMED and says "None may be used for selection *or*
validation in iteration 011" — read literally that leaves 011 with no build
population at all. It refers to "the declared development population" and never
declares it; the implementation resolved it as train 08-24 / score 08-25. An
undefined term settled by whoever implemented it first (§A1.0's own warning).

**TWO OOM KILLS, and the retries are STOPPED.** 10 G → `oom-kill` at 10 min;
14 G → `oom-kill` at 20 min, peak 14.0 G, past the launch pattern's documented
`<=14G` single-job ceiling. Not retried at 18 G: the output is ALREADY
non-adjudicable per the finding above, so climbing the cap would spend the
budget to produce numbers that cannot be used — and 18 G alongside
`pm-evaluation-pipeline`'s app.slice 16 G is the unbounded aggregate this
guard exists to prevent. **Fix the contamination first, then run ONCE.**
Memory is itself a live question: R-194 seam 15 already cut this index once
(12 GB for 1.7M rows) and it is over the ceiling again.

**MY MONITOR REPORTED THE FIRST OOM AS A SUCCESS.** `systemd-run --collect`
deletes the unit on exit, after which `systemctl show` returns DEFAULTS —
`Result=success`, `ExecMainStatus=0` — indistinguishable from a real success.
Only the absent artifact contradicted it. Never read a result from a collected
unit; keep the unit and read `Result`, or read the journal.

### 2026-09-01 ~08:20Z (coordinator) — MEMORY SLICED; AND THE CONTAMINATION
### IS A FORWARD PROBLEM, NOT A DEVELOPMENT ONE

USER: *"lets optimize the code to use less memory, or we truncate the data to
do it slice by slice. then if data is not good, check if there is good data
population to use. do check data before using."*

**CHECKING FIRST STOPPED ME EDITING THE WRONG FILE.** The obvious memory fix is
`phase2_arms._feature_pass` (hardcodes both coins, json-loads its whole source).
**`phase2_arms.py` is in `CODE_IDENTITY_FILES`**, the frozen btc candidate binds
`provenance.fit_code_sha256_prefix = 3d0b6c8c6dfe9466`, and hazard-plan §10.9
requires scoring that candidate **UNCHANGED**. Editing it mid-race would make
the forward scoring a different program from the freeze.

**AND THE IDENTITY HAS ALREADY DRIFTED:** current `e27cab9e5f6ce8e5`. One file,
one hunk — `2e1204f` added a refusal to `_stream_tape_rows` for a tape ending
without its closing `]`. **Provably inert here:** its own commit states the
accepting path is byte-for-byte unchanged, and our runs streamed all 1,764,206
rows *without* raising, so that branch never fired; the tape hash also matches
the freeze exactly (`c7ab02ebcf27d2fc`). Defensible, but it must be DECLARED in
any forward receipt, not discovered later — a receipt claiming "unchanged" over
a changed lattice hash is false on its face.

**The slicing therefore lives in the RUNNER**, which declares itself outside the
lattice. `--coin <name>` restricts the tape index (a plain dict keyed by slug),
drops the other coin's block the moment each pass returns, and frees the embargo
probe **before** the topup pass instead of after — both OOMs landed in
`[topup/eth]` holding FIT for both coins + the 638,917-row score index + ~640k
probe dicts simultaneously. Identity re-checked after the edit: **unchanged**.

**Numerically identical, and proven so:** `--dry-run --coin btc` produces
`Q1 auc 0.46386666666666665` — the same digits as the unsliced dry run. The
results loop was already per-coin; this slices WORK, never the estimand.
Seven controls added, including a KNOWN-BAD that a prefix (`bt`) matches
nothing, and a source check that the probe is freed before the pass.
**The dry harness did NOT cover the slice at first** — `--dry-run --coin btc`
ran both coins and exited 0, so the slicing would have reached a real run never
exercised through `main()`. That is exactly defect I11-2's shape; now wired.

**THE POPULATION SURVEY — btc gap contamination by day.** Estimated share of
decision rows with a gap between window start and cutoff (the interval queue
position is rebuilt over). The estimator reads 63.4% on 08-25 against 56.4%
measured, so it is slightly conservative and trustworthy for ranking:

| day | windows w/ gap | est. rows gap-before-cutoff |
|---|---|---|
| 08-19 .. 08-24 | 0 – 15.6% | **0 – 10.9%** (clean) |
| 08-25 | 80.2% | 63.4% |
| 08-26 | 63.8% | 47.7% |
| 08-27 | 70.1% | 52.8% |
| 08-28 | 64.6% | 48.4% |
| 08-29 | 32.3% | 18.8% |
| 08-30 / 08-31 | 70.3% / 73.8% | 57.1% / 65.0% |
| **09-01 (v4_1, live)** | 43.3% | **31.5%** |

**The bind, stated plainly.** The CLEAN days (08-19..08-24) are exactly the
days the prereg lists as CONSUMED. The AVAILABLE days (08-25 on) are all
heavily contaminated. The current tape already uses the cleanest available day
for training (08-24, 10.9%); the damage is concentrated in the SCORE split,
which is 08-25 alone at 63.4%.

**AND THIS IS NOT ONLY AN 011 PROBLEM — 09-01 IS 31.5% CONTAMINATED.** Every
forward validation day carries the same defect, because the btc break is
ongoing and `queue_ahead_missing` never fires on any of it. So fixing the flag
is a PRECONDITION FOR THE FORWARD RACE, not a tidy-up for a development run:
the frozen candidate is about to be scored on days where a third of btc rows
have a queue position rebuilt across a hole, silently marked as known.

**Recommended, still not ruled:** fix the flag first (it is dead weight today
and cheap to make correct), then exclude gap-crossed rows from BOTH the 011
development population and forward scoring. Excluding costs ~56% of 08-25 and
~11% of 08-24 — the development population survives it. Re-freezing is NOT
required for the flag fix per se, but the fix touches feature semantics, so
whether it lands inside or outside the frozen lattice is a rule-12 question.

### 2026-09-01 ~09:00Z (coordinator) — THE ERA LITERAL CLEARED IN THE ROW
### GENERATOR, AND A CORRECTION TO MY OWN ALARM

**FIRST, THE CORRECTION, because I raised an alarm and it was overstated.**
I reported `queue_ahead_missing` as a flag that "cannot fire" and put 56.4% of
btc rows under suspicion. Both were wrong:

- `queue_ahead_missing` fires on `denom = qahead + resting <= 0`. Every row in
  this population exists BECAUSE an order is resting, so `denom > 0` by
  construction. It is a **structural zero**, not a silent failure — a flag for
  a question this population cannot pose. **The USER identified this**, noting
  empty books are expected once an outcome is decided.
- The empty-book condition IS represented, by a different field:
  `queue_ahead_of_level = None` on **58,893 rows (18.9%)** — exactly the rows
  where `level_size` is empty. `terminal_window` is only 2.2%, so these are
  decided-but-not-expiring markets, matching the USER's reading.
- The 56.4% assumed corruption persists to window end. It does not: busy
  windows carry **2,700–4,900 `book` snapshots**, applied via
  `state.apply("book", …)`, so level state re-anchors continuously.

**And the gap discipline that DOES exist, fires:** `GAP_IN_HORIZON` excludes
10,456 fragment rows (0.92%) and 6,873 topup rows (1.06%) BEFORE features are
built (`_feature_pass` keeps only `status == "OK"`); `GAP_AT_CUTOFF` excludes
192 more. Labels and decision-instant state are both protected. **The data
behind 011 is sound.** I was wrong to tell the USER to hold the lane on this.

**WHAT WAS REAL, and is now fixed.** Both row selectors filtered the slug set
AND the gap intervals through the `fi.ERA` literal (`clob_v3_1`, closed
2026-08-30T05:30:01Z):

```
gaps = fi.gaps_by_slug(fi.ERA);  for slug in sorted(fi.covered_slugs(fi.ERA)):
```

On any later era that selects **nothing** — a forward tape would have built
EMPTY and reported success, which is worse than unprotected.

**NOT fixed by widening to "all eras", and the measurement is why.** All-era is
inert for `select_v2_era` (1,666 slugs either way, and 0 of 808 slugs in the
current artifacts change their gap list) but NOT for `select_stratified`:
TRAIN_DAYS 08-20..08-22 span the clob_v2 → v2_1 → v3 → v3_1 transitions, where
all-era gives 4,753 slugs against 4,543. There the era filter is doing real
purity work, and widening it would silently admit three earlier collectors.

So the fix is the `--freeze-epoch` pattern: **the era becomes an explicit
parameter, its default unchanged, and an empty selection REFUSES by name.**
Behaviour is inert BY CONSTRUCTION — the default passes the identical value —
and the selectors still return 471 / 30 windows. Three checks added (48, was
45): both selectors refuse a foreign era, plus a positive control that the
default still selects, so the guard discriminates rather than refusing
universally. My first pass wired the guard into only ONE selector and the
falsifier caught it.

**DECLARED LATTICE AMENDMENT (rule 12/13).** `harmful_exposure_rows.py` is in
`CODE_IDENTITY_FILES`, so the combined identity moves again:

| stage | combined | cause |
|---|---|---|
| freeze `b3f7f9f` | `3d0b6c8c6dfe9466` | — |
| before today | `e27cab9e5f6ce8e5` | `2e1204f`, inert truncated-tape refusal |
| **now** | **`ad535550d366347d`** | **this fix** (`harmful_exposure_rows.py` `c2e40100…` → `1bbd8e75…`) |

**No frozen artifact changes**: fragment, topup and the model artifacts are
DATA artifacts bound by content and are untouched; the default code path
reproduces them exactly. But any forward receipt MUST carry all three hashes
and state why "unchanged" is still true — a receipt asserting it over a moved
lattice hash is false on its face. ALL 15 GATES PASS.

### 2026-09-01 ~09:30Z (coordinator) — ACCRUAL RULE CLARIFIED BY USER RULING

USER: *"i dont care about collector version, as long as the data quality is
good, then we can use to test the model"* → *"can we update the rule and
corresponding docs"*. Updated in `ACCRUAL_RULE`, in `split_verdict`'s output
(new `era_role` field, so every verdict carries it), and in the governing plan.

**The rule is unchanged in form — FINISHED AND AFTER AND ADMISSIBLE AND
HEALTHY — and clarified in meaning.** ERA IS NOT A QUALITY VERDICT. The ledger
states `clob_v3_1 → v4 → v4_1` as *"distributional only; NO row-stamping
change"*: surviving rows are recorded identically across all three, so era
carries no fidelity claim about the data the model sees. **Among ruled eras,
quality alone decides.**

**No code behaviour changed, because that is ALREADY the operative behaviour.**
`clob_v4_1` is ruled admissible and every forward day is era-pure `clob_v4_1`,
so the era conjunct is satisfied from 09-01 and the quality bars govern. The
conjunct was never blocking the forward race.

**It survives as an INTERLOCK.** Its refusal — *"a collector version is not
admissible by default and silence is not a ruling"* — fires at the NEXT
boundary, so a deploy cannot start accruing days under an unvetted collector.
Free while every live era is ruled; this programme deployed one yesterday.

**Two things kept SEPARATE that I had collapsed:**
1. *Using* a quality-passing day's data — era-independent, the USER's point.
2. *Comparing* quality numbers across eras — still invalid. P1/P2/P3 are
   era-dependent in magnitude: at 3/3 a stall is logged in ~3 s; at 10/10
   sub-10 s stalls self-heal and never appear. Measured on the same feed:
   08-31 → 1,134 btc gaps / 27.3 s median cumulative; 09-01 → 84 / 9.7 s.
   I answered (2) when the USER was asking (1).

**A TABLE CORRECTION, because it changes a conclusion.** The bar REGIME differs
by day: `count_bar_v1_frozen` before 08-29 (`gap_rate_under_bar` governs),
`day_bar_v2` from 08-29 (P1/P2/P3 govern, gap_rate superseded). Applying v2
bars to v1-governed days is an anachronism that flips verdicts:

| day | claimed | verified |
|---|---|---|
| 08-28 | Pass | **FAIL** — `gap_rate_under_bar` 20.29/hr vs 15, 12 h over. Its P1 *would* be 114.1 (passing) but P1 is not its bar |
| 08-29 | Pass | **PASS** — `all_pass=True` under day_bar_v2 |

So **08-29 is the ONLY quality-passing, era-pure, post-freeze day the era
conjunct has ever excluded** — not 08-28 and 08-29.

**PROSPECTIVE ONLY (rule 11), and the USER stated this first.** The
clarification is issued while it is already known which days would pass, so it
grants nothing retroactively. 08-29 stays excluded — it was seen and does not
become forward validation. **09-01 remains the first possible forward day.**

**Provenance note:** the era-as-metadata framing arrived attributed to me
(*"under your intended rule"*). I did not state it — I implemented ADMISSIBLE
as a conjunct and described it as one. Recorded so the register does not carry
a ruling under the wrong seat; the ruling is the USER's, the clarification is
mine, and the correction of the 08-28 row is mine too.

### 2026-09-01 ~09:45Z (coordinator) — WHY 08-28 FAILS, AND THE HOLE IT EXPOSES
### IN day_bar_v2

> **CORRECTED 09:43Z BY THE CURRENT READ-FIRST ENTRY:** the historical
> measurement below stands, but “a gap poisons the rest of the window” and
> “09-01 passes both families” do not. The replay resets/resynchronizes after a
> gap, and the old count predicate fails on two hourly bins. Breadth remains a
> reported diagnostic, not a new gate or a whole-window contamination label.

08-28 fails on **exactly one** predicate — the gap **COUNT** rate — and passes
every duration bar:

| bar | 08-28 | bar | verdict |
|---|---|---|---|
| P1 lost s/hr | 113.89 | 120 | PASS (5% margin) |
| P2 windows ≥75 s | 0.00% | 5% | PASS |
| P3 worst 60 min | 291 s | 900 s | PASS |
| **gap count/hr** | **20.08** | **15** | **FAIL** ← its governing bar (v1) |

**The data issue is real, not a bar artifact: 186/288 btc windows (64.6%) carry
a gap.** 08-28 sits in the middle of the post-08-25 break. Its gaps are SHORT
(median 11.9 s cumulative per affected window), so total lost time squeaks
under the duration bar while two thirds of windows are touched.

**THE DISCLOSURE GAP: `day_bar_v2` HAS NO BREADTH PREDICATE.** `count_bar_v1_frozen`
caught breadth through the gap-count rate; v2 retired that (`SUPERSEDED_ON_V2 =
('gap_rate_under_bar',)`) and replaced it with three DURATION bars. So the
v1→v2 migration stopped governing on raw event breadth. A gap creates a blind
interval and a modeled queue reset/repost, but it does **not** leave stale state
through the rest of the window. **08-28 is the proof that high any-overlap
breadth can coexist with passing duration bars; it is not proof that 64.6% of
whole windows are corrupt.**

**NOT a proposal to change the bars.** They are pre-registered and frozen, and
09-01 is the first forward day — retuning a bar now, knowing which days pass,
is choosing after seeing (rule 11) and would void the race. This is a
**DISCLOSURE item for the forward receipt**: the governing bar set scores
duration, not breadth, so a passing forward day can still be substantially
gap-affected, and the receipt must carry windows-affected beside P1/P2/P3
rather than instead of them (Q-DA-69's original point, now with a second
instance).

**09-01 currently passes its GOVERNING v2 family** — projected P1 60.8 s/hr
(bar 120), with 51 windows touched at 9.37 h. Its superseded v1 count predicate
does **not** pass: the average was 10.99/hr, but two individual hours exceeded
15. This does not affect the live race. Watch the projected P1, which moved
34.4 → 42.3 → 60.8 s/hr across the day.

### Immediate order

0. ~~Rule B1~~ / ~~day-closed conjunct~~ **BOTH DONE.** The rule is one line in
   `ACCRUAL_RULE` and is enforced at the function AND the call path.
0h. **Carry windows-affected in every forward receipt** beside P1/P2/P3 —
   as a disclosure count of short blind intervals/resets, not as a count of
   fully contaminated windows. For an open day report both affected/elapsed
   and affected/288; only the latter is the closing-day denominator.
0g. **A forward tape build must NAME its era explicitly** — the default is
   still `clob_v3_1` and will refuse loudly on v4_1 windows rather than
   building empty. That refusal is the fix; passing `era="clob_v4_1"` is the
   caller's job, and the day range must not straddle a boundary.
0f. **The lattice drift must be declared in any forward receipt** — freeze
   `3d0b6c8c6dfe9466` vs current `ad535550d366347d`; the current development
   artifacts remain content-bound and reproduced.
0e. ~~Diagnose the stopped 09:34Z attempt~~ **DONE — the retry COMPLETED and
   wrote its declared artifact** (`54f899d` → `e326782` → `0b1f6bb`;
   `iter011_conditional_value_v1__coin_btc.json`, 11:23:34Z, 24/24 declared
   cells). The two 011 items now open are both inside the result, not before it:
   **the Q4 incumbent-loading defect** — Q4 is the decision metric,
   `incumbent_null_applicability` declares it `comparable:true`, the incumbent
   did not load, and until that closes there is **no economic result** — and
   **implementing R-306's ruled conjunction + worse side for Q3**, which is
   frozen in `plans/ITER011_PREREG_AMENDMENT_A1.md` and absent from the code.
   The standing caution holds and was applied: do not infer success from the
   service wrapper; the artifact and its own identity/population fields were
   read before any number here was quoted.
0d. ~~Move `pm-evaluation-pipeline` into `research.slice`~~ **DONE at 13:02Z**
   (R-374, `e9f4834`), verified here at the unit: `Slice=research.slice`,
   `MemoryMax` unchanged at 16G, inactive at edit time. **Residual, not closed:**
   the `Slice=` line lives only in the installed unit — the repo copy
   `live/pm_research/ops/pm-evaluation-pipeline.service` has no `Slice=`, so the
   guard is live but not reproducible from git (R-361 LIVE-3's shape).
   **CLOSED at `8b47dff` (R-376):** both mirrors synced and verified identical
   at 13:53Z, and the second unguarded 16G unit (`pm-measurement-pipeline`) was
   found by looking for the class and moved too.
0b. ~~Confirm the tier1:full lane actually clears 08-25~~ **DONE — the 146-hour
   block is cleared** (R-374, `e9f4834`): after the B2 `[0,1]` fix the lane
   committed 08-26..08-28, then 08-29 COMPLETE at 09:01:48Z and 08-30 at
   10:01:55Z. Confirmed here in the journal: the 11:20, 12:20 and 13:02Z runs all
   finish `IDLE` — caught up, nothing hung. 08-31 waits on its upstream
   measurement-lane trigger and is a mixed-era day.
0c. De-flake `v4 behaviour` — replace wall-clock margins with injected time.
1. Preserve the running collector and close the first full v4.1 UTC day.
2. Obtain an independent re-review of `1aaac18`: it claims RR1/RR3 closure, but
   the latest filing reviewed its parent and did not release them. **It is no
   longer HEAD**; the re-review target is the commit, not the tip. **Held
   deliberately** until BE's batch commits, so one round covers both at a single
   pinned tip (rule 5 as completed by R-377). **The reviewer is now a Claude
   session** (R-375), so the round's weight rests on execution — suites,
   mutation audits, known-bads, the operator walk — rather than on reading:
   agreement from a same-model reviewer is consistency, not confirmation.
3. Close the two 011 items the completed run exposed, in this order: **(a) the
   Q4 incumbent-loading defect** (no economic result exists until it closes),
   **(b) implement R-306's conjunction + worse side for Q3** — the ruling exists
   and the per-coin evidence is preserved, so it applies without re-running.
   Then **(c) raise the permutation count**: every surviving p sits at the 1/501
   floor and holm is 24 × 0.001996 = 0.0479, so the design has no headroom to
   carry a verdict. The result stays **development evidence** throughout — the
   artifact computes `is_a_validation=false` at G=0 complete UTC days.
4. Keep fair-price, skew and replay work build/freeze-only behind their stated
   gates. No PnL, capacity, promotion or forward verdict is claimable.
5. **USER-ONLY, waiting on the USER's hand:** apply
   `workspace/DRAFT_CLAUDE_MD_AMENDMENT.md` to `CLAUDE.md` — the state-file
   ownership exception (R-274, ruled 2026-08-28 and never landed) and rule 9's
   false Binance parenthetical (R-253). No seat may make this edit. When hunk A
   lands, `SEAT_PROTOCOL` rule 6's "amendment is pending" clause comes out —
   coordinator's edit, and not before the amendment is actually in the file.

Historical handoff entries follow unchanged.

## 2026-08-28 ~15:19Z (MEM) — R-289: the checker's chair has an empty-set trap too

### What happened

My last sweep recorded the committed-null conclusion as *"checked three times
independently — BE, the coordinator, and me."* **That overstated it when I wrote
it**, and the correction is the interesting part.

**The coordinator's R-288 check was itself vacuous.** Its parse matched all 12
cells but **guessed the observed field name** (`observed` / `observed_increment`
against the artifact's `observed_increment_cents`), so every cell defaulted to
`0`, the filter never fired, and *"0 negative with p<0.10"* was **a field-level
vacuum wearing the shape of a confirmation** — recorded in the register as
independent evidence.

**Mine, an hour earlier, was a cell-level vacuum** — the parse matched **zero**
cells, so every filter was trivially empty. I caught it by implausibility (zero
cells in a 12-cell artifact is visibly wrong), added a count falsifier, and
re-ran. **My disclosure is what sent the coordinator back to theirs.**

### The matched pair is the lesson

**The same four-line check produced two vacuums, in two seats, in one hour, each
vacuous a different way — and neither was visible from inside its own run.**

| | failed at | caught by |
|---|---|---|
| MEM | **cell** level — parse matched 0 cells | implausibility of the count |
| Coordinator | **field** level — matched all 12 cells, read none of them | **only** the other seat's disclosure |

**What the pair proves that either instance alone could not: a count assertion
would have caught mine and missed the coordinator's.** The check must assert
**both** that the population was read **and** that the fields it filters on were
read as the type it filters them as.

**The rule, in its canonical form:** *a verification claim entering the register
must assert that its parse actually read the population and the fields it filters
on. "Found nothing" from a reader that touched nothing is the empty-set trap in
the checker's chair.* That is rule 15/16 applied to **verification itself**
rather than to the code under test — the seventh sibling of the day's
truncation/vacuity class, this one in the verifier's chair.

### The conclusion itself never moved

**BE's original check was real.** The honest ledger now reads **BE's original +
MEM's asserted check + the coordinator's corrected check**, agreeing exactly:
12 cells, three negative-increment cells at p 0.2334 / 0.2774 / 0.4208, **zero
negative under p<0.10**.

So the committed null's defect remains **conservative in the safe direction**,
the *"10 of 12 chance"* conclusion **stands**, and **no summary may say the
canonical null was wrong.** What was wrong was the *accounting of the evidence*,
which my own flag had overstated — corrected in-band, original line kept as the
record of the overstatement.

### Unchanged from the previous entry

Chain **restored scientifically, not yet mechanically** (47j red until the
fit-time re-stamp). Post-release cycle carries four items. **Round 3 fired at
`a63d717`** with the **O1-adverse condition live again** before tonight's arming.
**Tonight 00:06Z: the first verdict governed by released day-bar v2.**
**Tomorrow 00:00Z: the boundary.**

## 2026-08-28 ~15:15Z (MEM) — R-277..R-288: the chain restores, and a defect that was conservative

### The chain restores — in two states, not one

**Scientifically CLOSED:** the re-gate determination returned **IDENTICAL** on
the branch pre-declared before it ran, independently confirmed in the
coordinator's own run. **Not yet mechanically closed:** seam **47j stays
deliberately red** until the fit-time re-stamp. **Both are true and they are not
the same statement** — a red seam here is *a correct instrument reporting an
unfinished mechanism*, not a live defect.

**Restored:** v2.3, the freeze receipt and the increment-null stand **on the
re-derivation**; the do-not-cite interim lifts; BE's trajectory hold lifts. The
fit7 manifest stays **frozen and unedited**.

**Three method marks worth copying:**

- **Subject identity was checked first** — tape sha and byte count identical on
  both sides. *Comparing verdicts about different tapes proves nothing.*
- **The comparator was falsified before the answer arrived** — real-vs-real
  DIFFERENT at 24 fields, one-mutation DIFFERENT at exactly 1, byte-copy
  IDENTICAL at 0. **A comparison that cannot fail is not evidence** (rule 16,
  again).
- **Honest scope kept:** the two gate fixes are **not inert in general** — on a
  heterogeneous or large-header tape the new gate *refuses where the old passed
  silently*. tape6e is homogeneous with a small header, so they were correctly
  invisible **here**. The chain closes **and** the fixes still matter; either
  half alone misreports.

**A finding rode along, worse than it sounds: the verdict the whole chain rests
on was never in git.** 44 receipts tracked; the verdict cited *by sha* lived only
in the working tree. Both sides are now committed **byte-unchanged** — and the
distinction is stated: **committing preserves a frozen artifact; it does not
edit it.**

### The committed null: a real defect, and the conclusion stands

`phase2_increment_null`'s `sign_flip_p` was **two-sided where one-sided was
meant**, and it fed Holm across the canonical null's 12 cells — the numbers the
freeze receipt cites. **Say this precisely, because "same bug, committed
artifact" reads like an emergency until someone checks the sign:**

**For positive effects the two-sided p is ~double the one-sided, so every
surviving cell survives *more easily* under the correct test.** The defect was
**conservative in the safe direction**; the committed "10 of 12 chance"
conclusion **stands**.

**Dependence checked three times independently** — BE first, the coordinator's
four-line check, and mine at 15:15Z: 12 cells, three negative-increment cells at
**p 0.2334 / 0.2774 / 0.4208**, and **zero negative cells under p<0.10**.

**My first parse matched zero cells and would have reported "no negative
significant cells" vacuously** — the rule-16 class appearing inside my own
verification. The count falsifier (*must be 12*) is the only reason the check
means anything.

**BE did not touch the module** — editing it re-prices a committed receipt — and
routed the decision instead. The fix plus a **superseding vN+1 receipt** queue to
the first post-release cycle. **No summary may say "the canonical null was
wrong."**

### The post-release cycle is now four items

(1) annotation-merge **wiring**; (2) the manifest **re-stamp** — what closes 47j
mechanically; (3) the **tranches-persistence** decision; (4) the increment-null
**one-sided supersession**. Each queued deliberately, not deferred by neglect.

### Also on the record

- **A2 (the 2B reconciliation) was corrected twice, then confirmed on fresh
  data** — the settlement audit's **escape hatch did not fire**. The equivalence
  control moved from **decorative to load-bearing** (frozen-snapshot confound
  removed) and is now executing. An amendment corrected twice and then confirmed
  on data it had not seen sits in a different evidential position from one
  merely written well.
- **The trajectory run completed: 457,268 events, every anchor holds** — under
  the slice, parity/lifecycle only, economics excluded by the user's dispatch.
  Two honesty marks kept: the independence agreement is reported **as a result**
  rather than assumed, and the inputs are declared **easy** — a pass on easy
  inputs is a pass on easy inputs.
- **Two more user rulings landed in the A1 file** (Q3 gate-governs joins Q2 min).
- **Six truncation-class siblings today**, the sixth found by reviewer patterns
  as routed.

### Board

**Round 3 is FIRED at `a63d717`** (`REQUEST_BATCH3` under `reviews/`, scope
everything since `4a3d457`). Its four asks: the **011 hold decision**, **2B
freeze-fitness**, **O1-adverse flagging before tonight's arming** — the same
pre-ruled condition that postponed the last boundary, **live again** — and
re-execution of the pre-round-3 closures.

**Tonight 00:06Z: the first verdict governed by released day-bar v2** (DA
files). **Tomorrow 00:00Z: the O1 boundary**, coordinator cadence 22:30Z /
23:56Z / 00:00Z on 08-29. **2B goes to the user for freeze after this filing if
fit.**

## 2026-08-28 ~14:20Z (MEM) — R-254..R-276 swept: two releases, one open chain, and a second settlement correction

**A 23-entry gap** (R-254 → R-276, 51 commits). Reading the arc rather than
transcribing it; everything below is verified at the register entry or artifact
it cites.

### 1. Day-bar v2 is RELEASED — and released is not "the day passes"

The Codex batch-2 filing (`7f583d6` at tip `4a3d457`) verified with **no review
errors**. Day-bar v2 **governs coin-days ≥ 08-29**, the **R-256 register-side
inadmissibility interim lifts**, and **the 08-29 verdict re-runs under released
code**.

**Keep the distinction.** The filing **explicitly does not pre-judge 08-29**.
The bar is now *allowed to judge*; what it returns is a separate fact. A summary
that reads the release as a result would invent a verdict nobody computed.

### 2. O1 is CLEARED — the boundary re-arms

**2026-08-30T00:00:00Z**, per `O1_DEPLOY_RUNBOOK`. Coordinator cadence on 08-29:
**22:30Z confirmation, 23:56Z prep, 00:00Z deploy + era stamp + within-cause
verification** (never a throughput A/B). **The `clob_v3_1` working-tree hold
stays until the boundary** — `collect_pm.py` remains deliberately modified and
must not be cleaned, restored or committed by any seat.

### 3. The provenance chain on the committed result is OPEN

**BE's seam 47j fired: the gate that signed fit7's tape verdict is not the gate
that exists.** Live gate sha `58e48820d3cb209a` against the fit7 manifest's
`gate_code` `1da60b56e1fb2801` — **match False**; scoring against that manifest
now **refuses**, naming both values. Cause taken from git rather than inferred:
`da_state_tape_verify.py` changed twice since fit7 (`91a8949`, `f43359b`), both
**substantive gate-defect fixes**. R-225's content-binding did exactly what it
was built for — *the gate is bound by content, the fit by assertion*.

**BE's precision is the record, verbatim: it does not claim the verdict is
wrong — the chain no longer closes.** Those are different statements, and the
second is the one that holds. Prior evidence cuts toward unchanged, but
**spot-lines are not a re-gate**.

**Ruled: v2.3 and the freeze receipt are NOT to be cited as gate-verified** until
DA's **re-gate determination** — now the **queue head**, above round-2
verification — lands. **Both branches are pre-declared**, which makes it a test
rather than a repair: *identical* restores the chain and v2.3 stands on a
re-derivation recorded by a **superseding annotation** (the frozen receipt
untouched); *different* means provenance is **broken**, the **user is informed
immediately**, and nothing downstream is cited until re-derived. Trajectories
stay held.

### 4. This morning's settlement estimand is itself superseded

**I recorded full-window TWAP-vs-open at 10:21Z. That reading is refuted by the
repo's own passed reconstruction**, and I verified it at the artifact
(`live/pm_research/EXP_RESULTS_2026-08-20.md:10-17`, read 14:20Z):

| convention | n | agree |
|---|---|---|
| **S60(T) vs S60(t0)** | 1465 | **99.8 %** (99.9 % on >0.5bp) |
| meanS60[t0,T] vs S60(t0) | 1465 | **86.9 %** |

The gate required ≥99.0 % pooled / ≥99.5 % on the subset — **passed** — and the
artifact's own words are *"the full-range reading scores 86.9 % and is
refuted."* **The averaging window is w = 60 s, not the full 300 s range.**

So the likely estimand is **S60(T) ≥ S60(t0)**. **The tension is stated, not
silently resolved:** the market *description's prose* says "TWAP of the time
range"; the repo's *reconstruction* says S60 endpoints. DA reconciles A1.3
against the resolution artifact with the exact RTDS topic and boundary reader
pinned, unless a new independent same-population audit supersedes EXP_RESULTS.
**Do not let any summary retain the full-window TWAP as settled.**

**Consequence for the transform:** with S60 endpoints, the terminal-60 s
part-realisation is restated and **PM Identity/microprice are direct event
probabilities — no forced path integral**, which is what my morning "path
average" note implied. **2B is not fit to freeze.**

**Second settlement correction in one day, both caught pre-freeze, both by
review of a draft.** The reviewer knew the repo's own artifacts better than the
seats did — DA's amendment and the coordinator's ratification had read the
*description* and not the repo's own settlement foundation.

### 5. The user ruled on CLAUDE.md

**Option (a): `CLAUDE.md`'s program-tracking paragraph defers to
`SEAT_PROTOCOL.md` for P-2026-003** — fresh readers land on the register first,
**MEM's exclusive state-file ownership stands**, and CLAUDE.md's general rule
remains for single-seat programs. **The rule-9 parenthetical fix rides the same
edit.** The coordinator drafts the text; **the edit is applied by the user's hand
or on their explicit instruction — never by a seat unprompted.**

Note the compounding: the parenthetical is wrong about Binance, **and
"Chainlink TWAP-vs-open" is also not the right replacement** if the S60
reconstruction holds. The drafted text needs that second correction folded in.

**Ratified alongside it, and it binds this seat:** *a relayed "the user approved"
is not actionable for user-authority matters — the register cite is.*

### 6. Also on the record

- **011 stays dark**, four executed findings: Q4 **mislabels the candidate's own
  value as "increment vs incumbent" with no incumbent input**; A1.5 weights / n /
  first-crossing unimplemented; Q3 min without CIs; the receipt guard validates
  **keys, not contents** (24 empty dicts pass). Next 011 review is **non-fit
  again**.
- **`truncation read as complete` — five siblings**, the last two found
  *pre-emptively* by DA sweeping its own instruments. Sharpest form: **head
  proves presence, never absence.** Remedy: **quotes of rulings now carry line
  numbers**.
- **Escalation channel fixed** (R-273): user-escalations route through the
  coordinator **in the same breath** — an escalation that was *right in
  substance* sat two hours in the wrong channel. Companion ruling: **the
  argument's author is the seat that wants the work**, and **peer instruction is
  not user approval**.
- **BE's dry-run harness paid on first execution** — a live `NameError` that
  would have killed the real run at the first arm, after ~20 wasted minutes on
  1.1M rows, invisible to every component test. **The rule-17 class, caught by
  the instrument built for it.**

## 2026-08-28 ~10:21Z (MEM) — R-253: a false premise died before it could price anything

### The correction

**Settlement is CHAINLINK TWAP-vs-OPEN, ties UP — not Binance.** Verified at
`data/pm_5min/markets.jsonl` by my own run: **17,734 records, 17,734 mention
Chainlink, zero mention Binance.**

*(The register's 17,727/17,727/0 was correct as of its run one minute earlier;
the file grew 7 records in between. A live instance of rule 8 — every quoted
population carries its n **and** as-of, because the tape grows during
measurement. **The ratio is what carries, not the integer.**)*

Verbatim: *resolve Up iff the Chainlink TWAP over the title's range is ≥ the
price at the **beginning** of that range; otherwise Down.*

### Why this changes work, not just wording

**The settlement event is `P(TWAP over window ≥ the window's own open)` — a path
average against its own open, not a terminal-price comparison.** A
transformation built on terminal price would **price the wrong event**. And
**tie → UP is pinned by the venue**, not chosen by us.

That is the difference between a premise correction and a footnote: FP2 asked
for the price→probability transformation to be pinned *before* freeze, and the
transformation now has to be named against **this** event.

### What the pre-declaration bought

**DA's R-244 "closest to the settlement source" reading is VOID** — superseded
in-band. Note the *direction*: a positive increment from the bookTicker
challenger would now be a **genuine cross-venue lead — a stronger claim than the
voided reading allowed**. The coordinator's **decision-value complement survives
unchanged**.

**This is the whole argument for pre-declaring both readings.** They were on
record before any number existed, so the false one died **while there was still
no sign to build a story around**, and **nothing was ever scored on it** — the
draft was never frozen, no challenger ran.

**Two facts, deliberately kept separate because neither cancels the other:**
DA **erred first** in the draft **and corrected first**, filing against its own
rows; the coordinator **ratified the false reading on the register at R-244 and
so co-owns it**, corrected in the same entry.

**Rule 9 still applies, through a corrected door:** the tautology risk was never
Binance — it is that **`Identity` (the PM book) already prices this event**.
Incremental-to-Identity stands with its reason fixed.

Whether Binance flows into Chainlink's *aggregation* upstream is a mechanism
question about Chainlink's feed, **not the settlement rule** — and no instrument
of ours reads Chainlink at all.

### ⚠ CLAUDE.md carries the same false claim — user action, no seat edits it

**`CLAUDE.md` line 130**, reliability rule 9, says the target is derived from an
input *"(PM binaries settle on a Binance-derived price)"*. **That parenthetical
is false.**

- **No seat may cite it as fact.** Until the user's own edit lands, the
  `claudemd_rule9_parenthetical_is_FALSE` flag in `STATUS.yml` is the
  **correction of record**.
- **The rule itself is unaffected and still binds** — only its example is wrong.
- **`CLAUDE.md` is the user's file. No seat edits it** — not the coordinator,
  not MEM, and a peer cannot authorise it. Suggested replacement text is in the
  register and the flag.

### Also landed: rule 17, earned rather than filed

**`SEAT_PROTOCOL` rule 17 — "suite-green is not pipeline-wired"** (verified at
`SEAT_PROTOCOL.md:79`). A control that cannot **run**, distinct from rule 16's
control that cannot **fail**. It was adopted only **after batch 2 demonstrated
the both-halves closure** — DA's seam red→green with the producer's 100.0 s
charged, BE's evaluator 0 → 12 refs with a receipt-level all-cells guard — so it
lands with a cite instead of an anecdote. **Closure needs the wiring, an
artifact-level guard refusing output produced without it, and a seam test that
runs the integration the way production does.**

### Board

Batch 2's **gating halves are verified**; it completes on DA's FP1 (landed),
the 2B amendment on the corrected basis, and parity hardening. **One re-review
when complete**, then the O1 re-check and the **08-30 arming decision**. Deploy
remains **postponed to 2026-08-30T00:00:00Z**; tonight only the 00:06Z old-bar
verdict.

## 2026-08-28 ~10:09Z (MEM) — R-251: the deploy postponed itself

### Tonight changed — read this before anything else

**O1 moves to 2026-08-30T00:00:00Z.** Tonight there is **no arming and no
deploy**. The `clob_v3_1` working-tree hold **stays exactly as is** — and
unchanged is *correct*, not merely convenient: Codex's recommended fix is
**consumer-side** (the validator accepts a validated finite producer end), so
**committed v4 is untouched** and no new collector surface appears. **Tonight's
00:06Z verdict under the old bar runs unaffected** (btc expected FAIL,
`ACCRUES=False`).

**No user ask was needed, and none was made.** R-240/R-245 had already ruled the
deploy *"subject only to no adverse Codex filing before arming."* DB2 is a
verified adverse O1-relevant finding, so the postponement follows **by the
standing rule** — the condition was written before the finding existed, so
nothing was decided after seeing it. **This is what a pre-registration looks like
when it pays off operationally.**

**Consequences priced, not discovered later:**

- 08-29 runs on the **old collector**, so the post-O1 **P1 band expectation
  (~55–80 s/hr, and its `<45` / `>120` branches) moves to 08-30**.
- Under the day-bar hold, **08-29's v2 verdict is inadmissible** until release.
- Re-verdicting 08-29 **after** a release is **legitimate** — the bar predates
  the day and nothing is selected — so **the accrual clock pauses; it does not
  void.**

### Both holds maintained — and the findings are worth reading, not just counting

**Filing** at `reviews/CODEX_REVIEW_BATCH1_2026-08-28.md` (`641e326`, merged
`3ea72bc` — **merge not rebase**, so peer SHAs stay register-citable).
**Coordinator verified every claim before routing: no claim failed verification,
no review errors.** The reviewer also **independently confirmed BE's 126 and the
R-250 correction.**

- **DB1 — a coin-day published `PASS` over a 4,000-second outage.** Per-coin
  `all_pass` is computed *before* P1/P2/P3 attach. This is the R-238(a)
  recorded-not-enforced defect **recurring at the per-coin level after the
  whole-day path was closed**: fixing one path did not fix its sibling.
- **DB2 — the seam neither side owned.** Confirmed by execution: a
  producer-shaped `gap_open_at_exit` row (finite `gap_end_ns`, *exactly what
  committed v4 emits*) fed to `day_bar_v2()` **raises REFUSED**. **Both suites
  are green in isolation** while the integration always refuses when O1d fires.
  Neither owner's tests could see it; that is why it took a third party.
- **I11-1 — still live at HEAD, same blocker with a new face.**
  `h['Q2_sign'].get('auc')` at `runner:666` against a dict keyed
  `Q2_p_pos`/`Q2_p_neg`; post-`b3f082e` it would `AttributeError` where the tip
  `KeyError`'d.
- **I11-2 — the 24-cell evaluator is unit-proven and never invoked.** Zero
  references to `build_cell` / `assemble_family` / `sign_flip_null` /
  `cluster_disclosure` in the runner; Q4 composed then **discarded**.
- **I11-3 — a one-class head reports `status=OK`.**
- **Parity gaps:** `matched_control` **ignores its `cancels` argument** (0/1/6/99
  all yield 12); `battery()` emits two anchors; no external-arm interface.

### A new defect class, adjacent to rule 16 but not the same

**I11-2 is a control that cannot RUN, where rule 16 covers a control that cannot
FAIL.** Every test of that evaluator passes; none of it executes in the path that
produces the artifact. **Green suites prove the unit, not the wiring**, and a
coverage claim that counts *tests* rather than *call sites* cannot tell those
apart. Recorded as `defect_class_unit_proven_unwired`.

Closure requires **both** halves: wire it into `main()` **and** ship an output
known-bad that **refuses a receipt lacking all 24 cells**. The second is what
makes the wiring self-policing instead of a one-time fix.

### 2B freeze deferred — and freeze-after-review is now twice vindicated

**Do not present the draft.** **FP2: a dollar mid is not a probability** — the
price→probability transformation must be pinned before freeze (reference, tie,
horizon, vol lookbacks). The draft also leaves decision-bearing terms unfrozen
("systematically later", "materially below", budgets, min-n, statistic, alpha,
weighting), needs a declared **common eligible universe**, and needs a controlled
synthetic fixture for the constant-lag falsifier. **FP1** confirmed by execution:
`FairPrice(value=60000.0, status=OK)` constructs with **no invariant
enforcement**.

**Worth stating plainly: FP2 would have been frozen in.** That is the second
vindication of freeze-after-review (R-243) — the first was 011's Q4 algebra —
and both were caught by reviewing a **draft** rather than amending a frozen
document.

### Batch 2 routed

**DA, seam-first:** (1) DB2 consumer fix **plus the real seam test** — fake O1
socket → actual emitted ledger row → `day_bar_v2` consumes it, which is exactly
the test whose absence let two green suites hide a broken integration; (2) DB1
per-coin P1/P2/P3 with the quality/accrual split in the per-coin artifact;
(3) FP1 record-boundary invariants; (4) the 2B amendment; (5) parity hardening.
**BE:** I11-1 print path, I11-2 wiring **plus** the output known-bad, I11-3
head-level `UNEVALUABLE`. **One re-review when complete.** The **arming decision
for 08-30 follows the next filing.**

## 2026-08-28 ~10:03Z (MEM) — standing convention: `updated:` is a rolling window of three

**Ruled by the coordinator after MEM raised it and did not execute it.** The
`updated:` field had reached **255 lines / 18.7 KB with 15 PRIOR generations**
(10 written in one day), so a cold reader met `flags:` at **line 261 of 3225**.
It had become a second narrative history duplicating these HANDOFF sections, in
front of everything else in the file.

**The convention, binding on every future sweep:**

1. `updated:` keeps the **newest three** entries.
2. Older entries move to **`workspace/STATUS_UPDATED_ARCHIVE.md`** — **moved,
   never deleted**.
3. **On each sweep, move the overflow in the same commit**, so the field never
   regrows.
4. The archive is **append-only**, one dated batch per move.

**Effect of batch 1:** `flags:` now begins at **line 77 of 3041**; the field is
5.0 KB instead of 18.7 KB. 13 entries archived.

**Nothing was lost, and that is checked rather than asserted:** the archive
documents the exact join rule, and reconstructing the field as it stood at
`1bc65cc` from the kept entries plus the archive body reproduces it
**byte-for-byte (18,709 = 18,709 chars)**. If an automated reader turns out to
need the full chain, it is in the archive and in git history — that recovery
path is why moved-never-deleted was the design rather than trimming.

## 2026-08-28 ~10:00Z (MEM) — R-250: the count was never a disagreement, and my instruction was wrong

### Correcting this file first

The previous entry told a future reader to take the script's **134** and treat
BE's **126** as a superseded hand figure. **That instruction was wrong and would
have propagated the error.** It is superseded here; the old text stays as the
record of a wrong instruction this file gave.

**There was never an instrument disagreement about how many tests exist.** There
were two *different measurements*, one mislabelled. The pre-fix
`falsifier_count.sh` requires a module argument (`${1:?module}`); the
coordinator's invocation passed none, the script **died at the arg check** with
stderr suppressed by that command's own `2>/dev/null`, and its `||` fallback —
`grep -c "known_bad\|ok("` over **source text** — produced 81/38/15, which was
then recorded under the instrument's name. **Static source-grep and runtime PASS
lines are different measurements.** BE's 126 (anchored `"^  PASS "`, executed)
was correct at `e72dd4c`.

**Current truth — verified by my own run of the fixed script at HEAD `1444f84`,
not taken from the entry:**

| module | falsifiers |
|---|---|
| `phase2_iter011.py` | 81 |
| `phase2_iter011_run.py` | 39 |
| `phase2_annotation_merge.py` | 11 |
| **total** | **131** |

The +5 on the runner are the new Q2 falsifiers.

### My own miss, named

I verified that the register **said** "by the counting script." I did not verify
that the **script produced it** — and then wrote an instruction telling future
readers which number to trust. **CLAUDE.md rule 16 says verify at the artifact
the claim names**; the claim named the script, and I checked the entry citing
it. **When a state file is about to tell readers which of two conflicting
numbers to trust, run the instrument.** It took one command.

### Two lessons adopted for both seats (R-250)

- **A count without a commit ref is not a measurement** — 126 was quoted
  unref'd, 134 was mislabelled.
- **An instrument's name may only be attached to numbers the instrument actually
  produced** — CLAUDE.md rule 16's "know what *kind* of document you are
  reading," applied to one's own tool output.

### BE's conduct is the part worth copying

BE **refuted both natural explanations by measurement** — the unanchored-grep
delta was exactly zero, and never-executed `ok(False)` branches fit one module
while failing the runner at 43 ≠ 38 — and then **refused to put a
plausible-sounding wrong reason into the register.** The true cause was
*invisible from its environment*, because the other seat's command hid it.
**Declining to supply a cause you cannot see is correct even when a plausible one
is available.**

The **"held open, already paid" framing stands**: chasing the wrong number is
what put BE inside the helper where the bare fallback lived. And the **rule-15
soft spot I flagged dissolves with the cause** — the register claim had not
drifted from the code; it **named the wrong instrument**.

### Q2 both-sides verified, and how it composes with the min ruling

BE's fix is verified at `b3f082e`: the gap was real (p_pos 0.92 / p_neg `None`
→ cell 0.92 before the fix). Now **min adjudicates only when both sides are
measurable; otherwise the cell is UNEVALUABLE and still occupies its Holm slot.**
That composition is the thing to quote — the ruling and the guard do different
jobs and neither replaces the other.

### The tooling shape, named

**A helper built this morning to enforce discipline** — don't type counts from
memory — **acquired the capability to start a research run nobody asked it to
have, and was found by *use*, not by review.** Fallback removed; `--selftest`
only, under a timeout.

### Unchanged

Codex round executes at `e72dd4c` as scoped; `b3f082e` closes the flagged `:231`
path and is available for verification. Tonight: **ON but not unconditional**,
22:30Z confirmation, arming ~23:55Z, 00:00:00Z deploy, 00:06Z verdict under the
old bar.

## 2026-08-28 ~09:57Z (MEM) — R-249: one gate closes, and it closes by ruling

### Gate arithmetic — read this precisely

The 011 fit **was** blocked by **(a)** Codex `HOLD RELEASED` and **(b)** the
user's Q2 ruling. **(b) is now satisfied** — *ruled*, not dissolved, and not
discovered to have been unnecessary. So fit clearance now blocks on **(a) alone
and nothing else: a clean review does clear the fit.**

I am spelling that out because this is the update most easily misread. The
previous entry was written *against* collapsing two gates into one; this entry
closes one of them legitimately. R-249 carries the sentence explicitly so the
next reader can neither **re-collapse** the gates into one that was never there,
nor **re-split** them and hold the fit against a ruling the user already made.

**The ruling:** Q2's cell statistic is `min(AUC(p_pos), AUC(p_neg))` — the worse
side; half a working head cannot carry a cell; the family stays 24. Recorded as
a ruled block under A1.4 **in the frozen file**, so the gap is closed where the
design lives rather than in a message. BE's pre-ruling implementation at
`phase2_iter011_run.py:231` is authorized.

### The one-side flag was real — and BE's fix is rule 16 applied on the spot

The subtlety routed into the round turned out to be a live defect (`b3f082e`).
`report_arm` **filtered `None` out of the two sign-head AUCs and took `min()` of
what remained**, so one side could carry the cell. **Demonstrated before
fixing:** `p_pos` AUC 0.92 beside `p_neg` AUC `None` produced a cell of **0.92**
— the surviving side sailing past the `UNDERPOWERED` machinery **as though the
pair had been measured**.

Sign discrimination needs **both** sides. If either is missing or underpowered
the **cell** is unevaluable, with the reason in `Q2_cell_status` /
`Q2_cell_detail` and the requirement stated in `Q2_cell_rule` **so it travels in
the artifact rather than living in a commit message.**

Four falsifiers — and the fourth is the new house rule applied immediately:
three refusal cases (`p_neg` None, `p_pos` None, `p_neg` UNDERPOWERED → all
`None`) **plus both-evaluable yielding the worse side** (0.61 of 0.92/0.61),
**so the guard is proven not to be a wall.**

### Rule 16 is adopted

**`SEAT_PROTOCOL` rule 16: a control that cannot fail must never be mistaken for
a control that passed** (verified at `SEAT_PROTOCOL.md:71`). It consolidates the
four named instances — a fixture supplying what the code should produce; a guard
shown only to refuse, so boundary positive controls must **admit**; an anchor
that includes the arm name and therefore passes nothing and fails nothing; a
falsifier that enshrines the defect as spec — and is **phrased on the control
side deliberately, because the next instance will wear a shape none of these
four had.** The coordinator's completing clause: **every control ships both
directions — it fires on the bad case *and* admits the good one.**

### The unresolved count discrepancy already paid for itself

The 126-vs-134 count is **still not reconciled**. But holding it open instead of
quietly accepting a number is what sent BE looking — and looking found **a worse
defect in its own tooling**: `falsifier_count.sh` **fell back to running a module
bare** when `--selftest` produced no PASS lines. For the runner, bare means
`main()` — **the heavy data path, launched from the session shell**, which
R-148(3) forbids and which the now-binding resource cap makes a live hazard. BE
triggered it accidentally; it ran two minutes before timing out; no stray process
survived, verified. The fallback is removed — the helper now only ever invokes
`--selftest`, under a timeout.

**The principle worth keeping: a helper for counting tests must not be able to
start a research run.** And the meta-lesson: a small discrepancy held open is
cheap, and this one bought a resource-cap violation path that nobody was looking
for.

### Tonight, unchanged

**ON but not unconditional** — an adverse O1-relevant Codex finding before
~23:55Z arming postpones the boundary. 22:30Z confirmation, arming ~23:55Z,
00:00:00Z deploy, 00:06Z verdict under the old bar (btc expected FAIL,
`ACCRUES=False`).

## 2026-08-28 ~09:53Z (MEM) — R-248: batch complete, and two gates that are not the same gate

### The batch closed and the round fired

BE's five A1.8 steps landed (`50277fb` / `6f559fc` / `9ace8c1` / `f9fb032` plus
frozen A1); **all three suites green in the coordinator's own run**, not from
BE's report. Batch tip `e72dd4c`; scope request filed at
`workspace/reviews/REQUEST_BATCH1_2026-08-28.md` — the R-239 location, and **no
state-file collision this time**.

### Fit is blocked by TWO independent gates — do not treat either as the only one

1. **Codex HOLD RELEASED** at the batch tip.
2. **The user's ruling on Q2's cell statistic.**

BE implemented Q2 as **`min(AUC(p_pos), AUC(p_neg))`** — the *worse* side, so
**half a working head cannot carry a cell**. Pre-declared before any number,
conservative, family-preserving. But the choice **fills a gap in a user-frozen
amendment**, so it is the user's: min / mean / separate cells (the last would
**change the 24-cell family**). It is **blocking fit clearance and is not a
review matter** — a clean review does not by itself clear the fit.

One subtlety was flagged *into* the round: at `:231` a side whose AUC is `None`
**drops out of the min**, so the reviewer must check the one-side-unevaluable
path cannot let a single side carry a cell past the `UNDERPOWERED` machinery.

### The bias falsifier checks both directions — that is the part that matters

The A1.1 Option-1 bias algebra was **independently hand-verified**: on 2 harm /
1 good / 1 zero with `m_good=2`, `p_zero=0.25` → bias **+0.5000 exactly**
(amended 2.0000 vs superseded 1.5000). And the falsifier asserts **both**
directions — **with no zero mass the amended and superseded forms agree.**

That second assertion is the real check. **An amendment that changed the answer
everywhere would be a different estimand, not a correction.** A one-directional
falsifier would have passed either way.

### Two BE self-catches, one lesson

- **A falsifier had enshrined a defect as the spec.** The old assertion
  `signed_v_cancel({}) == 0.0`, filed as *"0, not a crash"*, **was the fail-open
  written down as intended behaviour** — policed by the very test meant to catch
  it. It is now the *documented defect* at `phase2_iter011.py:371`, **inverted
  into a refusal test**.
- **The runner's `row()` helper manufactured the exact malformed pair A1.3
  bans** (zero shares with nonzero value), so **every prior test ran on rows that
  cannot occur.** The strictness caught its own test harness on first contact —
  **which is the argument for the strictness, not against it.**

**Filed into the standing defect vocabulary**, because this is now the fourth
face of one error: *a fixture that supplies what the code should have produced*;
*a guard shown only to refuse may be refusing everything*; *an anchor that
includes the arm name passes nothing and fails nothing*; and now *a falsifier
that enshrines the defect as spec*. **All four are a control that cannot fail
being mistaken for a control that passed.**

### Counts recorded from the script, not the message

Falsifier counts **by the counting script: 81 / 38 / 15 = 134.** BE's message
said 81 + 34 + 11 = **126**. Recorded **from the script**, per BE's own
`001991c` lesson — read the count, never hand-type it. The script's numbers are
*larger* (the over-delivery direction), which is why this is **queried, not
blocking**. **Unresolved:** a later reader takes the count from the script and
treats 126 as a superseded hand figure.

### House rule adopted

**`SEAT_PROTOCOL` rule 15** — *"a register entry citing a property of code
carries a check behind it in that code's suite"* (DA's standard, MEM's proposal
to make it a rule, coordinator's adoption; cites R-247). Verified at
`SEAT_PROTOCOL.md:66`. Filed as **consolidation** — the register stays
authoritative.

### Tonight

**The deploy is ON but not unconditional.** Per R-240, **an adverse O1-relevant
Codex finding before ~23:55Z arming postpones the boundary.** Otherwise
unchanged: 22:30Z confirmation, arming ~23:55Z, 00:00:00Z deploy, 00:06Z verdict
under the old bar (btc expected FAIL, `ACCRUES=False`).

## 2026-08-28 ~09:48Z (MEM) — R-246/R-247: pinning what "identical" means, and refusing the first exemption

### The lane-4 battery, and why its definition matters more than its count

`da_replay_parity_battery.py` — **14 checks, 0 failing, reproduced in the
coordinator's own run** (`7815c2f`). Typed synthetic stubs, zero filesystem
reads, 0.10 s / 18 MB, **correctly self-classified as light class** under
R-148(3), so no slice was needed.

**"Bit-identical" is now defined over `replay_traj_canon_v1`** — sorted keys,
compact separators, UTF-8, `allow_nan=False`, events ordered by `(t, seq)` and
never dict order. This reuses the `annotation_canon_v1` recipe for the same
reason: **an unpinned signature rule makes every valid comparison fail
indistinguishably from a real difference.** The canonicalisation is
semantic-preserving *in the direction that matters* — it removes only
representation noise, while floats serialise by repr with **no tolerance**, so
"differently ordered but wrong" still breaks parity.

**The arm name is excluded from the canonical bytes, and that exclusion is
asserted as its own check.** Including it would make the anchor **pass nothing
and fail nothing** — the decorative-anchor trap the spec warned about.

The falsifier structure is the part worth copying: the perturbation check (one
extra cancel breaks parity) is **paired with a same-arm digest reproduction**, so
a break is a *real difference* rather than run-to-run noise; an enabled predictor
that cancels is proven non-identical, so the battery can tell a real difference
from none; an uncancelled generation fills normally, so `STALE` is not simply
what the harness always says; the matched control is checked **non-empty**;
determinism is crossed over `PYTHONHASHSEED` so it cannot inherit blocker-7's
class; and an empty run is **NOT EVALUABLE**, never seven passes. R-235
separation held throughout — **DA built the checker, BE's arms remain the
checked.**

### Signed zero stays unnormalised — the first exemption is the one that counts

`0.0` vs `−0.0` **stays unnormalised** in the parity canon (R-247(2), DA at
`6b25b7e`, suite 14 → 18). Two reasons, both kept:

1. Normalising it would be **a tolerance by another name** — the first exemption
   in the one comparison whose entire value is admitting none.
2. A signed-zero digest difference between real arms is **informative**: it
   betrays a **different computational route to the same zero**, which is exactly
   the coupling class the anchor exists to surface.

The edge is **declared in-file and asserted in-suite**, so a future firing reads
as a **finding, not a flake**. Verified claims behind the ruling: ULP-apart
values digest differently, identical values reproduce, NaN refuses at
serialisation. **The normalisation question is explicitly in scope for the Codex
round** as a parity-contract matter — any change there is a *ruled contract
change*, never a quiet loosening.

### A standard worth carrying beyond this program

**"A register entry citing a property of my code should have a check behind
it."** The register had cited a property of DA's code; DA answered not by
agreeing but by **asserting it in the suite**. That is rule 15's logic pushed up
a level: **the register is an instrument too**, and a claim in it about code
behaviour that no test pins can drift away from the code without either one
noticing.

### Provenance upgrade completed

The idle-dispatch narrow reading (**coordinator dispatches; seats never
self-dispatch; MEM acquires no plan work by being idle**) is now
**register-backed at R-247(1)** — and I verified the entry exists and says so
*before* upgrading the flag, which was the entire point of the line being
upgraded. The superseded provenance line is kept as the record of how the claim
was sourced. R-247(1) adopts the framing and ratifies the refusal to let an
in-session confirmation stand as register-backed, as the R-14 pattern applied to
MEM's own mandate.

### Tonight

**Runbook pre-checked ~14 h ahead** (R-246): working tree holds `clob_v3_1`,
HEAD holds `clob_v4` — the deliberate hold, exactly as documented — and
`pm-collector-clob.service` is active. Sequence, structural verification
(within-cause, never a throughput A/B) and abort path all defined.

**Batch:** DA has three deliveries complete (`d97c23e`, `612346b`, `7815c2f`)
plus the signed-zero hardening, all riding this round. **Outstanding: BE's A1.8
steps 2–5 only** (2, 2b, 3 landed; mid-flight on 4–5); idle-notify armed on BE;
the Codex round prompt is pre-drafted so the round fires on BE's flag.

## 2026-08-28 ~09:44Z (MEM) — R-243..R-245: freeze-after-review, and a story that cannot be chosen later

### The 2B draft rides the batch — the 011 lesson pointed forward

DA's **Identity artifact** landed (`d97c23e`) and the coordinator verified it by
**its own run, not DA's report**: 21 selftests, 0 failing; authorization stated
**in-file** against the user's committed plan `d506a06`, so the relay was checked
against the user's own text before building and that reading is auditable in the
file.

Four design points worth keeping, each a defect class paid for elsewhere in this
programme:

- every refusal is `value=None` **plus a tallied status** — a zero is
  indistinguishable from a real 0.0 price;
- `read_as_of()` keeps **unknowable-at-t** and **inadmissible** as *separate*
  refusals — collapsing them would hide look-ahead behind a data-quality
  message. A future-dated record is **valid**; as-of is a *consumption* rule;
- boundary positive controls **admit** exactly-at-minimum depth and
  exactly-at-bound freshness — **a guard shown only to refuse may be refusing
  everything**;
- the complement check reports **NOT CHECKED** on an inadmissible side rather
  than passing.

**The 2B protocol draft rides the current batch by ruling: Codex reviews it
before the user is asked to freeze it.** That is the 011-preregistration lesson
applied *forward*, and the reasoning is the durable part — **an amendment to a
frozen document costs a user act; a review comment on a draft costs nothing.**
Waiting costs hours, and the challenger race clock starts at freeze either way.

### The second challenger is named — and both readings of a win are pre-declared

**Binance USDM bookTicker mid** (`612346b`), named **before** the review.
Against DA's own closed-set clause that was the right call: the reviewer sees a
real candidate, not a placeholder, and **naming it after review would have been a
new family.**

Tape claims verified at the data: 213 hourly files per symbol from `20260819_12`,
columns exactly as declared, freshness arithmetic consistent. **Era floor is the
ledger boundary of record** — `recv_ns >= 1787579334881534478`
(2026-08-24T13:48:54Z, hf_ws_v2 stamp): pre-boundary instants are
**inadmissible for this challenger, not merely noisier**, and §7.4's
admissible-count parity already tests the strictly smaller population that
creates.

**Both interpretations of a positive increment are pre-declared, and both are
kept — neither compresses into the other:**

1. **DA's, binding in the draft:** the challenger sits **closer to the
   settlement source** than Identity, so a positive increment is **not
   forecasting skill**. It says the PM book **lags** the settlement venue — a
   latency/microstructure fact that must not be narrated as alpha.
2. **The coordinator's complement, equally pre-declared:** lane 2's deliverable
   is the best admissible point-in-time fair price **for decisions** — toxicity
   residuals, predictor features, quoting — and "the PM book lags the settlement
   venue" is precisely the class of fact a fair-price successor **exists to
   capture**. Not alpha, but **decision value**.

**What is now excluded is choosing the story after the sign.** The
admit-versus-one-challenger-family call is the **user's**, at freeze time,
post-review, with both readings already in hand.

### Two standing user directives

- **The aggregate resource cap is binding on all research work.** Verified live
  at the instrument rather than read from a doc: `research.slice`, MemoryMax
  18.4 G, CPU quota 1200 %, ~3 G in use at the check. **Enforcement is at
  clearance time, not crash time** — every heavy-run clearance carries the
  `systemd-run --slice=research.slice -p MemoryMax=…` pattern *in the clearance
  itself*. Named risk points: BE's eventual 011 fit/score after HOLD RELEASED,
  and DA's at-scale bookTicker processing.
- **Idle seats take the next admissible plan job.** **Note the subject:** this is
  a standing practice for the **coordinator** to dispatch idle seats. It is not a
  licence for a seat to self-dispatch outside its surface — **MEM does not
  acquire plan-execution work by being idle**; it takes such work only on
  dispatch. Batch boundary rule came with the dispatch: lands before BE flags →
  rides this round; later → opens the next; **no rushing half-work to catch a
  boundary.**

DA is now on the **lane-4 seven-arm parity stub battery** (`7815c2f`): typed
stubs, **bit-identical** anchor (after the summation-order finding, "close"
cannot be distinguished from "differently ordered but wrong"), perturbation
falsifier, `PYTHONHASHSEED` determinism so it cannot inherit blocker-7's class,
and an empty run must not report seven passing arms. **Nothing scored.** Seat
separation holds: **DA builds the checker, BE's arms remain the checked.**

### Gate model, restated at the user's request

**Fixes verified is not HOLD RELEASED.** Both holds lift only on the reviewer's
explicit words at the exact batch commit, however thorough the fixes. Work ruled
**safe** proceeds on its own gates instead — the 2B draft's gate is a user
freeze, post-review. Companion phrasing for the other hold: **cleared to build is
not cleared to fit.** Same failure, different nouns.

**Batch outstanding: BE's A1.8 steps 2–5 only.** Cadence unchanged: 22:30Z
confirmation, ~23:55Z arming, 00:00:00Z deploy, 00:06Z verdict (btc FAIL
expected, `ACCRUES=False`).

## 2026-08-28 ~09:31Z (MEM) — R-242: A1 frozen before the first number; DA to fair price

### The sequencing is the result here

The R-238 blocker-1 **Q4 algebra flaw was data-independent** — it could be
established as wrong from the estimand alone, with no data read. So it was
fixed by a **user-frozen amendment before any iteration-011 number existed**,
rather than discovered afterwards and argued over. That ordering is the whole
reason the pre-fit review was worth running.

**Amendment A1 FROZEN by the user** (R-242, 09:29Z;
`plans/ITER011_PREREG_AMENDMENT_A1.md`). Drafted by BE, **frozen by the user and
only the user** — BE does not amend a frozen document it authored, and it
committed-but-did-not-push a user-facing draft without the user's word, recorded
as correct boundary conduct.

- **§A1.1 = OPTION 1:** separate `p_positive` / `p_negative` heads,
  `value = p_pos·m_harm − p_neg·m_good`, `p_zero` implied and reported. The
  frozen form's bias was `m_good·P(V=0)`, downward — hand-checked by the
  coordinator, matching the reviewer.
- **The Holm denominator is FIXED at 24, and unevaluable cells occupy their
  slots.** That is what actually closes the shrinkable-family blocker: a cell
  that cannot be evaluated still costs its share of the correction.
- **`PERM_SEED=20260828` with sorted-key consumption (R-234)** — blocker 7's
  determinism lesson carried *into the new design*, not left behind as a one-off
  repair on the old instrument.
- Feature fence **name-bans `any_fill_ahead`** (BE recorded its own earlier
  `FENCE_REVIEWED` admission as wrong); strict-refuse target construction;
  action-unit `n` with `1/rows_in_generation` weighting;
  `UNDERPOWERED_MIN_N=100`, ridge 10.0.
- Arms, heads and family unchanged (2 × 4 × 3 = 24). **No clock effect:** 011
  has no forward clock, and the hazard candidate's race clock (`b3f7f9f`) is
  untouched.

**Verified here, not accepted:** the frozen preregistration is **byte-untouched
since `3b71d3e`** — `git diff` is empty. The amendment is a separate document
and rule 13 held.

### Cleared to build is not cleared to fit

BE proceeds **A1.8 steps 2–5 red-first**. The **fit/score hold still stands** and
lifts only on Codex's **HOLD RELEASED** at step 6. Same distinction as the
day-bar hold below: permission to work on a thing is not permission to produce a
number with it.

### DA dispatched to the fair-price lane

On the user's directive that the hazard line proceeds per
`HARMFUL_FILL_HAZARD_TOXICITY_PLAN.md`, DA turns its own lane-2 spec (`6fc96e2`,
ruled safe-as-spec) into the **typed Identity artifact** — dual timestamps
(source vs local-knowledge, **whose gap is exactly where look-ahead enters**),
strictly-as-of consumption, admissibility statuses, red-first selftests — and
**drafts** the 2B challenger protocol (≤2 predeclared challengers, incremental
to Identity, multiplicity recorded) as **DRAFT-FOR-USER-FREEZE**. **No challenger
scoring until frozen.** A challenger that cannot produce both timestamps is
**inadmissible, not degraded** — absence must never read as zero freshness.

### Housekeeping that matters for whoever writes these files next

- **The reviews directory now has its first filing**
  (`workspace/reviews/CODEX_REREVIEW_DAYBAR_V2_2026-08-28.md`). The R-239
  location works and the next round's prompt names it explicitly, so the
  collision that interrupted my previous sweep has a home. Reviewer text already
  inside these files stays **verbatim**.
- **The seat-handoff identity check is ratified as a standing pattern** and binds
  future MEM sessions: **a state-writer confirms rather than assumes across a
  seat handoff.** A sweep request from an unfamiliar sender is executed only
  after the seat and the cited artifacts are confirmed. The check is cheap; a
  state file written from an unverified claim is the failure this programme
  keeps eliminating everywhere else.

### Unchanged and still true

Day-bar v2 remains **held** for 08-29 (**fixes verified is not hold released**).
Tonight: 22:30Z confirmation, ~23:55Z arming, **00:00:00Z deploy**, 00:06Z
verdict under the **old** bar with btc expected to FAIL and **`ACCRUES=False`**.
`collect_pm.py` stays held at `clob_v3_1` until the boundary.

## 2026-08-28 ~09:24Z (MEM) — R-236..R-241 swept: a reviewer, two HOLDs, and an epoch

**Sweep note first, because it changed how I write here.** Between my last sweep
and this one, the Codex filing (`df1ef73`) and R-238 (`8a74571`) both wrote into
`STATUS.yml` and this file. That is the collision shape R-239 names, and the
fix is already ruled: review filings belong in `workspace/reviews/`. **Their
text is left verbatim** — the two reviewer-authored flags
(`iter011_prefit_review`, `day_bar_v2_judgment`) and the 08:46Z section below
are theirs; superseding state sits *beside* them, never over them.

### The user designated a system reviewer

**Codex (tmux `pm-codex`) is the system reviewer** (R-238, the user's word);
pm-co coordinates follow-up. **Standing protocol** (R-239): build →
commit+push → Codex reviews → Codex commits+pushes its review → coordinator
**verifies every claim by execution** (a claim is either a reproduced defect or
a review error, established by running, filed either way) → fixes land
**red-first** → re-review → only then proceed. **R-240 refinement, the user's:
finish ALL fixes in a round, then ONE review round — never several per round.**
The batch waits on BE's 011 half; contents so far `99d0573`, `c288ed1`,
`f8581b6`, `9bcc208`.

### HOLD 1 — iteration 011: no fit, no score

Five blockers. **The first is data-independent, which is why it is being fixed
before any number exists rather than argued about after one:** the frozen **Q4
algebra is wrong** whenever `P(V_cancel=0|preventable) > 0` — `(1 − p_harm)` is
not `P(V<0)` under zero mass while `m_good` is measured on `V<0` only. A
**superseding preregistration amendment** is required: BE drafts, **the user
freezes**, and that is sequence step 1 — nothing lands before it. The rest:
target construction **fails open**; `any_fill_ahead`, an *outcome* field,
admitted by the feature fence; Q2/Q3 predicting on realized conditional subsets
so action-time Q4 cannot compose; and metrics/Holm accepting subsets, **so the
frozen 24-cell denominator can shrink**. Slices 1–4 landed and fit/score
nothing.

**R-237's pre-run ruling stands** (a gap BE raised *before* running, which is
the right moment): Q2/Q3 incremental cells carry `NO_INCUMBENT_COUNTERPART` as
a **declared status** — rule 4 applied to null cells — because the incumbent
never decomposed sign from magnitude, and inventing a baseline would be the
inverse of rule 9. **The frozen document was not edited.**

### HOLD 2 — day-bar v2 must not judge 08-29 yet

DA's five re-review blockers are **closed at `f8581b6` and verified by
execution** (R-241: suite 63 reproduced in the coordinator's own run; launcher
passes the ruled epoch; the coverage guard is `coverage_observed is True` only,
so `None`/`False`/omitted/malformed all refuse; structural validation precedes
coin filtering with a counted `n_structural_bad`; P3 exactness checked twice, by
breakpoint argument and independent recompute).

**The hold is still in force.** It releases only on Codex's explicit *HOLD
RELEASED* at the next batch review, and **no 08-29 v2 verdict is admissible
until then. Fixes verified is not hold released** — the two are separate events
and collapsing them is how a held instrument quietly starts judging days.

**P3 grounding column corrected in-band** (R-241): 08-26 278.6 → **283.2**,
08-27 256.3 → **258.9**; 08-25's 301.2 unchanged, so §2's anchor sentence
stands; both corrected values still pass ≤900. No verdict changes, no threshold
moves, no clock effect; superseded values kept as provenance.

**Keep this framing exactly, because the compressed version is wrong:**
Q-DA-115's implementation-vs-table "match" was **agreement between two runs of
the same defect** — the table's P3 column came from the same defective aligned
stepping the implementation used. It must **not** become "validated against the
table." Validation against a grounding table is only as good as the instrument
that built the table.

**DA filed that correction against itself, unprompted, in the same message that
reported its own fixes** (Q-DA-116, `233c95e`), and adopted the rule from it:
**a filing may not say VALIDATED unless the entry point was exercised the way
its launcher invokes it.** Ten defects had followed the "validated" claim, all
in *consumers* — `all_pass` not reading the bars, the CLI not reading its own
renamed keys, the launcher not passing the epoch, a coverage guard's `is False`
letting the default `None` through (N/A-vacuity *inside a guard added the day
before to close a fail-open*), validation running after coin filtering so a
reversed interval scored −50 lost seconds and passed.

### Race accrual now has an epoch — and day quality is split from it

`entirely_post_freeze` is governed by the **freeze-commit epoch** (`b3f7f9f`,
1787897340 = 06:09:00Z) per the receipt's own `clock_starts` clause, so a
mid-day freeze means **08-28 must not accrue** toward the btc candidate's five
days. **DA corrected the coordinator's ruling text and the correction is the
substance:** "day quality unaffected, accrual changes" was not true of the
implementation, because `entirely_post_freeze` fed `all_pass` — a
**healthy-but-early** day would have read as **bad** rather than as
good-but-not-counting. `split_verdict()` now reports `day_quality_pass` /
`post_freeze_pass` / `race_accrual_eligible` separately, falsified three ways.
Material effect tonight: zero. **08-28 reports `ACCRUES=False`.**

### Tonight is ON

Both O1 conditions were met **~13.5 h early** (R-240): coordinator behavioral
tests 10/10 driving the *real* `PMCollector._market` from a **git-extracted** v4
copy against fake sockets — the held working tree never touched, asserted by a
`COLLECTOR_VERSION` check — plus DA's `gap_open_at_exit` integration at
`f4fafe6`. **Boundary deploy ON for 2026-08-29T00:00:00Z**; **22:30Z is now a
confirmation check, not a decision point**; arming ~23:55Z; subject only to an
adverse Codex filing in between. Tonight's 00:06Z verdict runs under the **old**
bar.

### Closed since the last sweep

- **Freeze receipt v2 LANDED** (`68dca00`) — and I verified it at the artifact
  rather than from its commit message, as promised:
  `THIS_IS_A_CITATION_CORRECTION_NOT_A_RE_FREEZE` anchors
  `race_clock_start_commit: b3f7f9f`, `race_clock_UNCHANGED: true`,
  `multiplicity_UNCHANGED: true`, and states outright that reading v2's commit
  date as a new freeze "would hand the candidate days it did not earn." The
  null pointer now reads `163bd36`. **v1 frozen untouched.**
- **The kill mystery is resolved** — reviewer PID-stops, not OOM, not the code.
  Kept as a lesson pair: BE's boundary-nailing and its refuted-by-measurement
  memory hypothesis were **correct method on an external cause**, while a
  released transient unit's **default** fields (`Result=success`) were read as
  an observation and sent a hunt after a swallow that does not exist.
- **Lanes 2+4 specs delivered** (`6fc96e2`), spec only, no scoring authorized.
- **`SEAT_PROTOCOL.md`** (`a5d010b`) consolidates six seats and 14 standing
  rules — **consolidation, not new law**; the register stays authoritative.

## 2026-08-28 ~08:46Z — independent pre-fit review: HOLD 011 before numbers

The user explicitly requested review of the committed 011 machinery before any
fit. Three fit processes launched concurrently by another workspace session
during the review were stopped by exact PID; no `*iter011*` derived artifact
exists. Commit `bac5469` subsequently added a silent-success output guard, but
its `__main__` check also makes an explicitly requested `--selftest` print GREEN
and then exit nonzero because a self-test correctly writes no result artifact.

**011 blockers:** (1) the frozen composition is algebraically wrong when
`P(V_cancel=0 | preventable)>0`; that is known before data and must be amended
before fitting, not decided after observing the zero fraction; (2) missing or
inconsistent label fields fail open to no-fill/zero value; (3)
`head_populations` consumes generators twice; (4) the feature fence explicitly
admits the target-side `any_fill_ahead`; (5) metric functions do not enforce
aligned lengths/domains and report prediction count as population count; (6)
Q2/Q3 models are applied only to realised conditional subsets, so the four
heads are not row-aligned and Q4 cannot be composed for decision rows; the
runner emits no Q4, matched-random, incremental-null, or Holm results; (7) all
decision rows carry unit weight despite the declared cancellable-generation
action unit; (8) the 24-cell family does not define how multi-metric heads or
budget-independent Q1-Q3 metrics map to one p-value per cell, and status cells
can shrink `holm()`'s denominator; (9) exact minimum-n/ridge/weighting/output
constraints and standalone code identity are not fully frozen or bound; (10)
the new output guard breaks the self-test-only command.

**Day-bar/O1 blockers before the 08-29 judgment:** P1/P2/P3 are appended after
both per-coin and whole-day `all_pass` are computed, while the superseded raw
gap-count predicate remains binding; a full elapsed day with an empty ledger
passes P1-P3; structurally malformed JSON rows and `gap_open_at_exit` are
silently ignored; P3 is 5-minute-aligned rather than the declared maximum
rolling hour; the CLI renderer reads removed breadth keys; and the default
freeze epoch is 2026-08-24T15:04:28Z rather than the admitted candidate's
2026-08-28T06:09:00Z freeze. The O1 patch itself compiles and its legacy suite
is green, but the new O1a-d paths lack committed behavioral tests and its
`gap_open_at_exit` output is currently invisible to the governing day bar.

**Reviewed and internally consistent:** freeze receipt v2 preserves the
`b3f7f9f` clock, binds the canonical null and unchanged survivor conclusions,
and is correctly marked unvalidated. The canonical null's deterministic-order
repair is real. Lane 2/4 documents are useful specifications but need typed
join/units/target fields and an unambiguous replay base/output schema before
implementation.

## 2026-08-28 ~08:37Z (MEM) — TODO trued up; progress recorded WITH its definition

**`plans/STATEFUL_HARMFUL_CANCEL_TODO.md` trued up at `9669203`.** Six boxes
ticked, all in §0.1, each carrying inline the commit or code line that proves
it. **No prose edited, nothing un-ticked, no box ticked on judgment** — five
ambiguous ones were reported to the coordinator instead and ruled there.

**Countable progress: 39/113 = 35 %**, as of this entry.

| section | done/total |
|---|---|
| §0.1 correctness blockers | 6/7 |
| Phase 0 | 7/7 |
| Phase 1 (+ correctness tests) | 8/9, 7/7 |
| **Phase 2 core** | **6/6** |
| Lanes 2A / 2B / 2C | 0/24 |
| Phase 3 (+ parity, arms) | 0/23 |
| Phase 4 · Phase 5 · §10 | 0/10 · 0/7 · 0/5 |

**Two progress numbers now exist and they measure different things** — record
which one is being quoted, because a figure without its definition is how a
claim gets misquoted:

- **39/113 countable** (this one) — literal ticked boxes in that one file, each
  requiring an artifact. Unweighted by effort, importance or evidence strength.
- **~55–60 % evidence-weighted** (coordinator's) — a different measure blending
  evidence maturity with the file. Not a competing value for the same quantity.

**Read the shape, not the ratio.** The original Phase-2 comparison is **complete
at 6/6**. The 24 open lane boxes were **added by the user's own plan at
`d506a06`** — the denominator grew, the work did not stall. A bare "Phase 2 =
6/30" understates finished work and overstates the backlog's age; that framing
was corrected on this basis and the correction goes to the user.

**§0.1 item 5 stays open on purpose:** the annotation mechanism is proven on
real sidecar bytes (`63d9c7e`) but wiring rides the first lattice-touching 011
commit, so a fresh generation still drops the field. **Proven is not wired.**

**Recount after any commit that ticks boxes** — the flag carries an as-of for
exactly that reason.

## 2026-08-28 ~06:58Z (MEM) — the flagged residual is ruled: freeze receipt v2, clock unmoved

**Supersedes, in-band, the "raised, not ruled" close of the 06:56Z entry below**
(which stands unedited). The coordinator ruled it as rule 13 by its own
rationale: **BE authors `harmful_phase2_lgbm_btc_freeze_v2.json`** carrying the
canonical null pointer (`163bd36`) and canonical p's, a `supersedes` block
naming v1 as a **citation correction** (verdict-bearing statements unchanged —
true / true / false), and an **explicit clock-unmoved clause**. **v1 stays
frozen and untouched.**

**The clock-unmoved clause is the load-bearing part.** The freeze *event* stays
at `b3f7f9f`, so the race clock and the multiplicity-1 record run from the
original freeze commit; v2 corrects citations and never re-freezes. A
superseding receipt that quietly restarted the clock would hand a candidate
that has been accruing since 06:09Z five fresh days — which is exactly what
that clause forbids, and worth naming because it is the plausible way this
correction could have gone wrong.

Flag is `RULED-IN-EXECUTION`; it advances to `LANDED` when BE's commit ref
arrives.

## 2026-08-28 ~06:56Z (MEM) — blocker 7 closed by finding something; one residual flagged

### The re-binding did its job by failing

**R-234 (`8da983e`, ruling) → fix `e694511` → canonical receipt `163bd36` → DA
close `81eb368` → R-235 (`7ec5f4e`).** Blocker 7 was bookkeeping — re-bind the
null to the current chain. It surfaced a real defect instead, which is the
argument for doing bookkeeping at all.

**All 12 increments bit-identical (max |Δ| exactly 0), but 11 of 12 p-values
moved** (max 0.0245) on the same declared design, same `PERM_SEED=20260827`,
same `N_PERM=2000`, same data. **Mechanism verified, not inferred:**
`sign_flip_p` assigns signs in *list* order; the list came from
`set(cbw)|set(bbw)`; set-of-strings iteration order varies per process under an
unpinned `PYTHONHASHSEED`. Reversed insertion order → same increment, different
p (0.7964 vs 0.8184); three interpreters gave three orders.

**The sentence worth keeping:** `PERM_SEED` pinned the RNG but **not the data
order the RNG was applied to.** Every run was an independent Monte-Carlo draw
(SE ≈ 0.011 at n=2000) while the pinned seed advertised exact reproducibility
the instrument never delivered. A pinned seed beside an unpinned iteration order
is not reproducibility; it looks exactly like it.

### Why this was a repair and not a re-selection

Rule 11 bars choosing after seeing, so the conditions were declared **before**
the canonical run: `sorted(wins)` + pinned `PYTHONHASHSEED` (sorting is the
*unique* canonical order, so the chooser cannot steer the draw — the fix is
mechanically direction-blind); **acceptance pre-committed sight-unseen — the
canonical p's become the numbers of record whatever they are, one run, no third
look**; both prior draws preserved as independent MC estimates of the same true
p; falsifier shipped (two different `PYTHONHASHSEED` values must now give
identical p) and the reversed-insertion known-bad **kept live**, because it must
keep showing p differing *without* the sort or the repair is untested.

**Result: survivors unchanged** — btc LGBM @5 % Holm **0.00600**, @10 % Holm
**0.03298**; 10 of 12 chance on the joint reading; state arm @10 % a near-miss
again (raw 0.0090, Holm 0.0900). **DA's framing is the load-bearing part and is
preserved verbatim in state: that nothing flipped was not knowable beforehand
and is not why the run is accepted**, and it does not retroactively make the old
draws reproducible. DA's refusal to make the one-line fix on its own judgment
("not mine to make — a frozen instrument's numbers") is recorded as the model
escalation for this class.

### The ~1e-11 delta, explained after six appearances — and deliberately not fixed

The persistent per-cell scorer delta DA had reported six times without
explaining is **non-associative float addition over identical terms**: DA
accumulates net in *row* order, BE in *score-descending* order
(`harmful_action_eval.py:76`). 17,143 synthetic terms give 1.455e-10 between
orderings of the same values (`fsum` exact at 3.6e-12) — the observed magnitude.
Both orders deterministic, neither a defect, and **the six-generation agreement
is tighter than the raw delta suggested.**

**Standing instruction, ratified: the two orders stay different.** Harmonizing
them would reduce the cross-check's independence — which is the entire value of
a second implementation. It is in `STATUS.yml` as
`verifier_delta_1e11: EXPLAINED-DO-NOT-HARMONIZE` precisely so a later reader
does not tidy the two implementations into agreement and quietly delete the
independence that makes six agreeing generations mean anything.

### Also landed

- **Blocker 6 → mechanism PROVEN, wiring scheduled.** BE's receipt-side merge
  (`63d9c7e`) proves canon agreement on DA's **real committed sidecar bytes**
  (both sides `7acbca07…`), implements DA's three amendment clauses, and
  exercises live `BINDING_STALE` against the real receipt. Wiring into
  `stage_score` is deferred by ruled **option (a)** — it rides the first
  lattice-touching 011 commit so one fit8/score8 cycle pays for both. **Until
  that ride lands the mechanism is not in the write path**, so a fresh
  generation still drops the field. §0.1 now **6 of 7**.
- **011 slices 1–2** (`77cc76c`, `e597630`+`2febf2d`): target builder + four
  heads with 27 falsifiers, then head metrics each on its **own** population
  with failures reported. **Fits nothing, scores nothing, advances nothing.**
- **The 06:26Z `FIT_CODE_REF` refusal** is recorded as the R-228(1) guard firing
  in production against a real mis-launch nobody staged — rule 15's opposite
  case, evidence *for* the instrument rather than an incident.
- **BE's positive control failed on first build a third time** (symmetric values
  summing to zero saturated p at 1.0 both ways — a control that cannot fire
  proves nothing); rebuilt asymmetric, it shows ~0.06 p-spread from iteration
  order alone. **The habit is the record, not the instance.**

### One residual I am flagging, not ruling — receipts are not my surface

R-235 rules freeze-receipt impact **none**, and on verdict-bearing statements
that is correct and I verified it: `increment_survives_joint_reading_at_0_05`
reads true / true / false in `b3f7f9f` and stays true / true / false under the
canonical draw. **The residual is narrower:** the freeze receipt also *quotes*
p-values the canonical instrument no longer reproduces (`increment_holm_p`
0.021989 @5 %, 0.017991 @10 %, vs canonical 0.00600 / 0.03298) **and resolves
its declared null to `artifact_commit: e7caaeb`**, which `163bd36` supersedes.

Rule 13's stated reason for superseding in-band is that **automated readers
resolve receipt fields** — and `artifact_commit` is exactly such a field. The
receipt is frozen and must not be edited, so if anything is owed it is a
superseding freeze receipt v2, and that is BE's and the coordinator's call. It
is in `STATUS.yml` as `freeze_receipt_null_pointer: FLAGGED-FOR-OWNER` so the
divergence is discoverable from the state files instead of only by reading two
artifacts side by side.

## 2026-08-28 ~06:25Z (MEM) — DA's v2.3 verification + R-233 swept; one earlier fact corrected

**DA both-coin verification CLOSED** (Q-DA-113, `846e1ca`; ratified in R-233,
`166679c`): **15/15 cells reproduced independently at the commit** `fd1e949` —
btc worst |Δ| 1.273e-11, eth worst 3.638e-12, the same per-cell float-noise
pattern for a fourth generation.

**The part that is new, not just repeated:** this is the **first verification
that is self-attesting on runtime identity.** It emits `RUNTIME_IDENTITY`
naming its own feature-code bytes, and wrong-tree modules refuse — DA's
Q-DA-112 sweep applied to DA's own stack. Every prior verification hashed the
repository copy while running whatever tree the process happened to import;
this one proves which code it ran. That is R-230(3)'s defect class closed on
the verifying side, not only the producing side.

**CORRECTION to the 06:12Z entry below** (in-band; that entry stands unedited):
it recorded that v2.3 carried `da_caveat_field` = `RESERVED`. True when read at
06:09Z, **not true now.** DA re-applied the Q-DA-79 caveat at `cd23ebd`, taken
from the committed owner sidecar with its signature verified under
`annotation_canon_v1` — *not re-authored*, which is the whole point of the
contract. Read at the receipt just now: the field carries the Q-DA-79 content
plus a `BINDING_STALE` block naming `declared {btc 645,851}` against `actual
{btc 311,640, eth 299,703}`, `applied_by` "DA by hand, contract clause 4, until
BE's sidecar merge lands". Contract clause 4 working as designed: the caveat is
**merged and marked stale** — not dropped (which loses a caveat someone relied
on) and not carried silently (which is how one population's magnitudes come to
describe another's receipt). DA verified only that field changed; everything
else byte-identical to `fd1e949`.

**The blocker is still open and the count is the argument.** Third consecutive
hand application. BE's `annotation_canon_v1` merge — proven-agreement gate
against DA's real sidecar bytes, expected `BINDING_STALE` — is next in its
queue and is what closes it.

### Tonight, in order — and what is expected before it happens

1. **00:00:00Z — O1 boundary deploy** per `plans/O1_DEPLOY_RUNBOOK_2026-08-29.md`
   (now carrying the amended single-band citation, `6c59b75`).
2. **00:06Z — per-coin verdict on 08-28, OLD count bar.** **Expectation stated
   in advance** (R-233, from the live ledger as-of 05:31Z): **btc expected FAIL
   at ~153 s/hr; eth expected PASS.** Recorded here and in `STATUS.yml` so the
   result is read against a fixed prior rather than explained afterwards — **if
   btc passes tonight, that is a surprise to explain, not a relief to absorb.**
3. **08-29 — the first day under day-bar v2**, and the pre-declared
   discriminating test on the amended single band (~55–80 / <45 / >120 s/hr).

### Open queues at this write

- **BE:** `annotation_canon_v1` merge (closes blocker 6), then 011 development
  — target builder + heads, known-bads first, against the FROZEN preregistration
  (`3b71d3e`).
- **DA:** blocker-7 increment-null re-binding at the v2.3 chain; day-bar v2
  implementation (P1/P2/P3, falsifiers, per-slug vs coin-level dual reporting);
  lanes 2/4 specs.
- **Coordinator:** the boundary deploy, then the 00:06Z verdict to the user.
- **User:** the CLAUDE.md code-relay amendment (R-233 ask) — until it lands,
  fresh seats writing the state files are behaving correctly.

## 2026-08-28 ~06:15Z (MEM) — freeze receipt swept into state

**Supersedes, in-band, the "not yet written" line in the 06:12Z entry below**
(rule 13 — that entry stands unedited as provenance). The rule-12 freeze
receipt landed at `b3f7f9f` while that sweep was being written.

**Read at the artifact** (`data/pm_5min/derived/harmful_phase2_lgbm_btc_freeze_v1.json`,
`kind: RULE_12_FREEZE_RECEIPT`), not from a report:

- **`status`: FROZEN — ADMITTED TO THE FORWARD RACE, MARKED UNVALIDATED.**
  Candidate = `LGBM_PINNED`, `coin_scope` btc, thresholds
  `CAUSAL_FROZEN_FROM_TRAIN` per budget, frozen before scoring.
- **Rule 12 satisfied inside the receipt:** `fit_code_ref` `e12e2c7` +
  `fit_code_sha256_prefix` `3d0b6c8c6dfe9466`; `freeze_commit_parent`
  `fd1e949` (= v2.3, so the chain is continuous); pipeline in repo; declared
  nulls carried inline; **`multiplicity_at_freeze` = 1**.
- **Race:** clock starts **at the freeze commit**; bar = **5 later complete,
  passing btc UTC days** (per-coin, R-232 §9.4); `auto_entry_of_other_candidates`
  false.
- **All three btc budgets carried, not the two that survive:** @5 % +5,189.8 c
  Holm 0.0220 survives; @10 % +6,608.6 c Holm 0.0180 survives; **@15 %
  +1,053.7 c Holm 1.0000 (raw p 0.6677) — indistinguishable from chance.**
- **eth negative at every budget** (−847 c / −1,311 c / −1,283 c), so the
  btc-only scope reads as a consequence of the evidence rather than an
  arbitrary restriction.

**Two caveats the receipt deliberately refuses to conflate** — worth carrying
forward verbatim, because one is covered and one is not:

1. **Population** — development data, G=0 complete UTC days, one 14.4-hour
   span inside the consumed range. *Covered by "marked-unvalidated".*
2. **Budget non-monotonicity** — *covered by **nothing***. An advantage present
   at two budgets and absent at the third, with no mechanism offered for why a
   larger cancellation budget should erase it. Marked-unvalidated speaks to
   where the evidence came from; it says nothing about an object that behaves
   like this.

**Sharpened in the receipt:** the arm beats **matched random** at all three
budgets while beating the **incumbent** at only two — the concrete form of
"beating random is not beating the incumbent" for this candidate.

**Cluster-unit disclosure, carried in the null and easy to miss:** rule 8's
ruled unit is the UTC day, but the unit *used* is the **window**
(`weaker_than_ruled: true`), because G=0 leaves the ruled unit with no
replicates. Windows within a day are not independent, so **these p-values are
optimistic — evidence, not a significance certificate.** Quote the increments
with that attached or not at all.

**What changes operationally:** the programme now has a candidate whose clock
is running, which raises the cost of a lost forward day from "slower" to
"directly delaying a verdict". That is the same day the O1 deploy and day-bar
v2 exist to protect.

**Also fixed since the 06:12Z entry:** the stale day-bar citation in the O1
runbook — superseded in-band at `6c59b75`, so what gets read at 00:00Z tonight
now carries the single-band reading.

## 2026-08-28 ~06:12Z (MEM) — state sweep: STATUS.yml + HANDOFF current through R-232 + v2.3

**Why this entry exists.** Both state files had stopped at R-228 / `d506a06`.
`R-229`, `R-230`, `R-231`, `R-232`, the O1 collector package, the O2 day bar
and iteration-011 appeared **zero** times in either file before this write —
measured, not assumed. Everything below is sourced from the commit or artifact
named beside it (rule 16); nothing from conversation.

**State-file ownership moved to the MEM seat** by coordinator standing division
this session: BE/DA commit artifacts and report facts, MEM sweeps and writes
`STATUS.yml` + `HANDOFF.md`, so the post-chain refresh habit (the `a46ae64`
pattern) cannot race a coordinator tick. See the unreconciled CLAUDE.md
conflict under *Watch out for* — that one is the user's to settle, not ours.

### Done — landed and verified at the artifact

- **Receipt v2.3 — `fd1e949`.** fit7/score7 from `e12e2c7` under the R-230
  chain. **Sixth consecutive numerically identical generation:** 1,046 leaves
  compared, max abs delta `0.000e+00`; sole differing leaf `da_caveat_field`,
  predicted before the run. Supersedes
  `phase2_four_arm_v2.SUPERSEDED_BY_v2_3.json` (the v2.2 bytes, preserved
  unedited *before* anything ran — rule 13). Six fits now agree
  (`ef9b775` / `19b0611` / `43f777d` / `97b7183` / `e12e2c7`).
- **The population/reach disclosure is generator-owned for the first time.**
  Read out of the receipt for this entry, not from a report:
  `population_and_reach` = label `da_development_topup`, `G_complete_utc_days`
  **0**, `is_a_validation` **false**, `intervals_claimable` **false**,
  `dates_present` `[2026-08-25]`, 611,343 rows, span 14.41 h — *computed* from
  the rows actually scored against rule 11's bar. R-229's top debt is closed by
  mechanism. BE deliberately did **not** hand-attach `be_receipt_notes` this
  cycle; re-attaching would reopen the habit the fix exists to end.
- **R-230(1) confirmed in production, not only in seams.** `val_models.json`
  records `{"btc": true, "eth": true}` and is hashed into the lattice — 14
  `file_hashes`, up from 13 — so score7 *required* both val models. The branch
  where a deleted val model silently degraded arm C to hazard-only ranking is
  closed for this run.
- **Iteration-011 preregistration FROZEN by user ruling — `3b71d3e`** (R-232(4)).
  Two arms (composed-linear, composed-LGBM) → 24 Holm cells. §9.2: Q4 alone may
  not advance a candidate, explicit user sign-off required, reported *always*
  including on failure. §9.3: no auto-entry to any forward race. §9.4: G bar =
  ≥5 complete UTC days **per coin**. §10 separates what BE decides from what BE
  does not (rule 14). **Nothing fitted or scored.**
- **O1 collector package committed-not-deployed — `6786a02`, runbook `cb85ebd`.**
  a) ping 10/10 → 3/3 (77.1 % of the btc loss was client-side detection lag);
  b) cause-aware exponential backoff with full jitter, SLOW_CONSUMER floored at
  2 s base, ladder resets on a working connection; c) first-message subscribe
  confirmation (10 s bound, distinct `SUBSCRIBE_UNCONFIRMED` cause, hot recv
  path unchanged); d) `gap_start` falls back to scope-start coverage, never the
  error instant. `COLLECTOR_VERSION` `clob_v3_1` → `clob_v4`. **Live process
  untouched until the boundary restart.**
- **Day-bar v2 PRE-REGISTERED — `dfa0977`, amended `368345b`** at 06:04Z,
  before any day it judges. Per coin, all must hold: **P1** lost s/hr ≤ 120;
  **P2** windows with ≥75 s gap-intersect ≤ 5 % (≤14 of 288); **P3** max
  rolling-60-min lost seconds ≤ 900. Raw breadth = diagnostic, **no bar** (it is
  near-invariant under O1 and dominated by sub-2 s gaps). Applies to days
  ≥2026-08-29 only; nothing retroactive.
- **`annotation_canon_v1` RATIFIED** (R-232; DA at `417423a` + `2642b90`) with
  `allow_nan=False` and the float-repr residue pinned to CPython 3.12+, an
  unknown `canonical_form` refusing with a *distinct* cause from
  signature-mismatch, and agreement proven before first use. The real sidecar
  is committed with `population_independent: false`, so `BINDING_STALE` is the
  **expected** production outcome — the interesting path is exercised.

### In progress

- **DA — both-coin independent verification of v2.3** (the R-230 dispatch), then
  the day-bar v2 implementation: P1/P2/P3 into the day-verdict tool, coin-level
  mandatory, falsifiers shipped with the checker (rule 15), verdict artifact
  naming `DAY_BAR_V2_PREREGISTRATION.md` and its commit as the governing bar.
- **The rule-12 LGBM freeze receipt — not yet written.** Ruled by the user at
  R-232(3): `LGBM_PINNED` enters the forward race **btc-only** and **marked
  unvalidated**; the state arm stays out (survived nowhere); eth has no
  candidate (its LGBM increment is negative on the decision metric despite the
  best AUC). The receipt owes: builder committed (hash + ref), full pipeline in
  repo, declared nulls *inside* the receipt, multiplicity at freeze (= 1), and
  the words "admitted without clearing validation; development evidence only".
  Clock accrues **from the freeze commit**: ≥5 later complete passing btc UTC
  days.
- **Blocker 6 — annotation-survival mechanism.** Contract ratified, mechanism
  not built: v2.3 again carries `da_caveat_field` = `"RESERVED for Q-DA-79
  post-gap queue-validity finding"` (read at the receipt, 06:09Z). DA's Q-DA-79
  content is **not** in the current receipt and needs re-application by its
  owner, as in v2.1/v2.2 (`8b4bcee`, `e14639a`). Third consecutive cycle of hand
  re-application — the standing ruling is that this needs a mechanism, not a
  reminder.
- **Blocker 7 — increment-null re-binding** (DA, after v2.3): regenerate
  `phase2_increment_null_v1.json` (`e7caaeb`) against the current provenance
  chain with the *same* declared design / seed / n_perm as R-217. A re-binding,
  **not** a re-selection; the R-218 result of record does not move.

### Next — in order

1. **Tonight 00:00:00Z — the O1 boundary deploy** (coordinator-owned; collectors
   are outside worker surfaces, R-110). Sequence, verification and abort path
   are in `plans/O1_DEPLOY_RUNBOOK_2026-08-29.md`.
2. **00:06Z — the per-coin verdict on 08-28, under the OLD count bar.**
3. **08-29 is the pre-declared discriminating test.** Amended single band
   (`368345b`, supersedes the earlier two-model reading): **~55–80 s/hr** = the
   fix worked as modelled; **below ~45** = something else also improved, most
   plausibly O1b's backoff residual, which neither model prices; **above ~120
   (P1 FAIL)** = the detection-lag *diagnosis was wrong*, not "the fix
   underperformed". That last branch survives the correction unchanged and is
   the one worth having declared.
4. **Lanes 011/012 — build and preregister only.** Outcome-driven selection
   stays serialized; no scoring under 012; no new-feature scoreboard on consumed
   days.

### Watch out for

- **`live/pm_research/collect_pm.py` shows MODIFIED on purpose.** The v4 patch
  is committed (`6786a02`) but the working file is held at `clob_v3_1` until the
  boundary, because `Restart=always` would load v4 **unstamped** on any mid-day
  auto-restart and manufacture an unrecorded era boundary inside a live day.
  **Do not clean, sweep, checkout, restore or commit that file** (`8dd9831`,
  `cb85ebd`). It is a safety hold, not drift.
- **Same class: the tape7+ arming runbook.** Archiving the ruled gate-verdict
  locator is **step one of a future arming**, never tidying (R-227 / R-229). A
  guard's output that looks like something to clean up is exactly what these two
  notes exist to protect.
- **The runbook's day-bar line points at superseded numbers.** `cb85ebd`
  (05:49Z) cites "~30 vs ~79 vs >120 s/hr readings, doc §3"; the O2 amendment
  `368345b` (06:04Z) replaced that two-model test with the single band in item 3
  above. The pre-registration document is authoritative; the runbook line is
  stale by 15 minutes. Flagged for its owner (coordinator), **not** edited here.
- **Provenance ≠ reach.** R-225/R-228/R-230 hardened the chain, not the claim.
  Six identical generations of a **development** result on G=0 complete UTC days
  is still not a validation. Do not let chain strength read as claim strength.
- **CLAUDE.md is not yet reconciled with the state-file ownership split.** It
  still instructs *every* session to update `STATUS.yml` + `HANDOFF.md` after
  each completed step — a user-level rule no coordinator or peer can revoke.
  Until the user amends it, **a fresh seat writing these files is behaving
  correctly**: sequence around such a write, never treat it as a violation or
  revert it. The amendment sits with the coordinator to raise with the user.
- **Filed debt, named rather than patched** (BE, in-band at `fd1e949`): the
  receipt carries no pointer to its own null artifact
  (`phase2_increment_null_v1.json`, `e7caaeb`) — a reader has to know to look.
- **`STATUS.yml` `focus:` was rewritten in this sweep**; it had been frozen at
  2026-08-21 (sigma pipeline / route_a framing). The prior text is preserved
  inside the new block as PRIOR FOCUS, and Route A's counts are carried forward
  explicitly **unverified at their 2026-08-21 as-of** — re-read them before
  quoting.

## 2026-08-28 (coordinator) — conditional-value + modular integration plan recorded

**DOCUMENTATION ONLY; NO CANDIDATE FROZEN OR SCORED.** Updated
`live/pm_research/plans/HARMFUL_FILL_HAZARD_TOXICITY_PLAN.md` and
`STATEFUL_HARMFUL_CANCEL_TODO.md`, with iterations 011--013 cross-referenced in
`PM_STRATEGY_OPTIMIZATION_LOOP.md`.

The next cycle has four separately owned lanes: (1) fill hazard plus
conditional harmful-sign/favourable-vs-harmful magnitude, (2) a timestamped
unconditional fair-price successor with `Identity` mandatory, (3) frozen skew
as inventory/placement control, and (4) a common stateful action-value replay.
The full comparison has seven arms, including neutral skew, incumbent,
hazard-only, conditional cancel, cancel x skew, cancel x skew x fair residual,
and decision-matched random cancellation.

**ORDERING:** first close the recorded Phase-2 hash/drift/module-root and
generator-owned-disclosure seams; conditional value and fair-price engineering
may then run in parallel while skew/replay parity is built against typed stubs.
Final scoring consumes only immutable module artifacts. Existing 08-20..25
harmful-fill days remain consumed; every changed candidate starts a new forward
clock and needs at least five later complete UTC days.

**FAIR-PRICE WARNING:** `plans/BE_BELIEF_PLAN.md` is refuted provenance and is
not an implementation source. A successor carries only `Identity` and the
unconditional `E[Y|state]` ownership rule; toxicity owns the fill-conditional
residual so adverse selection is not counted twice.

Prior active-state summary (DA): **receipt VALIDATED 15/15; increment-null
10-of-12 chance; 08-27 EXCLUDED; per-coin live; BTC evidence pack delivered;
freeze awaits the user.**

## 2026-08-28 (BE, ~04:35Z) — R-228 chain CLOSED; receipt v2.2, replication BIT-EXACT

**AUDIT #9: the fail-open class one level down.** R-225's guards were added and
still passed VACUOUSLY. Four fixes at `97b7183`, eleven behavioural known-bads
RED pre-fix and green after:

1. **Completeness.** The hash loop iterated whatever `file_hashes` held, so an
   EMPTY map ran zero iterations and read as a satisfied check; a manifest
   listing one artifact of fourteen left thirteen unverified. Now checked
   against an EXPECTED set derived from `empty_coins.json`, which is itself in
   that set and hash-verified. And `fit_code_ref` must RESOLVE to a real commit
   CARRYING the recorded bytes — it had only ever been compared to the env value
   the scorer was launched with, so all-zeros matched all-zeros. FAIL-CLOSED: a
   git failure refuses, never skips.
2. **Identity lattice.** Four result-bearing modules sat outside it, including
   `harmful_exposure_rows` — which owns BOTH `any_fill_ahead` (the valuation
   gate) and the latency cut (the estimand). A PARTIAL identity read as a whole
   one. Bound alongside: the scoring top-up, its build receipt, the frozen
   incumbent. The SCORE side now captures and rechecks identity like the fit;
   only the fit did, so the stage producing the published numbers was the
   unguarded half.
4. **Lock leak.** The identity capture sat between acquisition and the `try`, so
   an identity failure raised with the lock HELD.
5. **Supersession is GENERATOR-OWNED** (protocol_version, supersedes), and the
   block reports a missing predecessor explicitly.

**RESULT: v2.2 (`c47eb83`), REPLICATION BIT-EXACT.** 1,046 shared leaves, max
absolute numeric delta **0.000e+00** — the v2.1 rerun gave 8.3e-17; this is
zero. Sole differing leaf `da_caveat_field`, predicted before the run. Fit
manifest + parity + empty_coins committed at `ff80ebd` so the chain verifies at
committed artifacts (rule 12). v2.1 preserved unedited at
`phase2_four_arm_v2.SUPERSEDED_BY_v2_2.json` (`974a1bf`) before anything ran.

**FIVE FITS AGREE**: ef9b775 / 19b0611 / 43f777d / 97b7183 on every drop cell
and both purge deltas; the last three on every scored number. Four consecutive
pre-registration matches at 578,917 / 505,904.

**GUARDS CONFIRMED IN PRODUCTION**, not only in seams: `fit manifest OK: 13
artifacts`; score-side identity captured before load; `released
phase2_fits.lock` (third consecutive run leaving none); `supersedes
present_at_write True` — the generator VERIFIED its predecessor rather than
asserting it.

**MY POSITIVE SEAM WAS CERTIFYING THE HOLE.** Seam 42f asserted "a WELL-FORMED
manifest is ACCEPTED" using `file_hashes: {}` — the exact vacuity the audit
found. I wrote that seam so the guard would be a gate and not a wall, and in
doing so blessed the defect as the DEFINITION of well-formed. A guard's accept
path must be exercised with something actually valid, or the positive control
certifies whatever hole it contains.

**HAND-ATTACHED DISCLOSURE, third cycle running.** Generator ownership covers
protocol_version and supersedes ONLY. `be_receipt_notes` — carrying
development-not-test, G=0, not-a-validation — was re-added BY HAND, marked
`ADDED_POST_GENERATION`. Kept because a receipt without it can be misread as
validation, which is worse than a hand-added field that admits what it is.
**Top next-cycle debt: the generator must own the population/reach fields** —
they are precisely what a fresh generation drops and precisely what their
silent absence would convert into an apparent validation.

**TELEMETRY CAVEAT, so no false improvement enters the record**: fit6's unit
reported `2.2M memory peak`, impossible for a job holding a 1.1M-row index
(score5 reported the same). cgroup peak-accounting artifact; the real peak is
~14G as measured throughout.

**FOR DA**: `da_caveat_field` is RESERVED again. Q-DA-79 must be re-applied by
DA, and the deeper problem — a peer's annotation survives no regeneration by
construction — needs a MECHANISM, not a reminder. On the debt list.

**UNCHANGED AND STILL GOVERNING**: `da_development_topup` is DEVELOPMENT not
test; G = 0 complete UTC days on one 14.4-hour span inside the consumed range;
no interval claimable; **THIS IS NOT A VALIDATION**. R-225 and R-228 hardened
provenance, not reach. The increment null (`e7caaeb`) stands: 10 of 12 cells
chance, `PLUS_PRED_STATE_V1` surviving nowhere.

**NEXT**: DA verifies both coins against v2.2; freeze remains **with the USER**.

## 2026-08-28 (BE, ~02:53Z) — R-225 enforcement chain CLOSED; receipt v2.1, numbers identical

**AUDIT #8's CRITIQUE AND WHAT IT COST.** The user's finding: BE's seams proved
hashes CHANGE and none proved scoring REJECTS a missing or mismatched hash. An
instrument that moves is not a gate that bites. Four enforcement fixes at
`43f777d`, each with a behavioural known-bad that FAILS against the prior code
(red 7 -> green 0 on the same harness):

1. **Measured identity ENFORCED.** `fit_code_sha256_prefix` and
   `fragment_sha256_prefix` were WRITTEN into the manifest and never COMPARED,
   so a manifest carrying no measured identity was ACCEPTED. Two holes closed:
   a key absent from the compare list was never checked, and
   `m.get(k) != now.get(k)` passes VACUOUSLY when both sides are None — an
   absent binding read as agreement. "Missing" is now a distinct refusal from
   "mismatched", and a well-formed manifest must still be ACCEPTED (seam 42f),
   so the guard is a gate and not a wall.
   Also: the receipt reported the **SCORER's** identity under the **FIT's** name.
   It now reads the fit's identity FROM ITS MANIFEST, with the scorer's beside it.
2. **Fit-path TOTAL absorption bound**, callable. Ten 0.9% categories
   aggregating 9% passed every per-status check. Seam 31e's source-text search
   for the word "absorption" replaced by driving the guard.
3. **The lock did not exclude, and was never released.** Check-then-write let
   two processes both pass `if _lock.exists()`. Now `O_CREAT|O_EXCL` with a
   dead-holder reclaim that RE-ATTEMPTS the atomic create. Release in a
   `finally`, ownership-checked. Seam 44a spawns two concurrent acquirers and
   requires exactly one winner.
4. **Identity captured BEFORE load, RECHECKED at write**; drift is a REFUSAL.

**RESULT: fit5/score5 -> RECEIPT v2.1 (`2fbf233`), NUMBERS IDENTICAL.** 980
leaves compared against the superseded receipt; **max absolute delta 8.327e-17**
— floating-point last-bit noise. Every net, threshold, AUC, rho and gate outcome
reproduces, including eth LGBM_PINNED's `beats_NET=False` at 5% and 15% (the
seeded null replays too). The eight differing leaves are ALL structural
provenance and no numbers. The enforcement changed what is attested, not what
was measured — divergence would have meant R-225 altered the thing it was only
supposed to witness.

**DETERMINISM NOW SPANS THREE FITS**: fit3 `ef9b775`, fit4 `19b0611`, fit5
`43f777d` agree on every drop cell and purge delta; fit4 and fit5 additionally
reproduce every scored number. That is what makes the two-stage registration a
control rather than a coincidence.

**SUPERSESSION, rule 13.** v2 preserved by RENAME at
`phase2_four_arm_v2.SUPERSEDED_BY_v2_1.json` (sha `0f61d38814d5aeaa`, unedited,
`ecb8707`) BEFORE score5 could overwrite it — the R-216 defect was one command
away from repeating and was caught before the run, not recovered after.

**STANDING RULE DISCOVERED (bit twice).** R-225 makes fit and score code
byte-identical BY CONSTRUCTION, so **output PATH and receipt SCHEMA are both
FIT-TIME decisions**. Changing either between fit and score alters
`fit_code_sha256_prefix` and the manifest correctly refuses its own score. The
v2.1 supersession fields are therefore `ADDED_POST_GENERATION` and say so in the
artifact. Pre-launch checklist must confirm PA.OUT is writable-without-overwrite
BEFORE the fit.

**INSTRUMENT SELF-CRITICISM, recorded because it recurs.** BE's first red
harness was SOURCE-TEXT based and misreported in BOTH directions: a false GREEN
on the missing total-form bound before the fix, four false REDs after, because
the guards had moved into callables while their properties held. Tenth
appearance of the source-grep class, this time in BE's own test. Rewritten to
drive real code, which is when the red->green evidence became trustworthy. Same
lesson as the audit, one level up.

**BATTERY**: 456 checks, 0 failing across 9 suites. Earlier in the day BE also
found **F1: `python3 -m phase2_embargo` ran ZERO checks and exited 0** — four BE
commit messages had cited that silent rc=0 as "GREEN". Fixed; the coverage claim
was wrong even though the module's 8 checks pass.

**UNCHANGED AND STILL GOVERNING** (carried in the receipt, not just here):
`da_development_topup` is DEVELOPMENT not test; **G = 0 complete UTC days**, one
14.4-hour span on 2026-08-25 inside the consumed range 08-20..25; no interval
claimable; **this is not validation**. The increment null (`e7caaeb`) is
unaffected — it reconciles to these same `net_cents`.

**FOR DA**: `da_caveat_field` is a RESERVED placeholder again on this fresh
generation. Q-DA-79 must be re-applied BY DA; BE has not copied or reconstructed
it. Ready for two-track re-verification against v2.1.

**NEXT**: DA re-verify v2.1; freeze decision remains **with the USER**.

## 2026-08-28 (DA, ~01:xxZ) — day one EXCLUDED; per-coin live; receipt validated; btc evidence pack delivered

**WHERE THE PROGRAMME ACTUALLY STANDS** (the audit flagged this file stale —
R-222..R-225 lived only in COORDINATION):

**RESULT STATE.** The Phase-2 four-arm receipt is **VALIDATED**: DA reproduced
**15 of 15 cells** (btc 6, eth 9 incl. incumbent) with an implementation sharing
no scoring/thresholding/evaluation code — **worst delta 1.3e-11 cents**,
populations, drops, cancellation counts and harm/sacrifice/rho all matching
(Q-DA-107). **INCREMENT-NULL (R-217/e7caaeb): 10 of 12 cells are chance after
joint reading.** Holm independently recomputed, all 12 agreeing to 1e-9;
**only btc/LGBM_PINNED @10% (0.01799) and @5% (0.02199) survive at 0.05.**
eth's increment is **negative in the point estimate and indistinguishable from
chance** (−1310.82 @10%, p=0.24), rebuilt from DA's own numbers. **Cluster
disclosure is honest: G=0 complete UTC days, window unit, declared
weaker-than-ruled and OPTIMISTIC — evidence, not a significance certificate.**

**FORWARD RACE.** **08-27 (day one) is EXCLUDED (R-222)** — verified FAIL on
`gap_rate_under_bar` alone (554 gaps, 23.08/hr vs bar 15, 17 of 24 hours over,
3297.2s lost); complete (288/288 every coin) and post-freeze otherwise. **eth is
lost with it** under the frozen whole-day rule; **BE is prohibited from scoring
it**; the incumbent's clock restarts at the first per-coin-passing day.
**PER-COIN VERDICTS ARE LIVE from 08-28** — the prospective boundary held, and
08-28's first run correctly reported NOT-ESTIMABLE rather than passing vacuously.

**btc COLLECTOR — evidence delivered, design pending (Q-DA-108, `91eb3f5`).**
**77.1% of btc's post-step lost time is DETECTION LAG, not reconnect**:
PING_TIMEOUT median gap 11.305s of which **10.005s** elapses before the client
notices (`ping_interval=10`); reconnect itself ~1.3s. **Control group decisive**:
btc 1,734 gaps / 10,012s vs **36 gaps / 261s for the other SIX coins combined**
in the same process, one connection per market; PING_TIMEOUT is btc-only.
**Queue telemetry REFUTES the slow-consumer story**: `ws_ever_paused` 0 across
all 1,734 disconnects, lag median 2.1ms, and depth LOWER post-step (80) than
pre-step (123). This **refines DA's own earlier claim** — the trigger may be
peer-side, but the COST is client-side detection latency. R-181/182 embedded in
the artifact so the design does not re-walk the refuted shard premise.

**FOUR THINGS OPEN, none of them DA's to decide.** (1) **The FREEZE — the
user's word**, held. (2) **btc mitigation design** — coordinator drafts, user
rules; no deploy, boundary-only if ever. (3) ~~R-214 staging-then-promote~~ **DONE 2026-08-28 (`32fe17b`)**: gate writes
to staging, promotion only when the WORST rc across both checkers is 0, plus a
default-on refusal (rc 6) to arm onto an occupied locator. Seam 13/13.
(4) **BE's bound debts** — callable `gap_contains`, callable absorption guard,
seam 30 rewritten to call the real function, the increment-null reconciliation
refusal given a falsifier. **R-225's four guard-wiring findings are CLOSED** (BE's enforcement chain, receipt v2.1). **R-228 (user audit #9): items 1/2/4/5 are BE's and are CLOSED 2026-08-28** — the fail-open class one level down: an EMPTY `file_hashes` map passed the completeness loop vacuously, a manifest listing one artifact of fourteen left thirteen unverified, an all-zeros `fit_code_ref` matched an all-zeros env value, result-bearing code/data sat outside the identity lattice, an identity failure after lock acquisition leaked a live lock, and the supersession fields were hand-added rather than generator-owned. 11 known-bads authored RED pre-fix, all green post-fix; the POSITIVE seam that had blessed `file_hashes:{}` as well-formed is rebuilt to use a genuinely complete manifest. **Item 3 was DA's staging fail-open and is FIXED 2026-08-28 (`da_await_gate.sh`, seam 15/15, red-then-green proven against the pre-fix wrapper).**

**RUNBOOK — ARMING A FUTURE TAPE (tape7+). STEP ONE, BEFORE ANY ARM:**
`da_await_gate.sh` REFUSES (rc 6) onto an occupied ruled locator, by design
(R-214, ruled default-on). So the FIRST line of any future arming is to archive
the incumbent verdict deliberately:
`mv data/pm_5min/derived/da_tape_gate_verdict_v5.json \
    data/pm_5min/derived/da_tape_gate_verdict_v5.SUPERSEDED-<tape>-<ref>.json`
**NEVER before that moment, and never as tidying.** Through fit5/score5 the
tape6e verdict at that locator is **LOAD-BEARING**: the rerun re-fits and
re-scores THE SAME certified tape, and `score5`'s `assert_gate_passed` READS it.
Archiving it early would break the rerun, not tidy up for it. A byte-identical
archive already exists (`da_verdict_tape6e_ed9d572.json`), so when the moment
does come the move loses nothing.

**DA's NEXT ACTION.** When BE's superseding receipt lands after fit5/score5,
**re-run the independent scorer** (`scratchpad/review/da_recompute*.py`);
numbers should be identical to ~1e-11 and **any divergence is a finding**.

## 2026-08-27 (DA, ~17:xxZ) — R-219 adversarial review CLOSED: 15/15 cells reproduced independently

**DONE.** **(1) INDEPENDENT RECOMPUTATION (Q-DA-107).** Fifteen cells — btc 6,
eth 9 including the incumbent — recomputed with DA's own implementation and
matching to **<= 1.3e-11 cents** (float noise, NOT "to the cent"). Populations,
drops and **cancellation counts identical in every cell**; harm/sacrifice/rho
independently derived (LGBM@10% rho 1.441946745689). **The eth negative
increment rebuilt from DA's own numbers** (candidate minus incumbent, never read
from the artifact): −847.27 / −1310.82 / −1282.47 at p 0.38/0.24/0.28 — **real
in the point estimate, indistinguishable from chance.** **Independence boundary
declared and AST-verified**: shares the feature builders, `encode_row`, the
committed artifacts and the certified tape; shares NO scoring, thresholding or
evaluation code. **(2) NINE RECOMPUTED PASSES (Q-DA-106):** provenance chain
(every link, incl. the checker file still hashing to what the verdict recorded),
embargo 60.309452056884766 bit-identical, rows/action bit-identical, **Holm all
12 cells to 1e-9** (survivors: btc/LGBM @10% and @5% only — "10 of 12 chance" is
correct), reconciliation effect <=2.7e-11, increment arithmetic, **rule 7 at
source** (`cut = t_start + L/1000`, only later tranches valued), **rule 2**
action unit end-to-end, and a cross-check nobody claimed: the tape's **289**
GAP_AT_CUTOFF rows split **97 fit + 192 score**, reconciling the receipt's
per-split exclusions to the tape header. **(3) THREE FINDINGS, none touching a
reported number:** hardcoded `reconciles_with_receipt: true` literals (rule 10 —
the guard is real but records an assertion where the delta was available); that
guard is **inline and uncovered by the selftest** (rule 15, third
cannot-be-called instance, belongs in the bound BE commit); `any_fill_ahead` has
two definitions across layers (**measured harmless** — 0 suppressed values in
330,202 rows).

**METHOD NOTES WORTH KEEPING.** A review BINDS TO A REF — the working tree had
already drifted from `19b0611`, and the first `stage_score` read was of the tree.
And **DA's own independence check was a grep that matched its own docstring**
(*"I do NOT import phase2_arms"*) and reported independence BROKEN — the seam-30
defect, self-inflicted, inside the check certifying DA's own work. **Vocabulary
is not identity; the AST parse is the real check.**

**NEXT / WATCH OUT FOR.** **00:06Z tonight** — 08-27 judged whole-day under the
FROZEN rule; expected FAIL, eth lost with it; per-coin from 08-28. **R-214
staging-then-promote UNIMPLEMENTED** (write-new + mv, before the next arming).
**Bound BE debt**: callable `gap_contains` + callable absorption guard + seam 30
rewritten + the increment-null reconciliation refusal given a falsifier. The
freeze is the USER's word.

## 2026-08-27 (DA, ~13:4xZ) — tape6e GATED ALL-PASS, confirmed two ways; join drop is a filter, not a lookup failure

**DONE.** **(1) tape6e ALL-PASS** (Q-DA-104). Both predicates that forced the
rebuild flipped exactly as pre-registered: **at-g0 4 present / 4 FLAGGED** (was
4/0), **at-g1 493 present / 0 flagged** (was 493/1), count **289 vs 289**,
ledger sha = pin, `builder_ref ed9d57299a80`, `gate_code head=6a476a9
dirty=False`. `n_rows` 1,764,206 unchanged — the population did not move, only
the flags. **(2) INDEPENDENT COUNTER AGREES ZERO/ZERO** — cross-tab collapsed to
a **single cell** `{'GAP_AT_CUTOFF': 289}`; no disagreeing row in either
direction across 1.76M rows. **This is the R-213 citation; seam 30 is not one.**
Exactly the five disputed rows moved and nothing else; partition intact. The
header corroborates independently (`OK` −4, `GAP_AT_CUTOFF` +3, **`PRE_WINDOW`
+1** — the reverted at-g1 row was warm-up, as the census said). **(3) A RACE IN
DA'S OWN WRAPPER, SELF-REPORTED BEFORE ANYONE ACTED:** the verdict promotes to
the ruled locator at the end of `run_gate`, so `assert_gate_passed` ACCEPTED
while the counter was still running — fitting was permitted on the gate alone,
inverting R-213. Fit held on DA's confirmation; **R-214 adopts
staging-then-promote** for the NEXT arming (write-new + mv; not yet
implemented). **(4) JOIN PROBE** (Q-DA-105) confirms BE cell-for-cell: 605,256 /
578,917 / 26,339 = PRE_WINDOW 26,227 + GAP_AT_CUTOFF 97 + NO_LEVEL_HISTORY 15;
**near-miss 0, truly absent 0**. **Every one of the 26,339 has an EXACT tape
match** — the join key is sound, `t_start` round-trips with zero drift, and a
**filter was counted as a lookup failure**. Not a tape6e regression (the filter
predates it; the rebuild moved 4 rows). R-216 rules the 97 excluded
deliberately — unreliable decision-time data, all four arms identically.

**IN PROGRESS.** BE: accounting fix → fit3 → score → receipt.

**NEXT / WATCH OUT FOR.** **00:06Z tonight** — 08-27 judged whole-day under the
FROZEN rule (R-211(2)); expected FAIL on `gap_rate_under_bar`, **eth lost with
it**; per-coin verdicts from 08-28. **R-214 staging-then-promote is UNIMPLEMENTED**
— do it before the next arming, write-new + mv. **Bound BE debt** (R-210/213):
one commit lifting `gap_contains` to module scope AND making the absorption
guard callable, seam 30d–f rewritten to call the real function with genuinely
different inputs, greps replaced by behavioural assertions. Q-DA-99's btc gap
cause remains unfixed **by design**.

## 2026-08-27 (DA, ~12:0xZ) — tape6d gated and REFUSED; rebuild armed at ed9d572 on a pinned ledger

**DONE.** **(1) tape6d gate REFUSED as pre-registered** — `gap_count_matches_expected`
FAIL (286 vs 289) and `half_open_containment_landed` FAIL (493 at-g1, 1 flagged).
Verdict bridged to the ruled locator; `assert_gate_passed` refused; fit blocked
by the machinery. **(2) ROW-LEVEL DIFF — both hypotheses refuted.** Status-masking
refuted (the 4 rows carry `OK`, nothing competes). Adjacency refuted from the
ledger alone (btc 1632 raw gaps, **0 touching, 0 overlapping**). **My own
replacement mechanism — the complement interval `(g0, g1]` — was WRONG and
withdrawn one test later**: 493 rows sit on a g1 and only 1 is flagged, so the
upper edge agrees 492/493. The tidiness was the tell. **(3) CENSUS gave the
denominators a disagreement-only diff cannot**: at-g0 **4/4 systematic**; at-g1
splits perfectly by t_start sign **but the negative cell is n=1 and the tape holds
exactly one such row — the sample is EXHAUSTED, not thin**. Interior 285/285
agreed, outside 1,763,424 clean: **the edges are the only disagreement classes**.
Classes partition to n_rows and both disputed numbers reconstruct from
independent cells (289 = 4+285, 286 = 285+1). **(4) BE's ed9d572 mechanism beats
mine and retro-explains the n=1**: the projection subtracted t0 from ~1.79e9
where a float ULP is 2.4e-7s, so edge equality survives for some values and not
others — one cause, both edges; never a warm-up *convention*. Verified at the
call sites: one comparison, feature path handed `gaps=()` (retired, not
corrected). **(5) SEAM 30 does NOT test the fix** — source-text greps, a local
re-implementation, an `X and X` tautology that cannot fail, an unused import.
Root: `gap_contains` nested in `main()`, uncallable — same class as Q-DA-98,
same file. Ruled non-blocking; **DA's counter is the citation**; debt bound to one
BE commit. **(6) `gate_code` was absent from the written verdict** (verify()
computed it, write_verdict dropped it) — third *present-upstream/absent-
downstream* instance; fixed with the class (unlisted field REFUSES the write).

**IN PROGRESS.** `da-gate-tape6e-ed9d572.service` polling. **Ledger PIN**
`ledger_pin_tape6e.jsonl` sha `6cb3a027e25fb5df…`, 3494 lines, chmod 444,
write-once. Registration **measured against the pin**: 289 / at-g0 4 / at-g1 493.
Gate AND counter both read the pin. Ruled locator cleared by RENAME so a refusal
leaves absence. Persistent Monitor watches the gate log (two session-scoped
watchers were killed — R-147(2) again: durable work in units, notifications are
not durable).

**NEXT / WATCH OUT FOR.** On landing: expect at-g0 4/4 FLAGGED, at-g1 493
present-and-UNFLAGGED, count 289, ledger sha = pin. **00:06Z tonight**: 08-27
judged whole-day under the FROZEN rule (R-211(2)) — expected FAIL, eth lost with
it; per-coin from 08-28. Q-DA-99's btc cause is unfixed **by design** (R-211(1):
a mid-day collector change would stamp an era boundary inside day one).

## 2026-08-27 (DA, ~10:2xZ) — R-211(3) per-coin rule shipped; tonight's unattended path had two defects

**DONE.** **(1) R-211(3) per-coin forward-day verdicts** (`c8a655b`), 13h inside
the deadline. `verdict_granularity(day_token)` takes **only** the date — no
override parameter, because *a granularity a caller can choose is one that gets
chosen after the numbers are visible*. `PER_COIN_RULE_FROM_DAY = 20260828`,
Class A, prospective-only. Same bar of 15; **no CHOSEN value moved**. Day-level
`all_pass` stays whole-day-strict under BOTH rules so an un-updated reader
**fails safe**; per-coin verdicts in `per_coin`, and the artifact names its own
rule. Invariance proven old-vs-new on CLOSED day 08-26 (identical, 708 bytes,
cross-checks the recorded 508 gaps / 18 hours over bar). **The first invariance
proof was a FALSE PASS** — it printed IDENTICAL while diffing two EMPTY files
after both runs crashed on a repo path I broke; the check now refuses on an
empty side. 17 → 27 checks. **(2) Tonight's 00:06Z path had two defects**
(`817f7b0`): `main()` returned 1 for both a computed FAIL and an uncaught
exception (now 0/1/**4=instrument failure**), and `da_midnight_verify.sh` exited
with its last `echo` — always 0 — so a crashed verifier reported SUCCESS to
systemd (the R-199 swallow-class, still present in the one script that runs
unattended). **My first fix was wrong and my own falsifier refuted it**: `rc>=4`
only catches exceptions inside `main()`; a module dying on import exits 1. The
check is now **positive evidence** — a parseable verdict artifact naming the day
it claims to have verified, temp-written and promoted only on validation.
*Absence is never success.* `DA_MIDNIGHT_LOG` makes the script rehearsable
without touching the nightly record; per-coin verdicts print in the log too.

**IN PROGRESS.** `da-gate-tape6d-b3ef93c.service` polling; background watcher
reports when it fires.

**NEXT / WATCH OUT FOR.** 00:06Z tonight: 08-27 judged whole-day under the
FROZEN rule (R-211(2)) — expected FAIL on `gap_rate_under_bar`, **eth lost with
it**, the price of not choosing after seeing. From 08-28 per-coin. **Q-DA-99's
btc cause is unfixed by design** (R-211(1): no mid-day intervention — a
collector change writes a stamp boundary INSIDE the day and would manufacture a
mixed-era day); mitigation is an OPS between-days task on the coordinator's
desk, deploy only at a UTC boundary. Standing pattern now: **every unattended
runner proves it ran by leaving an artifact, never by an exit code.**

## 2026-08-27 (DA, ~10:1xZ) — REVIEW-COMPLETE given; two vacuous load-bearing predicates fixed; day one failing MID-DAY

**DONE.** **(1) R-209(1) — two of six load-bearing predicates were VACUOUS and
it was mine.** The builder splits skips into `pre_emission_skip_counts` (:371,
the R-203 split); the gate kept deriving them from `state_status_counts` (:442).
**Reproduced against the previous commit before fixing**: 100 `NO_ARCHIVE_PATH`
skips vs 60 rows → `no_rows_skipped_by_builder` True, `absorption_within_bound`
True. I had verified BE *emitted* the key and never checked my gate *consumed*
it — guard-vs-consumer, one turn after auditing BE for exactly that. Fixed
`c36a794`: union-read both sources (a reverted split stays caught), absence on a
V5 artifact FAILS, absorption reports UNCOMPUTABLE not 0.00%, `verify()` refuses
on a header lacking the key. **(2) R-209(4) was worse than filed** — `_accumulate`
never *returned* `at_g1`, so `half_open_containment_landed` went silently N/A on
the list path, on top of the UnboundLocalError. GC import must stay
function-local (`da_gap_at_cutoff_count` imports the verifier). **(3) R-210**:
absorption bound is TOTAL (per-status is evadable by status proliferation — two
names at 0.59% build, 1.19% total refuses); LOAD_BEARING = SIX. Both verified by
execution and pinned as regressions. **(4) Verdicts now carry `gate_code`
{file,sha256,head,dirty}** — the await unit runs the gate BY PATH, so armed-at is
not ran-at; `dirty` matters because a working-tree verdict is reproducible from
no ref. **(5) Seam test was writing into the PRODUCTION gate log** — six stub
triplets landed under a live `armed` header and I misread them as the armed gate
refusing. Fixed `55e0a24`; correction appended in-band, log stays append-only.
Suite **87 → 101**, contract 8/8, seam 6/6.

**IN PROGRESS.** `da-gate-tape6d-b3ef93c.service` armed and polling (289 ·
provenance b3ef93c · 133 gapped slugs · verdict →
`da_verdict_tape6d_b3ef93c.json` · 6h deadline · worst-rc). Background watcher
reports when it fires.

**NEXT / WATCH OUT FOR.** **Q-DA-99 is time-critical and unanswered**: day one
(08-27) is FAILING verify-first *while it runs* — 19.78 gaps/hr vs bar 15, btc
79/121 windows (65.3%) gap-affected, eth 0/121. **btc-only step change on 08-25**
(76→665/day, other coins improved), **no version change** (clob_v3_1 throughout),
**restart did not fix it** (08-26 04:38), **not host starvation** (load 1.25).
The failure mode flipped: `SLOW_CONSUMER_1013` (our consumer slow) →
`PING_TIMEOUT`/`NO_CLOSE_FRAME` (connection dies with no close handshake) —
points away from our code toward the path/peer for the btc market. One WS
connection per market (`collect_pm.py:14`) is why btc alone suffers. **Salvaging
hours 11–23 expires in hours**; exclusion is the coordinator's call, not DA's.
**Do NOT edit `da_await_gate.sh` in place while armed** — bash reads scripts
lazily from a byte offset; write-new-and-`mv` (the running process holds the
inode). Post-tape6d the wrapper arms against a named gate ref and refuses on
drift (ruled, unimplemented).

## 2026-08-27 (DA, ~00:2xZ) — 08-26 closed FAILING; four instrument defects fixed; day one accruing

**08-26 CLOSING VERDICT: ALL PASS = FALSE** (Q-DA-82, `abc7455`). post-freeze
PASS 278/278; complete_tape FAIL (**short 9 per coin**, = the ~43-min crash
outage); gap_rate FAIL (**508 gaps, 21.17/hr, 18 of 24 hours over, 50.7 min
lost**); **btc 178/278 = 64.0% gap-affected vs eth 0.4%.** Already inadmissible
per Q-DA-72, so this is the formality plus the final counts for the OPS
diagnosis. **The "+1 drip" closes as crash debris — my 9→10 reading came from
comparing against a moving elapsed baseline on an open day, not from windows
disappearing.**

**FOUR DEFECTS IN DA's OWN INSTRUMENT, all fixed (`ab0fad4`), selftests 13→17:**
1. `gaps_per_hour` divided by a **gap-derived** denominator, not elapsed time —
   30 s after the boundary one gap read as "1.0/hr", **~120x flattering**, and
   was relayed upward as possible abatement. Now elapsed-based; **no rate at all
   below one full hour.**
2. `day_closed` came from the selector's tape-derived predicate and called
   **08-26 open 30 s after 08-26 ended**, while driving the complete_tape
   branch. Now **calendar** closure, flags selector disagreement.
3. `complete_tape` **passed vacuously** on 0-of-0. Empty expectations can no
   longer pass.
4. **THE TIMER FIRED TOO EARLY — this one would have failed DAY ONE.** The
   collector records each window until `start + WINDOW_S + GRACE_S`, so a day's
   last window is still recording until **00:01:30** and gzipped after. Counts
   read 277/278 at 00:00:30 and 278/279 minutes later. **Timer moved to
   `00:06:00 UTC`** — verify only after the grace window.

**NEXT AND ONLY OWED ITEM: DAY ONE (08-27) VERDICT at 2026-08-28T00:06Z.**
Timer `da-midnight-verify.timer` is armed and daily. **BE SCORES NOTHING UNTIL
THAT VERDICT PASSES.** If the timer is gone, run by hand AFTER 00:06:
`python3 live/pm_research/da_forward_day_verify.py verify --day 20260827`.
**Open question it settles:** whether the btc degradation abated. As of 00:13Z
08-27 shows 4 gaps in 811 s with btc 2/2 windows affected — **not** abatement,
but n=2 is nothing. Do not read hour-zero quiet as a trend; that was defect (1).

## 2026-08-27 (DA, ~06:xxZ) — gate hardened after R-187; seam testing is now the rule

R-187 found two defects in DA's own state-tape gate; **the seam test it
mandated then found two more, also DA's.** Fixed at `334775d`, selftests
27 → 37. Details in STATUS.yml under `state_tape_gate`. The generalisation of
record: **owner-declares now covers the COORDINATE SYSTEM** — layout and clock
basis — not just the field list, because two modules can agree on every field
NAME and still disagree about where the fields live and what clock they are on.

**The lesson to carry, tested against DA's own work: all four defects survived
27 passing selftests, every predicate falsified and firing, and died to a
SINGLE REAL BUILDER ROW.** Per-module selftests certify a module's idea of the
world, not the world. Any DA instrument that sits at a seam should round-trip a
real artifact from the module on the other side before it is trusted — the
hand-made fixture is exactly where the shared assumption hides.

## DA's TWO OPEN GATES — both pre-registered, both waiting on someone else

**1. STATE-TAPE GATE (R-185) — the corrected Phase-2 freeze WAITS on this.**
`python3 live/pm_research/da_state_tape_verify.py verify --tape PATH
 [--gapped-slugs N]` · commit `cbfc44d` · 27 selftests.
Written and falsified **before BE's tape existed**. Expectations are READ from
`da_pred_state_v1_schema.json`, which is emitted by running the builder —
**builder → schema → gate, nothing transcribed at any link**, so a change to
the family reaches the gate automatically. Regenerate the schema with
`harmful_state_features.py --schema`.
Targets the five silent failures of the first tape (R-184/185): undeclared
reduction · guardless pin · zero-imputation · `bn_recv_ns` omitted (constant
freshness) · `gaps` omitted (unreachable `GAP_AT_CUTOFF`) · plus the embargo.
**Fails closed by design:** a tape lacking `feature_asof`/`decision_time` FAILS
the knowledge-time predicate rather than passing it vacuously, and an
unlabelled tape leaves the embargo **NOT CERTIFIED** rather than assumed clean.
**DA CHECKS, DA DOES NOT SPECIFY** (upheld as the rule for this review): if DA
tells BE how to satisfy the gate it stops being independent. Both sides read
the schema; neither reads the other's intent.

**2. DAY-ONE VERDICT — BE SCORES NOTHING UNTIL IT PASSES.**
`da-midnight-verify.timer`, daily, **00:06:00 UTC** (not 00:00:30 — the last
window records to 00:01:30 then gzips; verifying earlier undercounts). Next
fire **2026-08-28T00:06Z** against **08-27**, the day that closes then.
Hand-run after 00:06 if the timer is gone:
`python3 live/pm_research/da_forward_day_verify.py verify --day 20260827`
**Open question it settles:** whether the btc degradation abated. Do NOT read
hour-zero quiet as a trend — that was a defect in DA's own rate denominator,
since fixed (no rate reported below one elapsed hour).

## FORWARD-RACE ADMISSION SEQUENCE — confirmed, do not re-derive it

Settled with the coordinator 2026-08-26 ~18:5xZ after DA raised that the two
readings could not both be true. **Reading (a) is the ruling:**

* **2026-08-27T00:00:30Z** — CLOSING verdict on **08-26**. Already inadmissible
  (Q-DA-72, permanent complete-tape failure), so its value is the **final drip
  count** and the **end-of-day degradation figure** for the acting-OPS
  diagnosis. This boundary merely OPENS 08-27.
* **2026-08-28T00:00:30Z** — **DAY ONE's admission verification**, against
  08-27 as the day that closed.
* **BE MAY SCORE DAY ONE ONLY AFTER THAT VERDICT PASSES.** The coordinator is
  binding this into the day-one register entry so the gate cannot go out of
  order a second time (it did once — BE scored before DA verified, which turned
  an ordinary exclusion into a post-hoc call on a visible result, Q-DA-69).
* **The btc-gap risk deadline is NOT moved by this.** 08-27's tape starts
  ACCRUING at tonight's boundary, so the user's fix-or-accept decision wants
  making **before 00:00**, even though the verdict on it arrives 24h later.

`da-midnight-verify.timer` is `*-*-* 00:00:30 UTC` — **daily**, so it covers
both boundaries without further action. It logs the day that closed AND the day
that opened, to `derived/.da_midnight_verify.log` (append-only).

## 2026-08-26 (DA, ~18:4xZ) — Phase 2 verified + ratified; queue-validity null; verify-first is now an instrument

**Q-DA-78 verdict: ALL PASS.** BE's rebuild verified against v3 — 337 built vs
337 expected, 0 missing, 0 extra, forward reservation intact, consumed fragment
not re-entered. Arms scored, **Phase 2 ratified (R-175)**.

**Q-DA-80 (queue validity, R-173(2)): H0 FALSIFIED, magnitude negligible.**
195 OK rows of 645,851 (**0.030%**) begin after a PM gap but before their book
resync; `resync_lag_s` p50 and p90 both round to **−0.0 s**, worst −1.135 s,
**0 slugs left unresynced**, btc-only. **Consequence for Phase 2: NONE — the
ranking stands unqualified.** Caveat appended to both reserved fields
(`harmful_phase2_winner_FREEZE_ASK_v1::reserved_Q_DA_79`,
`phase2_three_arm_v1::da_caveat_field`), atomically, key sets asserted
unchanged. The pre-committed reading held in both directions: it stopped
inflation of a large result and equally stops claiming this small one vindicates
anything.

**TONIGHT'S DUTY IS NOW MECHANICAL — use the committed instrument:**
```
python3 live/pm_research/da_forward_day_verify.py verify --day 20260827
```
Exits non-zero on any failed predicate. Validated against known answers: it
reproduces Q-DA-69's 08-25 (673 gaps, 28.04/hr, btc 231/288) and Q-DA-72's
08-26 exactly. **Read `windows_gap_affected` beside gaps/hour, never instead —
28.0/hr sounds survivable while 80.2% of btc windows does not.** It states
reasons and does not exclude; the exclusion is the coordinator's to rule.

**LIVE RISK GOING INTO DAY ONE:** the btc degradation has NOT abated —
08-26 at ~18:40Z reads 19.26 gaps/hr, 12 of 19 hours over bar, **btc 127/212 =
59.9% gap-affected, eth 0.0%.** Four consecutive days now (btc 15.6 / 80.2 /
59.9; eth 1.4 / 0.7 / 0.0). **On current form 08-27 fails the gap predicate**,
and that is the most expensive day to lose. Cause unidentified; OPS's
commissioned measurement, seat unstaffed. Flagged to the coordinator before
midnight so it can be accepted or acted on in advance rather than read post hoc.
Also noted: 08-26's deficit grew 9 → 10 short per coin during the day, so one
additional window went missing after the crash outage. Too small to chase, but
a slow drip is exactly what a nightly check exists to catch.

## 2026-08-26 (DA, ~16:0xZ) — v3 re-base landed; btc n=166; verdict pending

**R-173 ruled for the row-level convention and the correction was DA's** — BE's
build was right; the 133-slug disagreement my verifier reported was a
DEFINITION, not a bug. **v3 (`53f12a9`) pre-registered before the build and met
exactly: OK 337, BINANCE 9, candidates 346. btc test n = 166 (was 33 — the
underpowered framing is superseded); eth 171.**

**Reversibility deliberately preserved:** v3 carries `pm_gap_s` /
`pm_gap_intervals` per slug (137 slugs, 2,098.6 s). **Filtering `pm_gap_s > 0`
reconstructs v2's 204 exactly** — the ruling demoted the window-level reading
rather than destroying it, so the alternative population stays examinable.

**OWED, unchanged in substance:**
1. **Verdict on BE's landing rebuild** — the watcher fires automatically and the
   verifier resolves to v3 (checked live). Hand-run fallback unchanged.
   **A mismatch stops the Phase-2 receipt, not the runs.**
2. **Verify-first on 08-27** at midnight — day one of the forward race.
3. **Q-DA-79 queue-validity test** — design declared, NOT run; non-blocking,
   after the verdict.

**Hard-won this hour, all three now fixed at the mechanism:** the watcher must
be a systemd unit (two harness background watches were killed mid-poll); its log
must be append-only from birth (re-arming it once ERASED the rejection verdict it
existed to hold); and the verifier must resolve the HIGHEST-version receipt at
run time (pinned to v2 it would have judged the rebuild against a manifest R-173
had already overruled — failing loudly, confidently and wrongly).

## 2026-08-26 (DA, ~15:3xZ) — R-171 verifier ARMED, waiting on BE's build

**OWED BY DA, exactly two things:**
1. **Verify BE's built top-up population** the moment
   `data/pm_5min/derived/harmful_exposure_rows_v3_topup.json` lands.
   **A watch is armed as a SYSTEMD USER UNIT** — `da-topup-verify.service`
   (check: `systemctl --user is-active da-topup-verify`; progress and verdict:
   `cat data/pm_5min/derived/.da_topup_verify.out`). It polls for the file,
   waits for the size to be STABLE, then runs the verifier, and **survives this
   session ending.** It is a unit and not a background job because **two
   harness background watches were killed mid-poll on 2026-08-26** while the
   artifact was still coming — the R-147(2) lesson (units survive, nohup does
   not) applied to DA's own instrument. **IF THE UNIT IS GONE TOO, RUN IT BY
   HAND — the verifier is committed and needs no setup:**
   `python3 live/pm_research/da_topup_population_verify.py verify`
   It exits non-zero on any failed predicate and REFUSES if the artifact is
   absent. Per R-171 **a mismatch stops the Phase-2 receipt, not the runs.**
   File the verdict under `Q-DA-*` **either way — a clean pass is a result.**
2. **Verify-first on 08-27** when it closes (midnight UTC). Day one of the
   forward race. **DA verifies BEFORE anyone scores** (R-153(2), hard
   precondition). The three predicates and the method are in `Q-DA-69`/`Q-DA-72`;
   08-26 is already ruled inadmissible for every line.

**The verifier (`da_topup_population_verify.py`, `c727f46`, 15 selftests) is
PRE-REGISTERED** — every predicate fixed and every falsifier run while the
artifact did not exist. Expectations come from DA's own pinned manifest in
`da_development_topup_v2.json`, never from the built dataset's own summary
fields. Predicates: slug-set IDENTITY (not counts), n_ok, **no reserved forward
tape (>= 1787702400)**, no consumed-fragment re-entry (<= 1787650200), t0 inside
the open interval, declared end 1787702410.0 covering the last window, non-empty.
**The forward-tape check is the one that matters** — that breakage is
unrepairable by construction, since the tape would be consumed.

**Q-DA-77 low, queued to BE:** `build_topup_rows.py:86` writes the final path
directly (`write_text(json.dumps(built))`) — non-atomic, against Phase-0's
atomic-write requirement, and the `json.dumps`-then-write shape is the R-148
allocation burst on a hundreds-of-MB artifact. **This is why the watch requires
size stability rather than mere existence; a hand-run verification should wait
for the file to stop growing too.**

## 2026-08-26 (DA, ~06:1xZ) — top-up v2: population UNCHANGED; scan race filed

**Q-DA-76 (`08a4923`)** — `da_development_topup_v2.json` supersedes v1 (rule 13;
**v1 untouched on disk**, mtime still 05:36 through two rebuilds incl. a crash).
BE's Q-DA-67 fix changed a file v1 pins, so the hash had to be refreshed.
**Population RE-DERIVED and compared, not assumed: 346/204 both,
`slug_manifest_sha256` IDENTICAL, 0 of 346 slugs differ.** The fix corrected the
selector without moving the population. v2 also declares
**`era_end_s = 1787702410.0`** for population `da_development_topup` — **BE reads
it from the receipt** (R-156(1)) rather than inventing one — and records
`era_floor_verified_at_ledger` from the new refusal guard.

**Reading note now in the register:** `markets.jsonl`'s pinned hash moves between
ANY two builds (append-growing registry). For a growing input a pinned hash is
**state-at-build provenance, not a reproducibility anchor**; the guarantee is the
population check. All 346 slugs still resolve, 0 missing.

**Q-DA-75 (`e45569b`, OPEN, LOW, BE-owned)** — `_bn_gap_index` globs bookTicker
with no upper bound, so builds open files the live collector is rotating; killed
one v2 attempt. **Build-fragility only, NOT reproducibility — measured:** in-span
gaps identical, added gaps after the cut, `last` only grows, so a fully-past
population's verdict cannot move. Fires only at hour boundaries, so it will look
intermittent. Fix: bound the glob by the declared end (exists post-R-154, never
reaches that function). NOT `except FileNotFoundError: continue`.

**Q-DA-74 → R-157:** my own Q-DA-73 cross-coin hazard was overstated (the fit is
per-coin, the evaluator generation-native — nobody makes that comparison). The
real exposure is fit-side: rows carry **no `sample_weight`**, so each generation
enters weighted by row count (up to 150x), invisible to a correct evaluator.
Ruled into Phase 2 as `w = 1/n_rows(generation)` for candidate fits only.

## 2026-08-26 (DA, ~06:0xZ) — R-153 ruled all five; day two FAILS verify-first; eth audited

**All five DA filings ruled in R-153 (`f264e60`)**: Q-DA-67 upheld (fix owned by
BE; Q-DA-68's keying is the reference implementation), Q-DA-69 adopted in full
(day one STAYS IN with the 80.2%/28-per-hr disclosure on every citation;
**DA-verifies-first restored as a hard precondition for days 2-5**), Q-DA-70/71
accepted, Q-DA-68 ACK.

**Q-DA-72 — day two (08-26) FAILS DA verification, filed BEFORE scoring.**
61 window files vs 70 elapsed slots, **short by 9 per coin**, from the R-147(3)
outage. Permanent; cannot self-heal. Do not score 08-26 as a forward day. The
btc/eth divergence is now three days — btc 15.6% / 80.2% / 50.0% gap-affected
vs eth 1.4% / 0.7% / 0.0% — which **refutes the host-wide cause DA itself
proposed**; cause stays with the (unstaffed) OPS measurement. v2.1 race stays
**G=1**; next candidate forward day is **08-27**, which is also the
harmful-cancel line's day one if the freeze lands first — one degraded day
would cost both lines at once.

**Q-DA-73 — eth stewardship: sound, and the stronger population.** Streamed the
full 1.24 GB v3.4 set (~90 MB RSS, 95 s; scanner validated element-for-element
against `json.load` on the 21 MB file first). **Headline reproduces exactly:
471 windows, 1,125,289 OK rows.** **eth carries 471,079 ACTIONS vs btc 355,165
(+32.6%) on 13.7% fewer rows** — at row level eth looks smaller, at the
decision unit it is larger by a third. **HAZARD: rows-per-action is btc 1.7169
vs eth 1.1169, a 1.537x differential** — a row-level btc-vs-eth comparison
flatters btc by ~54%, i.e. masks exactly the underpowering R-153(3) recorded.
Evaluate at actions (rule 2); tail p99 7, **max 150**.

**Lane state:** PRED_STATE_V1 stays built-not-scored until BE runs Phase 2.
Standing duty: DA verifies each forward day BEFORE it is scored.

## 2026-08-26 (DA, ~05:4xZ) — top-up materialised, PRED_STATE_V1 built, two blocking finds

Commit `c4621af`. **DONE:** (1) **R-145(3) top-up receipt** —
`data/pm_5min/derived/da_development_topup_v1.json`, 346 candidate slugs, all
counted by status. **Yield is thin and btc-specific: btc 33 OK / 137 PM_GAP /
3 BINANCE (19.1% usable); eth 171 OK (98.8%).** G=1 (08-25 only) so no
intervals. Era floor pinned to the R-145(3) LITERAL, never `max(ledger)`.
Derived rows NOT built (blocked on Q-DA-67) and the receipt says so — it pins
INPUTS, not outputs. **BE's Phase 2 is unblocked on DA's side; its btc
development population is 33 windows, recorded before any score exists.**
(2) **PRED_STATE_V1** — `live/pm_research/harmful_state_features.py`, §4
family declared in full, 67-check battery, AST guard proving no policy state
is emitted. Two bugs the synthetic battery passed and real tape killed:
exec-vs-cancel attribution was pairing trades within 50 ms and mis-filing ~80%
of executions as cancellations (replaced with CONSERVATION, no tolerance
parameter, 100% attribution); and `same_side_fill_share` was ALWAYS 0.0
because `fold_side` returns a TAKER side while rows carry a MAKER side.

**BLOCKING, both filed, neither mine to fix:**
* **Q-DA-67** — `harmful_exposure_rows.v2_era_bounds()` reduces the collector
  ledger with `max()`. R-147(2)'s restart row moved the era floor forward
  **39.4 h**, and `select_v2_era` now admits **0 of 926** windows. **BE's
  Phase-0 cent-exact repro cannot reproduce the v3.4 population today** — it
  fails closed, as an unexplained empty rather than a named refusal. Do not
  re-run the repro until this is ruled: a zero-window build looks like a
  failure of the MODEL when it is a failure of the SELECTOR.
* **Q-DA-69** — the R-141 day-one admission act never ran DA's verification
  step; BE scored first. Ran it: post-freeze PASS, complete tape PASS, **gap
  rate FAIL** — 08-25 at **28.0 gaps/hr** vs bar 15 and a prior five-day range
  of 1.4–4.1, **231 of 288 btc windows (80.2%) gap-affected vs eth 0.7% the
  same day**; still elevated on 08-26 (20/hr). The eth control refutes a
  host-wide resource cause. **Not decided by DA:** excluding a day that
  returned a NEGATIVE removes evidence against the candidate, and the call is
  now post-hoc because scoring preceded verification. Recommendation: keep day
  one IN with the 80.2% as a disclosed caveat, and **restore DA-verifies-first
  for days 2–5 while they are still unscored.**

**Also fixed (Q-DA-68, DA surface, maintenance):** `stamp_boundaries_ns`
treated every collector restart as a stamp boundary; now keyed on
`(collector_schema_version, stamp_point)` vs the preceding run per R-147(2).
Measured both arms — 08-26 (the discriminating day) 54/55 → 55/55,
`joint_covered` unchanged. Added `check_days()`: a dashed day token matched no
directory on EITHER tape and returned a silent empty, which made the first
invariance run compare two empty sets and call them identical. Selftests
40 → 53.

**NOTE for whoever holds OPS:** the dispatch table still prints the superseded
`MemoryMax=8G`; R-148(3)/R-150 govern (`--slice=research.slice`, ≤14G/job,
18.4G aggregate, `MemoryHigh` forbidden, venv python explicit).

## 2026-08-26 — OB dynamics loop CLOSED; stateful phase dispatched (R-145)

The fine-feature loop closed at I5 (commit `f1ceec9`, five receipts):
**reduced fine CONFIRMED** (PM + L1 imbalance + 10–250 ms mid-move), extended
HELD @5%-only, depth20/PM-thinning/btc-lead rejected-null-flagged. Five specs
consumed the 08-24/25 fragment — scoreboard ban permanent. **The freeze of
reduced(+extended) is the USER'S decision, open; Phase 0 gates it.**

Next phase governed by `live/pm_research/plans/STATEFUL_HARMFUL_CANCEL_TODO.md`
(`e3d3aaf`), dispatched in `workspace/COORDINATOR_DISPATCH.md` (rewritten,
R-145): **BE** Phase 0 manifest/repro (THE blocking item: cent-exact
reproduction of the reduced-fine receipt from committed code) then Phase 2
model comparison; **DA** Phase 1 `PRED_STATE_V1` builder + the declared
development top-up receipt (slugs strictly after `1787650200`, strictly before
08-26 00:00 UTC — development only, never forward) + R-141 daily-admission
check; **DE** Phase 3 stateful cancel/hold/repost state machine + parity
battery vs `QR_SKEW_ONLY` (buildable now); **OPS** heavy-run standing rule
(`systemd-run --user -p MemoryMax=8G -p OOMScoreAdjust=1000`, R-145(4)) +
recv_ns-degradation measurement + collector watch. Q-OPS-13 answered-adopted;
Q-DE-14 closed superseded. Tape from 08-26 00:00 UTC on is untouched, reserved
for forward validation (≥5 complete UTC days after the freeze).

**Read `live/pm_research/FLOW_MODEL_STATE.md` FIRST** -- it is authoritative for the flow model and twelve documents defer to it. All work is on branch
**`mm-research`**; nothing is on `main`. Sigma remains **Revision 5 / PRICING
HOLD**, while the offline measurement stack is complete through contracts
**v22**. `route_a_v1` has one OOS test day; per-symbol `route_a_v2` is
pre-registered and begins primary evaluation on 2026-08-22. Neither is
authorized for probability-level use.

## SESSION ROLES 2026-08-23 (v2) — FOUR plane sessions, one coordinator

**Re-assigned by the user. The programme now runs as four worker planes plus a
separate COORDINATOR seat**, superseding the earlier DE-worker/coordinator
split. Everything produced under that split — the DE plans, `DE_MODULE_PLAN.md`
Revision 2, `DE_PLACEMENT_POLICY_PLAN.md` Revision 3, `DE_PLAN_REVIEW_LOOP.md`
iteration 1 — stands unchanged.

| plane | session |
|---|---|
| COORDINATOR | tmux `pmmm-coordinator` |
| DA | tmux `pmmm-da` |
| BE | tmux `pmmm-be` |
| OPS | tmux `pmmm-ops` |
| DE | tmux `cta` |

**➜ The cross-session interface is `workspace/COORDINATION.md`.** Ownership map,
coordinator-gated decisions, active FILE LOCKS and the dispatch ledger live
there; read it before touching another plane's code. This file (`HANDOFF.md`)
and `STATUS.yml` are **coordinator-owned** from now on to stop concurrent
writes — send state to the coordinator rather than editing them.

**OPEN, and it corrects this document:** both hourly timers have been FAILING,
not returning `IDLE`, for ~21 h — `COORDINATION.md` dispatch D-1. `tier1/` holds
only `day=2026-08-20`, btc/eth/sol/xrp, `lane=measurement`, and **no `full` lane
receipt has ever committed**, so Tier-2 has never run for real. The claim below
that the timers own catch-up is currently **false**, and OOS days are not
accruing through the committed lane. Raw tape is intact; OPS holds the repair,
with `measurement_batch.py` and `tier1_pipeline.py::normalize_clob` locked to it.

The interface between sessions is the repo files, not conversation history.
Consequences, per the split this program already runs (coordinator writes
decision rules; worker runs measurements and builds):

- **DE owns (unchanged):** the two DE plans and their revisions, the
  DE review loop (`DE_PLAN_REVIEW_LOOP.md`, running), DE replays/probes
  (warning-window envelope, cancel grid execution, composed skew×cancel,
  cross-window correlation), and DE-side implementation when authorized.
- **Coordinator ratifies (do not self-decide):**
  1. `CANCEL_POLICY_PROTOCOL.md` — verdict bars, grid, cooldowns, and the
     §8.1 envelope branch threshold are DECISION RULES; the DE session
     prepares the draft, the coordinator freezes it before measurement.
  2. Cross-plane contract changes in `DE_MODULE_PLAN.md` §6.2 — above all the
     NON-ADDITIVE `DecisionProblem.belief` widening (a migration touching a
     shared type).
  3. SP-Params choices surfaced by the plans (γ ladder, flat band + hysteresis
     grid, `r=60` handle, cancel-by deadline, total capital, κ_$,
     ScenarioLossLimit).
  4. Three iteration-1 design calls recorded in the plans as proposals, open
     to coordinator re-cut: HALTED blocks even risk-reducing `CROSS` (carry is
     the designed degradation); feasibility prices CONTINGENT `L_adv`
     (position + worst-case fill of resting quotes); one shared REDUCING-ONLY
     state for cap-breach / `r<60` / DEGRADED.
- **Standing discipline unchanged:** rules are not re-cut after their answers
  are visible; nothing measurable recorded as settled; plans stay DESIGN.

## SESSION 5 2026-08-23 — the decision module has a plan

**The DE plane is now planned at two levels.**
`plans/DE_MODULE_PLAN.md` covers the **module structure** — all five DE modules
(ActionSpace · Constraints · DecisionScheme · Allocator · Actuator, all
unbuilt), what measurement already pins in each, and a demand-driven build
order. Key structural decisions: ActionSpace is five verbs
(`PLACE/CANCEL/CROSS/MERGE·SPLIT/NONE`) over **one** signed Up-equivalent
exposure (the exact identity makes complement verbs a double-count), and
placement is a level-policy, not a price; Constraints is an oracle whose caps
price `L_adv` side-aware (a `|net|` cap is not a risk limit); the composed rule
policy registers as solver plugin `RulePolicy_v1` with a **no-belief-inputs
manifest** (the optimizing seams stay empty because FlowAndFills is unvalidated
and Route A is HOLD — measured blockers, not taste); utility is deliberately
unchosen; coupling is `{Up,Down}` ATOMIC (exact) plus same-coin SHARED_RISK
with **unmeasured** correlation; the Actuator is where the tape ends, and its
deployment-measured ack latency **selects the operative τ rung** of the cancel
ladder rather than triggering re-research. Buildable now: the ActionSpace/
Constraints vocabulary inside the replay harness, and the **cross-window
correlation measurement** (retires DA-plan falsifier #2, decides the
Allocator's character).

**`plans/DE_PLACEMENT_POLICY_PLAN.md` is now Revision 2** (Rev 1 in git
history). It absorbs the three measurements that landed after Rev 1 — its own
§7.1 answered `SKEW_ROBUST`, the exact one-book identity, Layer-1
negative-for-never-cancels — and adds the lever Rev 1's menu lacked:
**cancellation**, the only lever that acts on fill quality. The composed policy
is four questions: WHERE to rest (skew, measured) · WHEN to leave (cancel,
unmeasured) · WHEN to cross (`N*` backstop) · WHEN to stop (`r≈60` schedule).

**The pre-committed execution order (plan §8), falsifier-first:**

1. **Warning-window distribution, POLICY-FREE**, on the existing `edge_l1_v1`
   fill set under both queue bounds: the share of negative drift sitting on
   fills whose warning exceeds each cancel latency `τ`. One number decides
   whether any cancellation policy can work on this tape *before one is
   built*. The mechanism case: drift ~92 % complete by 5 s, flow clustered at
   75–350 ms (a burst has a first trade), and the queue is a warning buffer —
   depth ahead leaving by *churn* is visible to a depth-depletion trigger even
   when it never trades.
2. Freeze `CANCEL_POLICY_PROTOCOL.md` (trigger family T-FLOW/T-DEPLETE/T-MID/
   T-AGE, `τ` ladder 0–1000 ms, named re-post rule, every grid cell reported),
   then run it on static `JOIN_BBO`. Three axes mandatory; excised fills get
   counterfactual markout rows.
3. Composed skew × cancel replay — including whether the fronted reducing
   side's zero-warning exposure shows up per side (tension recorded in plan
   §4.6).

No code was written; no measurement was run. Next session starts at step 1.

## SESSION 4 CLOSE-OUT 2026-08-23 — the first determinate answer

**`live/pm_research/FLOW_MODEL_STATE.md` is authoritative for what the flow model
believes.** Anything conflicting with it is stale. Eleven probes, **350 selftest
checks**, all green. Collectors up 47.9 h, **five UTC days** on disk.
`route_a_v1` day 4 of 10; `route_a_v2` primary evaluation open. Both frozen.

### A passive maker who never cancels LOSES on both verdict coins

Layer-1 markout against **book mid**, decomposed — replacing the settlement
estimand, which measured hold-to-expiry drift rather than spread capture:

```
btc h=5s  n=10,294  markout -0.532 [-0.797,-0.287]  spread +0.642  drift -1.175
eth h=5s  n= 1,999  markout -1.243 [-1.726,-0.759]  spread +0.778  drift -2.021
```

Spread capture is **real, positive, stable**. Post-fill drift is **1.8x larger on
btc, 2.6x on eth, negative.** Six of eight cells negative, interval excluding
zero.

**This closes a loop the fee structure opened**: takers pay ~225 bps to cross, so
anyone crossing is heavily informed. "The fee does not kill MM on cost, it loads
the question onto adverse selection" was the prediction; adverse selection is
roughly double the capture.

**READ IT NARROWLY.** Every simulation in this corpus rests the order until
filled or the window ends — **nothing ever cancels**. So this is *"a maker who
never cancels loses here"*, not *"market making loses here"*. The gap is the
whole DE question, and **simulating a cancellation policy is the highest-value
unmeasured lever left** — same harness, one more rule, data in hand.

### Inventory: control is load-bearing, and placement skew works

`net` does **not** self-balance — reversion half-lives 519–2726 s, all longer
than the window. Placement skew cuts terminal `|net|` **76–89 % (btc)**,
78–81 % (eth), cash at risk ~13x. The published 15x was the optimistic end of a
**narrow** band, properly bounded.

**Two-sided quoting only works where flow is thick**: two-sided ÷ one-sided is
btc 0.101, eth 0.199, but **doge 1.173, hype 1.752** — on thin coins the second
quote makes inventory *worse*. Third independent argument for btc/eth-only.

**The queue is a risk filter.** `NEW_BBO` symmetric is a random walk at ~9.4x the
risk; the same property is a liability when flat and exactly what you want when
reducing, which is why the skew is asymmetric.

### Inventory is THREE things in THREE planes

Dependency **SP ← DA ← BE ← DE**; **BE must never read DE**.
*What do I hold* → `DA-State` (`plans/DA_INVENTORY_STATE_PLAN.md`).
*What may I hold* → `SP-Params` → `DE-Constraints`.
*What do I do about it* → **`DE-DecisionScheme`**
(`plans/DE_PLACEMENT_POLICY_PLAN.md`) — **and it IS the placement policy.**
`BE-FlowAndFills` is inventory-agnostic by rule.

### Also settled

- **The two books are ONE book, exactly** — `bid(Up)+ask(Down)=1.0000`,
  1,081,800 checks, **zero violations, worst deviation 0.00000**.
- **Hawkes censoring was our grid** — venue clock is milliseconds; floored at 10
  ticks, clustering runs **75–352 ms**, two estimators agreeing independently.
- **`f_r` binning replaced** — body 4x60 s absorbs the unidentifiable term *by
  construction*, terminal 12x5 s.
- **Terminal confound partially broken** — uniform artefact refuted at 6–7x.
  TWAP **favoured, not established**.
- **U9 closed at n=13** — `MNAR-suspect` stands.

### Open, sorted by what would actually move it

**Permanent:** queue-position inference, sub-millisecond structure, own impact,
ack delay, hidden liquidity.
**Calendar:** layer generalization (~10 days), maker-edge sign (~25–30x data),
rebate `rho`.
**Cheap and unmeasured — next:** a cancellation policy.
**Unreconciled:** settlement census `+0.173` vs Layer-1 `-0.53` on btc. Different
estimands, different populations, not a contradiction — reconciliation
unmeasured and deliberately unnarrated.

### Method lessons, each paid for

1. **A gate that cannot fire is not a gate** — three written, including an
   algebraic identity and a threshold against a denominator 60–1000x too large.
2. **A SHA-256 is a change-detector, NOT a conformance checker** — committed code
   conformed to no frozen protocol while the snapshot verified clean.
3. **The name is not the definition** — five instances, two self-inflicted.
4. **State the population of every denominator** — six instances, three read as
   findings.
5. **A hardcoded day list cannot survive a running collector** — `DAYS` went
   stale **four times in three days**, the last within twelve hours of being
   fixed. Now DERIVED from disk, with `provenance(sampled=...)` recording days
   actually **sampled**, since `select()` takes the earliest slugs and a new day
   can be globbed without entering the sample. **Compare on `days_sampled`.**
6. **Do not slice source by index to edit it** — two files broken that way today,
   once deleting four functions including the conformance checker. Anchor to
   exact strings.

## Read this first

The sigma **engineering pipeline is finished and frozen**. Collectors are
supervised; the all-coin Tier-1 batch, knowledge-time leak canary, model-free
Tier-2 terminal markout and fixed-grid calibration scaffold are immutable,
resumable and commit-last. Hourly user timers run the measurement lane at minute
20 and Tier-2 at minute 40. At this checkpoint both correctly returned `IDLE`:
2026-08-20 is not eligible until the adjacent 2026-08-21 UTC day closes. No
partial smoke is a research result.

The empirical state has not changed: `route_a_v1` has one of ten required OOS
days and all 84 gates are `INSUFFICIENT_EVIDENCE`. `route_a_v2` is per symbol
and horizon, uses signed-x conditional variances with no cross-instrument
pooling, and needs all 126 gates to pass on primary days from 2026-08-22.
Pre-freeze rows are design/training data, not fresh validation evidence. Leave
sigma code untouched while days accrue and continue independent mechanism work.

Reading order:
1. `live/pm_research/PM_ARCHITECTURE.md` — the entry point; explanatory structure.
2. `live/pm_research/contracts/contracts.yaml` (**v22**) — machine-readable
   source of truth for types. The prose defers to this file, not the
   other way round.
3. `live/pm_research/MEASUREMENT_PIPELINE.md` and `EVALUATION_PIPELINE.md` —
   current Tier-1/Tier-2 runbooks and claim boundaries.
4. `live/pm_research/SIGMA_ROUTE_A_RESULTS_2026-08-20.md` — the measured,
   strictly-forward Route-A result and current verdict.
5. `live/pm_research/SIGMA_ROUTE_A_PROTOCOL.md` — protocol frozen before fit;
   includes the non-analytic post-run embargo-wording erratum.
6. `live/pm_research/SIGMA_ROUTE_A_V2_PROTOCOL.md` — pre-registered
   conditional-variance successor; evaluation begins 2026-08-22 and no v2 fit
   exists yet.
7. `live/pm_research/GFF1_RESULTS.md` — frozen v3 side-convention PASS evidence.
8. `live/pm_research/EXP_RESULTS_2026-08-20.md` — earlier model results.
9. `live/pm_research/SIGMA_PLAN.md` — **REVISION 5, canonical.** One consumer
   matrix, one PRICING law (route A) and one DIAGNOSTIC decomposition (route B),
   never summed, now enforced as a TYPE boundary. **Read §2.3 then §1a** — the
   route decision scopes everything, and §1a says where each consumer's number
   actually comes from. v1/v2 text is in git history.
10. `live/pm_research/SIGMA_PLAN_REVIEW.md` — first implementation-readiness review.
11. `live/pm_research/SIGMA_PLAN_REVIEW_ITER2.md` — review of Revision 2.
12. `live/pm_research/SIGMA_PLAN_REVIEW_ITER3.md` — review of Revision 3 and v14;
   historical input to Revision 4.
13. `live/pm_research/SIGMA_PLAN_REVIEW_ITER4.md` — review of Revision 4/v15; its
   six items are applied in Revision 5/v16.
14. `live/pm_research/SIGMA_PLAN_REVIEW_ITER5.md` — pre-measurement verdict:
   MEASUREMENT GO / PRICING HOLD**, plus the frozen fit sequence.
15. `live/pm_research/sigma_kernels.py` — executable model **fixture**, not a
   frozen spec. `--selftest` checks exact arithmetic under a **declared and
   still UNVERIFIED** sampling convention; it does not establish that convention
   against the Chainlink streams.
16. `live/pm_research/plans/BE_FLOWANDFILLS_MODEL_PLAN.md` — **flow-and-fills
   Revision 4, canonical and frozen**; per-coin marked flow, execution reach,
   observable queue bounds, and separate development/validation states.
17. `live/pm_research/FLOW_MODEL_PROTOCOL_V4.yaml` — machine-readable freeze;
   development fitting is allowed now while promotion still requires 10
   complete forward UTC days per coin.
18. `live/pm_research/FLOW_FILL_DEVELOPMENT_RESULTS.md` and
   `flow_fill_development.py` — two-hour B0–B3/mark/Hawkes/fill development run;
   explicitly not decision eligible.
19. `live/pm_research/FLOW_INTENSITY_RESULTS.md` and `flow_intensity.py` —
   corrected same-state descriptive `f_r`/`f_p` evidence and executable guards.
20. `live/pm_research/plans/` — BE_BELIEF, MEASUREMENT, PRELIMINARY, and
   historical plan inputs.

Before any book/trade/queue analysis, read
`live/pm_research/DATA_COLLECTOR_AUDIT_2026-08-20.md` — current collector
verdict, live evidence and acceptance boundary. **Read its v3 section: the v2
addendum's "repair successful" is withdrawn, and the root cause is now measured
rather than hypothesised.**

### Session close-out 2026-08-21 01:34 — what the monitoring loop established

**The binding constraint is now the calendar, not the code.** `route_a_v1`'s
frozen gate needs **10 OOS test days**; 2026-08-19 is training-only, so the count
is **1 of 10** (08-20 complete, 08-21 in progress), tracking to ~2026-08-29.
Nothing about the collectors or the spec blocks that; it has to elapse.

**The Route-A prices lane is self-healing, which is not the same as clean.**
I had been reporting it healthy on the strength of `open_gaps=[]` at each check.
Over 11 hours it actually logged **58 gaps and 26+ reconnections** — roughly one
11–13 s gap every 20 minutes — under four causes: `GLOBAL_SOCKET_SILENCE` 38,
`TOPIC_STALE` 7+, `PEER_TOPIC_RECONNECT` 3, `CONNECTIONCLOSEDERROR` 2. Per-hour
counts `15:4 16:2 17:4 18:8 19:4 20:6 21:10 22:10 23:4 00:4 01:2` — a level in
the 2–10/hr band, no trend. Every gap closed; none ever left open.

**Why that matters:** an 11–13 s gap landing on a decision time breaks the
protocol's ≤5 s predictor-staleness rule for that horizon, and a long one breaks
the 90 % coverage rule for the whole window. **This is a candidate mechanism for
`route_a_v1`'s 374 `s30_window_coverage` exclusions (19 % of windows)** — the
MNAR risk that is still unaudited. One outlier so far: a **44.8 s `TOPIC_STALE`
pair at 22:28**, 3.5× any other, which at 14.5 % of a 310 s span fails the 90 %
rule outright for every coin and horizon.

**CLOB lane, closed out.** 46 disconnects across all eras; `ws_ever_paused` has
read True **0 times**, deepest backlog 254 of 65,536 (0.4 %). Cumulative loss
149.6 s across 36 windows, **zero unclosed gaps**. 1013s are bursty around
2–3/hr with whole hours of quiet. Gap-touched BTC windows sit at ~22 %.

**A discipline note worth keeping.** Twice I called a direction on one interval —
a "rising 1013 frequency" and a "30 % gap-touched share" — and both dissolved
within the hour as the denominator grew (30 → 25 → 23 → 21 → 22 %). Report burst
minutes and per-hour counts; do not describe a direction until several hours
separate the observations. Neither claim reached these documents, which is what
saved them.

**Correction, so nobody re-derives the wrong urgency.** I claimed twice in
session that the counts-only exclusion ledger meant selection-audit information
was "not reconstructible later". That is **wrong**: the protocol reads immutable
rotated `.csv.gz` archives plus jsonl byte snapshots, so re-running it reproduces
the identical exclusion set and window identity can be recorded then. Building
the exclusion ledger is a **scheduling choice, not a race**.

### Post-close-out implementation 2026-08-21 — completed through Tier-2

**`route_a_v2` is now frozen before its evaluation data.**
`SIGMA_ROUTE_A_V2_PROTOCOL.md` (SHA-256
`c75fd12e74e8400f3761111028a14f75ddc6ae6e2629dd7fc13d1cf5e116456a`)
keeps the v1 mean and replaces only pooled residual variance with three signed-x
tercile variances estimated from historical strictly-forward residuals, with a
fixed 30-row shrinkage weight. Primary evaluation begins 2026-08-22. It needs
10 future evaluation days, 30 rows per cell and 126 passes: conditional mean,
conditional-variance calibration and paired Gaussian quasi-score for every
symbol/horizon. It is **PRE-REGISTERED / NOT FITTED / PRICING HOLD**.

**The Route-A selection ledger is now executable.** Commit `d8a8481` first
recorded excluded-window identity and an S60 range proxy. The current extension
normalises the audit unit to one `(slug,horizon)` candidate, expands window-wide
failures across all six horizons, records both accepted and excluded rows, and
joins separately hashed price-gap cause/version intervals without allowing them
to affect eligibility. A temporary full run over 2,943 final resolutions emitted
17,658 unique candidate keys (exactly six per resolution), 14,644 accepted and
3,014 excluded, with zero duplicates. Those counts include the incomplete
2026-08-21 day and are verification figures, **not a new v1 result**.

**The post-sigma measurement foundation is complete.** `da_state.py`,
`tier1_pipeline.py`, `coverage_ledger.py`, `replay_canary.py`,
`daily_pipeline.py` and `measurement_batch.py` implement the closed-day Tier-1
DAG. All requested coins are preflighted before writes; partitions are
code/schema/input-addressed, immutable and merge-never-overwrite; coverage facts
remain separate from the frozen admissibility rule; the cross-coin receipt is
published only after exact validation. Interrupted valid staging is reusable
but never mistaken for completion.

`evaluation_pipeline.py` implements Tier-2. It requires a complete `full`
Tier-1 receipt and the exact frozen G-FF1 v3 PASS artifact. It emits one
model-free gross terminal maker observation per parent trade and exactly one
calibration row per `(slug,r_s)` on the nine-point frozen grid. Knowledge-time
quote selection matches `StateView`; invalid and unavailable states remain
named rows instead of disappearing. Every markout summary carries both per-fill
and share-weighted estimates per coin and phase. With too few day clusters,
manifests explicitly say `DESCRIPTIVE_POINT_ESTIMATE` and CI unavailable.

Contracts **v20** add R-BATCH, R-DERIVE, R-GROSS, R-DUAL and R-ONEROW plus the
batch/evaluation carrier types and orchestration modules. The two user timers
are installed and active. Their first 2026-08-21 invocations returned `IDLE`
because the adjacent day was still open. That is the expected readiness gate,
not a failure. Partial smoke counts remain wiring evidence only; the first real
all-coin receipts will be created automatically after the boundary closes.

### G-FF1 v3 passed — `side` is the taker's

The frozen v3 run is **PASS**: agreement **600/600 = 1.0000**, Wilson95
**[0.9936, 1.0000]**, with every one of seven coins and five moneyness buckets
perfect. There are zero excluded rows. The side-evidence artifact and SHA are
mandatory Tier-2 inputs, so terminal markout cannot run on an assumed sign.

**The exchange ABI in the older docs does not apply to this tape.**
`0xe111180000d2663c0091e4f400237545b87b996b` replaces `(makerAssetId,
takerAssetId)` with one asset id plus an explicit `uint8` side. Signatures were
verified by keccak-256, not taken from a lookup:
`OrderFilled(bytes32,address,address,uint8,uint256,uint256,uint256,uint256,bytes32,bytes32)`
and `OrdersMatched(bytes32,address,uint8,uint256,uint256,uint256)`.

**`fee_rate_bps = 0` is a websocket artefact, and the fee schedule is now
MEASURED.** `taker fee = 0.07·p·(1−p)` $/share, matching to four decimals on
**600** transactions across the full moneyness range — 1.75 ¢/share at `p=0.5`,
0.63 ¢/share at `p=0.1`. **The taker pays and the maker does not**: 600/600
taker legs carry a fee, **744/754 maker legs carry zero**. `BE_FLOWANDFILLS_PLAN`
§12.1 had already derived the formula from a single transaction; this confirms
it at scale and adds the incidence, which was not previously established. The
Q5 reading `0.07·min(p,1−p)` (3.5 ¢/share) is **REFUTED at 2×**.

Consequence for the model, and it is not the comfortable one: makers pay no fee,
but a taker crossing at ATM pays ~0.50 ¢ half-spread + 1.75 ¢ fee = **~2.25
¢/share, about 225 bps on a $1 binary**. Nobody pays that casually, so every
maker fill is against a counterparty who expected more than 2.25 ¢ of move.
**The fee does not kill market making on cost; it loads the entire question onto
adverse selection.**

Still open: the maker **rebate** (zero fee is NOT evidence of a rebate paid, and
`OrderFilled.fee` is unsigned so a rebate could not appear in it), and **10 of
754 maker legs that DO carry a fee** — verified under an unambiguous test
(`taker == exchange address`), so this is a real residual class, not a
classification artefact. Mechanism unexplained.

(A tempting corollary — that fee presence explains the residual
mismatches — was tested and **refuted**: fee is present on validated and
mismatched rows alike, so it discriminates nothing.)

**Why `gff1_v1` is on the record as superseded.** v1 hard-coded the BUY reading
of the amount pair and returned `226/226 = 1.0000` — on a sample that was
**100 % BUY, zero SELL validated**, with a 0.548 mismatch rate. The frozen
mismatch ceiling is the only reason that one-sided result did not read as clean.
An order's `makerAmount` is what its creator *gives*, so the pair is ordered by
direction; under the BUY-only reading a SELL decodes to a price above 1, which
is impossible in a prediction market. 235 of the 274 mismatches reconciled
exactly under the inverted reading, which is what confirmed the diagnosis.
**Keep the guard discipline: it caught a defect that pooled agreement hid.**

**Open, characterised, unexplained:** 27 residual legs (5.4 %) where size
matches *exactly* and only price differs — the websocket price sits on the tick
grid (0.12, 0.65, 0.05) while the chain effective price does not (0.115862,
0.649168, 0.048008). Direction resolved for all 27 regardless. Not a direction
failure; a price-comparison artefact of unknown mechanism.

The v1/v2 failures remain useful provenance: v1 validated only BUY rows; v2 had
473 validated clusters and missed its frozen sample/mismatch guards. Neither is
retroactively called a pass. Full evidence is in `GFF1_RESULTS.md`.

### Flow-and-fills Revision 4 — development runs now; validation still waits

`plans/BE_FLOWANDFILLS_MODEL_PLAN.md` is now the only authoritative flow spec;
`FLOW_MODEL_SPEC_REV2.md` is explicitly historical. The machine-readable freeze
is `FLOW_MODEL_PROTOCOL_V4.yaml`, made before the primary period beginning
2026-08-22. Revision 4 separates `DEVELOPMENT` from `VALIDATED`: existing hours
may test the estimator and queue mapping now, while promotion still requires at
least 10 complete forward UTC days per coin.

Revision 4 retains the Revision 3 state corrections and closes the missing fill
seam:

- `lambda` is per-coin **event count intensity** in events/s. Side is a
  conditional mark, never the realized-next-side covariate in total intensity.
- Actual monetary mark is `size * native_execution_price`; USDC/s is derived as
  count intensity times the conditional monetary mark. It has no point-process
  compensator and never substitutes for an underpowered count model.
- `MICRO_002` and `MARKET` are labelled subprocesses. Their independence is a
  tested null after cause-specific baseline time changes, not a prerequisite
  for estimating ex-micro flow. If dependence exists, use a two-type model.
- Hawkes is an optional residual in baseline operational time. It is admitted
  only after residual clustering survives Holm correction on a complete B0–B3
  baseline and at least 10 forward days. Retention then requires forward NLL
  improvement, stable branching (`n<1` or spectral radius `<1`), and improved
  residual calibration.
- Execution price and size are marks. For an exact frozen shadow action they
  determine cumulative shares that reach the action level; unconditional
  arrival rate never determines notional or fills by itself.
- Public level-total L2 cannot identify exact queue position. Every action
  therefore returns the optimistic front and conservative trades-only
  back-displayed fill quantities. The midpoint is forbidden and collector-gap
  paths remain explicit unavailable rows.

The original `f_p` profile is **WITHDRAWN**. Its numerator used folded execution
price while its denominator used midpoint dwell, so it did not estimate one
conditional rate. `flow_intensity.py` schema v2 now uses the exact same
250 ms-lagged Up-midpoint intervals for arrivals and exposure; collector gaps
kill state until a new quote matures. All 31 semantic/control selftests pass.
On the corrected six-window design sample, execution price would have selected a
different bin for **6.9% of BTC** arrivals and **38.1% of HYPE** arrivals, which
shows the defect was material. The replacement shape is descriptive only and
is conditioned on `r` in the forward model.

`flow_fill_development.py` now runs the first executable lane on 24 consecutive
five-minute windows per coin (2026-08-20 17:45–19:45 UTC): 80,714 admitted
arrivals. Within-design held-window NLL says B1 beats B0 on all seven coins; B3
beats B2 on six, while B2 is mostly unsupported/neutral. The exploratory
operational-time Hawkes grid selects branching 0.40–0.55, but it resets each
market with no warm-up and is stamped `DEVELOPMENT`. At 15 seconds the 5-share
join-touch any-fill bracket is very wide: HYPE 71.3% front versus 2.4% back;
BTC 94.6% versus 76.9%. This is quantity evidence only—no fill-conditional
markout or P&L verdict. Full results are in `FLOW_FILL_DEVELOPMENT_RESULTS.md`.

Contracts **v22** retain R-FLOW and add R-FILL, `FlowAction`,
`QueuePositionBound`, `FillQuantityBound`, `FlowActionFillFit`, and a separate
non-decision `HawkesDevelopmentDiagnostic`. `BE-FlowAndFills` now requires a
`VALIDATED` action-fill artifact and remains unavailable.

**Next:** freeze and implement the conditional M1–M4 mark-law families, then
freeze the candidate code before scoring post-cutoff days. Continue accumulating
primary days in parallel. Do not promote the provisional Hawkes or fill
parameters; the 10-day minimum and forward gates still apply.

### Flow-model evidence audit trail — pre-Revision 3

Charter `live/pm_research/FLOW_UNCERTAINTY_LOOP.md`; plan
at that time (now superseded by Revision 4); probe `flow_uncertainty.py`. Coordinator
writes the decision rules, research agent runs the measurements — the split is
deliberate and has already stopped two rules being re-cut after their answers
were visible.

**The plan is a first-principles rebuild.** The old G-FF1..G-FF4 chain is
replaced. From the identity `net = half_spread + rebate − maker_fee − AS`, the
**sign is independent of queue position** — queue enters only via `E[N]` and the
conditioning inside `AS` — so the old chain put an unidentifiable quantity
(`Q_ahead`) ahead of a question that does not need it. New order: cost schedule →
sign → marginality → scale. `E[outcome − ℓ | fill]` needs **no fair-value model**,
which decouples this module from the 10-day sigma clock entirely.

**Closed so far:**

- **U1a `CLEARED`** — `size` is shares at 6 dp, exact against chain 600/600.
  **The volume layer is UNBLOCKED.**
- **U1b `CLEARED` / SINGLE-ACTOR — but the pooled share is a BTC ARTEFACT.**
  **One address**, present in all seven coins, at exactly 0.02 shares and
  **99.98% SELL**, carrying **0.0145% of notional**. 300-transaction unstratified
  draw: top-1 = 100%, distinct = 1, HHI = 1.0000.
  **CORRECTED 2026-08-21 by the intensity fit — the "16.3% of events" figure I
  recorded here is POOLED and hides a 45x range**, because btc is 64% of the
  pooled denominator:

  | coin | arrivals | micro share |
  |---|---:|---:|
  | btc | 270,404 | **2.0%** |
  | eth | 56,265 | 22.4% |
  | xrp | 24,107 | 59.9% |
  | bnb | 16,925 | 78.2% |
  | hype | 16,160 | **90.0%** |
  | pooled | 423,134 | 18.3% |

  **On btc the count layer is barely touched; on hype it is not contaminated by
  that actor, it largely IS that actor.** So R-DUAL is not a uniform reporting
  convention -- for thin coins the count layer is close to unusable and the
  notional weighting is the only meaningful one. This is the SIXTH instance of
  the denominator/population defect, and it was inside a number this file
  already carried as established.
  **THE INVERSION still holds:** raw-count intensity is contaminated by an
  economically empty class; notional-weighted intensity is not. Rule **R-DUAL**:
  every intensity AND every **signed** flow quantity (imbalance, side mix,
  signed volume) is reported both ways, exclusion published beside the retained
  set. Signed quantities are the fragile ones -- the contamination is ~100%
  one-signed and does not average out.
  **What the address is doing is NOT established and must not be narrated.**
- **U2 `CLEARED`** — tick composition: 0.001 exists only in the tails (6.75% at
  `p<0.15`, 6.73% at `p>=0.85`), **absent from the middle three buckets**. Where
  0.001 is available the spread is 1 tick in **99.9%** of quotes, so the 1-cent
  spread is a **CONSTRAINT, not a convention** — makers step inside the moment
  the venue allows. `γ_tick` is collinear with extreme moneyness and must be an
  interaction inside the tail buckets, never a main effect.
- **U3 + U3a `CLEARED` via the bound branch** — gap exposure concentrates at
  window **open** (31.7% of lost seconds in the first 30 s, 3.2x mean). KS
  occurrence was refused as **insufficient power**, not uniformity
  (`D=0.132, p=0.312`, min detectable `D=0.190` at n=51). Bound: **0.155% worst
  decile, 0.0488% overall**, `clob_v3_1` only. **That bounds EXPOSURE, not
  FLOW.** The earlier “long gaps are quiet” reading used a window-mean baseline
  against first-decile gaps and is withdrawn; phase-matched U9 does not support
  it.
- **U4 `CLEARED / REPLACED`** — the stale-book `+0.45 c/share` maker markout was
  ~2.6x too high. The model-free terminal identity gives in-window per-fill
  `+0.165 c` and share-weighted `+0.173 c`; after excluding the single-actor
  0.02 class, per-fill is `-0.211 c` while share weighting stays `+0.172 c`.
  **CORRECTED 2026-08-21 (U10/U10b) — THE SIGN IS UNDETERMINED, NOT NEGATIVE.**
  Window-clustered bootstrap, 931 windows, 10,000 resamples:

  | figure | estimate | 95% CI | |
  |---|---:|---|---|
  | per-fill, all flow | +0.165 | [-0.377, +0.734] | spans 0 |
  | per-fill, ex-0.02 | **-0.211** | **[-0.849, +0.457]** | **spans 0** |
  | 0.02 class alone | +1.987 | [+1.529, +2.440] | **excludes 0** |
  | share-wtd pooled | +0.173 | [-0.251, +0.596] | spans 0 |

  All seven per-coin CIs span zero on both weightings; the permutation test on
  coin labels is p=0.0482 but **names no coin**, so the surviving set is empty.
  **"On real flow, makers lose per fill" is NOT SUPPORTED and is withdrawn** --
  it was published in commit `6a0e593` and relayed as a finding. The only
  interval in the whole analysis that excludes zero is the **+1.987 against the
  single-actor class**, tight precisely because it is one counterparty behaving
  consistently, and carrying **~$91 of capacity over two days**: the sole
  statistically distinguishable maker edge here is the un-harvestable one.
  **What survives is the ESTIMATOR finding, which needs no interval:** the two
  weightings diverge in sign on the same fills, so a single-weighting spec would
  have reported "+0.165, makers profitable" and never revealed the dependence on
  one counterparty. Keep that apart from the economic claim -- conflating them is
  how `+0.45` survived two sessions.
  Window clustering misses day-level common factors, so these intervals
  **understate** uncertainty, and nothing on real flow excludes zero even so.
  **PROGRAMME STATE: cost settled bar rho; SIGN MEASURED AND UNDETERMINED;
  marginality and scale sit behind a sign that only more days resolve** -- so
  answering the queue question perfectly would still not say whether there is
  anything to harvest.
  Per-coin signs are mixed and two days permit no clustered CI.
- **U5/U7** — the rare maker fee is a thin per-address tier: 0/~10/50 bps.
  No in-transaction rebate appears in 600 receipts, but periodic/off-chain
  rebate remains `Unavailable`.
- **U6 `UNRESOLVED`** — cross-price chain-leg order is non-random but misses its
  frozen clearance bar: 49/63 = 0.778, Wilson95 [0.661, 0.863]. Same-price time
  priority, 59% of adjacent pairs, remains invisible; this cannot identify
  counterfactual `Q_ahead`.
- **U8 `CLEARED`** — spread is one tick on BTC/ETH but 3–7 ticks on the thinner
  coins. The pooled one-tick headline was BTC denominator dominance. Spread
  width is **not** an edge predictor: equal-width per-coin markout signs flip,
  consistent with wider spreads pricing adverse-selection risk.
  **WITHDRAWN (U10): that mechanism is NOT supported.** All seven per-coin CIs
  span zero, so scattered signs across spread widths is exactly what
  all-coins-near-zero plus sampling noise produces. Calling CIs-spanning-zero
  "signs flip" asserts structure the intervals deny. What survives is only the
  NEGATIVE result: **spread width does not predict edge.**

**Fee schedule — see `fee_structure_known`.** Taker pays `0.07·p(1−p)` $/share
(n=600, four decimals); maker pays zero on 744/754 legs. Crossing at ATM costs
~2.25 c/share (~225 bps), so the fee does not kill MM on cost — **it loads the
question onto adverse selection.**

**Two cross-cutting defects, both recorded with binding consequences:**

1. **The quote guard `0.0 < bid < ask < 1.0`** appeared **independently in both
   the coordinator's and the agent's code**, against the same tape, and excludes
   exactly the deep-tail quotes where the 0.001 tick lives. Caught once, only
   because 84 observed transitions contradicted a reported 0.00% share. Cost:
   124,772 quotes (5.2%), all from the tails. **Any quote filter must print its
   exclusion count beside its result.**
2. **`coin_msg_rate_hint` is a cumulative counter, not a rate**, despite the
   name (`collect_pm.py:489`); using it produced a spurious 3.26x. **Confirm any
   collector telemetry field against its definition in the collector source
   before use — the name is not the definition.** A real rate exists in the
   heartbeat `rate_msg_s`.

**Open:** U9 remains `UNRESOLVED` at seven phase-matched `PING_TIMEOUT` gaps;
five more in one collector era are required. The checked-in uncertainty ledger
and reproducible U1–U9 probes are at commits `6a0e593` and `6e125dc`.

### Queue and type tests — 2026-08-21. C1 CLOSES A LEAD STRUCTURALLY.

Protocol `QUEUE_AND_TYPE_PROTOCOL.md` (frozen before measurement), probe
`queue_and_type.py` (34 checks), results `QUEUE_AND_TYPE_RESULTS.md`.

**C1 — cancellations and the fill bracket: `UNIDENTIFIABLE`.** The coordinator
proposed crediting cancellations as an independent source that could narrow the
bracket. **It is not, and the reason is structural rather than empirical.**
Cancellation *volume* is abundant — saturation p50 2.0-13.2, with **86-99% of
actions saturated** — so crediting it collapses the pessimistic bound onto the
optimistic one (btc and eth agree to three decimals, 0.946/0.946 and
0.848/0.848). What displayed L2 withholds is cancellation **position**. Credit
all and you get FRONT; credit none and you get BACK_DISPLAYED; the interior
needs an ASSUMPTION, not a bound.
**THE BRACKET WIDTH IS THE QUEUE-POSITION AMBIGUITY RESTATED, and cancellation
data cannot reduce it because the missing quantity is the same one.**
**CORRECTED 2026-08-21: that is right about INFERENCE and wrong about what it
implies.** Queue position is an OUTPUT OF THE PLACEMENT POLICY -- new-BBO puts
you at the front, joining an existing level puts you behind its depth -- so
FRONT/BACK is the span across POLICIES, not an epistemic bracket, and it
collapses once a policy is named. The strategy defines the measurement rather
than waiting on it. What is genuinely unobserved is narrower: whether a
new-BBO quote WINS THE RACE against others doing the same, which depends on
latency we do not observe -- so FRONT is an upper bound on that policy, not a
guarantee. Next step is a POLICY COMPARISON (fill AND fill-conditional markout
per placement rule), not a sweep over an assumed parameter.
Consequence is close to "fill is not determinable from data we can collect" —
but NOT for the expected reason: displayed depth ahead does not trade through,
it **churns**, and we cannot tell whose.

**Two defects in the coordinator's own rules, raised and upheld:**
- **R1 — the reconciliation gate was a TAUTOLOGY.** `cancelled` is definitionally
  the residual, so the identity balanced by construction, and a 1% threshold
  against gross churn running **60-1000x trade volume** could never fire. A gate
  that cannot fire is not a gate. Re-anchored to the independent
  `last_trade_price` stream, the residual against **traded volume** is
  **2.3-12.1%** (SOL one share in eight) — trade volume with no matching
  displayed decrease. Consistent with hidden liquidity or sequencing; not
  separable here and NOT narrated. Sixth instance of the denominator/population
  defect, inside a rule written to guard against exactly that class.
- **R2 — `MATERIAL` could not distinguish tightening from DEGENERATION.** The
  rule as written granted `MATERIAL` on a 97-100% width reduction; the agent
  **declined the win it was entitled to** and reported `UNIDENTIFIABLE`, because
  the bound had not tightened. Taking it would have published "cancellations
  narrow the bracket by 97%", which is false. A saturation guard is now required:
  the credited bound is a bound only where `cancelled_at_level < queue_ahead`,
  which holds in **1-14%** of actions.

**C2 — bivariate Hawkes on {MICRO_002, MARKET}: `RETAIN`. The coordinator's
motivating hypothesis is REFUTED.** The 2x2 branching diagonal dominates the
off-diagonal on every coin — `market<-market` 0.18-0.45 against cross terms
0.02-0.18 — so market self-excitation SURVIVES being modelled alongside the
micro actor, and the scalar 0.40-0.55 was not cross-excitation wearing a
self-excitation label. The Hawkes layer stays. A1 is not contradicted: cross
terms are non-zero, just smaller. Separately the micro actor is strongly
**self**-exciting (0.18-0.35 on five coins).
**Corrects a published number:** the scalar figure OVERSTATES market
self-excitation for most coins — only eth reaches 0.45, four sit at 0.18-0.35.
**Scope:** intervals are grid-quantised and conditioned on the selected
half-life, so they show fit STABILITY, not sampling uncertainty (btc's
degenerate [0.180, 0.180] proves it). `RETAIN` means not-deletable-on-this-
evidence, NOT a validated branching estimate.

**C2b — the instrument floor, and it blocks the obvious next step.**
Websocket-frame batching is **REFUTED**: 17.6% of btc market-market pairs fall
under 5 ms and 12.0% under 1 ms, but **not one shares a frame and not one has a
zero gap**, on any coin. However the sub-millisecond gaps pile up at **0-50 us
with a 26 us median**, which is **16.2x** the Poisson expectation. `recv_ns` is
stamped at PARSE time, so several messages arriving in one TCP segment are
stamped microseconds apart by processing cadence — distinct frames, distinct
timestamps, **no market information in the spacing**. The test rules out
batching at the websocket-message level and NOT at the transport level, and
26 us is more consistent with the latter.
**So neither branch is established for btc**: not a frame artefact, but its
grid-floor selection cannot be read as clustering either.
**DO NOT EXTEND THE HAWKES GRID LOWER — it would make this worse**, letting the
fit chase into the region where timestamps carry processing cadence rather than
arrival time. Prerequisite: establish a **timestamp-resolution floor** (the
shortest interval at which `recv_ns` differences reflect venue timing) and
truncate the grid there. Until then btc branching stays **CENSORED**.

### Residuals — open, in priority order

1. **Accumulate OOS days.** Tier-1/Tier-2 infrastructure is complete and the
   timers own catch-up. Do not turn partial partitions or design days into a
   result; rerun frozen v1 only at its formal boundary and evaluate v2 only on
   primary days from 2026-08-22.
2. **Finish the flow candidate while days accrue.** The B0–B3, mark-census,
   exploratory-Hawkes and queue-bound development path already runs under
   `FLOW_MODEL_PROTOCOL_V4.yaml`. Freeze/implement the conditional M1–M4 law
   families next, without treating the two-hour receipt as forward evidence.
   Hawkes and action fills remain unvalidated until the completed forward
   time-change and ten-day gates say otherwise.
3. **`PING_TIMEOUT` classification.** Phase-matched U9 is unresolved at n=7;
   retain `MNAR-suspect` and wait for five more same-era gaps before amendment.
4. **Phase 0A 5 — S30/S60 internal sampling semantics.** This still gates Route
   B only; the route-A fit remains descriptive until 10 OOS days.
5. **Downstream model work waits on data.** Tier-2 deliberately leaves sigma
   forecasts, walk-forward isotonic calibration and inferential intervals
   unavailable. Attach them only when the frozen day/fold requirements exist.

### Collector: the 1013 is VENUE-SIDE — measured, not inferred

**Resolved 17:46:41 UTC.** `clob_v3_1` samples the `websockets` Assembler, which
pauses reading from the transport once its inbound backlog passes a
65,536-frame high-water mark — and a paused transport is exactly what fills a
server's send buffer. On the first 1013 under v3_1:

```
ws_ever_paused      False        <- never stopped reading
ws_queue_depth_max  133          <- 0.2% of the pause threshold
lag_ms_max_interval 1.8 ms
```

**We were draining at 0.2% of capacity while the venue said its send buffer was
full.** Every client-side cause is now excluded by measurement: loop stall
(1.8 ms), gzip (off-loop, none in flight), write backpressure (`writer_wait=0`,
`q_hi=1`), memory (RSS 260 MB stable), and network throughput (11.7 Mbps
sustained; one BTC socket is 0.24 MB/s).

**Two successive repairs failed because neither addressed the cause.** The v2
write-queue decoupling and the v3 gzip offload were both real defects worth
fixing — the gzip stall was 1.8–1.9 s of measured loop block — and neither was
the answer. I asserted the gzip finding as the root cause; that was an inference
written as a measurement, and it is corrected in `2d5503f`.

**What this changes.** The acceptance boundary as written — one full busy UTC
day with zero `SLOW_CONSUMER_1013` — tests something we do not control and is
probably unachievable. The pre-registered *alternative*, a cause-aware exclusion
rule with enough complete independent days, is now the operative path. That
branch existed before any of this was known, which is what makes the finding
actionable instead of a dead end.

**The exclusion rule already has its input**, and there are **two loss
mechanisms**, which is what makes the pre-registered *cause-aware* framing
load-bearing rather than stylistic:

| mechanism | cause codes | pattern | missingness |
|---|---|---|---|
| venue send-buffer / slow-consumer label | `SLOW_CONSUMER_1013` | 12 of 14, **all BTC**, bursty | **MNAR** — activity-correlated |
| venue server cycling | `CONNECTIONCLOSEDOK` (1001), plausibly `PING_TIMEOUT` / `NO_CLOSE_FRAME` | hits whichever sockets a restarting server held, across coins | plausibly **MAR** |

They need different handling. A 1001 going-away gap can be excluded and the rest
stays representative; a 1013 gap cannot, because it lands preferentially on the
busiest windows, so *excluding it is itself a selection* and the excluded set
must be reported next to the retained one — the lesson of the original MNAR
incident. A rule keyed only on seconds-lost would treat the two as
interchangeable.

Loss to date: ~40 s across 10 windows, worst `btc-updown-5m-1787247000` at 21.5 s
(5.5% of its 390 s). Tally by coin: **btc 12, sol 1, eth 1** — an earlier note
saying every disconnect was BTC was true when written and is now superseded.

Posture: stop fixing this client-side; keep `clob_capture_clean: false`; leave
`DISK_WORKERS` and `ping_timeout` alone — there is no client-side hypothesis
left to test.

### Historical — `clob_v3` deployed 16:31:26 UTC

**The v2 repair did not close the failure.** `clob_v2_1` logged a
`SLOW_CONSUMER_1013` on BTC **5.8 minutes after deployment** and finished its
80-minute run at `retries=5 slow=5`, all BTC, over 4.82M messages — **one drop
per ~16 minutes**. The v2 addendum called the repair successful at 15:12; the
ledger contradicted it at 15:16.

**Root cause, measured:** `gzip_atomic` ran **synchronously on the event loop** —
**1,818–1,915 ms** to compress a ~180 MB BTC shard at level 6, every five
minutes per coin, during which **no socket is drained** and the venue's send
buffer to us fills. The v2 write-queue repair was real but was never the binding
constraint: `writer_wait=0`, `q_hi=1` across 4.8M messages.

`clob_v3`: gzip off-loop on a dedicated disk pool; **disk and HTTP executors
split** (they shared the default 20-worker pool, where a stalled `urlopen` could
starve a shard write and reproduce the same 1013 by a second path); an
**event-loop lag probe** reported per heartbeat and **stamped into every
disconnect**; `gap_open_at_exit` so a gap running to window end is no longer
indistinguishable from a lost close record; `markets_force_cancelled` replacing
the misleading `active_markets_drained` (2 reported, 14 actually drained); and a
narrow chunk-loss window on cancellation closed.

**Why the lag probe matters more than the fix.** A 1013 has two candidate causes
with *opposite* remedies — the loop stalled (offload work) or the socket rate
genuinely exceeded capacity (shard connections across processes). Nothing
previously distinguished them, so every diagnosis was an argument. Now it is a
number in the disconnect record.

Selftest is 12 checks including a **control**: the same gzip inline must stall
the loop ≥100 ms and ≥20× the off-loop figure, or the off-loop test proves
nothing. Measured **211 ms on-loop vs 0.5 ms off-loop, 393×**.

**Acceptance is unchanged and not yet met:** one full busy day with zero `1013`,
or a cause-aware exclusion rule with enough complete independent days. Compare
against the v2_1 baseline of one drop per ~16 minutes. A few clean minutes prove
nothing — that was exactly v2's error. Never pool v2 and v3 rows without the
`collector_version` field the ledger records.

## Done this session

**Route-A sigma candidate measured — DESCRIPTIVE / PRICING HOLD.** The
preregistered `route_a_v1` run produced **9,332 admissible rows**, **5,796
strictly-forward OOS rows**, and all **42** independent fits (7 symbols x 6
horizons). Settlement direction agreed **1,560/1,560** after admissibility
filters. An independent post-run audit found zero timing, formula, uniqueness,
fold, coefficient or source-hash violations. Only 2026-08-20 is an OOS test
day, so every one of the 84 gates is `INSUFFICIENT_EVIDENCE`. The point
diagnostics are not reassuring—42/42 conditional-mean effects and 40/42
conditional-variance effects exceed their frozen tolerances—but one regime-day
cannot refute the law. Full result: `SIGMA_ROUTE_A_RESULTS_2026-08-20.md`.

**E-M6 settlement truth — PASSED, the foundation gate is cleared.**
`S60(T) vs S60(t0)` reproduces the winners Polymarket actually paid on
**1,465 windows at 99.8%** (99.9% restricted to `|margin| > 0.5 bp`). This
settles the open `w = 60 s` vs `300 s` ambiguity: the full-range reading scores
86.9% and is **refuted**. Reading the same grid at knowledge time gives 99.3%,
and that 0.5 pp gap is the size of the look-ahead a careless backtest banks.

**E0 data integrity audit** (`exp_e0_data_audit.py`, 7 checks) quantifies every
known incident rather than assuming it benign: the duplicate-collector overlap,
the ~16 min market-side outage, the 8 malformed resolution rows, restart shards,
TWAP gaps, knowledge-time lag per coin, and the up-rate drift confound.

**Architecture v12 + machine-readable contracts v22.** Six planes, a structural
diff checker, version-bound migration records, and executable DA/EV pipeline
contracts through the commit-last Tier-2 boundary. Twelve external review
iterations. **Two of my own artefacts were
proven unsound during that loop and replaced** — worth knowing because both
failures were of the same kind, a checker that reported success without
checking:
- the first contract checker: an invalid ref exited 0, generics were invisible,
  and narrowing a type produced an identical inventory. All three reproduced.
- the path-keyed allowlist (M11-1): entries authorised **any** change at that
  path. Reproduced `CouplingSource -> CompetitionState` passing. Replaced with
  migration records bound to (operation, key, old, new, version).

**Collector MNAR bug found; second repair deployed, acceptance pending.** The
hot loop
allocated an `asyncio` timer per message; at BTC's rate that dominated and
backed the server's send buffer up into `1013 slow consumer` disconnects. **27
of 47 disconnects were our
own doing, and 32 of 47 were BTC** — i.e. the loss was concentrated in exactly
the busiest intervals, which is missing-not-at-random. The initial short probe
read 0 post-fix drops, but extended observation of the repaired 10:55 process
found **13 further 1013 closes across 11/41 recent completed BTC windows**, plus
seven other BTC retries. `50dd889` replaces that path with a minimal receiver
and bounded ordered writer queue; `2deb8e8` adds active/rate/age telemetry.
The first versioned high-load run lasted **19m23s**; its last heartbeat reported
**908,843 messages** (**552,166 BTC**) with zero retries, slow closes or writer
waits. That is a successful smoke test, not the required full busy day. Never
pool an unpaired statistic across repair eras, and do not use this tape for
flow/fill/queue inference yet.

**Collector lane audit.** Discovery grids are complete, resolutions are current
(1,963 final, zero give-ups), TWAP parsing has zero malformed rows or negative
knowledge lags, and capacity is ample. The price socket nevertheless has
unreplayed global gaps: recent full-horizon admissibility is 224/273 (82.1%).
That is sufficient row flow for filtered Route-A accumulation, not evidence
that excluded regimes are ignorable. `prices_v2` now detects global and
per-topic silence at 8 s and persists exact topic gap boundaries; its first
real outage recovered both topics after about 11.5 s. It reduces future loss
but does not repair historical missingness. See
`DATA_COLLECTOR_AUDIT_2026-08-20.md`.

**SIGMA_PLAN Revision 5 reviewed and measured.** The route split and fit
specification remain frozen; the next action is more OOS days, not Revision 6.

## What was withdrawn — do not cite these

**1. "The book beats our model at every horizon" — WITHDRAWN, not held.**
The 2026-08-20 run showed the book winning by a stable 2.5–3.2 Brier points at
all six horizons, which read as a uniform information deficit and prompted the
conclusion *"no alpha, therefore pure market making"*. That model was
**mis-anchored**: `E_t[X_T]` used the trailing S60, which lags spot by
`w/2 ≈ 30 s`, while being paired with a *conditional* variance law. The
resulting `σ_eff` was ~2.6× too small at `r = 30`. The candidate
`P̂ = 2·S30 − S60` gained **−0.0101 Brier pooled, at every horizon** on one test
day, but Revision 3 correctly shows that coefficient is a biased trend
extrapolator under the Brownian fixture. It establishes that the lagging-S60
direction was wrong, not that `alpha=2` is the final anchor. The residual verdict must be re-read
on the corrected specification before it means anything.

**2. The FLB edge — downgraded from an edge to a rounding error.**
"+3.6 c/share at `p ∈ [0.15, 0.35)`, stable" was the measurement that
recommended the Option-B re-scope. It was computed on `book` snapshots, which
are **p90 6.2 s stale**. Rebuilt from `price_change.best_bid/ask` (the
executable quotes): `b̂ = 1.145` where the stale read inflated it to 1.182, the
walk-forward gain is **0.0004 Brier**, and the effect is **one-sided** — a drift
signature rather than a genuine bias.

**3. Everything book-derived in `PM_DEEP_REVIEW.md` inherits defect 2**,
including the "+95 bps maker gross / +136 with rebate". Treat as unverified
until re-measured on `price_change` quotes.

**4. Five premises I had briefed into the flow/fills work were wrong**, all
corrected in `plans/BE_FLOWANDFILLS_PLAN.md`: trades are NOT double-reported
(zero dupes); the modal spread is **1 tick**, not 2–4 (ATM runs 6–8 c); the
1.7 s lag is the **signal clock only** — the book itself arrives in **47 ms**,
which materially softens the session-1 FATAL-1 latency finding; `side` is the
**taker's** (90.8%); and `price_change` carries **post**-change quotes.

## ⛔ Decision point A/B is still open, but B lost its prior

Session 1 closed on Option A (fix the mechanism program) vs Option B (re-scope
to FLB harvest), with the reviewer *and the data* recommending B. **The data
that recommended B was measured on stale books** (withdrawal 2 above). Re-frame
the choice before making it; do not read `PM_MM_PLAN §17`'s recommendation as
current.

## Immediate next step — accrue OOS days and rerun route_a_v1 unchanged

No manual normalization step remains. The supervised collectors and hourly
measurement/evaluation timers accumulate and materialize each eligible closed
day. Preserve `route_a_v1` and rerun it unchanged as immutable days accrue. A
formal verdict needs at least **10 OOS test-day clusters** and 30 OOS rows in
every frozen conditioning cell; because the first day is training-only, that
normally means at least 11 collected days. Do not respond to one-day point
effects by changing the cells, tolerances or functional form.

Keep `route_a_v2` separate: it is per symbol/horizon, begins primary evaluation
on 2026-08-22, and requires all 126 frozen gates. Rows through 2026-08-21 may
train a fold but may not contribute to its headline score or interval.

Phase 0A step 5 may proceed in parallel and gates Route B only. Do not add
structural `k/v/Omega` terms to the Route-A residual. Probability-level use and
the estimator integration remain on HOLD until all per-fit gates pass.

In parallel, continue the mechanism/flow work from the reproducible uncertainty
ledger. Keep cause/version gap facts, per-coin primary reporting and both
per-fill/notional weightings. The 1013 is measured as venue-side; the operative
path is the pre-registered cause-aware exclusion, not another client tuning
cycle. No two-day point estimate is an inferential profitability result.

### Historical — iteration-4 boundary review and application

Revision 4 makes the right central decision: **Route A's reduced-form residual
is the whole pricing variance; Route B is a structural diagnostic; never add
them.** Iteration 4 retains that decision and narrows the HOLD to six integration
problems:

1. Route A is documented as independent of internal S30/S60 kernel semantics,
   but `ReducedFormLaw` and `pricing_var` still require its convention to be
   `VERIFIED`. Remove that Route-B gate from Route A.
2. `pricing_var` and `conditional_mean` bypass `check_request`; the helper also
   accepts future-issued laws and reversed target intervals. Build one atomic
   request-to-distribution API with unavoidable temporal/link checks.
3. Negative residual variance and NaN evidence can price, while infinite rates
   throw. Validate every domain and return a typed refusal.
4. High p-values are treated as proof of conditional mean/homoskedasticity.
   Pre-register per-symbol/per-horizon equivalence/calibration gates with effect
   sizes and confidence bounds.
5. Route B's `anchor_var` changes with empirical alpha because it absorbs
   squared model gap, and it exposes `model_total_var`. Separate bias/MSE and
   make the diagnostic a type that cannot satisfy a pricing protocol.
6. `PathLaw` is not a discriminated route union; YAML schedules lack offsets;
   kernel coefficients lose their seconds unit; `CalibrationCurve` is stale;
   and the source of operational dynamics shape is unresolved.

Shipped checks passed at the time of that review (kernel **41** pre-repair,
checker 13, v14→v15 migration), but focused
adversarial probes reproduce every issue. Full evidence and acceptance tests are
in `SIGMA_PLAN_REVIEW_ITER4.md`.

### ITER4 APPLIED — Revision 5 / contracts v16 / boundary rewritten

Every ITER4 probe was **reproduced before acting**: `resid_var = -1` priced; NaN
cluster counts and NaN p-values passed (every ordered comparison against NaN is
false); a **future-issued law with a reversed target interval** returned `True`;
`anchor_var` moved with the empirical alpha; `model_total_var` was exposed;
pricing took no request; an infinite rate raised `OverflowError`. All correct.

**M4-1 — the one that mattered, and it was a self-contradiction.** The plan said
route A regresses published streams and needs no internal kernel; the code
refused route A unless the convention read `VERIFIED`. That gate removed the very
advantage that selected route A. Route A's precondition is now
**`StreamProvenance`** — stream identities, point-in-time reads, units, alignment
at *published* timestamps. `SamplingConvention` gates the structural arm and
nothing else, and there is a test that a well-formed route-A law prices **while
every convention is still UNVERIFIED**.

**M4-2 — one atomic query.** `pricing_distribution(law, request, observables)` is
the only pricing entry. It validates every temporal and identity invariant
*before* computing either moment and returns mean and variance from **one**
validated fit. v15 exposed `pricing_var` and `conditional_mean` separately and
neither called the checker, so correctness rested on every future caller
remembering a pre-call. Now also refused: a law issued after the request, a
reversed target interval, and observables newer than the knowledge cutoff. Every
boundary refusal carries `since` and a machine-actionable `cause`.

**M4-3/M4-4 — the gate was numerically and statistically unsafe.** Positive,
finite, integer validation throughout; NaN and ∞ refuse instead of passing or
raising. And **failure to reject is not equivalence**: `GateEvidence` carries a
verdict (`PASS` / `INSUFFICIENT_EVIDENCE` / `MODEL_REFUTED`), an effect size and
a tolerance, and `PASS` requires the |effect| confidence bound *inside* the
tolerance. A p-value gate treats "not enough data" as "verified", which is
exactly backwards at a ten-cluster minimum.

**M4-5 — bias had crept back into the variance, on route B.**
`cond_var_at_model` now takes **no alpha**, so an empirical anchor can no longer
change the alleged conditional variance. Route B returns a **distinct type** with
no pricing protocol; v15's "no total is reachable" tested two key *names*.

**M4-6 + §1a.** `PathLaw` is a real discriminated union; `WeightAtOffset` gives
schedules their support; `KernelCoefficient` carries SECONDS; `c(r)` is the
route-agreement ratio. And **"diagnostic" means "not a probability input", not
"not operational"**: §1a maps every consumer to its source, and the four *shape*
consumers read **route A's own horizon profile**, so route B feeds no control.

**Verify:** `python3 live/pm_research/sigma_kernels.py --selftest` (45 checks) ·
`contract_check.py --selftest` · `contract_check.py HEAD WORKTREE`.

### Historical — the pre-measurement directive (now executed)

This directive produced `route_a_v1` and is retained as provenance. Four review
rounds had each found a real defect, and the pattern is worth
naming: **each error was the previous error one level of abstraction higher.**
v1 used a lagging anchor; v2 replaced it with a trend extrapolator and buried the
bias in the variance; v3 named the bias but kept two incompatible estimators; v4
chose one route in prose and contradicted it in code. Every one was caught by an
adversarial probe rather than by a test I had written. Nothing further is a
specification task.

1. **Phase 0A 6 — FIT THE ROUTE-A LAW.** Regress observed `x_T` on observed
   `(S30, S60)` per horizon and per symbol, cross-fitted, day-blocked, embargoed;
   emit `GateEvidence` with an effect size and tolerance, not a bare p-value. Do
   **not** estimate `Ω` on this route — it is inside the residual.
2. **Phase 0A 5 — S30/S60 semantics** against the 1 s Binance tape. Gates route B
   only, and may run in parallel; it must not block a valid route-A fit.

Estimator implementation and any probability-level use remain on **HOLD** until
the repeated step-1 run has enough OOS days and produces a law whose gates all
read `PASS`.

### Historical — Revision 3 review and ITER3 application

Revision 3 gets the central mathematics right: under the declared Brownian
fixture, `alpha*=2700/1801`, the conditional anchor variance is `8.2590 sigma²`,
and the `2/-1` extrapolator's `9.5139 sigma²` is unconditional MSE containing a
known squared-bias term. The false ordered bracket, hidden nugget and several v13
carrier defects are also repaired. Keep those changes.

The iter-3 review found six remaining integration blockers:

1. Direct regression on `(S30,S60)` estimates a reduced-form mean and total
   residual, while the plan separately adds structural `k`, `v` and `Omega`.
   Choose one route; combining them double-counts forecast error.
2. The fixture compares every supplied empirical `alpha` with the Brownian
   `alpha_star`, labels it biased, and pulls the corrected mean back to the
   Brownian fixture. It cannot express the empirical anchor the plan requires.
3. The plan/contract type `Omega` in bps², but the code treats it as a multiple
   of `sigma²`. With `sigma²=4`, an identity covariance contributes 9.9867 bps²
   instead of 2.4967. Non-PSD inputs can produce negative total variance.
4. The default convention is `UNVERIFIED`, yet `settlement_var` returns a number
   and discards the status. The “weight schedule” cannot represent arbitrary
   temporal weights/support, and negative rates are accepted.
5. v14 carries request timestamps but does not enforce request/law instrument,
   target, horizon, knowledge or link equality. Checker green is structural.
6. Canonical guidance still conflicts on whether semantics gates the anchor,
   whether alpha is estimated or assumed, and whether regression residuals are
   the whole law. Tracking also retained the overclaim that `alpha=2` “fixes” it.

Shipped tests pass (kernel 24, checker 13, v13→v14 migration clean), but the
adversarial cases above fail semantically. Full evidence and acceptance tests are
in `SIGMA_PLAN_REVIEW_ITER3.md`.

### ITER3 APPLIED — Revision 4 / contracts v15 / fixture rewritten

Every ITER3 probe was **reproduced before acting**: pricing under `UNVERIFIED`,
`bias_coeff 0.200833` against an empirical `α`, the 4× `Ω` unit error
(9.9867 vs 2.4967), negative rate → −4.691, non-PSD → −120.905, the one-slot
`Unavailable`, the `KeyError` on an unknown convention, and the
`1799/1200`-vs-`2700/1801` contradiction inside my own file header. All correct.

**M3-1 — THE ROUTE DECISION, and it scopes everything else.**

|  | **Route A — reduced form** | **Route B — structural** |
|---|---|---|
| object | fitted law of `x_T` on `(S30, S60)` | `σ²k_law + σ²v(r) + uᵀΩu` |
| needs sampling semantics? | **no** | **yes** |
| identifies `Ω`? | **no** — it is *inside* the residual | **yes** — the lag-0 nugget |
| delivers | a **pricing** law | the **decomposition** |
| status | **PRICES** | **DIAGNOSTIC ONLY** |

**Route A prices; Route B diagnoses; they are never summed** (`R-ROUTE`,
`PathLaw.estimand_route`). The consumer matrix decides: the only LEVEL consumer
is the BE-Belief fallback, which needs `Σ(r)`, not its parts. `c(r)` is
redefined as the *agreement* between routes, `Σ̂_A/model_total_B ≈ 1` — a
model-adequacy diagnostic, not a term in either.

**And OLS is not a free lunch.** It gives the best *linear projection* and a
*pooled* residual — the conditional mean only if that mean is linear, the
conditional variance only under homoskedasticity. Otherwise it is an
unconditional forecast MSE, which is the same category error we removed from the
Brownian variance line one revision earlier, one level up. So route A ships with
**gates**: cross-fitting, ≥10 day clusters, a residual conditional-mean test and
a heteroskedasticity test. `pricing_var()` refuses if any fails.

**`Ω`'s identification has an answer** (§9-2a): contemporaneous moments give **3
numbers for 4 unknowns**, so it is *not* identified from them. Under route B it
is the lag-0 discontinuity of the bivariate cross-variogram — the **nugget**,
already in the per-symbol table — which needs a VERIFIED convention and is
entangled with `ŵ = 47 s`. **`Ω`, the nugget and `ŵ` are one problem, not three.**

Also applied: **M3-2** `AnchorSpec.selected` (MODEL|ESTIMATED), horizon-indexed,
bias measured against the *selected* estimand so a fitted `α` is unbiased with
respect to itself, `model_gap` kept as a diagnostic, `conditional_mean`
implemented. **M3-3** `Ω` is bps² once, PSD-validated, `RateQuantity` separates
bps²/s from terminal bps². **M3-4** fail-closed on status, rates, PSD, unknown
conventions; `Unavailable{reason, since, cause}`; conventions are `(offset,
weight)` schedules. **M3-5** `check_request` **evaluates** the comparisons with 8
negative fixtures — a typed timestamp that is never compared is documentation.
**M3-6** the scan now covers plan, code, contracts, STATUS and HANDOFF.

**Verify:** `python3 live/pm_research/sigma_kernels.py --selftest` (41 checks) ·
`contract_check.py --selftest` · `contract_check.py HEAD WORKTREE`.

**Revision 4's proposed next steps (superseded by the iteration-4 boundary
review above):**
1. **Phase 0A 5 — verify S30/S60 semantics** against the 1 s Binance tape. Gates
   route B entirely; does **not** gate route A.
2. **Phase 0A 6 — fit the route-A law**: regress `x_T` on `(S30, S60)` per
   horizon and symbol, cross-fitted and day-blocked, and report both residual
   diagnostics. Do **not** estimate `Ω` on this route.

Estimator implementation remains on **HOLD**.

## Then

- **G-FF4 queue bracket** (`plans/BE_FLOWANDFILLS_PLAN.md`) — potentially
  **programme-ending**. If the bracket on `Q_ahead` is wide enough, MM
  profitability is not knowable from data we can collect. Effort saved by D3
  goes here.
- Structure review loop is at 12 iterations and **not converged**
  (LOCAL 11 / SPREADING 1 / STRUCTURAL 1). Iteration 12 produced *zero*
  documentation change — it was a checker patch labelled as an architecture
  version. Semantic rules/producers/ports, scenario type consistency,
  incentive-to-outcome wiring, decision-snapshot equality and ModuleManifest
  coverage all remain unresolved.
- `pm_backfill.py` — historical windows. Gamma canNOT resolve these; enumerate
  via paginated CLOB `GET /markets`. Note calibration may **never** cross the
  2026-08-07 rule change (snapshot → 60 s TWAP, after the Stanford/SMU
  manipulation study).

## Standing rules (each one paid for)

- **Check the data before using it, every time — no lane is trusted by default.**
  Before any analysis reads a lane, verify *for the exact rows it will consume*:
  coverage and the gap ledger over each window, predictor staleness at each
  decision time, collector version / repair-era boundaries, and whether the lane
  is cleared for that class of inference (`clob_capture_clean`). Report the
  excluded set beside the retained one, and characterise it on the statistic the
  model actually estimates. Paid for four times: the FLB edge measured on p90
  6.2 s stale books; the v2 "repair successful" the collector's own ledger
  contradicted four minutes later; the prices lane called "clean" on
  `open_gaps=[]` while logging 58 gaps in 11 h; and the exclusion-MNAR reading
  that reversed sign once a variance statistic replaced a displacement one.
- No design decision that a measurement on existing data could settle may be
  recorded as settled until that measurement is run.
- Read book state from `price_change.best_bid/ask`, **never** `book` snapshots.
- Read everything at knowledge time (`recv_ns`), never payload timestamps.
- Dedup prices by `(timestamp, symbol)`, raw by message identity — `recv_ns`
  differs per process so exact-line dedup does **not** catch a duplicate
  collector. Check with `ps -eo pid,etimes,cmd | grep live/pm_research`;
  pgrep patterns must include the `pm_research/` path segment.
- **Read fees from the CHAIN, never from `fee_rate_bps`.** The websocket field
  is `"0"` on all 446,412 trade events and that is an **artefact**, not a zero
  fee. There is no three-way conflict any more: the docs' 7 % is correct, and
  `taker fee = 0.07·p·(1−p)` $/share is confirmed to four decimals on 600
  sampled transactions across the whole moneyness range. **Incidence: the taker
  pays and the maker does not** — 600/600 taker legs carry a fee, 744/754 maker
  legs carry zero. The `0.07·min(p,1−p)` reading (= 3.5 ¢/share at p=0.5) is
  **REFUTED**; it is 2× too large. Still unknown: the maker **rebate** (a zero
  fee is not evidence of a rebate, and `OrderFilled.fee` is unsigned so a rebate
  could not appear there), and the 10/754 maker legs that do carry a fee.

## Watch out for

- `resolutions.jsonl` holds 8 garbage rows from the first hour — filter through
  `is_final`, dedupe by slug keeping the final row.
- Thin windows exist (a BTC window with `volumeNum = 2`). Stratify by volume;
  never average across dead and live windows.
- The tick regime changes away from the money (0.01 → 0.001), which makes the
  tick a first-order economic parameter far from ATM.
- `PM_THEORY_CHECK_ORCHESTRATOR.md` §2's claim that "MM-under-obligation theory
  doesn't exist" was a **search failure** — the body is principal–agent MM
  contracts (El Euch et al.; Baldacci et al.). Marked superseded; don't cite it.
- Trading access (US status, CLOB auth) is a deployment question, out of scope
  until the gates pass. Don't let it leak into research design.

## Coordinator tick — 2026-08-23 ~22:50 UTC (R-70..R-76)

**Register re-counted with a real instrument.** `live/pm_research/register_count.py`
(8 selftest checks). My previous grep was wrong in BOTH directions: it did not
know DISCHARGED/SUPERSEDED/UPHELD, and it read the whole row so a resolution
word in a row's *prose* closed it. Honest count was **53 open ASKs / 39
resolved**, not the ~44 I had been reporting. Six closed this tick → **48 open**.
OPS and DE registers are now **clear**; DA 25, BE 23.

**Rulings.** R-70 register instrument + ASK/FILING taxonomy, closed resolution
vocabulary, status-cell-only reads. R-71 Q-DE-8 ADOPT (coupling independent,
B6's limits ride with it — 21.2% real overlap, decorative intervals). R-72
enum is `MINT|MERGE`, rides a batched v23→v24 record; `de_actionspace.py`
selftest must stay matching neither side. R-73 `verify_landing_evidence.py`
adopted as OPS state source (20/20, not the 15/15 OPS reported). R-74 BE's
declination of DE's 20th migration record UPHELD — *a migration record
authorises a change, it does not validate one*; §9 UNION amendment ratified,
non-additive stays 19. R-75 Q-BE-18 annotate §8a, no edit, no verdict moves.
R-76 dispatch channel: long multi-line `send-keys` pastes without submitting.

**Two of my own instruments failed during the tick that was auditing
instruments.** (i) `grep -m1 'Revision [0-9]+'` on `OP_PLANE_PLAN.md` returned
"Revision 1" by matching *prose about* Revision 1 — the exact mechanism behind
four consecutive stale OPS state blocks, reproduced live, which is what decided
R-73. (ii) `register_count.py` found a false positive in **itself** on first
real use (`Q-BE-26` resolved itself on the word "declined" in its own body,
because its author wrote no status cell). Both are now fixed and both carry a
selftest case.

**In flight:** DA — MEASUREMENT_PLAN iteration 4 vs frozen Rev 7 + triage 25
register rows. BE — execute the §8a annotation (R-75) + triage 23 rows, marking
FILINGs for ACK. OPS — 08-22 Tier-2 exponent measurement (tier1 08-22 now
present; tier2 still 08-20/08-21 only). DE — draft the v23→v24 `MINT|MERGE`
change record.

**Still the user's call:** `STOP-MM-VIABLE`.

## Coordinator tick — 2026-08-23 ~23:20 UTC (R-77..R-80)

**Register:** 95 filed / 50 resolved / **44 open ASKs** (DA 20, BE 25; OPS and
DE clear). DA triaged 5 down; BE's count rose 23→25 because BE files as it
triages, which is what an honest triage looks like. BE is marking every row
`ASK:`/`FILING:` so the FILINGs can be ACKed in one pass.

**R-77 — lens coverage is now a precondition of termination.** MEASUREMENT_PLAN
ran to 5 iterations, MUST-FIX 25/4/2/7/5. The rise at iteration 4 is *not* the
R-60 expanding-set pathology: the set is eight and frozen, iterations 4-5 both
record "No new lens", and the count rose because **currency** and **cross-plane
consistency** fired for the first time. But it exposed a hole in R-61 — both
close routes look only at recent iterations, never at coverage, so a loop could
close by re-running exhausted lenses while a fresh one never fires. No loop now
closes while any lens in its frozen set has never run. **7 of 8 have run;
decision-readiness has not**, and that is the lens whose output the coordinator
consumes — invisible from my side by construction. DA runs it before close.

**R-78 — OPS refused my 08-22 premise and was right.** `twap/day=2026-08-22` is
the *08-21* run's forward dependency (`measurement_batch.py:469-473`), not a
readiness signal; every other tier1 stream for 08-22 is absent and
`newest_eligible_day = 2026-08-21`. `NEXT_DAY_CLOSED` opens 08-22 at **2026-08-24
00:00 UTC**. Ratified: presence of an artifact is not evidence of the state it
is named for. OPS holds — sampler detached and gated, prediction on file
(143 s CPU · 11.2 GB · FITS), cap not raised, collectors untouched.

**R-79 — the false-positive class is named: VOCABULARY-IN-DISCUSSION.** *Any
document that discusses a rule contains that rule's own vocabulary, so searching
for the vocabulary finds the discussion, not the state.* Four instances in four
days, three of them mine (Revision-1 prose, "terminal markers", OPS's
`Status.*FROZEN`, `register_count.py` on "declined"). State comes from a
producer's criterion, a structured field, or an instrument with a selftest.

**R-80 — I verified R-75 against the wrong artifact.** BE's §8a annotation is on
disk at `EX1_PREDICTION_PROTOCOL.md` §8a.2 and was before I asked; I grepped
`BE_BELIEF_PLAN.md` because §1.2 lives there. R-36 amended: verify the artifact
the *claim* names, never one inferred from the subject matter.

**In flight:** DA — decision-readiness lens + triage. BE — ASK/FILING marking
pass + triage. DE — **B3, `EV-Replay` plan + harness** (v24 draft accepted, M-1
held as the only §1 entry). OPS — gated to 00:00 UTC, reports 08-22 after.

**Still the user's call:** `STOP-MM-VIABLE`.

## Coordinator tick — 2026-08-24 ~00:10 UTC (R-81..R-84)

**Register: 98 filed / 61 resolved / 37 open ASKs / 0 FILINGs** (DA 23, BE 14).
BE's marking pass let **11 FILINGs ACK in one ruling** (R-81) — 50→61 resolved
with zero adjudication. Open ASKs 48 → 37 across two ticks.

**R-82 — decision-readiness fired and R-77 was right in the worst way.** DA ran
the eighth lens: *"Seven of eight obligations never left the document."* A 12.5%
surfacing rate, on the most-reviewed plan in the programme, by the most rigorous
plane. Invisible from the coordinator side by construction — every instrument I
own measures what reached me. Corroborated independently: register 95→98 filed,
`Q-DA-39/40/41` present as rows. **R-77 extended: decision-readiness runs in the
first two iterations of every loop AND before close** — requiring it only before
close leaves obligations trapped for the loop's whole life, which is what
happened here across five iterations. BE is now running it over its own surface.

**R-84 — §8 has been reporting GROSS behind a discharged blocker, and it reaches
`STOP`.** Verified at `FLOW_MODEL_STATE.md:59`: taker pays, maker does not —
**744/754 maker legs zero**, n=600. One correction to DA in the conservative
direction: 744 of 754, *not* 754 — ten legs were charged (1.3%), so the exception
must ride with every citation. **DA carries §8 gross → net, and the STOP dossier
does not go to the user until settled either way.** Not a claim it flips the
verdict; markout is dominated by adverse selection, not fees.

**R-83 — a stale header misled my accounting twice.** `EV_REPLAY_PLAN` line 3
says "Revision 3", line 9 says "Revision 7"; R-67 read the stale line and this
tick's "draft the PLAN first" premise descended from the same misread. DE's
artifact defect *and* my R-79 (4th instance) — I read a revision from prose
instead of a field, while the programme now holds two revision-pinning
instruments (`da_freeze_pin.py`, 8 checks). DE's census correction accepted and
it sharpens rather than softens: **five replay dialects is now eight**, three
added this session. DE names the pattern adopted-or-debt in Revision 2.

**Next tick's first business:** `Q-DA-39` (A-CALIB-1 owed, never asked, on a
committed panel), `Q-DA-40` (v24 assembling without §5's three "Incompatible
artifacts" inference rules), `Q-DA-41` (R-7 condition 4 — trigger nobody filed,
no detector; 23 of 36 above the 0.5 pp reference, mean 0.645 pp), plus **R-7 is
half-landed in v23 exactly as R-3 is** → two rule bodies in v24.

**OPS:** 08-22 opened at 00:00 UTC; measurement timer fires **00:20:16**,
sampler alive (pid 508268, 10h), prediction on file. Result expected ~00:30.

**Still the user's call:** `STOP-MM-VIABLE` — now gated on R-84.

## Coordinator tick — 2026-08-24 ~00:40 UTC (R-85..R-88)

**TWO RECUSALS OUTSTANDING — the coordinator does not decide either.**

- **R-87** — DA challenges whether **R-7's licence survives** a bite distribution
  centred 1.3× above its reference. R-7 is the coordinator's ruling → **routed to
  OPS** for `SURVIVES`/`DIES`. Coordinator ruled only the mechanical half:
  `r7_drift_check` is extended to police the bite-vs-reference comparison
  regardless of the answer — *a condition with no detector is a sentence*.
- **R-88** — BE found a **plane-order inversion live in v23**: `GateEvidence.gate
  : GateId`, `GateRegistry` produced by EV-Gates, `GateEvidence` produced by
  BE-Uncertainty → **BE-Uncertainty must emit a GateId that does not exist until
  EV-Gates registers it. BE reads EV**, inverting the one direction the
  architecture forbids. **R-74's §9 union is what put it there.** → **routed to
  DA**. If DA finds against R-74, R-74 gets amended.

**R-85 — `calib_panel` admissible as a STALE-QUOTE panel with `r` demoted to a
nominal label; INADMISSIBLE as an r-indexed freshness ladder.** `quote_status ==
AVAILABLE` on **36,288/36,288** is zero-variance and therefore not evidence; p50
staleness **57.8 s at the r=2 s rung**; **627 s** max on a **300 s** window;
`r=2`/`r=10` share one quote event in **96.1 %** of windows, so the short rungs
are one measurement reported twice. `EXP-BLEND` ΔBrier not citable as it stands.
A-CALIB-1 is owed, bound **adopted from measurement (Class C), not chosen** —
that an honest bound refuses the panel is information about the panel.

**R-86 — strike §5's G-branch; keep `ci: Unavailable` pinned.** At **G=2** it is
the correct answer, not a limitation. Claim ladder filed as **debt with trigger
G ≥ 7 day-clusters**, named in `CONTRACTS_BATCH_v24` §3.

**R-82 now confirmed on TWO independent surfaces** (DA's and BE's, different
methods) — it is not a property of one document.

**DA's self-correction, made before the ruling and against its own filing:**
23-of-36 receipts → **9 of 14 coin-days, mean 0.645 pp, 2 day-clusters**. The
original pooled 8 pre-R-7 `leak_canary_v1` files with 14 coin-days × 2
content-addressed twins, and carried **two `INVALID_UNBOUND_GUARD` verdicts that
exist only in v1** — on the very coin-days R-7 reclassified. Cause named against
itself: `MEASUREMENT_PLAN` §4 lists `twap`/`coverage`/`windows` as
co-resident-generation datasets and **omits `canary`**. Now a MUST-FIX there.

**08-22:** sampler alive (pid 508268, 10.7 h), logged `activating` at 10.73 GB
against an 11.2 GB prediction; measurement ran 00:20:25, evaluation 00:25:36,
both success; tier2 still 08-20/08-21. OPS reporting the regime.

**Dispatch mechanism:** OPS and BE both had text queued unsubmitted (OPS's for
>1 h; two bare-Enter attempts failed). **Replaced the queued line instead of
retrying the submit** — all four planes went WORKING. R-76 amended in practice:
after two failed submits, replace, don't retry.

**Still the user's call:** `STOP-MM-VIABLE` — gated on R-84.

## Coordinator tick — 2026-08-24 ~01:10 UTC (R-89..R-93)

**BOTH RECUSED VERDICTS ARE IN. One killed a coordinator ruling; one upheld one.**

**R-89 — `R-7` IS DEAD.** OPS ruled `DIES`. Primary: R-7's Poisson fit came from
**14 coin-days over 2 clusters**, and DA's corrected population is **the same 14
coin-days** — R-7's own basis, read correctly instead of double-counted, does not
support R-7. Secondary: **Var 1.363 vs λ 1.857, ratio 0.734, under-dispersed** —
a Poisson-calibrated threshold is mis-set permissively. **OPS ruled against its
own interest and said so**: the amendment is what stops an unbound-guard coin-day
aborting the whole day, the direct fix for the 26-hour outage.
**Vacated, not amended — nothing may cite R-7 as authority.** A distributional
re-grant is unavailable at G=2 (same wall as R-86). The amendment runs
**PROVISIONAL/UNLICENSED** pending a **mechanism** re-founding, **routed to BE**
(DA proposed it, OPS killed it, the coordinator granted it). If BE cannot make it
stand, the amendment goes and the outage risk returns to the coordinator as an
open item — the correct place, since the licence was granted on a distribution
that could not carry the weight.

**R-90 — NO plane-order inversion. `R-74` stands, unamended.** DA adjudicated;
all three facts verified at the files, not on report: `contracts.yaml:1567-1571`
(`GateId` = {protocol,name,version}), `:1648` (`entries: dict[GateId, Gate]` — a
map **keyed by** GateId indexes identity, it cannot confer it), `GFF1_PROTOCOL.md:5`
(`Gate G-FF1` declared in a frozen protocol, BE-owned). BE emitting
`GateEvidence.gate` is BE reading a frozen protocol. Interface split **refused**.
Mint rule adopted: **protocols mint, `GateRegistry` indexes, no plane confers gate
identity at runtime** — additive, no record. *The recusal's value is identical to
what it would have been had DA gone the other way; R-74 was made on BE's argument
in one tick and happened to be right.*

**R-91 — the defect BE actually felt IS real.** `contracts.yaml:868-890`:
`GateEvidence` carries `decision_eligible: bool` produced by **BE-Uncertainty**, a
worker plane, while `R-ADMISS` reserves the selection decision to the coordinator.
**Ruled off the worker type** → non-additive → **v24 §1 beside M-1**, and the first
real use of R-74's `- !remove <element>` syntax. `admissible` stays *only* as the
evaluation of a coordinator-set rule — and **under R-85 `A-CALIB-1` does not exist,
so on calib rows `admissible` is currently a worker decision too. R-85 and R-91 are
one obligation.** Third fact-vs-decision defect this session: **a boolean is the
easiest place in a schema to hide a decision.**

**R-92 — 08-22 regime: `full-stall 0.0 s · high 0 · max 0 · oom_kill 0` →
UNTHROTTLED.** Receipt ABSENT (not committed), quotes 0 of 7, still on btc — in
progress, not failed. **10.73 GB observed against an 11.2 GB prediction filed
before the run.** First Tier-2 day with the throttling question settled in advance
rather than reconstructed after.

**R-93 — 4 more FILINGs ACKed** (Q-BE-40/41/42/43). Register: **106 filed / 65
resolved / 41 open ASKs / 0 FILINGs.**

**Still the user's call:** `STOP-MM-VIABLE` — gated on R-84.

## Coordinator tick — 2026-08-24 ~01:30 UTC (R-94..R-98)

**R-94 — the canary amendment is RE-FOUNDED and STANDS; the licence stays dead.**
BE argued from mechanism as asked, executed on the shipped rule with no data:

```
event_only  disagree  delta   PRE-R-7                POST-R-7
       566         0    0.0   INVALID_UNBOUND_GUARD  BOUND_ZERO_SCORE_DELTA
       566         5    0.0   BOUND_ZERO_SCORE_DELTA BOUND_ZERO_SCORE_DELTA
```

Zero disagreements with zero harm was fatal; five with the same zero harm was
fine — **the strictly safer observation punished more harshly**. Non-monotone in
the evidence: an *ordering* defect, no distribution, and **unbreakable by G=2**,
which is what clears R-89(b)'s bar on distributional re-grants. Licence dead,
amendment alive — they were separable and BE separated them. **`R7_PROVISIONAL`
stays up** until the stale artifacts clear; the flag is what makes them visible.

**R-95 — the coordinator built half the assignment on a conflation and BE refused
it.** "Why should one coin-day's unbound guard not condemn six others?" presumed
a forgiveness R-7 never granted (`event_only == 0 → INVALID`; an unwired guard is
not evidence). **Real defect: `INVALID_UNBOUND_GUARD` is returned by TWO arms** —
unwired guard *and* fail-closed counter inconsistency. **Ruled: split the status.**
Had BE argued as posed, it would have justified a property the amendment lacks and
the coordinator would have ratified it.

**R-96 — two artifacts run on the vacated licence, and two committed receipts
assert it.** `r7_drift_check()` polices a dead licence; the constants encode the
vacated basis. OPS's new `R7_PROVISIONAL` scan found the consequence already on
disk: **5 receipts rest on the amendment** (08-20/doge, 08-21/sol, 08-22/hype) and
**2 assert `drift_verdict: WITHIN_LICENCE`** — false statements in immutable
artifacts, corrected by **annotation beside (R-28), never edited**. 08-22/hype is
the first coin-day reclassified *after* vacatur — the flag caught its own first
live instance on the day it was built. **Class named (BE's second hit): a vacated
basis surviving in the code that implements it. Vacating a rule is not
self-executing** — every vacatur now carries a sweep for code, constants, receipt
fields.

**R-97 — RECUSED, routed to DE: R-87 and R-89 contradict each other and both are
mine.** R-87 ordered `r7_drift_check` extended to police condition 4; R-89 vacated
the licence condition 4 conditions. DE decides extend / narrow / retire. Weighing
note given to DE: an ordering property is checkable **statically**, so the honest
outcome may be a *different instrument* — R-87 ordering the wrong thing rather
than too much of it.

**08-22:** tier1 measurement **COMMITTED** (08-22/hype reclassified); tier1 full
and tier2 absent, quotes 1 of 7; regime **UNTHROTTLED**, full-stall 0.0 s.

**Register: 109 filed / 66 resolved / 43 open ASKs / 0 FILINGs.**

**Still the user's call:** `STOP-MM-VIABLE` — gated on R-84 (§8 gross→net).

## Coordinator tick — 2026-08-24 ~01:40 UTC (R-99..R-101)

**R-99 — DE's R-97 verdict adopted: `r7_drift_check` NARROWED to one arm, my
R-87 extension order VACATED AS MOOT.** DE decided from the code and explicitly
discounted the framing in the recusal note — which is the only reason the
confirmation counts. Verified at the file before adopting: `replay_canary.py:55`
`REFERENCE_DELTA_PP`, `:75` `"lambda": 1.857`, `:85` `R7_DRIFT_LAMBDA_TOLERANCE = 2.0`,
`:461` `lo = licensed / tolerance`.

| arm | polices | disposition |
|---|---|---|
| λ-tolerance vs licensed 1.857 | the fit R-89 killed | **RETIRE** |
| variance/mean Poisson-likeness | same dead fit | **RETIRE** |
| "no coin-day ever shows nonzero delta" | the amendment's **construction** | **KEEP** — runtime witness |

**Commissioned instead: a STATIC MONOTONICITY SELFTEST on `classify()`**, run with
every canary — *an ordering property cannot drift with data, but it can be
silently reintroduced by a code change.* Better than what the coordinator ordered.
OPS implements, DA records condition 4's retirement, R-96 annotations stand.

**Three coordinator rulings corrected by the planes this session — R-7's licence
(OPS), R-87's instrument (DE), R-95's conflated premise (BE). None by the
coordinator.**

**R-100 — 08-22 Tier-2 COMMITTED; memory scales ~LINEARLY.** Predicted (filed
before the run) 143 s CPU / 11.2 GB `FITS`; measured **139.0 s / 11.87 GB**, cap
untouched. Regime **UNTHROTTLED** — full-stall **0.3 s across 51 min**, high/max/
oom_kill 0; R-26 met, numbers usable as-is.

```
TIER-1 build   00:39:36 -> 01:27:58   CPU 2,895.4 s   peak  8.16 GB
TIER-2 proper  01:28:03 -> 01:30:19   CPU   139.0 s   peak 11.87 GB
```

**MEMORY 08-22→08-20 = 1.05** on two uncensored points — essentially linear. The
**≥1.52 previously reported was 08-21's peak pinning at the cap**: a censored
measurement read as a scaling law. Without the phase split the headline would have
been 3,882 s vs 08-20's 171.5 s and **manufactured an exponent of ~8**. OPS filed
a caveat against its own numbers unprompted: 08-22's Tier-2 ran warm in page cache
(same invocation as its Tier-1 build) while 08-20/08-21 were cold.

**`R7_PROVISIONAL` retargeted for R-94 the same hour its subject changed** —
carrying the amendment is no longer a finding; only `WITHIN_LICENCE` assertions
are (count 2 → 3 with 08-22).

**v24 §1 now carries M-1 and M-2** — M-2 (`- !remove decision_eligible`) verified
at `CONTRACTS_BATCH_v24.md:35` and `:57`, the first real use of the marker.

**Register: 112 filed / 67 resolved / 45 open ASKs (DA 28, BE 19) / 0 FILINGs.**

**Tier-2 days on disk: 08-20, 08-21, 08-22.**

**Still the user's call:** `STOP-MM-VIABLE` — gated on R-84 (§8 gross→net).

## DA — 2026-08-24 (R-94 tick)

**DONE.** All four queued items landed at **Revision 11** *before* this tick; the
banner said "iterations 1-5" and iterations 6-7 were unlogged, which is why the
coordinator opened two consecutive ticks on superseded state. **Now Revision 12,
banner derived from the logged iteration count.**

- **A-CALIB-1 WRITTEN** — `live/pm_research/config/a_calib_1.json`,
  `DRAFT_PENDING_FREEZE`. Bound **adopted, not chosen**: `max_quote_age = r`.
  **No free parameter.** Yield 22,318/36,288 = **61.5%**. Filed `Q-DA-47` for
  ratification — I cannot freeze my own config.
- **§4 canary omission fixed**; **§5 G-branch struck**, ladder debt at G ≥ 7;
  **§8 gross→net** landed Revision 10, all four copies.
- **R-94 recorded**: amendment survives on **ordering**, not distribution.
- **VACATUR SWEEP**: `classify()` docstring + status-site comment re-founded;
  `R7_LICENSE` marked `VACATED_R89_NOT_LIVE_PENDING_DE_ON_R87` and **left
  standing** (deleting it retires the drift check = decides DE's question).
  Executable AST identical but for the added constant; 27-cell truth table
  unchanged; selftests pass.
- **§1.2's "23 of 36" struck** → **13 of 21**, version-pinned, twin-deduped,
  population and as-of inline. Gate reads SATISFIED either way.

**BLOCKED.** `r7_drift_check` **HELD for DE** (`Q-DA-46`). R-87's bite-vs-reference
detector therefore **still does not exist** — §8 step 4's gate is satisfied by a
human comparison, not one the code makes.

**WATCH OUT FOR.** (1) A corrected statistic left standing at its own site — cost
three iterations here, in prose, the same class R-94 named for code. (2) **A
correction can age**: "9 of 14" was right when written and stale when a day
landed. Any count over a growing corpus needs population + as-of inline. (3) DA
register ids start at **3**; `Q-DA-1/2` never existed (no rows, no git history) —
not a loss.

**NEXT.** Iteration 8 proper vs frozen Revision 12, pinned at dispatch.
Decision-readiness runs in the first two iterations per R-77/R-82.

## SCOPE NARROWED — 2026-08-24 ~02:00 UTC (R-102, user-authorised)

**Read this before anything below it. The programme is on a four-item decision
path; everything else is debt.**

| # | item | owner | state |
|---|---|---|---|
| 1 | **B3 `EV-Replay` HARNESS** | DE | plan drafted (259 lines); **harness NOT built** — the only structural piece left |
| 2 | **Policy comparison** — new-BBO vs join-BBO, fill **and fill-conditional markout** | DE | not started; **not data-gated** (§7) |
| 3 | **§8 gross→net** (R-84) | DA | in flight — the only thing gating the STOP dossier |
| 4 | **`STOP-MM-VIABLE` to the user** | coordinator | blocked on (3) |

**Register rule, in force:** a row is an **`ASK`** only if it **blocks one of the
four**, and it must name which. Everything else is **`DEBT` with a named trigger,
closing on filing** — no ruling, no queue. R-86's claim ladder at `G ≥ 7` is the
template. Each plane triages its own rows once (DA 28, BE 19); more than three
genuine blockers from any plane is itself a finding.

**Why:** open ASKs went **37 → 41 → 43 → 45 → 48** in 2.5 hours; the coordinator
closes 5–6 a tick against ~10 filed. **101 rulings and 118 rows** now support a
programme whose central empirical finding is one markout table on **one UTC day**.
This is a coordinator design defect, not plane behaviour — treating every filed row
as needing adjudication rewards filing, and R-82 amplified it in the right
direction (planes were told to *find* unsurfaced obligations; they did, and every
finding landed in a queue with no exit).

**UNCHANGED — narrowing the queue is not relaxing the standard:** recusal (3
coordinator rulings overturned this session — R-7/OPS, R-87/DE, R-95/BE);
verification at the artifact the **claim names** (R-36, R-80); false-positive
analysis per check (R-79); frozen lens sets and pins; decision-readiness early
(R-77, R-82); instruments ship with falsifiers (R-59).

**Timeline honestly:** the policy comparison is answerable in **days** and is not
data-gated. The maker-edge sign needs **25–30× current tape — over a month** that
no amount of work shortens. Landing the harness buys the one answer available.

## DA — 2026-08-24 (R-102 triage)

**QUEUE DRAINED: 27 open → 2 BLOCKING ASKs, 0 untriaged.** Conformance holds,
no orphans. Everything else is DEBT with a named trigger, closed on filing.

**§8 IS NOT OUTSTANDING.** It landed at Revision 10 and is verified at
`MEASUREMENT_PLAN.md:776`. `Q-DA-42` had no closing status, so the register —
which is what the coordinator reads — never said so. **Now stamped DISCHARGED,
and NO STOP INPUT MOVED**: net is a lower bound and the 10-of-754 hedged-leg
exception is unfavourable, so it cannot flatter the gate.

**THE TWO BLOCKERS, both gating STOP going to the user:**
- **`Q-DA-14`** — STOP's metric says *"after fee ... under a STATED cancellation
  policy"*; neither is pinned. Absorbs `Q-DA-24` (same `h`+fee+cancellation
  triple). Seat conflict (BE-6: user / SP §4: coordinator) must resolve first.
- **`Q-DA-43`** — STOP's metric is measured *against book mid*, so `A-BOOK-1`
  is upstream of the user's number and was never frozen. Two bound links filed
  as debt, needing no separate ruling: `Q-DA-36` (v23 ratified with **no
  spec-resolver**, so `spec_hash` has no producer) and `Q-DA-38` (declared
  fields never reach the evaluator). Freezing without them is cosmetic.

**WATCH OUT FOR.** A row's status lives in-body by convention and nothing
enforces it — five DA rows carried empty status cells and
`da_escalation_conformance.py` passed them. **That is fail-open #14 in my own
instrument**, and it is how a discharged obligation reads as open for two ticks.

**NEXT.** Iteration 8 vs frozen Revision 12, pinned at dispatch.

## DA — 2026-08-24 (R-105) — **Q-DA-42 REINSTATED AS BLOCKING; I WAS WRONG**

**I stamped Q-DA-42 DISCHARGED yesterday and it was not.** The gross→net *edit*
landed at Revision 10; the row was never about the edit. `MEASUREMENT_PLAN.md:955`
says it plainly: carrying §8 to net makes **fee treatment** load-bearing on
STOP's metric, fee treatment is one of Q-DA-14's three unpinned inputs, R-35
reserved it to **the user**, and *"the STOP dossier does not go to the user until
that is settled either way."*

**How the error was shaped:** I discharged it on the grounds that net is a lower
bound and the 10-of-754 hedged-leg exception is unfavourable, so the number
cannot flatter the gate. True — and an answer to a different question. Whether a
number is conservative is not whether an input moved. **A check whose text reads
correctly while what it evaluates is different — the recurring class, mine this
time, and permissive in direction.** The coordinator held the correct position
for four consecutive ticks while I corrected them three times.

**BLOCKING now 3: `Q-DA-14`, `Q-DA-42`, `Q-DA-43`** — all gate STOP → user, and
Q-DA-42 merges into Q-DA-14 (it is why fee treatment is *live* not latent).

**LEG-NAMING CARRIED THROUGH.** `FLOW_MODEL_STATE.md:60` paired a half-spread and
a fee as ONE side while winning over every other document by its own precedence
rule — so the hazard outranked the caution. Row retitled **TAKER LEG ONLY**, with
**DO NOT SUBTRACT THIS FROM A MAKER NET** and a note beneath the table: a maker
net that subtracts 2.25 ¢ understates maker economics by the whole crossing cost,
the largest term in the model. `edge_vs_cost` is prose, not code, so nothing has
been computed wrong yet.

**R-105 ADOPTED:** every cited population carries `n` and as-of; applied at the
annotation (n=600 transactions, as-of 2026-08-23).

## Coordinator tick — 2026-08-24 ~02:10 UTC (R-104..R-108)

**DECISION-PATH ITEM 1 IS DONE.** `ev_replay.py` — 25,949 bytes, **selftest OK,
23 checks** — built within ~15 min of dispatch. **DE now holds the critical path
alone: item 2, the policy comparison.**

**R-102's effect, measured:** open ASKs **48 → 28**, resolved **70 → 93**. BE
triaged 28 rows to **1 blocker / 27 debt**; DE holds **zero** open ASKs. The queue
that grew every tick for three hours turned over in one — the growth was never the
filing rate, it was the absence of an exit.

**Also verified at the files:** OPS's R-99 commission —
`_classify_monotonicity_selftest()` at `replay_canary.py:694` (3 named checks) and
`R7_DRIFT_LAMBDA_TOLERANCE` now **0 occurrences**, presence-of-new *and*
absence-of-old each checked on its own terms. BE's `check_decision_as_fact.py`,
11,815 bytes, 20 checks, with the `admissible`-without-A-CALIB-1 falsifier.

**R-107 — Q-BE-4 UPHELD; the `STOP` dossier ships the HORIZON PROFILE, not a
verdict.** `edge_layer1_v1.json` carries `verdict: HORIZON_DEPENDENT`,
`horizons_s: [5, 15, 30, 60]` as its own top-level field — `FIRE_SIDE` at h=5,
`INSUFFICIENT` at 15/30/60, and nothing pinned the horizon. **The coordinator will
not pin it: whoever selects the rung selects the verdict**, which is a decision
disguised as configuration — the 4th fact-vs-decision defect this session and the
first inside the coordinator's own gate. The dependence ships as a **finding**,
because §1e shows `h=60` **discards 1,611 btc fills, all inside the terminal
minute** (p50 r 166 s → 190 s) and **cannot see the final minute by construction**.
"The effect fades" and "we stopped looking where the effect lives" are different
conclusions. **Item 4 unblocked on this ruling.**

**R-105 — two coordinator rulings cite stale figures; DA caught it.** 9-of-14
coin-days → **13 of 21** (canary now spans 3 days, 44 files vs 36); R-86's "two
day-clusters" → three. **No conclusion moves** (G≥7 trigger unaffected; a vacated
licence isn't revived by later data; R-94's foundation is population-independent).
New class: *correct when recorded, stale when new data landed* — neither R-79 nor
R-80. **Remedy adopted programme-wide: every cited population carries its `n` and
its `as-of`.**

**R-108 — BE pushed back on the coordinator's "101 rulings / one markout table"
framing and is partly right.** Conceded: a material share of those rulings is why
the table is trustworthy. What survives: the ratio was bad and the queue had no
exit — R-104 settled that. Adopted from BE's own admissions: **docstring content
does not become a register row.**

**Decision path:** ① harness **DONE** · ② policy comparison **IN FLIGHT (DE)**,
BE pre-reviewing the design before it runs · ③ §8 gross→net **IN FLIGHT (DA)** ·
④ `STOP` to user — **unblocked by R-107, waiting on ③**.

## ★ THE POLICY COMPARISON IS RUN — 2026-08-24 ~02:40 UTC (R-109)

**Decision-path item 2 is answered. `policy_comparison_v2.json`, protocol
`POLICY_COMPARISON_PROTOCOL.md` FROZEN 2026-08-22 *before* the run, 5 days
(08-20…08-24), 30 windows/coin/day, btc+eth, h=5 s, headline = paired FRONT−JOIN.**

### The answer: neither policy pays.

`m5_swm_cents` is **NEGATIVE on all ten coin-days, both arms**:

| coin | arm | per-day ¢/share |
|---|---|---|
| btc | JOIN | −0.526, −0.991, −1.048, −1.512, −1.514 |
| btc | FRONT | −0.516, −0.650, −1.055, −1.128, −0.990 |
| eth | JOIN | −0.966, −1.412, −1.913, −2.862, −1.338 |
| eth | FRONT | −0.878, −0.716, −0.912, −2.120, −0.932 |

**The policy lever narrows the loss and does not cross zero.**

### The difference: real on eth, absent on btc

Receipt note: *"window-clustered bootstrap; day-clustered refused below the
cluster floor (house rule)."* Recomputed from the receipt's own per-day cells,
t(4) on G=5 day-means:

| coin | per-day Δ (¢) | published (window-clustered) | **day-clustered** | days neg |
|---|---|---|---|---|
| btc | −0.060, +0.222, −0.041, +0.201, +0.406 | [+0.026, +0.251] excl. 0 | **[−0.098, +0.389] SPANS 0** | **2 of 5** |
| eth | +0.092, +0.583, +0.956, +0.398, +0.401 | [+0.282, +0.718] | **[+0.094, +0.879] excl. 0** | 0 of 5 |

**btc's advantage does not survive the correct cluster unit — its sign flips and
the pooled interval excluded zero only by averaging over that flip.** eth's
survives, barely (lower bound +0.094 at G=5).

**Standard ruled:** when the correct cluster unit is unavailable, **report the
point estimate with NO interval**. An interval on the wrong unit is a precision
claim the design cannot support. **G=5 is not G=2** — R-86's floor was written at
two clusters; the per-day means were already in the receipt.

### §7's prediction is REFUTED

§7 expected a trade-off — new-BBO wins fills, *plausibly loses markout* (quotes
when information is freshest). Measured: FRONT wins fills **5–6×** (btc 7,500 vs
1,400 shares/window) **and does not lose markout on either coin**. A pre-registered
prior was tested and found wrong.

### Caveats attached, not buried

**2026-08-24 is a PARTIAL day** — 21 windows vs 30, first hours UTC only — and it
is btc's most positive day. A partial day is a different population, not a smaller
one. **BE's pre-review was commissioned but the run executed at 02:07**; the review
still runs, since the receipt is re-analysable.

**Routed:** DE to confirm/dispute the arithmetic; BE adversarially, specifically on
whether **share-weighting** in `m5_swm` does work a per-fill statistic would not —
FRONT wins fills 5–6×, so a fill-weighted statistic flatters it by construction.

**Decision path:** ① harness DONE · ② **policy comparison ANSWERED** · ③ §8
gross→net IN FLIGHT (DA) — the last input · ④ `STOP` dossier to the user, shipping
the horizon profile per R-107.

## DA — 2026-08-24 (R-109) — §8 NET RESTATED. **THE CONCERN INVERTS.**

**Filed as `Q-DA-48`, its own row, not folded into a revision.**

**A net restatement cannot make these numbers look better: on the maker leg a
fee can only SUBTRACT.** All 20 arm-coin-day values are negative gross
(`policy_comparison_v2.json`, n=141 paired windows per coin, as-of 2026-08-24);
net is *more* negative.

- fee term at stated incidence 10/754 → **0.0232 c/share**
- fee term at the absolute bound (every leg at max `0.07·p(1−p)`) → **1.75 c/share**
- **across that whole range all 20 values stay negative** → **the STOP verdict is
  INVARIANT to the user's unpinned fee parameter**
- only sign-flipping term = **unmeasured maker rebate**: **>52 bps** flips the
  least-negative coin-day, **>286 bps** flips all twenty

**`Q-DA-42` STAYS BLOCKING.** The verdict is invariant; the NUMBERS are not, and
R-107 ships a horizon profile of numbers. Materiality at the claimed precision is
the user's call — I made the mistake of releasing this myself yesterday and am
not repeating it. The invariance **narrows the blocking question to one item: is
there a maker rebate above 52 bps?**

**PROVENANCE DEFECT (does not change the answer).** `744 of 754 maker legs zero`
cites *"n=600 transactions"* = the G-FF1 study, whose receipt carries per-leg
**side attribution only** and **no fee amounts**; scanning every artifact in
`data/pm_5min/derived/` returns **zero maker-fee fields**. The G-FF1 sample is
also **stratified** (9 per cell against strata of 595–99,172), so 1.3 % is a
within-sample rate, not a population rate. **The conclusion rests on the fee
term's SIGN, which is arithmetic — not on its magnitude.**

**R-105 APPLIED:** 08-24 is PARTIAL at 21/30 windows → 141 paired windows per
coin, not 150. "5 days" is not five equal days.

## DA — 2026-08-24 (R-110) — SURFACE FREEZE ACKNOWLEDGED. **§8 DONE. ONE CLAIM WITHDRAWN.**

**No new modules, plans, loops or checkers.** This tick added two register rows
and edits to existing documents — nothing built.

**§8 IS FINISHED** (`Q-DA-48`, filed 2026-08-24). Leg named at
`FLOW_MODEL_STATE.md:60` (**TAKER LEG ONLY** / **DO NOT SUBTRACT FROM A MAKER
NET**). The number: fee term **0.0232 c/share** at incidence 10/754,
**1.75 c/share** at the absolute bound. **On the maker leg a fee can only
SUBTRACT, so net ≤ gross and no fee treatment moves any estimate TOWARD zero.**

**WITHDRAWN THE SAME DAY, BY ME:** *"the STOP verdict is invariant to the fee
parameter."* `m5_swm_cents` **carries no interval at any level** — only the
paired difference does, and the frozen protocol header says **"levels are context
only."** STOP's verdict is defined on intervals excluding zero, so context-only
point estimates cannot establish verdict invariance in either direction. **Third
time in three days I have written a claim whose text reads correctly while what
it evaluates is different.** The 52/286 bps rebate figures are downgraded to
**INDICATIVE, NOT INFERENTIAL** for the same reason. Struck in both places
(R-28 annotate-beside), landing-verified, single occurrence, inside the
strikethrough.

**`Q-DA-49` FILED BLOCKING ON DE'S OPTIMIZER** — three sampling traps, stated
before the optimizer exists: (i) earliest-first truncation is **selection**, not
just mis-reporting (`N ≤ 60` never leaves 08-20); (ii) 08-24 is partial **and
chronologically last**, so earliest-first drops it first; (iii) **not
conditional** — `m5_swm_cents` has no interval and its own protocol calls levels
context-only, so an optimizer maximising a level optimises a quantity with
nothing to separate signal from noise.

**QUEUE:** blocking = `Q-DA-14`, `Q-DA-42`, `Q-DA-43` (STOP → user), `Q-DA-49`
(optimizer). Everything else debt with triggers. `Q-DA-47`'s trigger extended:
A-CALIB-1 is on neither the user's path nor the optimizer's (`edge_layer1` shows
no calib reference on the replay path) and becomes blocking if either changes.

## DA - 2026-08-24 (R-110) - SURFACE FREEZE ACKNOWLEDGED. **§8 DONE. ONE CLAIM WITHDRAWN.**

**No new modules, plans, loops or checkers.** This tick added two register rows
and edits to existing documents - nothing built.

**§8 IS FINISHED** (`Q-DA-48`, filed 2026-08-24). Leg named at
`FLOW_MODEL_STATE.md:60` (**TAKER LEG ONLY** / **DO NOT SUBTRACT FROM A MAKER
NET**). The number: fee term **0.0232 c/share** at incidence 10/754,
**1.75 c/share** at the absolute bound. **On the maker leg a fee can only
SUBTRACT, so net <= gross and no fee treatment moves any estimate TOWARD zero.**

**WITHDRAWN THE SAME DAY, BY ME:** *"the STOP verdict is invariant to the fee
parameter."* `m5_swm_cents` **carries no interval at any level** - only the
paired difference does, and the frozen protocol header says **"levels are context
only."** STOP's verdict is defined on intervals excluding zero, so context-only
point estimates cannot establish verdict invariance in either direction. **Third
time in three days I have written a claim whose text reads correctly while what
it evaluates is different.** The 52/286 bps rebate figures are downgraded to
**INDICATIVE, NOT INFERENTIAL** for the same reason. Struck in both places
(R-28 annotate-beside), landing-verified, single occurrence, inside the
strikethrough.

**`Q-DA-49` FILED BLOCKING ON DE'S OPTIMIZER** - three sampling traps, stated
before the optimizer exists: (i) earliest-first truncation is **selection**, not
just mis-reporting (`N <= 60` never leaves 08-20); (ii) 08-24 is partial **and
chronologically last**, so earliest-first drops it first; (iii) **not
conditional** - `m5_swm_cents` has no interval and its own protocol calls levels
context-only, so an optimizer maximising a level optimises a quantity with
nothing to separate signal from noise.

**QUEUE:** blocking = `Q-DA-14`, `Q-DA-42`, `Q-DA-43` (STOP -> user), `Q-DA-49`
(optimizer). Everything else debt with triggers. `Q-DA-47`'s trigger extended:
A-CALIB-1 is on neither the user's path nor the optimizer's (`edge_layer1` shows
no calib reference on the replay path) and becomes blocking if either changes.

## Coordinator tick — 2026-08-24 ~03:10 UTC (R-111..R-113)

**§8 IS DISCHARGED — decision-path item 3 complete.** Maker leg measured at zero
with a 1.3% exception; taker leg `0.07·p(1−p)`; both on-chain. **A fee can only
SUBTRACT on the maker leg, so net ≤ gross** — no fee treatment can move any
estimate *toward* zero, and the measured levels are a **lower bound on the loss**.
All 20 arm-coin-day values negative gross (n=141 paired windows/coin; **08-24
PARTIAL at 21/30** — "5 days" is not five equal days; as-of 2026-08-24).

**DA withdrew its own strongest claim the same day it wrote it.** *"The STOP
verdict is INVARIANT to the user's unpinned fee parameter"* — struck, because
`m5_swm_cents` carries no interval (only the paired difference does; the frozen
protocol says levels are context-only), and STOP's verdict is defined on intervals
excluding zero. R-109's standard applied against DA's own conclusion, unprompted.

### ★ The one term that could still flip the sign

**THE UNMEASURED MAKER REBATE.** Every other term is measured and points the same
way. §2: *"no per-trade in-transaction rebate found; that is **not** absence of a
rebate"* — every `ρ`-dependent estimand is `Unavailable`. **If market-making here
pays, it pays out of a rebate nobody has found yet.** DA is now searching for it;
a clearly-described negative is as valuable as a positive.

**R-111 — the coordinator's terminal-abstention reasoning was BACKWARDS, and the
axis was already tested.** POLICY_BOUNDS Lever T ran body-only (`r_cut=60`) for
JOIN: `GATE_FAILS` both coins, body ≈ base — and **R-50's inversion: the only
positive bins sat IN the terminal minute**. Abstaining there removes the
*profitable* region. DE kept the axis on a better licence: `abstention × FRONT` is
genuinely unmeasured because FRONT's fill mass is **formation-time**. *Fifth
coordinator correction this session (R-7 OPS, R-87 DE, R-95 BE, R-105 DA, R-111
DE).*

**`POLICY_OPTIMIZER_PROTOCOL` accepted; three choices adopted as standards:**
grid **FROZEN at ~20 cells, no cell addable once a number exists**; every axis
**cites its existing receipt** so the search cannot resell a closed finding
(depth-1 excluded on `DEPTH_FAILS`; cancellation's REACTIVE family **CLOSED, 8/8
coin-days**); and a **wiring must-fail** — `r_cut=300` must produce zero fills.
DE complied with an order it believes wrong (the cancellation axis) by
**pre-registering the expected null** and stating a non-null *"would challenge the
closure, not quietly override it."*

**BE's reconciliation landed**, including the framing guard now in force: *"FRONT
beats JOIN is incomplete without **and both lose at h=5**."*

**In flight:** DE — build the **simulated actuator** (§5, replay-side; **not** the
venue writer) and run **Stage A's 12 cells**, controls first. BE — adversarial
check on whether **share-weighting** in `m5_swm` does work a per-fill statistic
would not (Stage A varies size 5→10, which multiplies shares directly). DA —
**the maker-rebate search**.

**Surface freeze (R-110) in force. `STOP-MM-VIABLE` not put to the user: optimise
before concluding.**

## DA - 2026-08-24 (R-112) - THE MAKER REBATE: **FOUND, MEASURED, AND TOO SMALL**

**No modules built.** On-chain analysis ran ad hoc in scratchpad; deliverables
are one register row (`Q-DA-50`) and annotations to existing documents.

**SEARCHED, IN ORDER:** (1) all 901 on-chain receipts, **every one of the 12
event types enumerated and identified** - none is a credit/refund/rebate; the
fee recipient `0x115f48dc...` appears as `to` 901 times and **never as `from`**;
`OrderFilled.fee` is uint256 with observed min 0, so a negative fee is not
representable. (2) The venue's own documentation. (3) The Polygon RPC off-trade
test - **could not be run: 403/401 from all three configured public endpoints.
NOT-REACHABLE, not NOT-AFFORDABLE.**

**RESULT: A MAKER REBATE PROGRAM EXISTS** - which is why the in-trade search was
always going to be empty, exactly as U11 predicted. Crypto rebate share **20%**,
the lowest of any category; `fee_equivalent = C x 0.07 x p x (1-p)`; paid
**daily in pUSD**, min **$1** accrued; self-normalising to ~20% of the fee your
own fill generated.

**IT CANNOT FLIP THE SIGN, AND THIS IS A CEILING NOT AN ESTIMATE:**
`p(1-p) <= 0.25` by arithmetic, so the rebate **cannot exceed 0.35 c/share**.
Measured pro-rata on our corpus: **0.168 c/share**. Threshold to flip even the
least-negative coin-day: **0.5164 c/share**. **The maximum possible rebate falls
32% short.**

**BONUS: reproduced the `744 of 754` claim whose receipt I could not find**
(Q-DA-48's provenance defect). Decoding OrderFilled word 4 and splitting legs by
counterparty: **901 taker legs 100% charged; 1,056 maker legs, 10 charged,
1,046 at exactly zero.** The **10** matches U5 exactly - the denominator does
not. **10/1,056 = 0.95%, not 1.33%.**

**LIMITS, because the answer is negative.** (i) The thresholds rest on
context-only levels with no interval, so this is **INDICATIVE, NOT
INFERENTIAL** - the rebate is below the SCALE of the losses, which is not a
verdict. (ii) I found 0 of 218 addresses clearing the $1/day minimum **and then
withdrew it** - 901 receipts are a SAMPLE, so it bounds observed volume, not
their actual volume. (iii) Rebates pay in pUSD, not USDC.

**THE GAP I AM NOT CLOSING.** The venue runs **TWO** programmes. I measured
Maker Rebates. The separate **Liquidity Rewards Program** pays for orders
**resting near the midpoint with no fill required** - not a share of taker fees,
so **the p(1-p) ceiling does not bound it**. It is now the only remaining term
that could move the sign, and unlike the rebate I cannot bound it by arithmetic.
**ASK filed: is measuring it authorised under the surface freeze?**

## DA - 2026-08-24 (R-116) - LIQUIDITY REWARDS: **REAL, LARGE, AND NOT CLOSEABLE BY ME**

Filed `Q-DA-51` **BLOCKING on DE's optimizer**. Unlike the maker rebate this one
is **not bounded by arithmetic** - it pays the QUOTE, not the TRADE.

**WHERE I LOOKED:** venue docs; `rewards_registry.jsonl` (confirmed a **size
heartbeat** - 552 records, keys `{recv_ns,n}` only); `tier1/quotes` (**top-of-book
only**, score not computable there); **`raw/` - FULL BOOK DEPTH, ~50 levels/side,
25 GB**, which is what made this measurable.

**POOLS: $1M across AUGUST**, $550k to 5-minute markets, BTC $300k. **I counted
the markets rather than assuming: 288 btc 5-min markets/day** -> **$33.60 per btc
window**, $5.60 per eth window.

**RESULT 1 - IT SPLITS BY ARM.** 100% of the pool would cover the loss on **9 of
10 JOIN coin-days and 0 of 10 FRONT coin-days**. JOIN fills ~1,400 sh/window vs
FRONT's ~7,500: a fixed pool is a large fraction of a small loss.

**RESULT 2 - AT THE TESTED CONFIG IT DOES NOT CLOSE THE GAP.** The replay rests
`quote_size_shares = 5.0`. Against real book depth (698-1,382 shares within 3c of
mid) our score share is **median 0.69%** -> **$0.23/window vs a $7.36-$17.94 JOIN
loss = 3.1% coverage**.

**RESULT 3 - THE ONE THAT MATTERS.** Reward is **strongly concave in RESTING
size** while loss is roughly linear in FILLS. Score share by resting size (v=3c):
5 sh -> 0.69%; 50 -> 6.5%; 500 -> 40.9%; 1,400 -> 66.0%. Robust across
v=2/3/5c; `b` cancels in the ratio.

**THESE ARE NOT P&L FIGURES.** The loss is fixed at the 5-share config's measured
loss; resting more WOULD fill more. **The fill-vs-resting-size response is the
missing term and I have not measured it.**

**WHY IT IS DE'S:** the reward/loss ratio is a function of the resting-size
policy - the optimizer's own free parameter. **An optimizer maximising markout
alone will systematically under-quote, because markout prices the cost of being
filled and never the revenue of resting.**

**LIMITS:** `max_spread` not published per-market (2/3/5c sensitivity used, not a
known value); scored **one-sided** while the real rule takes `min(Q_one,Q_two)`
and needs two-sided quoting - not modelled; pool assumed uniform over 31 days;
**book sample is 48 snapshots from 8 markets on ONE day**; and at 40-66% of the
reward zone **other makers would react - the book is not static under our own
size**, which nothing here models.

## DA - 2026-08-24 (R-125) - REWARDS STOPPED; **FORWARD POPULATION IS BROKEN THREE WAYS**

**Rewards out of scope.** `Q-DA-51` -> DEBT, trigger "the user reopens the
rewards question". **Stopping point is written down in `Q-DA-52`**, so a
successor restarts from measurement: fills scale **sub-linearly** (elasticity
**0.50-0.83**, never above 1); the reward/loss ratio improves with size by
construction but **converges to 31-48%, not 100%**; and **even 100% pool capture
leaves -$19.97 to -$48.09 per window**. That last one is an arithmetic ceiling
like the rebate's. Also recorded there: my own Q-DA-51 Result 1 was **not
commensurable** - it set a reward at one resting size against a loss at another.

**`Q-DA-53` FILED BLOCKING ON BE'S FORWARD EVALUATION.** All counts as-of
2026-08-24.

- **(i) The admissible holdout is 2.2 hours.** Freeze 2026-08-24T07:30:44Z; PM
  tape ends 09:40Z, mm_hf 09:48Z. **btc n=26, eth n=26 windows, 07:35-09:40Z,
  08-24 only** - out of 1,384/coin across 6 days.
- **(ii) It is a single day-cluster, on the partial day.** Day-clustered
  inference is not computable on n=1; `DAY_BLOCK_UNAVAILABLE` is the correct
  answer. 26 windows is **below the 30/coin/day** the policy comparison used.
- **(iii) THE SILENT ONE: there is NO Tier-1 or Tier-2 data for 08-24.**
  quotes/trades/coverage and all of tier2 **stop at day=2026-08-22, two days
  BEFORE the freeze**; twap stops at 08-23. A forward run against Tier-1 returns
  **zero admissible rows**; one against `raw/` **bypasses knowledge-time
  truncation, the distiller and the coverage receipts**. Quiet either way.
- **(iv) No coverage receipts for 08-24**, so the blind-period accounting
  (30/112 btc hours, 15/112 eth) **cannot be computed for the forward span**.

**NOT broken, and verified rather than assumed:** the earliest-first truncation
defect is **fixed in `ev_replay`** - provenance carries `days_sampled` distinct
from `days_read` with `sampled_is_known: true`. **That fix is in that harness;
whatever BE uses must be checked separately.**

**NEXT:** answer on (a) Tier-1 vs raw, (b) whether a 2.2-hour single-cluster
partial-day holdout is accepted and under what inference. If not, the remedy is
**collecting more forward tape - a wait, not a computation.**

## DA - 2026-08-24 (R-126) - REGISTER MARKING PASS: **51/51, 5 ASK / 46 FILING**

Was 0/51 marked, so the counter read all 51 as open ASKs. Now every row carries
`**ASK:**` or `**FILING:**` in BE's format. `conforms: true`, `register_rows: 51`,
no orphans, no malformed keys.

**THE FIVE ASKs - all BLOCKING, each naming its gate:**
`Q-DA-14`, `Q-DA-42`, `Q-DA-43` (STOP -> user) - `Q-DA-49` (DE's optimizer) -
`Q-DA-53` (BE's forward evaluation).

**`Q-DA-48` marked FILING, and the demotion is recorded IN the row** rather than
done quietly: its materiality question is the same one `Q-DA-42` blocks on, so
tracking it separately was double-counting my own row.

**FOUND WHILE MARKING: THREE OF MY ROWS WERE STRUCTURALLY MALFORMED** and did not
match `_REG_ROW` at all - `Q-DA-24` and `Q-DA-41` had no status cell, and
**`Q-DA-42` had lost its closing pipe in my own R-105 edit**. That row is a
BLOCKING ASK, so **my single most important open row was unparseable by the
counter**. Repaired; all 51 now parse. This was my contribution to the miscount
BE audited, and it was invisible because the row still LOOKED right.

**Forward-population guard (R-125) stands unchanged:** freeze 07:30:44Z, partial
days never counted as clusters, chronological truncation, blind-period coverage,
n and as-of on every count. `Q-DA-53` carries the open findings.

## DA - 2026-08-24T10:37Z (R-128) - **CONTAMINATION FOUND BEFORE FORWARD EVAL STARTED**

**Marking pass CONFIRMED COMPLETE:** DA 51/51 Form-A marked, **5 ASK / 46
FILING**, `register_rows: 51`, `conforms: true`, no orphans, no malformed keys.
DA is the only plane fully marked and fully parseable.

**`Q-DA-54` (FILING): the register uses TWO marker conventions** - `**ASK:**`
(63 rows: DA 51, BE 12, OPS 1) and `**ASK: <text>**` (29 rows: BE 19, DE 10);
57 rows carry neither. A counter keyed to one form miscounts the other by 29.
**I do not give a whole-register ASK total** - my own ad-hoc parser gave two
different answers on the same file, so I report only DA's slice.

**`Q-DA-55` (BLOCKING): a positional selector cannot express a mid-day freeze.**
`select_by_day` correctly fixed CROSS-day truncation (R-9), but **earliest-first
survives WITHIN each day**. BE's freeze is mid-day (07:30:44Z), which makes that
load-bearing. btc 08-24 has 127 windows; positions 1-91 are pre-freeze,
**92-127 (n=36) are the admissible tape**. At the shipped `per_coin=30` the
sample ends **02:25Z, 5.1h before the freeze, ZERO admissible windows** - and
**the day still counts as `holdout_complete`, not partial**. `per_coin=90` still
misses it by 5 minutes; you need **>= 92**. Raising the number only shrinks the
contamination ratio - **the forward population must be selected by a TIME
PREDICATE, not by rank.** Not edited: DE owns the optimizer, BE is user-held.

**THE HOLDOUT IS GROWING WHILE WE COUNT IT.** Collection is LIVE (12 btc files
in the hour to 10:36:30Z). **Admissible went 26 -> 36 during this session**,
~12 windows/hour/coin. `Q-DA-53`'s "2.2h / n=26" is **superseded to ~3.0h / n=36
as-of 2026-08-24T10:37Z**. Every forward count must carry its as-of or it is
wrong within the hour - and the "wait, not a computation" remedy is already
working on its own.

**STANDING WATCH continues:** post-freeze days into training sets, partial days
as clusters, selector truncation, blind-period coverage, n + as-of on every count.

## DA - 2026-08-24T12:57Z (R-129) - **VERIFICATION PRE-REGISTERED; DE HAS NOT LANDED**

**DE has not landed** as-of 12:56Z (`warning_window.py` 04:38Z,
`policy_optimizer.py` 07:29Z - both predate the finding; no time predicate).

**MY TEST IS FIXED BEFORE THEIR RESULT EXISTS** - freeze-pin discipline applied
to my own verification, so I cannot move it after seeing theirs.
Script: `scratchpad/verify_admissible.py`. Computes from `raw/` filenames only,
**never calls DE's selector**, so independence is structural not promised.

**PREDICATE:** admissible iff `window_start_epoch >= 1787556644`.

**BASELINE (as-of 2026-08-24T12:56:56Z):** btc **64 admissible of 1,422**;
eth **64**; all on 08-24; span 07:35Z onward; **usable day-clusters = 0**;
correct inference verdict **`DAY_BLOCK_UNAVAILABLE`**.
Growth ~12/hour/coin - it went **63 -> 64 in the 53 seconds between two runs**,
so DE and I must compare **at a stated common instant** or disagree for nothing.

**MIDNIGHT HAZARD, FILED BEFORE IMPLEMENTATION (`Q-DA-56`):**
**2026-08-24 STRADDLES the freeze, so it is PERMANENTLY admissibility-partial** -
its first 91 windows can never become admissible. After midnight it will hold
~199 admissible windows/coin, **clearing any `per_coin` threshold** and getting
labelled `holdout_complete` while being a **truncated day**. That is Q-DA-55's
defect in different clothes: **a cardinality test standing in for a boundary
test.** Correct rule: **a day is admissibility-complete iff EVERY window of that
calendar day is admissible.** 08-24 never qualifies and **must never count as a
cluster**.

**SCHEDULING CONSEQUENCE:** the first complete admissible day-cluster is
**08-25, and it does not exist until 2026-08-26T00:00Z** - not tonight's
midnight, which only starts 08-25 accruing. Day-clustered inference on forward
tape stays `DAY_BLOCK_UNAVAILABLE` **until at least 08-26**; a multi-cluster
interval is later still. Anything presented before then is window-clustered at
best and must say so.

**ON VERIFICATION I REPORT:** the counts recomputed at a stated instant, whether
`holdout_complete` is derived from the filter or from a count, and whether 08-24
is excluded as a cluster.

## DA - 2026-08-24T14:40Z (R-132/R-133) - **FREEZE VOID FILED; UNIFORM GATE LANDED**

**`Q-DA-57` FILED BLOCKING** - the freeze is VOID because it was never
anchored: pin `c83d5132...` at 07:30:44Z, but both builders were **added** in
commit `3454f60` (09:57) and **did not exist in git before it**. Pre-edit HEAD
and post-edit working tree both hash to `dd9fe9b1...` - **the user's edits are
bit-neutral**. The difference is not the `exact_receipt_events` flag, not a
cooldown value, not a name-block truncation, so it is a **real feature-set
difference**. Remedy is BE's: re-freeze against committed code, new instant,
**`frozen_at` references a COMMIT HASH** from here. **Cost is hours, not
results** - forward was `DAY_BLOCK_UNAVAILABLE` regardless.

**UNIFORM GATE LANDED (R-133, cited in-file), maintenance under R-110.**
`da_hf_pm_alignment.py`: a window straddling a collector-restart boundary now
**fails joint coverage exactly as a data gap does**, admissible only by an
explicit `stamp_waiver` naming the window. New: `stamp_boundaries_ns`,
`window_stamp_uniform`, `hf_collector_run_defects`. Result now carries
`stamp_covered`, `stamp_covered_pct`, `stamp_straddling_windows`,
`stamp_waived_windows`, `hf_collector_ledger_defects`.

**FAIL-CLOSED CHOICE WORTH KNOWING:** `hf_collector_runs` drops malformed lines
so bad text cannot relabel raw data - right for reading, wrong for CERTIFYING,
since a dropped line may have carried a boundary. **While any ledger line is
unreadable, NO window is certified uniform.** An ABSENT ledger is different and
is treated as uniform-legacy, per the reader's documented semantics.

**GATE BITES ON REAL DATA:** boundaries 13:48:08Z and 13:48:54Z; of 175 btc
windows on 08-24 **exactly one (13:45Z) is refused**, 174 admitted - it fires,
and it does not over-refuse. **40 selftests pass**, including an R-42 mirror
test (edge vs interior must answer differently) and the absent/malformed/clean
ledger cases.

**VERIFIER PARAMETERISED:** `scratchpad/verify_admissible.py` no longer
hard-codes the instant - it reads `frozen_at_utc` from the candidate (and
reports `frozen_at_commit` as **ABSENT - Q-DA-57**), or takes `--freeze`. A
hard-coded instant would have verified the void freeze forever.

**Baseline as-of 2026-08-24T14:39:45Z:** btc **84 of 1,442** admissible, eth
same, 0 usable day-clusters, `DAY_BLOCK_UNAVAILABLE`.

**PENDING:** DE's timestamp-predicate selector - re-issued, confirmed never
landed (`select_by_day` byte-identical, `git diff` empty).

## DA - 2026-08-24T15:03Z (R-134) - **ASKs RE-TRIAGED 7 -> 5; STANDING READY**

**NEITHER HAS LANDED as-of 15:03Z:** only `be_adverse_move_candidate_v1.json`
exists (no v2 re-freeze receipt), and `git diff` on `warning_window.py` is still
empty with `select_by_day` byte-identical (no timestamp predicate).

**RE-TRIAGE — 2 closed, 2 narrowed, 3 unchanged. Register: 55 rows, 5 ASK /
50 FILING, `conforms: true`.**

- **`Q-DA-57` CLOSED** - ruled and adopted in full (R-132). The remedy is BE's
  and in flight; I hold the verification, not the remedy.
- **`Q-DA-42` FOLDED into `Q-DA-14`** - its substance is *why* Q-DA-14 is
  urgent (net made fee treatment live, not latent), not a second question.
  Same dedup I applied to `Q-DA-24`; tracking it twice inflated my own count.
- **`Q-DA-49` NARROWED** - limbs (i)/(ii) superseded by `Q-DA-55`/`Q-DA-56`.
  **Live limb (iii): `m5_swm_cents` has NO INTERVAL and its protocol calls
  levels "context only"** - an optimizer maximising a level optimises a
  quantity with nothing separating signal from noise. Untouched.
- **`Q-DA-53` NARROWED AND RE-VERIFIED THIS TICK** - limbs (i)/(ii) superseded,
  but **(iii)/(iv) re-checked and STILL TRUE: tier1 quotes/trades/coverage and
  tier2 calib_panel/markout_events ALL still stop at `day=2026-08-22`**, two
  days before the freeze. Forward eval on Tier-1 returns **zero admissible
  rows**; on `raw/` it bypasses knowledge-time truncation, the distiller and
  the coverage receipts. **Least-attended open row, and it does not self-heal -
  the distiller has to run.**
- **Unchanged and live:** `Q-DA-14` (STOP inputs unpinned, now carrying
  Q-DA-42's urgency), `Q-DA-43` (A-BOOK-1 never frozen - and R-132's lesson now
  applies to its eventual freeze too: reference a COMMIT), `Q-DA-55` (positional
  selector; upheld, re-issued, awaiting DE).

**VERIFIER NOW FOLLOWS THE ARTIFACT:** `verify_admissible.py` resolves the
**highest-numbered** candidate rather than a remembered path, so when
`candidate_v2` lands it verifies against the NEW instant automatically. A
verifier pinned to v1 would have gone on certifying the void freeze - the same
shape as the defect it exists to catch. Absence of any candidate is loud.

**Baseline as-of 2026-08-24T15:03:04Z:** btc **89 of 1,447** admissible,
0 usable day-clusters, `DAY_BLOCK_UNAVAILABLE`. (26 -> 36 -> 77 -> 84 -> 89.)

**STANDING READY:** on the v2 receipt, verify DE's selector against the new
instant read from the receipt. The split freeze waits on that and nothing else.

## DA - 2026-08-24T15:56Z (R-136) - **FALLBACK FIXED; MY OWN BOUND WAS 2x TOO SMALL**

**Fixed before the recompute, as instructed.** `scratchpad/m5_concentration.py`
scores a fill **only if the horizon is observable**: inside the window AND a
quote exists **at or after** it. Unobservable fills are **EXCLUDED and
COUNTED**, never approximated.

**Actual exclusion 4.98% btc / 4.50% eth** against the **2.46%/2.01%** I filed
as the bound. My bound used a narrower test (`ts[i] < tm`) and so measured a
smaller failure than the one I had just described. Dominant cause is
**quote availability (81,474 btc), not window end (2,394)** - the book goes
quiet, the same venue behaviour A-CALIB-1's staleness ladder found.

**CORRECTED (tape population):** btc worst-10% **80.7%**, bar -> **1.60% fills /
12.60% volume**; per-share **7.10% / 6.71%**. eth worst-10% **81.0%**, bar ->
**4.38% / 17.93%**; per-share **11.74% / 12.10%**. n = 1,599,690 / 341,828.

**Every figure moved slightly AGAINST the favourable reading** - truncated fills
had short horizons and understated drift, so dropping them raises what a gate
must sacrifice. **Conclusion unchanged: btc needs 1.60% of fills where a diffuse
tape needs 45% - 28x concentration.**

**PENDING BE:** which population the conditional-markout curve is on. On the
answer I recompute concentration on the matching population - the scorer is
built and parameterised, so it is a rerun.

**ALSO PENDING:** DE's timestamp-predicate selector (`Q-DA-55`), verification
standing ready against the latest candidate.

## DA - 2026-08-24T16:30Z (R-137) - **REPLAY-ARM CONCENTRATION: MY TAPE NUMBER FLATTERED IT**

**`Q-DA-61` FILED.** The commensurable number is in, and it is **~4.7x (btc) /
3.1x (eth) weaker** than the real-fill tape figure I filed first.

| population | btc worst-10% | btc bar -> fills/vol | eth worst-10% | eth bar -> fills/vol |
|---|---|---|---|---|
| real-fill tape | 80.7% | 1.60% / 12.60% | 81.0% | 4.38% / 17.93% |
| **replay arm (BE's)** | **53.2%** | **7.47% / 8.84%** | **52.8%** | **13.46% / 15.93%** |

n = 31,645 btc / 5,705 eth fills over 90 windows each, as-of 16:25:57Z.
Convention NOT reimplemented - `edge_layer1.decompose` called directly, so the
drift term is the one BE's curve conditions on.

**VALIDATION PASSES:** mean markout **-0.8654 c/share** vs `policy_comparison_v2`
three-day JOIN mean **-0.855** - 1.2% apart, so the replay I ran IS the policy
arm. (BE's -0.5325 differs because `edge_layer1.run` uses `iw.select`
cross-day-earliest-first; I used `select_by_day` on three fixed days. Different
SAMPLE, not different method.)

**Route not dead** - 7.47% against a diffuse 45% is still **6x concentration** -
**but the predictor's job is materially harder than my tape number implied.**
R-137 barring the cross-citation is what stopped that error reaching a decision.

**STRUCTURAL POINT FOR GATE DESIGN:** on the replay, cash-ranked and per-fill
ranked concentration nearly coincide, because the arm quotes a **FIXED 5 shares**
- no size variation. On the real tape a meaningful part of concentration was
**size** (worst 2.04% of fills were ~6.7x average). **The replay has no size
lever, so its concentration is PURE TOXICITY.** A gate designed against replay
numbers is asked to do by prediction alone what the real book could partly do by
sizing.

**ALSO THIS TICK:** `Q-DA-60` filed BLOCKING on the v2 re-freeze verification -
builder `sha256 e8a82b66` MATCHES (content-anchored, verified), numbers
recomputed, but `feature_schema_hash` is **vestigial** (nothing produces it) and
`frozen_at_commit` predates the freeze with `committed_at_freeze: false`.
Corrected my own Q-DA-57 mechanism: v1's builder **was never in the repo at all**.

## DA - 2026-08-24T18:00Z (R-140) - **RE-MARKING PASS: DA WAS ALREADY CLEAN; 3 ROWS CLOSED**

**The discipline did not slip on my side.** Audit before touching anything:
**61 rows, 0 unmarked, 0 unparsed** - every row filed since the last pass
carried its `**ASK:**`/`**FILING:**` marker as it was written.

**CLOSED THREE - all ruled AND remedied AND verified by my own sign-off:**
- `Q-DA-55` (positional selector) - R-129, `select_holdout` landed, verified.
- `Q-DA-60` (v2 freeze defects) - R-138, both remedies in `v2.1`, verified.
- `Q-DA-62` (straddle mislabel) - R-139, commit `f10d799`, signed off.

**DA FINAL: 61 rows, 4 ASK / 57 FILING, `conforms: true`, no orphans.**

**THE FOUR LIVE ASKs - none of them mine to close:**
- `Q-DA-14` STOP's `h` / fee treatment / cancellation policy unpinned (R-35
  reserved to the USER).
- `Q-DA-43` `A-BOOK-1` never frozen; STOP's metric is measured against book mid.
- `Q-DA-49` limb (iii): `m5_swm_cents` has **no interval** and its own protocol
  calls levels "context only" - an optimizer maximising it has nothing
  separating signal from noise.
- `Q-DA-53` limbs (iii)/(iv): **re-verified AGAIN this tick - tier1
  quotes/trades/coverage and tier2 calib_panel/markout_events ALL still stop at
  `day=2026-08-22`, two days before the freeze.** Forward eval on Tier-1 returns
  zero admissible rows. **This does not self-heal; the distiller has to run.**

**ON THE ~59 FIGURE - MY CONTRIBUTION IS 4.** Register-wide: 172 rows, 125
marked in one of TWO conventions - form A `**ASK:**` (BE 34, DA 61, OPS 1) and
form B `**ASK: text**` (BE 19, DE 10); **47 rows carry neither.** A counter keyed
to one form miscounts the other by 29 (`Q-DA-54`). **I still do not quote a
whole-register ASK total** - my own ad-hoc parser gave two different answers on
this file, and that has not changed.

**STANDING DUTY RESUMED:** forward-population watch until midnight - post-freeze
days into training sets, straddle days as clusters, selector truncation,
blind-period coverage, n + as-of on every count.

## 2026-08-25 — OB dynamics loop armed (coordinator)

User directive: reliable OB-dynamics tests, double-confirm, long optimization
loop. Charter: `workspace/OB_DYNAMICS_LOOP.md`. State: harmful-fill pipeline
clean (v3.4 dataset, all honesty counters 0); PM-only dead on honest labels;
reduced fine arm (imb + midbps) flips btc net positive, beats random max on
NET at all budgets. I1 in flight: three-arm PM_ONLY / +reduced / +extended
(OFI + big-print, mechanism-declared, sign-reviewed, commit fed230a).
Next: I2 depth20 depletion (semantics verified: 100ms snapshots, absolute
sizes, L1==bookTicker), I3 PM-side thinning, I4 confirmation pass
(time-shift null + per-hour stability). All development evidence; freeze =
user's call; multiplicity tracked in receipts (currently 2 specs).

### 2026-08-25 loop update: I1/I2 closed
Reduced fine spec CONFIRMED best (shifted-control + per-hour + reproduction
all passed, both coins). Extended = unconfirmed small @5% candidate (held).
Depth20 REJECTED (hurts both coins). Verdict table: OB_DYNAMICS_LOOP.md.
Next: I3 PM-side thinning. Multiplicity 3 consumed, I3 makes 4.

### 2026-08-25 ~20:25 UTC — loop HOLD + saturation
I5 lead positive all budgets (held-unconfirmed; 4b control killed externally
x2 — no third relaunch; possible collector-protection, flagged CPU/tape-purity
concern). Saturation report + freeze proposal in OB_DYNAMICS_LOOP.md.
Awaiting user: machine policy for heavy runs + freeze decision.

### 2026-08-28 — BE seat: provenance chain, batch-4, inert-arm run

**DONE**
- **R-275/277 provenance.** Seam 47j (mine) fired on its own: the loaded gate
  module no longer matched the `gate_code` fit7's manifest recorded, after two
  substantive gate defect fixes (`f43359b`, `91a8949`). DA's determination came
  back IDENTICAL, so v2.3's numbers stand on a re-derivation. **The chain is
  restored SCIENTIFICALLY but NOT MECHANICALLY**: the manifest was never
  re-stamped, so `assert_fit_complete_and_matching()` still refuses with "a
  different GATE produced the verdict". Correct behaviour; re-stamping is a
  fit-time decision (R-225). Seams remain 1 red — 47j, deliberately.
- **Instruments (`558e699`).** `falsifier_count.sh` line 21 ended in `|| true`,
  so a suite that crashed after printing PASS lines yielded a plausible count
  with rc=0. Fixed and given its own falsifiers — which immediately caught that
  `die` inside `$(...)` exits only the subshell, so three refusals reported the
  wrong reason. Seam 39d converted from a source grep to behaviour.
- **Batch-4 (`375190a`).** I11-B2/B3/B4 implemented red-first: first-crossing
  valuation (max-composed disagreed in SIGN: -90c vs +110c), head n as the
  ACTION count, a real candidate-minus-incumbent increment, the prereg §5(1)
  matched-random null, and a cell guard that validates contents (24 empty dicts
  used to pass). Two gaps STATED not picked: Q3's two slopes, and per-coin
  collapse vs §9.4's per-coin regime. Q2 is NOT a gap — A1.4 carries user
  ruling R-249.
- **Q4 incumbent (`5d9a6f8`).** Load-verify-apply, 11 falsifiers, dark until
  release. Note: the incumbent is **arm D (`linear_d_<coin>.json`)**, NOT
  `harmful_reduced_fine_candidate_v1.json` — that is arm A's frozen linear.
- **Inert-arm run (`f0e5272`, `d0ee2f2`).** BE's OWN producer (DA's arm code
  never imported into production). Full tape: 638,917 rows -> 457,268 actions;
  both arms 457,268 PLACE events; DA's real loader ACCEPTED both; anchors
  cancel-disabled==skew-only and crossed-PYTHONHASHSEED both hold; agreement
  with DA's stub event-for-event.

**WATCH OUT FOR**
- **systemd-run reports success for runs that did nothing.** One invocation
  returned `status=0, Memory peak: 256.0K` in 5.4s having read no tape at all
  (`python3 -` under `--pipe` got no stdin). The completed 914k-event run
  likewise reported `Memory peak: 512.0K`. **Exit status and memory peak from a
  transient unit are not evidence.** Check a number the work had to produce.
  Consequence: the inert run carries NO ceiling measurement; be-ceiling5's
  9.65G remains the only real datum.
- **11 of the 14 artifacts the fit manifest cites by hash are not in git**
  (total 1.15 MB, excluded by the blanket `data/` rule at .gitignore:9), incl.
  both `linear_d_*.json` the incumbent loader verifies. Unresolved.
- The v3.4 dataset **cannot** drive `harmful_stateful_policy.replay_policy`: it
  carries per-latency aggregates, no generation tranches. Not a blocker for the
  contract path (which needs opportunities), but real for Phase-3 integrated
  replay.

**NEXT**
- Post-HOLD-RELEASE fit cycle: annotation-merge wiring + the manifest re-stamp;
  the tranche persister is a candidate for the same slot.
- The two stated gaps (Q3 slopes, per-coin collapse) go to the user with the
  round-3 review.

### 2026-08-28 (later) — BE: round-3 findings, artifacts committed

**DONE (adds to the entry above)**
- **`edda820`** — the 14 artifacts the fit manifest cites by hash are now IN
  GIT (1.15 MB; they fell under the blanket `data/` rule, never a size
  judgement). Every staged blob was re-hashed out of the index against the
  manifest's recorded prefix before committing: all 14 match. This closes the
  hole under the R-280 incumbent loader, which could prove a file matched while
  no later reader could obtain the file.
- **`c3cd69c`** — seven review findings, red-first. **Two were defects shipped
  earlier the same day:**
  - `matched_random_null` was TWO-SIDED where the gate says *beats*. Measured:
    AUC 1.00 -> p=0.001996 and AUC 0.00 -> p=0.001996, identical. An
    anti-predictive head would have survived Holm as a discovery. Now
    one-sided, with `sided` / `no_skill_value` declared in-band.
  - **rule 17 inside batch-4:** `generation_weights` was written, unit-tested,
    and reachable only from its unit test while every fit passed `[1.0]*len`.
    So `n` was relabelled ACTION with a row-weighted estimator. LGBM took no
    `sample_weight` at all — the two arms were weighted differently while being
    compared as differing only in model class (R-232 9.1).
  - plus: stratum hour was window-relative (two rows an hour apart shared a
    stratum); the Q4 cell's p described increments while its statistic was raw
    net; the receipt guard accepted `p=0` / `n_actions=-999` / mismatched
    identity; intervals were claimable on a non-day unit; Q1's incumbent hazard
    comparator was missing.
- **`ca4fe7a`** — `sign_flip_null` had the SAME asymmetry (+120c and -120c both
  p=0.000500). Now one-sided; `p_two_sided` demoted to a reported diagnostic.

**CORRECTION TO `c3cd69c`'s OWN MESSAGE:** it claims the verdict-bound-to-the-
return fix was applied to the library suite. It was NOT — the edit sat inside a
guard that silently did not match, and the library was still printing
`GREEN: 0 failing` mid-function with later checks running below it. Fixed for
real in `ca4fe7a`. Found only because a falsifier refused to insert against the
anchor the false claim named.

**`phase2_increment_null.py` HAS THE SAME TWO-SIDED FLAW and produced a
COMMITTED result — but the conclusion does not depend on it.** Checked, not
assumed: every cell with p<0.10 has a POSITIVE increment; the only negative
cells are eth/LGBM at 5/10/15% with p 0.42/0.23/0.28. A two-sided p is ~2x the
one-sided one for a positive effect, so survivors would survive MORE easily
under the correct test. "10 of 12 cells chance" stands and was conservative in
the safe direction. NOT fixed here — re-pricing a committed artifact is a
decision (rule 13), queued.

**POST-RELEASE CYCLE now carries four items:** annotation-merge wiring, the
manifest re-stamp (which clears seam 47j), the tranches decision, and the
increment-null supersession.

**STATE:** library 102 falsifiers, runner 98, seams 185 pass / 1 red (47j,
deliberate). Nothing fitted, nothing scored.

### 2026-08-29 — BE: the R-293 fragment diagnostic, built and gated (NOT yet run)

**WHAT THIS IS.** A user-ordered DIAGNOSTIC on post-freeze btc fragments.
`DIAGNOSTIC_NEVER_EVIDENCE`, ONE run permitted, readings pre-registered before
any number existed (positive = weak comfort only; negative = ambiguous and must
NOT trigger a candidate change). **It has not been run.** The last gate is a
pre-run re-review.

**ARTIFACT CHAIN (each bound to the next by content, not by name)**
| artifact | key facts |
|---|---|
| `da_fragment_censoring_v1.json` | DA's; bounds, cutoff 1787973300, censoring statement |
| `ledger_pin_fragment_v1.jsonl` | DA's pin, sha `e1dcd4eb8a85a0b5…`, mode 444 |
| `be_fragment_exposure_rows_v1.json` | 253/253 windows, **0 reconciliation failures**, 482,224 rows (OK 472,413) |
| `be_fragment_state_tape_v1.json` | **REHEARSAL — do not use.** builder_ref names a commit not containing its builder |
| `be_fragment_state_tape_v2.json` | THE tape. sha `a6e841e8644265fc…`, 472,413 rows, embargo NOT_APPLICABLE |
| `da_verdict_fragment_6fe1c2c4.json` | DA's gate: all_pass FALSE, 17/19, failing exactly the ruled pair |

**KEY RULINGS ENCODED IN CODE (not just the register)**
- **R-303** fragments enter the **score** split; train is empty *by declaration*.
- **R-310** the gate binding is an **exclusion list**: every applicable predicate
  must pass except exactly `both_splits_populated` + `embargo_respected`, each
  carrying its citation in the consumer's own constant. Any other failure
  refuses. **Scope: this diagnostic only — result-bearing requires all_pass.**
- **R-306(2)** canonical row order is applied and labelled honestly:
  `deterministic: true, order_independent: false`.

**DEFECTS FOUND AND FIXED TODAY (all with falsifier pairs)**
- **T2** three readers accepted a **truncated array as complete** — 5 of 6 rows,
  no error. `harmful_rows_loader`, `be_inert_arm_run`, `phase2_arms`.
- **FD-R7** the `phase2_arms` repair MOVED FIT IDENTITY
  (`3d0b6c8c6dfe9466 → e27cab9e5f6ce8e5`) under ruling, with **semantic
  invariance proven**: full 3.17 GB tape streamed before/after, 1,764,206 rows
  and digest `ccbb470278cd724d…` identical. Evidence in
  `be_fitcode_rebind_v1.json`.
- **F2** the builder **certified an embargo against an empty split**
  (`CERTIFIED`, `gap_s: inf`) and wrote **invalid JSON** (`Infinity`). v1 is its
  own known-bad; v2 shows `NOT_APPLICABLE` and parses strictly.
- **FD-R3** my own **look-ahead**: the incumbent ran `RETROSPECTIVE_TOPK` while
  the candidate was frozen. Both arms now assert `CAUSAL_FROZEN`.

**WATCH OUT FOR**
- **`evaluate_policy` is row-order dependent at gmax ties** — measured 110c swing
  across shuffles. The one-line fix is blocked: `harmful_action_eval` is in
  `CODE_IDENTITY_FILES`. Canonical sorting is the interim.
- **The builder has no `--selftest`**: an unrecognised argument RUNS THE BUILD.
  Caught only by the overwrite guard on the default path. Unruled.
- **`_index_tape` is a named 8-line fork** of `PA.tape_index` (which takes no
  path). Equivalence asserted every suite run; retirement trigger in its
  docstring.
- Wrapper kills are routine here — **a dead watcher says nothing about a
  detached systemd unit**. Check the unit, never infer from the notification.
- systemd `MemoryPeak` reported 256.0K for runs that plainly worked, and 4.96 GB
  for one that did. Populated-here-not-there is **OPEN, not explained**.

**NEXT:** pre-run re-review → synthetic seam → the single `--score` run
(`--reason` required, existing output refused).

#### 2026-08-29 delta — ragged rows, and a docstring that lies

**`encode_row` PROMISES A RAISE IT DOES NOT PERFORM** (`phase2_state_schema_freeze`,
an **identity file** — filed, not fixed). Its docstring: *"If a None arrives with
no guard, that is a schema break and it raises."* The code appends `0.0`
unconditionally. Consequence, measured:

```
row declaring 1 of 45 fields -> 45-length vector, ZERO Nones
bn_feed_age_s   = 0.0
bn_feed_missing = 0.0      <- "NOT missing"
```

The guard flag is itself an absent field, so it also encodes 0.0. The model is
told the value is genuinely zero **and** present — the exact distinction the
guard pair exists to preserve. **A ragged row does not degrade the score, it lies
to it.** Fixing it moves `fit_code_sha256_prefix`; it sits on the identity queue
beside the `evaluate_policy` tie-order fix.

**R-313 — the diagnostic no longer trusts the gate on this point.** `_index_tape`
refuses any row whose declared field count ≠ the pinned 45. Base rate measured
BEFORE choosing the predicate (120k rows: every status declares all 45, so strict
equality cannot refuse a legitimate partial). Verified on the real tape:
**472,413 entries indexed clean, no refusal** — it discriminates rather than
refuses.

**R-310 binding needs no change** — tested, not recalled: a new predicate that
PASSES is accepted, one that FAILS refuses, the ruled pair alone is accepted.
That *is* the ruled semantics. (I had claimed it was name-strict; it isn't.)

**On re-binding a superseding verdict:** re-derive `all_pass` from the NEW
predicate table; never diff against the old. An artifact that happens to agree
with its predecessor is not thereby verified.

### 2026-08-29 (late) — the R-293 fragment diagnostic RAN. Once. It is not evidence.

**THE RUN HAPPENED AND CHANGES NOTHING.** That is not a hedge — R-293 fixed the
reading before any number existed, and the number came out positive.

| artifact | |
|---|---|
| receipt | `be_fragment_diagnostic_v1.json` sha `19286320e826d040…` |
| tape | `be_fragment_state_tape_v4.json` sha `14f77d413022a6a4…` |
| exposure | `be_fragment_exposure_rows_v1.json` sha `0a3f2e0b2cf7f788…` |
| gate | `da_verdict_fragment_v4.json` — 18/2/20, failing exactly the ruled pair |

| budget | candidate net | incumbent net | increment | candidate net beats matched-random |
|---|---|---|---|---|
| 5% | +13,661.1c | +6,864.7c | +6,796.4c | True (vs rand_max +949.7) |
| 10% | +21,185.6c | +11,114.5c | +10,071.1c | True (vs rand_max +196.1) |
| 15% | +24,551.9c | +17,349.8c | +7,202.1c | True (vs rand_max −169.2) |

**CORRECTED (R-326).** The last column is about the **CANDIDATE NET**, not the
increment. My original heading sat beside the increment column and read as
though the increment beat a null. **THE INCREMENT HAS NO NULL AND NO INTERVAL
IN THIS RECEIPT** — a paired-delta null was neither preregistered nor computed.
Verified at the artifact: no increment p_value exists.

Population reconciles exactly (442,964 kept + 29,449 named drops = 472,413 =
the exposure file's own OK count); `state_join_failed` 0; both arms alive; both
`CAUSAL_FROZEN_FROM_TRAIN`.

**"COULD NOT HAVE BITTEN" IS WITHDRAWN (R-326).** My tie check computes the
boundary at the NOMINAL top-k index (11,232 / 22,464 / 33,697), but the causal
policy cancelled by FROZEN THETA (9,782 / 19,921 / 29,774) — different indices,
verified at the artifact. So `tie_at_boundary=false` answers a retrospective
question about a threshold the policy never used, and it CANNOT show ties were
absent at the actual one. What stands is what the artifact already said
honestly: **deterministic via canonical sort, NOT order-independent.**

**READ IT AS THE PROTOCOL RULED, NOT AS THE NUMBERS INVITE:** WEAK COMFORT ONLY.
The censoring plausibly flatters. No admission, no re-freeze, no parameter, no
schedule; race, frozen candidate, admission rule and multiplicity untouched.
Three unconditional inadmissibility reasons stand (neither fragment is a complete
UTC day; fragment A is a selected mid-day slice; both carry burst-concentrated
feed loss). **This is a MODEL DIAGNOSTIC, not strategy performance** — reading
those cents as what the policy would have made is the likeliest misreading.

**REPRODUCIBILITY, measured:** tapes v2/v3/v4 have BYTE-IDENTICAL row content
(`e71db7b5fc5923f8…`, 472,413 rows) across five commits that changed the parser,
the embargo arithmetic, the stamps and the input-hash race — they differ only in
headers. The ledger pin held while the archive grew 19,162 → 19,848 paths.

**STILL OPEN (identity queue — each fix moves `fit_code_sha256_prefix`):**
- `encode_row` docstring promises a raise it does not perform; a ragged row
  reaches the model as confidently-zero rather than unknown.
- `evaluate_policy` is row-order dependent at gmax ties (110c swing measured).
  Canonical sort is the interim; it is deterministic, NOT order-independent.
- the builder has no `--selftest`: an unrecognised argument RUNS THE BUILD.

**METHOD NOTES THAT COST ME TIME TODAY:**
- A check placed AFTER the thing it guards is not a check (validator wired after
  `keptrow`; ten unit falsifiers could not see wiring order).
- An assertion comparing against a string that does not exist refuses
  everything, which is indistinguishable from strictness.
- `pgrep -f <word>` MATCHES ITS OWN COMMAND LINE — a liveness check that
  satisfies itself.
- **For anything promised, check that the ARTIFACT APPEARED. Never wait on a
  notification: a reaped wrapper produces silence identical to work in progress.**

---

## 2026-08-31T12:00Z — coordinator — **THE v5 DEPLOY IS HELD. Its premise is falsified.** (R-357, R-358)

**Read R-357 first.** Six Codex rounds and ~100 closed defects all audited
*whether the transition would be stamped correctly*. None audited *whether the
thing being deployed fixes anything.* It does not.

**DONE**
- **v5 held by me, not by Codex.** Nothing armed, no instant requested, v4 live
  and continuous (pid 3687786). Round-7 batch and the Codex round SUSPENDED.
- Premise falsified on four legs, each verified by execution — see R-357:
  the library pings unconditionally (so surviving silence proves the venue
  ANSWERS control Pongs); disconnects are 94.2% btc (a contract cannot be
  coin-selective); btc gaps are median 5.8 s = exactly interval+timeout with
  data flowing into them (local failure, not venue silence); and `PONG` is
  acknowledged in the READER TASK while v5's text `PONG` queues behind market
  frames — `ws_q_hi=344` at 301 msg/s is 1.14 s of a 3.0 s deadline, on the
  one socket carrying 94% of the failures. **v5 is directionally wrong.**
- Arithmetic nobody had done: **btc 230.1 s/hr vs a P1 bar of 120** — no
  contract fix reaches it. **eth is 13.8 s/hr and already passes under v4**,
  blocked only by the era ruling.
- Four further defects closed at `d0f2deb` (R-358): a MIDNIGHT boundary made a
  mixed-era day read pure and accruing; a near-midnight instant would have
  rolled back a HEALTHY v5; the half-landed COMPLETION branch could brick the
  append-only ledger permanently; the seen-set let a rolled-back version return
  as a plain transition (4th divergence with DA, 3rd resolved in DA's favour).
- **Q-DA-180 dispatched** — DA owns the consumer half of the era-purity gap and
  a LIVE wrong verdict (`da_dayverdict_20260829.json` asserts accrual for a
  `clob_v3_1` day; the `_v2` correction is unlinked in both directions, so no
  automated reader can resolve it).

**NEXT — a USER ruling, not a coordinator one**
1. **Recommended:** run `collect_pm_v5_shadow_probe.py` against a **LIVE btc
   slug at 3 s for ≥2 h**, alongside the running v4 collector. No boundary, no
   restart, no drop-in, zero cost to the validation clock, ~74 expected events.
   It settles the premise, the queue-ordering risk and the concurrent-flow
   residual together. The longest run ever done is 125 s, at the superseded
   10 s cadence, on an EXPIRED slug — i.e. exactly the condition where queue
   contention cannot exist.
2. Or re-aim at the **load** hypothesis (btc drain/parse cost, per-coin process
   split, or accepting btc's 230 s/hr as a per-coin P1 failure while eth
   accrues — `race_accrual_eligible` is already per-coin).
3. Or overrule and deploy anyway.

**WATCH OUT FOR**
- **A review scoped by the coordinator can only find what the coordinator's
  frame admits.** Every reviewer inherited "is the transition instrumented
  correctly" because I wrote the request that way. The premise sat one grep
  from the log for days. The finding came from the only question asked without
  naming what to look at: *would you sign this off?*
- **An artifact is finished where it is READ, not where it is written.** All
  four R-358 defects live in the hop between my emitter and DA's verdicts —
  the hop nobody owned end to end.
- **A repair path deserves MORE guards than the path it repairs.** The
  half-landed completion branch had fewer, and could brick what it existed to
  save.
- **A correction no reader can resolve is not a correction.** Rule 13's FORM
  (a vN+1 receipt) was followed on 08-29; its FUNCTION (a field an automated
  reader resolves) was not.
- Do not cite the "98.22% of disconnects were local PING_TIMEOUTs" figure as
  support for v5 again. The number is right; the inference from it was wrong.

---

## 2026-08-31T17:10Z — coordinator — **v4_1 MIGRATION SCHEDULED FOR 22:00:00Z. One USER ruling still blocks it.**

**What is deployed:** ping 3/3 -> 10/10, as era `clob_v4_1`. A ROLLBACK of
O1a, which was measured to have made btc ~2.6x worse. It does **NOT** repair
the 08-25 break — that is a REMOTE per-connection throughput limit at the
venue edge (`BTC_GAP_DIAGNOSIS_2026-08-26`), with our client exonerated
(`ws_ever_paused=False` across 1,106 disconnects). **Expect btc NEAR the P1
bar (~123 vs 120), not clear of it.** Cadence is not established as the cause
of the s/hr difference — the 10/10 days also differ by storm and by the R-351
contamination (DA's caution).

**Instant `2026-08-31T22:00:00Z`** — USER ruled "set as 9.1", read as *make
09-01 the first clean v4_1 day*, which needs the boundary BEFORE 09-01 begins.
08-31 becomes the mixed day and has already failed on btc, so spending it
costs nothing. 22:00Z is 2h clear of UTC midnight (audit A1).

**BLOCKED ON — and the gate enforces both:**
1. **`clob_v4_1` admissibility, USER ruling.** `require_target_admissible`
   reads DA's `ERA_ADMISSIBLE` and REFUSES while it is absent (verified live).
   **DA recommends ADMIT** (Q-DA-188): admissibility is about the DATA, and
   v4_1 changes only the keepalive cadence — row format, timestamps and
   sub-second validity are identical. **Do not edit that table to unblock a
   deploy; DA refused that ruling for the right reason.**
2. **Codex's final seam verdict.** Request filed (`0587ab7`); Codex was quota-
   blocked until 18:26Z. Re-dispatch scheduled 18:32Z.

**Scheduled (session-only, dies with this session):** 18:32Z re-dispatch Codex;
21:38Z migration window.

**Package state:** 13 gates green. v4_1 gate 47 selftests, shadow 13.
Byte pin verified against tree AND HEAD. `--pre-arm` against the LIVE system
refuses by name on the unruled admissibility — the mechanised precondition
working, not a defect.

**Watch-outs carried into the window:**
- **Same-IP confound:** the shadow opens a connection from the SAME HOST AND
  IP as the collector, and the 08-26 diagnosis does not rule out a per-IP
  component. It starts 20 min before the boundary, so harm from it would be
  CONFOUNDED with the deploy. Step 0's verify runs BEFORE the drop-in for
  exactly this reason — **if btc degrades between step 0 and step 1, remove
  the shadow and stop.**
- **v4_1 numbers are NOT comparable to v4 ones.** The cause mix shifts (~97%
  PING_TIMEOUT at 3/3 vs ~54% at 10/10), so a bar crossing at the boundary is
  a measurement change, not a regression.
- **The five-day clock must record the ERA of every accrued day** and never
  compare quality across eras (DA Q-DA-188). Nothing reads
  `race_accrual_eligible` yet, so this is a design input BEFORE that is built.
- **Freeze the content-liveness status rule before 09-01 is judged**, not
  after (rule 11). 08-31 stands `CONTENT_LIVENESS_UNRESOLVED`.
- A missed boundary is cheap; a bad one is not. **08-31 is already a failed
  day, so there is nothing to salvage by rushing.**
