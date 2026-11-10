# REVIEW 243 — the ledger is now a genuine second instrument

REV round 206. Filed 2026-09-11T20:55:34Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled. Fetched at 20:52Z. Worktree
detached at `58dc587`, `git status --porcelain` empty. Both executing refs are
`58dc587`, and all nine lane modules are byte-identical across them.

**The six gate modules did not move this round** — every blob is the one I drove
at REVIEW 242. Only `da_fair_value_ledger.py` changed
(`b95ad2e9a7b1` → `f813c5d5aac0`). So this filing is about the instrument, and
my gate measurements stand on identical bytes, re-driven anyway: 50/50, 27/27,
29/29, 22/22, 9/9, 25/25, all rc 0, plus the property map 8/8.

## THE RULING

**The ledger is now a genuine second instrument, and the six rows agree with
mine row by row.** All six are probed, `gates_satisfied` counts only probed
rows, and I confirmed the counter is load-bearing by breaking a probe and
watching the total fall.

One residual, named below and not blocking: a probe that runs and declares
**zero** properties would still read SATISFIED, because `all({})` is `True`. No
row is in that state today — 47 properties across six rows — but it is the
silent version of the failure that produced this round.

## 1. All six rows genuinely probed

`ref_head: 58dc587`, `gates_satisfied: 6`, `n_gates: 6`, `gates_probed: 6`,
`unprobed_gates: []`, falsifier 37/37 rc 0.

**I ran the six probes myself**, extracting each `PROBES[n]` source and
executing it directly rather than trusting the ledger's own harness:

| probe | rc | properties | false |
|---|---|---|---|
| gate 1 | 0 | 7 | — |
| gate 2 | 0 | 9 | — |
| gate 3 | 0 | 7 | — |
| gate 4 | 0 | 7 | — |
| gate 5 | 0 | 6 | — |
| gate 6 | 0 | 11 | — |

None crashes. 47 properties, none false.

**`row_status` is now airtight.** Driven across every branch:

| props | status |
|---|---|
| `probed: True, all_covered: True` | SATISFIED |
| `probed: True, all_covered: False` | CELLS_GREEN_BUT_PROPERTY_UNCOVERED |
| `probed: False` (the crash case) | **PROPERTIES_NOT_PROBED** |
| `probed` absent | **PROPERTIES_NOT_PROBED** |
| `probed: None` | **PROPERTIES_NOT_PROBED** |
| `probed: "true"` (truthy string) | **PROPERTIES_NOT_PROBED** |

`SATISFIED` is reachable only through a literal `True`, and the counter states
it a second time (`if status == "SATISFIED" and props.get("probed") is True`),
so an edit to either cannot quietly restore the default path.

**The mutation test.** I replaced gate 6's probe with a source that raises
before printing — exactly last round's failure — and re-ran `build`:

    gates_satisfied: 5 / 6   gates_probed: 5   unprobed_gates: [6]
    gate 6: PROPERTIES_NOT_PROBED   probed=False
    no_labelled_score_permitted: True

The total drops, the row is named, and the barrier goes back up. Under the old
rule this same input printed 6/6.

Two structural improvements I did not ask for and should record: the ledger now
drives each gate in a **temporary worktree cut at the ref** rather than in the
tree it runs from, and `all_covered` is **computed by the ledger** from the
probe's values (`all(props.values())`) rather than reported by the probe.
`score_is_evidence_permitted` is likewise computed —
`(satisfied >= 6) and step11_step6_freeze["satisfied"]` — and that second term
is measured by counting freeze declarations on every executing ref, which is
`0`, so it reads `False`. Correct: the §5 barrier being down is not permission.

## 2. Row by row, against my own measurement

| § | gate | my cells | my status | DA status | DA props | agree |
|---|---|---|---|---|---|---|
| 5.1 | settlement verifier | 50/50 | SATISFIED | SATISFIED | 7 | ✓ |
| 5.2 | sigma producer | 27/27 | SATISFIED | SATISFIED | 9 | ✓ |
| 5.3 | estimator wrapper | 29/29 | SATISFIED | SATISFIED | 7 | ✓ |
| 5.4 | forecast-action builder | 22/22 | SATISFIED | SATISFIED | 7 | ✓ |
| 5.5 | policy seam | 9/9 | SATISFIED | SATISFIED | 6 | ✓ |
| 5.6 | replay seam | 25/25 | SATISFIED | SATISFIED | 11 | ✓ |

**Six of six, agreeing on rows and not merely on the total.**

The row that matters most is 5.6, and it agrees at the property level too. Gate
6's list now carries `i_differing_action_populations_refuse` and
`k_the_action_population_is_REQUIRED_like_price_path` — the two entries that
were `False` at REVIEW 241 and were my blocker — both now `True`, and I drove
both myself at REVIEW 242 against these same bytes (omission is a TypeError;
every shareable placeholder refuses by name; the recomputation is what binds).

The agreement is between different instruments, which is the only kind that
counts. DA's probes are separate code building their own fixtures, not a re-run
of DE's falsifier — the ledger drives the module's falsifier for BEHAVIOUR and
runs DA's probes for PROPERTIES, and I read both sources.

## 3. The gate-2 adjudication: exclude-and-count is right

**The coordinator's reading is correct and the probe's original assertion was
wrong.** Three grounds, in order of weight.

**(a) §4 C2's grid rule is a selection rule, not an admissibility rule.** *"each
one-second grid value is the latest midpoint whose local-knowledge time is at or
before that grid instant; no interpolation or later tick may fill it."* With
`grid_end_sec = decision_sec - 1`, a post-decision row is never the
latest-at-or-before any grid instant. It is not selected. C2's typed non-OK
statuses are reserved for input that is *"unavailable, stale, pre-era or
malformed"* — a row after the decision is none of those; it is out of window.

**(b) Reliability rule 4 requires the count.** Exclusions are statuses, never
silent drops, and their counts are reported with every table.
`n_rows_after_decision` is exactly that field. Refusing instead of excluding
would also make C2 unusable in practice: any tape read after the fact contains
rows later than the decision instant, so the routine case would become a
refusal.

**(c) The module's own cells prove the property that actually matters**, which
is stronger than either reading of the words:

    rows AFTER the decision change nothing -- the estimate is bit-identical
        status OK, sigma equal to the same stream without them,
        n_rows_after_decision = 3
    a path FLAT until the decision and WILD after it measures ZERO_VOLATILITY,
        not the future

**Where DA's probe went wrong is instructive, and it is not a hallucination.**
`FUTURE_KNOWLEDGE_IN_SOURCE` is a real declared status in `be_sigma_30m.py`
(line 68) and it *is* emitted (line 216) — but for a different condition:
`if last_recv > decision_recv_ns`, a final structural check on the **sigma's own
last admitted knowledge time**, which is the plan's clause
`sigma_local_knowledge_ns <= decision_recv_ns`. It is unreachable on the normal
path, and the source comment says so ("structural, checked anyway"). DA's probe
mapped the right constant to the wrong condition.

DA has already corrected it in the right direction, not merely to green: the
property is now `h_post_decision_rows_change_NOTHING_and_are_COUNTED`, asserting
the estimate is unchanged and checking the count in two fixtures
(`n_rows_after_decision` of 1 and of 3). That is the property §4 C2 cares about.

## 4. Anything else reading satisfied for the wrong reason

I swept the status path for it. Clean on every point I checked:

- `all_covered` is computed by the ledger, not reported by the probe;
- `probed` must be literally `True` — a truthy string does not pass;
- the count is guarded twice, in `row_status` and at the counter;
- a failed worktree or an undriven module yields `LANDED_BUT_NOT_DRIVEN` or
  `PROPERTIES_NOT_PROBED`, neither of which counts;
- the ledger's own cells assert that no probe reports coverage from a
  source-text match and that every probe imports and drives its module — six
  probes each, and I confirmed the six sources do construct and call.

**One residual, and it is the vacuity case.** `probe_gate` returns
`all_covered: all(props.values())`, and `all({})` is `True`. A probe that runs,
prints valid JSON and declares **no** properties therefore yields
`probed: True, n_declared: 0, all_covered: True` → **SATISFIED**. Driven:

    row_status(present, DRIVEN_GREEN, {probed: True, all_covered: True,
                                       n_declared: 0}) -> SATISFIED

No row is in that state — the six declare 7/9/7/7/6/11 — and a *crash* cannot
reach it, because a crash gives `probed: False`. It needs an authoring mistake:
a probe whose assertions are removed or whose dict is never filled. That is the
silent form of exactly what happened this round, and it is the same class I
ruled on for `da_guard_register_v1.json` at REVIEW 168 — a predicate true
because it quantifies over nothing. One line closes it: treat `n_declared == 0`
as not probed, and add the cell.

**One independence note, not a defect.** Gate 1's properties are DA probing
DA's own module. Seven properties, all true, and I drove its 50 cells myself and
audited the settlement verifier at REVIEW 232/233 — so the gate is
independently checked, but *within the ledger* that row is one seat vouching for
itself. Rows 2–6 are DA probing another seat's module. Worth recording so the
row is not read as two instruments when it is one plus my drive.

## 5. The six rows

At `58dc587`, every count and property driven this round.

| § | gate | landed | cells | status |
|---|---|---|---|---|
| 5.1 | settlement verifier | 1 / 1 | 50/50 | **SATISFIED** |
| 5.2 | sigma producer | 1 / 1 | 27/27 | **SATISFIED** |
| 5.3 | estimator wrapper | 1 / 1 | 29/29 | **SATISFIED** |
| 5.4 | forecast-action builder | 1 / 1 | 22/22 | **SATISFIED** |
| 5.5 | policy seam | 1 / 1 | 9/9 | **SATISFIED** |
| 5.6 | replay seam | 1 / 1 | 25/25 | **SATISFIED** |

**SIX OF SIX, confirmed by two instruments that disagree about nothing.** The
§5 build barrier is down. §11 step 6 — the full-pipeline freeze — is still
unmet and still binding: `step11_step6_freeze` finds 0 declarations on either
executing ref, and `score_is_evidence_permitted` is `False` until it does.

## 6. Owed

- Hardening, none blocking: `n_declared == 0` must not read as covered; drop the
  dead `and` in `run_arm`'s guard (REVIEW 242 §1); the `compare_arms` exclusion
  assertion and the non-JSON-native value refusal (REVIEW 240 §2).
- When the step-6 freeze lands I will verify it as a count at a fetched ref and
  re-drive the six rows against the frozen blobs, since a freeze that pins
  superseded bytes is the failure this lane has already had six times.
- Standing: one review per day for the 09-09..09-13 records (REVIEW 228); the
  round-boundary landing sweep as counts at fetched refs.
