# Review — the USER's result-reliability audit, and whether the ranking result survives it

reviewer: pm-codex · filed 2026-09-04T10:24Z · tip `0ab344f` (audit at `0b970c3`)
executed in `~/ctaNew-wt-rev`. **No race seal opened** — the 09-01 and 09-02 feeds and sealed scores were never read; every measurement below is on the **08-29 development feed**, which runs the same code path. Nothing written under `data/`.

---

# ITEM 2 FIRST — **the ranking result is insulated, and the reason is not the one I expected**

## The two paths do not share an aggregation

`prof.py`'s defective object is a local dict:

```python
seen = {}
for line in open(p):
    r = json.loads(line); k = (r["slug"], r["side"], r["gen"])
    if k not in seen: seen[k] = r          # first row per action
```

`seen` feeds exactly one thing: the `fill_shares` / `filled_notional$` / `no_cancel_$` block — the withdrawn profitability numbers. **`seen` is never passed to `matched_volume()`.** The MATCHED_VOLUME table is computed from a different object entirely: `lo = {d: RC.load_two_arm_feed(Path(p), L)}`, and `matched_volume(b["rows"], b["cand"], b["inc"], th, L)` runs over **all** feed rows, doing its own `_gen_index` → `_select_by_threshold` → `_cancel_value`.

**One column of the MATCHED_VOLUME table is contaminated and it is not one of the nine values.** `base = agg[coin][3]/100.0` — the `baseline_$` column printed beside the arms — comes from `seen`. So does every percentage computed against it. The `cand_$ / inc@theta_$ / inc_match_$` columns do not.

## But "different path" is not the whole defence, and I nearly filed it as if it were

I tested the ranking against four aggregation rules on the 08-29 two-arm feed (537,881 btc rows, 299,386 actions):

| budget | SHIPPED first-crossing | A crossing-onward | B all rows in the gen | C prof.py's first-in-file |
|---|---|---|---|---|
| 5% | **−666.38** | **+406.20** | −1,261.32 | −1,043.79 |
| 10% | **−601.16** | **+1,210.89** | −702.49 | −463.64 |
| 15% | **−1,198.67** | **+681.29** | −1,485.29 | −1,734.38 |

**The sign is not invariant.** Under variant A — sum every row from the crossing onward, which is the natural reading of *"a cancellation prevents everything that happens after it acts"* — the statistic goes **positive at all three budgets**. If that were a legitimate estimand, the ranking result would fall.

**It is not, and this is the measurement that settles it.** A "row" here is not a fill: `harmful_exposure_rows.py:363-370` already sums every tranche inside that row's own 1-second horizon at or after the 50 ms cut. Multiple rows per action are overlapping decision-time snapshots of the same exposure. Measured on the real feed:

```
rows/action 1.797 | 26.9% of actions have >1 row
consecutive-row spacing within an action: median 0.170 s, p75 0.414 s
pairs closer than the 1.0 s FILL_HORIZON_S  =>  217,267 of 238,495  (91.1%)
```

**91.1 % of consecutive row pairs inside an action are closer together than the horizon they each sum over**, so variants A and B add the same tranches two or more times. The positive sign under A is a double-count, not a rival reading.

So the defence is stronger than "different path", and I state it in the form that can be checked: **of the four rules, exactly one counts each prevented fill once, and it is the one shipped.** A cancellation acts once, at the crossing; the row at that instant already aggregates its own horizon. Variant C — prof.py's — preserves the sign here, but only by luck: it takes the first row by `t_start` regardless of score, i.e. a *different decision point* from the one at which the arm actually crossed.

> **Answer to item 2: the ranking result is genuinely insulated.** The aggregation defect reaches the profitability block and the `baseline_$` column beside the arms; it does not reach the nine MATCHED_VOLUME values. But the insulation rests on the double-counting measurement above, not on path separation alone, and that measurement should travel with the result — because a reader who re-derives "prevented value" the intuitive way gets the opposite sign.

---

# ITEM 1 — the audit's four findings, verified at the artifacts

| audit claim | verified |
|---|---|
| `prof.py` keeps only the first row per `(slug, side, gen)` and labels those sums as total filled notional and no-cancel P&L | **Exact.** The `if k not in seen` dict, and the three accumulators built from `seen.values()`. And the actions really are multi-row: **1.797 rows/action, 26.9 % of actions above one** |
| the emitted scale is `preventable_shares`, not filled shares — only fills inside the one-second horizon at or after the 50 ms cutoff, earlier fills recorded as `stale_shares` | **Exact.** `be_forward_metric.py:622-624` emits `preventable_shares` and `level`; `harmful_exposure_rows.py:339-370` sets `h_end = t_start + FILL_HORIZON_S` (=1.0 s), `cut = t_start + L/1000`, `prev = [t for t in fut if t.t >= cut]`, `preventable_shares = sum(shares for prev)`, `stale_shares = sum(shares for t in fut if t.t < cut)`. Fills outside the horizon never enter `fut` |
| `be_read_cells.compute()` emits only BY_THRESHOLD and BY_COUNT and never calls `matched_volume()` | **Exact.** AST of `compute()`: `matched_volume` appears neither as a call nor as a name; the only conventions in its body are `BY_THRESHOLD` and `BY_COUNT` |
| no committed caller for `matched_volume()` exists anywhere | **Exact, and I applied the stricter test.** An AST census over all 175 modules counting **calls and bare references** (the blind spot that nearly made me misfile `frozen_contract_gate` at BE21) returns **0 sites** |

**Nothing in the audit is overstated.** One thing I would add rather than correct: `prof.py` imports its modules from `/home/yuqing/ctaNew-wt-be/live/pm_research` — a seat's worktree, not the committed tree — so the published numbers were produced by a script in a temp directory importing code from a working copy. That is visible from the file's third line without reading any of its logic, and it is the cheapest possible tell.

---

# ITEM 3 — the class, and a check cheap enough to run every time

**Yes, there is one, and the programme already owns every piece of it.** What was missing is that we have been applying the wiring test to *functions* and never to a *number*.

**THE PUBLICATION PROVENANCE CHECK.** Before any number reaches a document, ask three questions of its producer. All three are mechanical and the whole thing is one AST pass over `live/pm_research/` — seconds:

1. **Name the producer.** The artifact must say which function computed each headline statistic. If it cannot, stop: an unnamed producer cannot be checked.
2. **Census its committed call sites — calls *and* references — and require ≥ 1 reachable from a committed entry point.** Zero means the number was not produced by the committed pipeline. Run today: `matched_volume` → **0 sites**. That single number is the whole finding, and it would have taken one command.
3. **Check the producing path.** The artifact's `producing_code` (which `be_forward_day` already emits) must name a path inside the repo. A `/tmp/**/scratchpad/*.py`, or an import rooted in `~/ctaNew-wt-*`, is a scratch result regardless of how good the arithmetic is.

A number failing (2) or (3) is not necessarily wrong — the nine MATCHED_VOLUME values reproduce and I have just re-tested their estimand — but it is **not a pipeline result**, and it must be labelled as a scratch reproduction until it has a runner, a positive control and a durable artifact. That is the audit's own next step 1, generalised.

**On my own part in this, plainly.** I dispatched and filed six zero-consumer findings today — `counts_toward_race` with no reader, `require_operating_point`'s fences with no production call site, `sealed_shape_is_unusable` that could not fail, a map asserting itself, a control whose subject was the calendar, a p that agreed because both sides called the same function — and I did not run the same census against the programme's own headline. The reason is diagnosable rather than excusable: **every one of those checks ran because a round was dispatched, and the released result was never in a round.** So the standing practice must attach to **publication**, not to review dispatch — the check is owed by whoever publishes the number, at the moment they publish it, and its output belongs in the artifact beside the number.

Had it been run on 2026-09-04 at 09:00Z it would have returned `matched_volume: 0 committed call sites` and the profitability block would never have been published. I would put it in the coordinator runbook as a precondition for any number entering `RESULTS.md`.

---

## Findings

| # | sev | |
|---|---|---|
| **AUDIT-R1** | — | **The audit is correct on all four points and overstates none of them.** Verified at `prof.py`, `be_forward_metric.py:622-624`, `harmful_exposure_rows.py:339-370`, the AST of `be_read_cells.compute()`, and a repo-wide reference census returning zero |
| **AUDIT-R2** | MEDIUM | **The ranking result is insulated** — `seen` never reaches `matched_volume()`; the contamination touches the profitability block and the `baseline_$` column only. But the insulation rests on a fact not yet recorded anywhere: the shipped first-crossing rule is the only one of four that counts each prevented fill once (**91.1 % of intra-action row pairs are closer than the 1 s horizon**), and the intuitive alternative flips the sign positive. That measurement should be published with the result |
| **AUDIT-R3** | — | one addition to the audit: `prof.py` imported from `~/ctaNew-wt-be/`, a seat worktree — a second, independent tell that the number was not a pipeline product |
| **AUDIT-R4** | — | the publication provenance check above: name the producer, census its committed call and reference sites, check the producing path. One AST pass; run it before a number enters a document, not when a round happens to be dispatched |

## Disposition

**The audit stands as filed, and the withdrawal of the profitability block is correct.** The narrow ranking claim survives — I re-tested its estimand rather than re-running its arithmetic, and it survives for a reason that is measurable and was not previously written down.

The one thing I would change in what the programme now says: the ranking result should carry the double-counting measurement beside it. As it stands, the result's defence lives only in the fact that nobody has yet tried the intuitive aggregation. Someone will.

I estimate; the coordinator routes; the USER decides (rule 14).

## Discipline record

Executed at `0ab344f` in `~/ctaNew-wt-rev`; every step under `systemd-run --user --scope --slice=research.slice -p MemoryMax=8G`. **No race seal opened** — the 09-01/09-02 feeds and all `*_SEALED_scores_*` files were never read; every measurement is on the 08-29 development feed, which is the same code path. `prof.py` and `interim_report.py` were read as scratch scripts in a session scratchpad, not from any seat worktree; `~/ctaNew-wt-be`, `-da`, `-de` were never read. Nothing written under `data/`. Worktree clean.
