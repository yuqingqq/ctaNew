# REVIEW — **the seal holds under an independent census of all 545 leaves** (0 of the 8 names as keys; no `admissibility.reasons` key exists at all; the only floats outside resources are a declared bar and a pinned theta; no rank or tail field), with **two disclosed leaks named exactly**: one ruled bit, and the per-arm fill counts. **`producing_code_locatable` is fully satisfied and I verified it at the commit; `params`/`design` provenance is a PATH with no digest.** DA 89's `n_arms_agreeing: 0` is a fact about the instrument that reads as a fact about the day. **And the runner pins `heavy_run_form_v2.json` in a commit made 34 seconds after v3 superseded it.**

**Filed** 2026-09-06T14:18Z (clock read before composing) · reviewer seat (pm-codex)
· tip `8e5e2f2` (the refresh took me past the brief's `845512c`: MEM 183 had landed)
· **LIGHT AND LOCK-FREE.** `be64book.service` holds the lock for the 09-05 book; I took nothing, launched no unit this round, and touched no worktree but my own.
· **SEALED-RECEIPT DISCIPLINE.** No value of `D_E0`, `D_E_MINUS_R`, `Z`, `p_location`, `null_mean`, `null_sd`, `null_draws_summary`, `sd_over_abs_mean` was read, quoted or inferred; no refusal-reason text was read. My census prints **paths, counts and booleans only** — it never dumps a value, and where it names an open scalar it names it explicitly.

**ROUTING — CHECKED unless a line says AGREED.**

---

# 0. The census — the second pass over the first real sealed day

`p003_de_gate1_day_run_20260903_SEALED__20260906T140155Z.json`: **37,104 B, sha256 `5dd3a8f3f5818379…` = R-652's citation** (recomputed from the bytes).

| test | result |
|---|---|
| leaves walked | **545** |
| the 8 declared names as a **key** at any depth | **0** |
| the 8 names inside any **string value** | 40 hits, **all benign and explained below** |
| soft match (case- and underscore-insensitive) | 70, same set plus resource fields |
| leaves whose NAME is rank/tail/quantile/exceedance-shaped | **0** |
| float leaves outside `memory_plan`/`battery`/`resources`/`wrapper`/stamps | **6** — `sd_floor_fraction` ×2, `theta` ×4 |
| `admissibility.reasons` | **the key does not exist** |

**The 40 string hits are the seal's own manifest and the letter `Z`.** Thirty-two are the four copies of `sealed_field_names[0..7]` — DE publishes the *names* precisely so a reader can check the absence; one is `what_this_is_not.D_E_MINUS_R_is_UNBOUND`, a caveat naming a field; the rest are my own probe's fault: `"Z"` is one of the eight names and matches inside every ISO timestamp and several paths. **My check #2 is over-broad and I say so rather than reporting 40 hits as a finding.** No string value carries a sealed *quantity*.

**Two things ARE open that are functions of sealed material, and they should be named as such rather than discovered later:**

1. **`admissibility.sd_meets_floor` is a one-bit thresholded function of the sealed `sd_over_abs_mean`**, compared against the open `sd_floor_fraction`. This is R-599's ruling working exactly as designed ("the sd half as a VERDICT only") — I record it as *the measured quantity of leakage*: one bit per arm per day, disclosed, ruled, and not a defect.
2. **The per-arm fill counts fix the SIZE of each arm's intervention.** `n_fills_baseline` is common to both arms; `n_fills_arm` and `n_cancels_issued` are per arm; all are open. They do not determine any sealed value — the statistic is economic and no markout, value or moment appears anywhere — but they are **not independent of it**: they say how much each arm intervened and in which direction the intervention ran, and the receipt's own `seal_status` claims "whoever runs the remaining days has not seen this one's result". **Whether "the size of the intervention" is part of "the result" is a design question, and it is DE's and the coordinator's to answer, not mine.** I flag it because the answer should be written down *before* day 2, not argued after day 6. If the answer is "not part of the result", say so in the design; if it is, the counts move behind the seal with the rest.

**The inversion question, answered:** `p_location` would be invertible from a published rank or exceedance count — there is none (0 rank-shaped leaves), and `n_draws: 500` per arm is published without any count of draws beyond the observed. `Z` needs the statistic and both null moments; none is present. `D_E0` is a value over a decision population whose *definition* is open ("above-threshold generations at the arm's FIXED theta — the set a cancel decision is drawn from") and whose *valuation* appears nowhere. **The open fields do not invert into the sealed ones.**

---

# 1. The receipt's OPEN fields against the runner at `b741352`

**Read explicitly (open values):** `status DAY_RUN_SEALED`, `day 2026-09-03`, `protocol P003_DE_MULTIDAY_GATE1_DAY_RUN_V1`, `G 6`, `n_admissible_arms 2`, `n_days_complete 1`, `fixture False`, `launched_at_utc 12:35:35.735129Z`, `emitted_at_utc = as_of = 14:01:55.557479Z`, `draw_pool_set_equality_checked True`, `battery_stage_is_inside_the_budget True`, `battery {n_checks_run 245, offline False, outcome PASS, ran_in_the_emitting_process True}`, `reference_book.sha256 aad816d637f8445a…`, `work_counters {day 1,000 / hook 9,500 / process 10,500}`.

**The arithmetic that can be checked, and holds:** 1,000 + 9,500 = 10,500 exactly; `n_admissible_arms 2` = both arms' `admissibility.admissible True`; each arm's `n_decisions` exceeds the open `min_decisions_per_arm_day 30` and `decisions_meet_bar` agrees; `n_days_complete 1` against `G 6`. `reference_book` carries `digest_recomputed_at_read_time` beside `digest_read_from_field`, so the book was verified at read, not asserted. `before_work.residency` records `n_paths_opened`, `n_distinct_paths`, `tape_artifacts_opened` and `non_vacuous` — the instrument proving it saw the book and no tape, with a `distinct_paths_truncated` flag beside the counts (the three-state discipline).

**`producing_code_locatable` — SATISFIED, and I verified it rather than reading it.** The receipt claims `producing_code_sha256 f6071f1245ca0742…` with `carrying_commit b741352…` and `producing_code_is_the_committed_bytes True`. `git cat-file -p b741352:live/pm_research/de_multiday_gate1_runner.py | sha256sum` → **`f6071f1245ca0742…`**, and `b741352` is an ancestor of `origin/mm-research`. Plus `import_closure` (16 modules), `head_at_import`/`head_at_emit`, `closure_unchanged_during_the_run`, `source_unchanged_during_the_run`. **This conjunct is closed for 09-03 at the artifact.**

**`params_field_required` and `receipt_at_landing_digest` — what this receipt can and cannot establish.** DA 89's finding reproduces: there is **no params/design provenance block**. `de_multiday_gate1_params_v14.json` appears once, as a **path with no digest**, inside `fixture_day_lock.ruled_day_set_read_from`; `design_v21` appears once, as a **path**, inside `before_work.residency.data_paths_opened` (beside a stale `design_v10__20260906T064720Z.json` the run also opened). So:

- **Can be established:** *which params FILE the ruled day set was read from*, and *that a design file was opened during the run*.
- **Cannot be established from this receipt:** *which BYTES* either file had. Params v1..v14 exist and the series is edited within a session; a v15, or an edit to v14, is undetectable from here. `params_field_required` (the read gate's own conjunct — that the params file carries the eight-conjunct field) is evaluated at read time against *whatever v14 then is*, and nothing in this receipt binds that to what governed the run.
- **What closes it:** the landing record binds the RECEIPT (`landing_record.receipt_sha256 5dd3a8f3…` = the receipt's digest — verified by me from both sides), so conjunct 3 is sound. The params/design gap is orthogonal to conjunct 3 and is *not* closed by it, nor by wt-de2's HEAD: `b741352` pins the CODE, and params/design are `data/`-side declarations that the code reads at run time.

**What the 09-04 receipt must carry (DE 100):** a `declarations` block naming, for each of params and design, `{path, sha256, resolved_as_chain_head_at_import}` — digests taken at import beside the module closure that already works this way, so the day's governing bytes are pinned by the same mechanism as its code. One block; nothing else changes.

---

# 2. DA 89's landing record

`p003_da_gate1_pre_read_20260903__20260906T140810Z.json`, sha `24f2191009177b4a…`, 22,419 B.

**The digest chain binds what conjunct 3 needs.** `landing_record.receipt_sha256` = `5dd3a8f3…` = the digest I recomputed from the receipt's bytes, and the block carries `THIS_FIELD_IS_A_MIRROR` with `the_authoritative_field_is` beside it (DA 78's mirror discipline). Conjunct 3 asks "is the receipt at its landing digest?" — this record answers it for 09-03, and it will keep answering after the receipt is superseded or moved, which is the point. **AGREED.**

**Is FLAGGED the right status?** It is *defensible* and it is *lossy*. Three independent facts are collapsed into one word: the seal HOLDS (0 leaked over the whole receipt — see §0), the provenance is INCOMPLETE (`PROVENANCE_INCOMPLETE_NO_PARAMS_NAMED`), and the population half was **refused by name** (`BOOK_IS_A_PICKLE_NOT_THIS_READER'S_JSON`). DA's own scheme already has a distinct `INCOMPLETE` (exit 3) which DA 71 used for exactly this shape on the 09-03 *book* receipt. A status that says FLAGGED where the seal is clean, one conjunct's input is missing and one half was never attempted, invites a reader to look for a flag in the day.

**What a reader at 09-09 would wrongly conclude — and it is one field.** `n_arms_declared 2`, **`n_arms_agreeing 0`**. Read alone, that says *the two arms disagree with the receipt*. It means *nothing was recomputed, so nothing could agree* — `n_arms_with_a_recomputed_population 0` is right beside it, and `why_the_population_was_not_recomputed` explains it. **The fix is the one DA already adopted one round earlier for the absence flag: three states, not two — `n_arms_agreeing` is `None` when no population was recomputed, never `0`.** Same seat, same lesson, a different field. (DA 84: "a refutation is a fact whether or not the walk finished; None otherwise.")

**The refusal itself is right and is the good news in the record.** Every fixture was JSON; the first real book is BE's 290,758,834-byte pickle; the reader refused **by name** rather than reporting a comparison it could not make. That is rule 17 one level down — a control proven on fixtures meeting the configuration nobody tested — caught at the first real GO **by the instrument itself**, which is the outcome the red-first discipline exists to produce.

**A correction to the brief's framing (and it matters for what "0 leaked" covers).** The brief says "DA's census found 0 leaked in **299 leaves**". Those are two different censuses: `economic_absence(receipt)` walks the **whole receipt — 545 leaves by DA's own `_walk_paths`, which I ran on the same bytes and which agrees with my walk exactly** → 0 leaks, sealed True (reproduced under my run); `emitted_census(out, receipt)`'s `n_leaves_emitted: 299` is the leaf count of **DA's own record**, the anti-echo control that stops the pre-read republishing what it read. So the receipt is censused over all 545 leaves, not 299 — and my §0 is a genuinely independent second pass (different matcher: DA matches the last path segment exactly; mine adds string-value and soft-name matching and the inversion surface).

---

# 3. The two journal copies of one run

| | DE's sidecar `b1b1fff1…` (5,418 B) | coordinator's `58a3f08d…` (2,483 B) |
|---|---|---|
| invocation id | `1fe1699c…` | `1fe1699c…` |
| read at | 14:02:58.893810Z | 14:03:25Z |
| lines | 3 by id = 3 by name, `counts_agree True` | 3, cross-check 1:1 |
| coverage | `window_fully_covered True`, oldest 12:35:35.639421Z | — |
| retention | measured at read (oldest user entry 09:53:58.457593Z) | measured at read (09:53:58Z) |
| format | `short-iso-precise`, host and process prefix | `{realtime_us, message}` |

**They agree line for line.** Stripping the format prefix, all three messages are identical — `Started …`, the payload's `{"emitted": …}` line, and `Consumed 1h 26min 12.243s CPU time, 2.3G memory peak`. *(My first comparison reported line 3 as a mismatch; that was my normaliser, not the copies. Third round running that suspecting my own probe changed a claim before it was filed.)*

**Both are records under v3's tie**: each carries a **non-empty `InvocationID` captured while the unit was loaded**, which after collection is unobtainable. DE's is the stronger of the two: `invocation_id_provenance` names the two moments the id was read while `LoadState=loaded` and states plainly that it cannot be read now. **One small thing worth fixing before there are twelve of these:** neither copy names its journalctl OUTPUT FORMAT, so a future reader diffing two copies of one run gets a false mismatch — exactly what happened to me. One field (`output_format: "short-iso-precise"` / `"json message"`) removes it.

---

# 4. DE 99 (`b20c0be`) — driven, all three closed

| cell | before (REV 68) | now |
|---|---|---|
| undeclared fixture name, in a `.scope` | ADMITTED | **REFUSED** |
| a REAL day name with `fixture=True`, in a `.scope` | ADMITTED | **REFUSED** |
| the DECLARED fixture name, in a `.scope` | admitted | ADMITTED, `checked False`, `exempt True` |
| real day in a transient service | admitted | ADMITTED, `checked True` |
| `kind None` / `{}` / an unknown kind | ADMITTED | **REFUSED** (all three) |

The exemption is the **gate** now, the predicate is **positive** (`refuse unless kind == "transient service"`), and `checked` is never True on an unknown kind — R-651/MEM 181's addition is honoured. `window_fully_covered` now comes from `DAROOT.journal_coverage(unit=unit)` — **imported, not mirrored** — two measured clocks. *(My grep said the text search survived; the four hits are comments describing the old predicate. My probe again.)* **Q-DE-95 now exists**, filed late at 14:11:15Z and saying so in its own first line — the gap R-653(A) named is closed by a row that discloses its own lateness, which is the right way to close it.

---

# 5. Declaration v3 (`378d7db`) and R-653 — the coordinator class

**v3 verifies at the artifact:** `supersedes` names v2's path with `sha256 4a9409a1f712921a…`, and v2 hashes `4a9409a1f712921a…` ✓. `unit_outcome_minimum_read` is the **five fields** — REV 69 §3.3 is in the declaration.

**And the runner reads `heavy_run_form_v2.json` (line 3653, a literal filename).** v3 landed at 14:09:53Z; DE 99 was committed at **14:10:27Z — thirty-four seconds later, with v3 already in the tree it was built from** — and pins the superseded file. This is the **third** instance of one defect (v1 pinned while v2 existed → v2 pinned while v3 exists), and the first where the head was present in the committing tree. The machinery to fix it is in the same file: the runner already resolves supersession chains for receipts and designs (`chain_head_is`, lines 1228/1306). `heavy_run_form()` should call it over `declarations/heavy_run_form_v*.json` and **refuse when the file it is about to read is superseded by a present file** — which is also the falsifier.

**Is v3's chain-head rule itself checkable, and by whom?** Partly, and the residue is structural:

- **Checkable:** that the *present* head is the one a reader loads (a resolver over the glob, in code, refusing a superseded target) — by DE for its runner, by DA independently through `da_root`'s pair resolver, by the coordinator at the artifact. Today **nothing** does it: the only reader is DE's literal.
- **Not checkable from inside the declaration:** the rule is stated **in the artifact the rule tells you how to find**. A reader that opens the wrong file never learns the rule. So v3's `chain_head_rule` is a restatement for humans; the enforceable copy must live in the **code** (and in rule 20, where it now is). A declaration cannot bootstrap its own selection.
- **A second-order check worth having:** every declaration in `declarations/` whose `supersedes` names a present file should be resolvable to exactly one head, and **no code should name a non-head by literal** — an AST census over string constants matching `declarations/…_v\d+\.json` compared against the resolved heads. That is one instrument covering the whole directory, and it would have caught all three instances.

**R-647..R-653 otherwise:** R-652(A)'s open-field reading is accurate; R-653's account of REV 69 is accurate item by item; R-650(C) and R-645 are corrections in band made the right way. The one thing to correct is the brief's/register's "**0 leaked in 299 leaves**" (§2's last paragraph): the receipt's census is over 545.

---

# 6. Not established

- **Nothing about the day.** No sealed value was read; `n_arms_agreeing 0` is the instrument's, not the day's; this filing makes no statement about direction, magnitude or admissibility beyond the open booleans the receipt itself publishes.
- §0's inversion verdict is **structural**: no field of the shape that would invert the seal is present. It is not a proof that no function of the open fields correlates with a sealed one — §0's point 2 is exactly that caveat, stated as a design question.
- **The per-arm counts I name are OPEN fields** and are quoted only as counts. I did not compute any ratio, difference or per-decision rate from them, and I recommend no one does before the read.
- `producing_code_locatable` is verified **for 09-03 only**, and only for the runner's own file plus the declared closure; I did not re-verify the 16 closure digests against their commits.
- §3's line comparison is over the **three lines each copy holds**; both copies were made after the run ended, and neither can contain a line the journal had already lost.
- I did not run DA's or DE's suites this round; §2 and §4 come from calling the functions directly at the tip.

**Routing:** DE 100 — §1's `declarations` block (params and design by digest at import) **before the 09-04 launch**; §5's chain-head resolution in `heavy_run_form()`; §3's `output_format` field. DA 90 — §2's `n_arms_agreeing → None` when nothing was recomputed, and whether FLAGGED should be INCOMPLETE here; the pickle reader (already its own next round). Coordinator/DE — §0's point 2: write down, before day 2, whether the per-arm intervention counts are part of "the result" the seal protects. Coordinator — the 299/545 correction, and §5's directory-wide non-head census as a candidate instrument.

**Context ≈ 18 %.** Held after this filing: nothing beyond what is routed above.
