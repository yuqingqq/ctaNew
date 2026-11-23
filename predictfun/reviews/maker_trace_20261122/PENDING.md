# MAKER-TRACE-RECONSTRUCTION-v1 — pending work

## STATE 2026-11-23 13:11Z — TAKE-LOGIC v2 (ambiguity-aware; review af916aba6's next priority) — read this first

**Reviews processed:**
- af916aba6, verified at the code and data. Finding 1 holds: at |q| (0, 50], 127 ambiguous vs 109 classified reduce event-seconds; above 200, 19 vs 325. `chain_take.combine` makes any variant disagreement `timing_sensitive`.
- Earlier: 6bb458b30, 6ddad3533, 8f2a2e753, 567f741d9.

**Plan:** `take_logic_v2_contract.md` (≤ 0.025 h all-in; ledger 1.8164 of 1.90 before it).

**Builder:** `tools/take_logic_v2.py`.

**Next:** freeze, then the full run, then `v2/report_logic_chains_v3.md` (with the findings 2–5 corrections), then commit + push, then ONE review, then the watcher.


## STATE 2026-11-23 12:54Z — ROUND COMPLETE: LOGIC CHAINS v2 (correction 6bb458b30 + TAKE-LOGIC v1) — review pending — read this first

**Receipt:** `v2/report_logic_chains_v2.md`.

**Runs:** `mt_x_cc_post_v1`, `mt_x_cc_pt_v1`, `mt_x_tl_v1` (all rc 0).

**Review:** ONE Codex review of 6bb458b30..HEAD queued; `scratchpad/review_watch.sh <HEAD>` armed; auto-check cron 60c3d98d.

**When the review lands:** list it under "Reviews processed", verify its findings, implement its ONE next priority. Stop and ask the USER for any new resource; the task cap is 1.90 (USER, 2026-11-23).


## STATE 2026-11-23 11:50Z — LOGIC-CHAIN CORRECTION v1 (review 6bb458b30's next priority) — read this first

**Reviews processed:**
- 6bb458b30 (findings 1–7 verified at the code: union `post_lo` / B_constrained, take-side 'against', fill-anchored family omitted). Being implemented.
- Earlier: 6ddad3533, 8f2a2e753, 567f741d9.

**Plan:** `chain_correction_contract.md` (budget ≤ 0.025 h all-in; ledger 1.7989 of 1.85 before it).

**Agents:**
- A: `tools/chain_correct_post.py` (§1 / §4 / §5);
- B: `tools/chain_correct_pull_take.py` (§2 / §3 / §5).

Each builder has ≤ 12 CPU s.

**TAKE-LOGIC v1** (USER direction, 2026-11-23: solid proof of the take logic — inventory, time left, one-off vs repeated):
- contract committed 3d9c6b082;
- USER cap 1.85 → 1.90 (line 5.25 unchanged);
- builder C: `tools/take_logic.py` (≤ 20 CPU s; pilot into `v2/chains/take_logic_pilot_20261024_r1`).

The full run follows the freeze, bundled with the correction round under ONE combined review.

**Next:** freeze-commit, then one real run of each into `v2/chains/correction_v1/`, then the superseding `v2/report_logic_chains_v2.md`, then commit + push, then ONE Codex review, then arm `scratchpad/review_watch.sh <SHA>`.

**Auto-check:** cron 60c3d98d (:07 / :27 / :47) replaces 0a70a441.


## STATE 2026-11-23 11:13Z — LOGIC-CHAIN PACKET: USER raised the packet to 0.11 h (task about 1.82 of 1.85) — read this first

**Done (tests pass):**
- `chain_context` (173 / 173);
- `order_estimator` + contract (75 / 75; pilot r2 OK);
- `chain_take` (59 / 59; pilot OK).

**Running:**
- `chain_post_fill`: cost cut, then selftest2 + pilot2, which must give identical outputs;
- `chain_pull` v2: selftest2 + pilot2 into a new folder.

**Next:** one freeze commit (the contract + 5 tools + selftests), then full runs (estimator, take, pull, post_fill) under `MT_RUN=v2`, then one combined receipt and ONE Codex review.


## STATE 2026-11-23 08:31Z — LOGIC-CHAIN PACKET IN PARALLEL (USER raised the packet ceiling to 0.08 h, task cap 1.85) — read this first

**Five agents are running.** Resume any that a rate limit stopped.

| Agent | Files | Slot |
|---|---|---|
| 1 | `order_estimator_contract.md` + `order_estimator.py` (detailed episodes) | claude4 / claude3 |
| 2 | `chain_context.py` (shared helper) | claude3 |
| 3 | `chain_post_fill.py` | claude2 |
| 4 | `chain_pull.py` | claude4 |
| 5 | `chain_take.py` + the action enumeration | claude |

**Next:**
1. Integrate the pieces.
2. Freeze-commit the contract and all tools.
3. Run every tool under `MT_RUN=v2`.
4. One combined receipt and ONE Codex review.


## STATE 2026-11-23 08:16Z — REVIEW 6ddad3533 ANSWERED (f1e75e6f9); USER-directed ORDER-ESTIMATOR / decision-logic packet BUILDING — read this first

**Plans:** 7d71ca40c, 72c5e5841, 5cb2f0229.

**Running:** the agent building `order_estimator_contract.md` + `tools/order_estimator.py` + its selftest, with one pilot. Resume it with SendMessage if a rate limit stopped it.

**Next:**
1. Freeze-commit the contract, tool and selftest.
2. `MT_RUN=v2` full run (≤ 96 cases).
3. Receipt, then ONE Codex review.

**Budget:** ≤ 0.035 h all-in, including a 0.010 h review reserve (ledger 1.708 + review 0.010 → 1.718 before this packet; cap 1.85).


## STATE 2026-11-23 07:57Z — MATCHED-GAP RESULTS COMMITTED (55625bfa6); CODEX REVIEW QUEUED; nothing to run — read this first

**Auto-continue:** reply "waiting" unless a `Predictfun-Review-Only` commit appears after 55625bfa6. Then read it, verify it at the code, and report to the USER. Run nothing until the USER's go (11-14 rule). CPU is about 1.71 of 1.85.


## STATE 2026-11-23 07:47Z — MATCHED-GAP run done (c0f237ffe; v2/matched_gap/) but its PRIMARY carry view is DEFECTIVE — read this first

**Defect:** carried lots are not in the FIFO, so the carry view equals the zero view. The agent is fixing it (resume it if rate-limited).

**Do NOT report the carry-view numbers.** The zero-start view is valid.

**Next:**
1. Re-freeze.
2. Run with `--out matched_gap_r1`.
3. Receipt, which discloses the defective first output (kept, not deleted).
4. Commit, then ONE review.


## STATE 2026-11-23 06:58Z — USER GO for MATCHED-MARKET-TURNOVER-GAP-v1; contract frozen (46cd3cc56); tool agent building — read this first

**Running:** the agent building `tools/matched_gap.py` + selftest. Resume it with SendMessage if a rate limit stopped it.

**Next:**
1. Freeze-commit the tool.
2. `MT_RUN=v2` run via run_light (`tools/matched_gap.py run`).
3. Receipt, then commit, then ONE Codex review.

**Budget:** ≤ 0.10 h, including review (spent 1.705 of 1.85).


## STATE 2026-11-23 06:49Z — REVIEW 8f2a2e753 ANSWERED (ed94b9d16); WAITING FOR THE USER'S GO on MATCHED-MARKET-TURNOVER-GAP-v1 — read this first

**Nothing to run.** Auto-continue: reply "waiting for the USER's go" unless the USER has replied.

**The proposal:** a saved-data comparison of our CURRENT_ASYNC_FAST10 ledger vs the ex-ante top 3 / top 10 on the same 19 BOOK markets and intervals (≤ 0.10 h including review; fits 1.85 / 5.25). Per the 11-14 rule, it does not start without the USER's go.


## STATE 2026-11-23 06:08Z — WAITING FOR THE CODEX REVIEW of 567f741d9..0e21193f2 (queued); nothing to run — read this first

- **Done:** v2 (checker 39/39), the decision paths, the completed closes and the identifiability control, all committed and pushed.
- **Auto-continue:** reply "waiting" unless a new commit with `Predictfun-Review-Only: true` appears after 0e21193f2. Then read it, verify it at the code, and report to the USER; start nothing that needs new resources (CPU 1.68 of 1.85).


## STATE 2026-11-23 01:47Z — AMENDMENT 3 IN PROGRESS (USER approved: lifecycle fix + top-3 decision paths; cap 1.85 h / line 5.25) — read this first

**Running agents** (resume with SendMessage if a rate limit stopped them):
1. **lifecycle builder:** verdicts and lifecycle-side fixes for the 7 failing checks;
2. **BTC grid builder:** `pf_mt_btc.py`;
3. **decision-path builder:** `pf_mt_decision.py`.

**Done:** the v2 root is set up (`maker_trace_v2_20261123`).

**Next:**
1. Send the checker author the checker-side fixes from the lifecycle verdicts.
2. `MT_RUN=v2` chain: life for every day, then truth, econ, report, checks.
3. `pf_mt_btc.py day` for Oct 24–27, then `pf_mt_decision.py run`.
4. One combined receipt and review.

CPU: tools/cpu_ledger.py, cap 1.85.


## STATE 2026-11-23 01:19Z — RESULTS COMMITTED (1af85c56); WAITING — read this first

**Done:**
- results receipt `report.md` (claims in `claims.json`), pushed;
- ONE Codex review queued (message 01a205fc, range 5927971d..1af85c56).

**Nothing to run.** Auto-continue: reply "waiting" unless a new review commit appears.

**When a Codex review commit** (trailer `Predictfun-Review-Only: true`) lands on the local branch after 1af85c56:
1. read it;
2. verify its findings at the code;
3. report them to the USER together with the resource question.

The open item is the 7 failing lifecycle / book-size checks (R3 provisional). Fixing them and re-running the lifecycle, report and checks needs about 0.1 h beyond the remaining 0.05 h reserve (spent 1.341 of 1.45). That is a **USER decision**: start no run that needs it.

# (older notes below)

2026-11-22, session aea5bfb9. Nothing here is frozen or committed except the contract (5927971d).

## STATE 2026-11-22 20:25Z — FROZEN (this file's commit is the freeze commit) — read this first

**Freeze:**
- All modules, the report module, the checker, the tools, the declaration and the manifest are committed.
- End-to-end run 7 passes: 39/39 checks, 76/76 falsifier cases.
- Spent 0.503 h of 1.35.

**Next:**
1. `predictfun/reviews/maker_trace_20261122/launch_mt.sh extract 20261024 claude` (prefix, about 8 min wall).
2. `launch_mt.sh life-prefix`.
3. `launch_mt.sh project` (write-once gate).
4. If `run_rest` is true:
   - extract 20261023 / 20261025 / 20261026 / 20261027 on slots claude..claude4, then 20261108 when a slot frees;
   - `life <day> <slot>` for each day as its extract finishes (Oct 24 too);
   - `truth`, then `econ`, then `report`, then `checks`.
5. Results commit and ONE Codex review.

If the projection does not fit, stop and ask the USER.

## CURRENT STATE (15:20Z, superseded)

**Landed and tested** (all uncommitted, untracked):
- extract (selftest 100/100);
- econ (1,010 assertions);
- truth (v2 table d42c3946, per-run bands; lowercased signers: 59 checks, 43/43 mutants);
- report (323);
- lifecycle first batch (145/145);
- checker x_lo_hard alignment (73/73).

**Contract amendment 2 and the declaration updates are written.**

**In flight** (agents; resume with SendMessage if a rate limit stopped them):
- ~~lifecycle~~ **LANDED (`mt_life_fix3`):** book evidence sets `removal_hi` and keeps both `exhausted` and `cancelled_or_expired`; `size_exact` comes only from the S_FIFO rules. Selftest 145/145; the simulator shows 0 of 23,664 observed / constrained claims failed; the reverted-rule mutant is caught.
- **checker:** fixes for e2e items 4–13 (rehearsal tag `mt_checks_rehearse_e2efix`).

**Next, in order:**
1. When both land, tell the e2e agent to do its LAST reserved run (`tools/rehearse_e2e.py`, `PF_SLOT=claude3`). Every mt_checks check must pass and the falsifier must refuse every case.
2. `tools/build_manifest.py` (write-once).
3. Freeze commit: all `pf_mt_*` code + selftests, receipt (contract, declaration, manifest, launcher, tools, mt_checks, `rehearse_*`), with HANDOFF / STATUS hunks via `scratchpad/stage_edit.py`. Then fetch and push.
4. `launch_mt.sh extract 20261024 claude`, then `launch_mt.sh life-prefix`, then `launch_mt.sh project`.
5. If `run_rest` is true: extract 20261023 / 25 / 26 / 27 / 20261108 on the claude..claude4 slots; `life` per day; `truth`; `econ`; the report freeze commit; `report`; `checks`.
6. Results commit and ONE Codex review.

**CPU:** about 0.47 h spent of 1.35, with about 0.8 h projected for the run.

## Update (14:35Z, after the usage reset)

- All six stopped agents have been resumed; none had edited anything before the stop.
- The lifecycle batch above has been DISPATCHED.
- The `project.py` and `launch_mt.sh` fixes below are DONE, and the launcher's refusal paths are checked.
- Auto-continue: recurring cron job `0a70a441` (:17 / :47). It resumes rate-limited agents or the next pending step; it is session-only and expires after 7 days.

## Done

- **Built:** the modules `pf_mt_{common,extract,lifecycle,econ,truth}.py`, their selftests, `mt_checks.py`, and the tools (`cpu_ledger`, `project`, `build_manifest`, `rehearse_checks`), plus `launch_mt.sh`, `declaration.md` and contract amendment 1.
- **USER cap:** 1.35 core-hours, line 4.40 → 4.85. About 0.34 h has been spent on tests (`tools/cpu_ledger.py`).
- **Pre-run code review:** 5 reviewers, all reports in. Fix batches were dispatched to econ, extract, truth, checker and report, together with a new end-to-end rehearsal agent. **Those agents were STOPPED when the limit was hit; their edits on disk are partial and untested.**

## Lifecycle fix batch (review received, NOT yet dispatched)

**B1. Capacity contradiction never demotes classes.**
- For orders with an H3 violation, skip or demote the capacity posting bound, `tight` and `book_gone`, with a status.
- Treat violated (frame, cell) pairs as unknown cells for every other order.
- Assert the classes in selftest case f.

**B2. The `lo` rule is validated only on a simulator with settle ≤ match + 30 ms.**
- Simulate settle lags of 0–1,000 ms, out of order, with same-frame refills and pending clears, and require lo < match.
- If that fails, give `_hard` only the MAX_LAG lower end and keep the evidence `lo` as a scenario refinement.

**B3. Re-run the mutant campaign on the final code** (about 60 CPU s) and cite the log.

**Other fixes in the same batch:**
- **Persistence:** `observed` only if `persist_lo ≥ 0` (seams review #3).
- **Size upper bound:** investigate `x_size_ub` (simulator trace o412: ub 6 vs true 8).
- **Non-blocking:**
  - `hi_from_receipt` fills get `lo = −1`;
  - count empty frames inside [ws, we];
  - set `valid` after the record flags are merged;
  - align the `missing_evidence` span start (max fa);
  - align T1/T2 tie handling;
  - statuses for scenario contradictions (`ahead_lo` > `ahead_hi`; FIFO_LO `post_lo` after the first fill).

## Launcher / gate fixes (integrator, not yet done)

**`project.py`:**
- read `ex['rest']['hours']` / `ex['ws']['hours']`;
- project the prefix from the ledger's process CPU (per-hour sums run about 10% low);
- docstring caps 1.35 / 4.85.

**`launch_mt.sh`:**
- a retry suffix on EVERY stage;
- a run_light REFUSED must not consume the write-once tag;
- `FREEZE_CHECK=0` only for `selftest`;
- `life` refuses when there are no depth markets.

## Next steps (in order)

1. Resume the stopped agents with SendMessage (same batches) and dispatch the lifecycle batch above.
2. Fix the launcher and `project.py`.
3. Run the end-to-end rehearsal (`tools/rehearse_e2e.py`) on the real builders, then all selftests.
4. Build the manifest, then the freeze commit (code + state hunks via `stage_edit.py`).
5. Prefix (extract Oct 24 + one lifecycle market), then the write-once projection, then the rest.
6. Report module freeze commit, then report, then checks.
7. Results commit and ONE Codex review.

**Deferred (next round):** decode the cached Oct 23 on-chain calldata (original sizes and expiry for all makers) as an all-maker validation.
