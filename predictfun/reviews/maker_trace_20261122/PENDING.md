# MAKER-TRACE-RECONSTRUCTION-v1 — pending work

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
