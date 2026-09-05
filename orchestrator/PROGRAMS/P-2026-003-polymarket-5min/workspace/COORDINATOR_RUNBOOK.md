# COORDINATOR RUNBOOK — P-2026-003

Written 2026-09-03T03:40Z, R-495's consolidation. **Single writer: the
coordinator.** Purpose: everything a coordinator session needs that lives
nowhere else, so that clearing or losing a coordinator session costs nothing but
the re-read. `COORDINATOR_DISPATCH.md` is the 2026-08-26 phase dispatch and is
**stale** (it names a coordinator session that no longer exists and a phase that
has been superseded); it is kept as provenance. This file supersedes it for
operations.

> **Operational override — R-531, 2026-09-04T14:25Z:** the programme is
> USER-HALTED. Do not dispatch or start heavy work from this runbook unless the
> USER explicitly resumes the programme. The collectors continue running. The
> broad `V_oracle` survey never produced a result and must not be described as
> active or decision-bearing.

> **Later user-directed override — 2026-09-04T15:27:56Z:** offline planning and
> lightweight implementation have resumed under
> `live/pm_research/plans/HARMFUL_FILL_HAZARD_TOXICITY_PLAN_V2.md`.
> This does not authorise a
> raw-tape replay, model fit, survey, grid or other heavy run. The direct
> instruction is newer than R-531 but has no coordinator-register citation
> yet. Before any dispatch, reconcile it into the register and use the v2
> gates; do not resume the stale seven-arm/fair-price dispatch below. The v2
> plan/modules/docs are currently uncommitted, so the main-tree clean-worktree
> expectation below does not apply until they are reviewed and landed.

> **Long-loop checkpoint — 2026-09-04T16:08:46Z:** Gate 0 cleared with the
> capped pipeline-only receipt
> `p003_v2_gate0_smoke__20260904T160623Z.json` (3,557 canonical actions, 458
> exact fills, 200 matched draws, all six identities true; 347,080 KiB maximum
> RSS, 11.70 s, one CPU/3 GiB). The first global-index attempt reached its job
> cap and was stopped without an artifact; the successful retry used the exact
> interval-local three-file selector. Do not read either event as an economic
> result. Overall v2 progress is 1/7; next is the synthetic acting matched
> stateful control and then at most one equally capped Gate-1 smoke.

> **Gate-1 correction — 2026-09-04T16:27:24Z:** the iid-permutation acting
> control passed 18 synthetic checks, but its capped real smoke correctly
> refused after only 1 of 4,000 proposals matched the treated realised
> side/hour action counts (200 required; 3,999 rejected; 4m44s one CPU;
> 250.5 MiB peak). Failure receipt sha256
> `ede26d60fdb425e9d760adca48e24191620c2fb15a5fe70028e124e758b1ebc9`.
> Do not increase that rejection budget. Current work is the constrained
> exact-fiber switch null with declared mixing/support falsifiers; Gate 1 and
> all later gates remain uncleared.

> **Gated-stop checkpoint — 2026-09-04T16:39:29Z:** the constrained switch
> replacement also refused. Its 5,000 proposals yielded 2,443 exact-fiber
> moves, all four chains left treatment and 400 samples held 399 distinct
> states with every identity true, but ESS was 10.53 against the declared 100
> minimum. Receipt sha256
> `cdff1a14de7ecff3351dc90da224e2d44bad43d13ef67e66934b859d777a36a9`.
> Do not extend/tune the sampler on this consumed window. Overall progress is
> 1/7; Gates 2–6 do not start. A new control estimand requires user ruling and
> a fresh prospective declaration.

> **Verification closure — 2026-09-04T16:44:23Z:** all ten v2 module batteries
> pass under one CPU/1 GiB and the parent suite passes 223/223. The post-receipt
> parent static-pin compatibility fix leaves exactly one current-tree identity
> file different from each successful smoke receipt; the receipts retain their
> own source hashes and no real rerun followed the gated stop.

> **User-resumed Gate 1c — 2026-09-05T00:58:42Z:** the user authorised only a
> genuinely different control estimand. Before any new output, the governing
> plan fixed the sequential random action-quota construction, a planted-harm
> positive control, an under-quota known-bad refusal, exact action/source/
> lifecycle identities, 200 accepted draws in at most 1,000 proposals, and
> one-CPU/1 GiB synthetic plus one-CPU/3 GiB real-smoke caps. Do not increase or
> tune either failed sampler. Do not start Gate 2, a fit, broad replay, survey,
> grid or cache rebuild. A quota-smoke pass is only the acting comparator; Gate
> 1 still needs the complete lifecycle economic ledger.

> **Gate-1c fixed support refusal — 2026-09-05T01:10:57Z:** the new module's
> 14 checks and wrapper's eight checks pass, but the one authorised real smoke
> produced only 16 accepted proposals out of 1,000 versus 200 required; all 984
> rejections were `UNDER_QUOTA`, and only 16 distinct action sets existed versus
> 50 required. Receipt
> `p003_v2_gate1_quota_smoke__20260905T010921Z.json`, sha256
> `e10dec7167a1b61a17c87b3ff0d19cd6c11692a6280035181e9cf5f1985a2ab8f`.
> Every accepted mechanics/source/quota identity was true, but the matched null
> and aggregate metric are absent. Do not widen or alter the control on this
> consumed window and do not interpret the 16 audit draws. Overall progress is
> 1/7; Gate 1 remains refused and Gates 2–6 do not start without another direct
> user ruling and prospective design.
> Final consolidated verification at 01:16:00Z: all 12 v2 module/wrapper
> batteries and the 223-check parent diagnostic suite pass under one CPU/1 GiB.
> Receipt/current identity is 11/12: only the v2 plan differs because its
> prospectively hashed declaration was extended after the run with the result;
> no named source-code file changed.

> **User-resumed Gate 1d — 2026-09-05T05:03:29Z:** before output, the v2 plan
> fixed a different finite cyclic-phase acting control. Enumerate every
> within-side/hour rotation of the complete clustered score sequence, retain
> exact actual-issued-count phases, deduplicate assignments and require at least
> 200 distinct joint phases; then sample exactly 200 uniformly without
> replacement and full-replay them. No quota suppression, force-cancel,
> proposal limit or economic selection. Synthetic cap one CPU/1 GiB; one fixed
> real smoke one CPU/3 GiB, swap off, ten minutes. Do not change or pool the
> prior three failures, and do not start Gate 2.

> **Gate-1d green / Gate-1e declared — 2026-09-05T05:14:39Z:** complete real
> enumeration found 18 BUY and 40 SELL exact-count phases (720 joint), and all
> fixed 200 uniform without-replacement full replays passed. Receipt sha256
> `8a97102cc11f5f8c94f1545deb0df75a82d6bb44a6970fd5fc4faaf723074650`;
> 99.85 s, 338,448 KiB process RSS. This clears acting support only. Gate 1e is
> prospectively fixed to those exact phases and must reconcile fills,
> spread/adverse, rho, terminal inventory, lifecycle counters and per-fill maker
> fees. Missing required monetary terms force null strategy net and Gate-1
> refusal. One CPU/1 GiB synthetic; one CPU/3 GiB/no-swap/five-minute audit.

> **Gate-1e terminal stop — 2026-09-05T05:28:23Z:** the accounting core passes
> 12 synthetic checks and the pinned wrapper 11. Real receipt
> `p003_v2_gate1_economics_smoke__20260905T052605Z.json`, sha256
> `e78fe495846cf22e834b63e04aea445cf1616563cb932a11f304d3a7ba2abd42`,
> reproduced the 5,869-row / 3,557-action population, Gate-1d SHA, 720 support,
> exact 200 offsets and every score/action identity. Baseline, treatment and
> all 200 controls passed every gross ledger identity. All 202 per-fill maker
> fee ledgers are unavailable, so all strategy nets and the matched decision
> null are null; no public taker/trade fee or zero was substituted. Runtime
> 22.98 s, process max RSS 338,556 KiB under one CPU/3 GiB, swap off. Gate 1
> refuses at its prospective stop and overall progress remains 1/7. Do not
> start Gate 2 or dispatch the historical loops below. Resumption requires a
> reliable owned-order maker-fee/ack/fill source, a prospective amendment and
> fresh data. Gate-1e checkpoint regression: all 16 then-existing v2
> module/wrapper batteries and the 223-check parent suite passed under one
> CPU/1 GiB, swap off. R-531 remains the
> latest append-only register entry.

> **Gate-1f acquisition stop — 2026-09-05T05:50:05Z:** the offline input
> contract passes 11 synthetic checks under one CPU/512 MiB. Corrected receipt
> `p003_v2_gate1f_owned_source_audit__20260905T054941Z.json`, sha256
> `c99109943de37d37d2fc8358628640214d489752e96bb8ca4f86e144bf197f47`,
> supersedes `...T054848Z.json` after correcting only its Tier-1 distiller-path
> census. The fixed owned-execution manifest is absent. Public raw data reaches
> 09-05 and Tier-1 public trades 09-02, but neither binds an owned client order,
> venue ack, maker fill and exact fee. Decision metric null; Gate 1 refused;
> Gate 2 off. Do not dispatch more public-tape work as a remedy. Await an
> authenticated offline export produced outside this repo over at least five
> post-freeze complete UTC days; never add credentials/signing/order code here.
> Latest bounded regression at 2026-09-05T09:54:58Z: all 17 current v2
> batteries pass (182 checks total), and the parent suite passes 223/223,
> sequentially under one CPU/1 GiB with swap off. No gate changed.

---

## 0. Cold start — do these in order

1. **Read** `workspace/RESULTS.md` (what has been tested and what came out of
   it), then the **last five R-entries** of `workspace/COORDINATION.md` and the
   tail of its Q-filing table, then `workspace/SEAT_PROTOCOL.md`, then
   `STATUS.yml` + `workspace/HANDOFF.md`'s READ FIRST block. Do not read the
   whole HANDOFF — it is ~11.9k lines.
2. **Re-derive the seat→pane map** (never hardcode pane ids; they change when a
   session is recreated):
   `tmux list-panes -a -F '#{pane_id} #{session_name}'`
   Sessions: `pm-co` (coordinator, yours), `pm-be`, `pm-da`, `pm-memory`,
   `pm-codex` (reviewer), `pm-de`. At this writing: `%2 pm-co`, `%3 pm-be`,
   `%4 pm-da`, `%5 pm-memory`, `%6 pm-codex`, `%7 pm-de`.
3. **Verify quiescence at the artifacts, not from panes**: `git fetch` then
   `git rev-parse --short origin/mm-research HEAD` (call separately — two args
   fails), `git status --short` (expect empty in the main tree),
   `git worktree list | wc -l` (**34** at quiescence),
   `ls data/pm_5min/derived | wc -l` (**178** tracked entries at this writing),
   and that the three collectors are alive:
   `ps -eo pid,args | grep -E 'collect_pm|collect_hf|collect_hl'`.
4. **Read each seat's held state** from git, never from its pane:
   `git -C ~/ctaNew-wt-<seat> log --oneline origin/mm-research..HEAD`. **Never
   run a git command inside a seat's worktree** — read from the main tree with
   `git -C` / `git show <sha>:<path>`; the object store is shared, so a seat's
   HELD unpushed commit is readable by sha from anywhere.
5. **Re-arm what died with the old session** (§5).

---

## 1. What is session-bound and DIES on a clear

| thing | effect of losing it | how to restore |
|---|---|---|
| the `/loop` ScheduleWakeup | **the standing coordinator duty stops** — nobody checks the 00:06Z governed verdict | re-issue `/loop <the coordinator prompt>`. **CORRECTION (R-496): there is no prompt of record.** An earlier version of this line cited "R-486's dispatch"; the register contains **zero** occurrences of `/loop`, so the prompt was never written down and died with the session that held it. Reconstruct it from §5, which is the duty in prose, and **write the prompt you use into the register** so the next coordinator inherits it |
| the commit `Monitor` on `mm-research` | seat landings no longer wake the coordinator; you fall back to polling | re-arm a Monitor on new commits to the branch |
| background Bash tasks (verification batteries) | partial results in the old scratchpad; nothing in the repo is affected | re-run; batteries are idempotent and read-only |
| the scratchpad path | a NEW session gets a new dir under `/tmp/claude-*/…/scratchpad`; old drafts survive at the old path | nothing to restore — drafts are disposable, the register is the record |

**Nothing in git, `data/`, the seats' worktrees or the systemd timer depends on
the coordinator session.** The `da-midnight-verify` timer fires at 00:06Z
whether or not any session exists.

---

## 2. Dispatch mechanics (tmux)

One complete batch per seat per round (R-377 / R-378); nothing further while a
batch is in flight.

1. Write the dispatch to a scratchpad file (one paragraph, no fences).
2. **Check the pane is idle** — a pane showing `esc to interrupt` is working;
   skip it:
   `tmux capture-pane -p -J -t %N -S -3 | grep -oE 'esc to( interrupt)?|done [0-9:]+ [AP]M' | tail -1`
3. `tmux send-keys -t %N C-u` → `tmux load-buffer FILE && tmux paste-buffer -t %N -d` → `sleep 1` → `tmux send-keys -t %N Enter`.
4. Confirm ~8 s later by capturing the pane again.

**Seat pane scrollback is not retained** (history_size 0 on some panes): a
seat's narration cannot be recovered later. Git logs and the register are the
artifact — anything a seat holds only in its pane must be pulled into a register
entry before its context is cleared.

**To interrupt a working seat**: `tmux send-keys -t %N Escape` (twice if the
first does not take), then dispatch.

---

## 3. Register mechanics (`COORDINATION.md`)

Append-only. Entries are `### R-NNN — <UTC stamp> — coordinator — …`.

- Draft in the scratchpad with the literal `SSTAMP` on line 1, then
  `TS=$(date -u +"%Y-%m-%dT%H:%MZ"); sed "s/SSTAMP/$TS/" draft.md > stamped.md`.
  **Every time written into an entry comes from a `date -u` read, never an
  estimate** (two forward-estimated times cost two corrections, R-466/R-467).
- Insert with Python, never by hand:
  `assert '### R-NNN' not in s; assert '\x60\x60\x60' not in entry; anchor=s.index('### R-<NNN-1>'); sec=s.find('\n## 6. Build-readiness'); nxt=s.find('\n### ', anchor+10); ins = sec if (nxt==-1 or sec<nxt) else nxt; s=s[:ins].rstrip('\n')+'\n\n'+entry+'\n'+s[ins:].lstrip('\n')`
- `git pull -q --ff-only origin mm-research` **first**; commit by pathspec
  (`git add $R && git commit -q -m "…" -- $R`); push.
- **After every register commit**, check:
  - placeholders on the NEW entry only —
    `sed -n '/^### R-NNN/,/^## 6/p' $R | grep -c 'SSTAMP\|xZ\|TBD'` → 0
  - exactly one ratification fence — `grep -c '^\x60\x60\x60ratification' $R` → 1
  - the ratification check passes:
    ```
    sys.path.insert(0,"/home/yuqing/ctaNew")
    from live.pm_research import de_ratification_check as C, de_admissible_windows as daw
    mask=daw.load_mask("20260901")
    sup=daw.supply("20260901", {c: list(daw._grid("20260901")) for c in mask["coins"]}, mask)
    C.check(sup,"R-419",s)["verified_for_new_run"]   # must be True
    ```
- Shell gotchas that have each cost a retry: never chain `grep -c … &&`; no
  nested backticks inside a code span; `git rev-parse --short A B` fails with two
  args; `ugrep` refuses long alternations ("exceeds complexity limits") — use
  Python.

Python interpreter for everything here: `/home/yuqing/pricer-sol/venv/bin/python3`.

---

## 4. Verifying a seat's round (the battery pattern)

Never verify from a seat's report. Execute at the tip in a **detached scratch
worktree**, then remove it **from the main tree**:

1. `git -C /home/yuqing/ctaNew worktree add -q --detach $S/wt_x <tip>`
2. Mirror the data the suites read: symlink each entry of
   `data/pm_5min/*`, `data/pm_5min/derived/*` **and its dotfiles**, plus
   `data/mm_hf`. Remove any `derived/derived` symlink that appears (`derived/`
   is tracked, so a naive symlink lands inside it).
3. `find . -name __pycache__ -type d -prune -exec rm -rf {} +` **before each**
   execution (a mutant and its cache are not the same program).
4. Run **both launchers** — `$PY live/pm_research/<m>.py --selftest` and
   `$PY -m live.pm_research.<m> --selftest` — each under
   `systemd-run --user --scope --quiet --slice=research.slice -p MemoryMax=8G`.
   DA modules print `<module> selftests: N checks passed`; BE prints `  PASS`
   lines (129 = 129 at `669ef72`).
5. Mutants: string-replace on a scratch copy with `assert src.count(a)==1`, run,
   restore bytes, compare sha16. Each must go red **by name**, zero tracebacks.
6. **Snapshot `ls -la --time-style=full-iso data/pm_5min/derived` before and
   after** and diff it — the main `derived/` is 17 GB and cannot be copied, so
   the listing is the guard. `git worktree remove --force $S/wt_x` from the main
   tree; confirm the count is back to 34.

Do not prune another seat's worktree entries (BE's transient `be-r10-c3-stale`
entries appear and vanish during its rounds).

---

## 4a. The counterfactual question (standing practice, R-509)

Before accepting **any** token as evidence — a field, a number, a citation, a
sentence beside a value — ask:

> **Could this token have been produced with the claim FALSE?**
> If yes, it is not evidence, whatever it is made of.

That single question unifies every reading failure this programme has recorded.
It is **practice, not a program**: do not try to build one checker for it,
because what SETTLES the question differs by claim, and a checker can only
consult one oracle.

| claim is about | oracle — what must be consulted | instrument | status |
|---|---|---|---|
| a **value's production** | the function's **own source**, statically: compare the value set reachable on error paths against the set reachable on success paths | codomain check (`monotone None` not `True`; `rc None` not `127`) | **instrumented** |
| **another document** | the **cited artifact** — nothing in the citing file can settle it | `entry_names_this_era`: resolve the cite AND check the cited text NAMES the subject | **instrumented, ONE table only** |
| **behaviour** | a **running system** — neither source nor documents suffice | run the behaviour and record what it did (`fd0995c`) | **instrumented** |
| a **population** | a **statistical comparison of excluded vs retained** | — | **NONE. Live instance: DE53's 4.21% exclusion** |
| a **human ruling** | the ruling's author | citation check proves it EXISTS, never that it MEANS what is claimed | **not instrumentable; escalate** |

Each oracle has a case the others are structurally blind to, so a merged
checker misses most of the surface. If a single artifact is ever wanted, build
a **router**: classify each claim-bearing token by which oracle settles it and
**refuse a token whose oracle is NONE** — a claim no oracle can settle is
precisely the one that gets believed.

**Two corollaries the register paid for.**

- **A citation check is not a claim check.** A `grep` landing on the right line
  proves the line exists, never that it says what the citer said. R-232 carries
  ZERO occurrences of `clob_v3_1`, and that cite kept 08-29 out of the race for
  three days.
- **Agreement between seats is evidence about the seats, not about the claim,
  unless the seats read different SOURCES.** Independence is a property of the
  sources, not of the readers. Three seats once produced one overstatement from
  one summary; this is R-495's non-independence error (replicated statistics)
  in a second domain (replicated citations).

---


**Two things a coordinator must never read as evidence (R-510(B)):**

- **A suite count is not coverage.** "52 checks passed", "129/129 green" report
  that the SUITE ran, never that the CODE is right. Every serious defect the
  reviewer found on 2026-09-04 was in code whose suite was green. Quoting a
  green suite as verification is the counterfactual question failing on the
  coordinator's own practice.
- **A reviewer's AGREEMENT is not a second observation.** Route every reviewer
  finding as **CHECKED** (it went to the artifact — a second observation) or
  **AGREED** (it read the same summary — the same observation, twice). Only
  CHECKED counts. The reviewer reads the same artifacts every other seat does.

## 5. The standing duty and its `/loop` prompt

The coordinator runs a self-paced loop: verify every landed filing at its
artifact, keep seats non-idle (a recorded standby counts, R-381), escalate USER
items with facts and a recommendation, and check the nightly governed verdict.
**The 00:06Z `da-midnight-verify` unit fires daily**; the next run after this
writing is **Fri 2026-09-04 00:06:00 UTC**, on the LANDED chain.

The check, each morning, at the artifacts and never from a seat report:
`systemctl --user show da-midnight-verify.service -p ExecMainStatus -p Result -p ExecMainExitTimestamp`;
`data/pm_5min/derived/da_dayverdict_<day>.json` (as_of after the run,
`write_reason` = scheduled unit run, the four forward-race conjuncts,
`content_liveness_rule` with its status, `blackout_mask_and_complement`,
`blackout_mask_artifact.status == WRITTEN`); the matching
`da_blackout_mask_<day>.json`; DA's preflight read-only; BE's scorer outcome;
then a register entry, commit by pathspec, push, and ONE PushNotification.
**The accrual itself is a USER call (R-409, R-486 (6)) — state the facts and
escalate; never accrue or refuse it yourself.**

---

## 5a. Resetting the seats

The stop / consolidate / clear / reload operation is a **skill**:
`.claude/skills/seat-reset/SKILL.md` (invoke as `/seat-reset`; add
`+coordinator` to include this session's own reset). It carries the four-question
stop dispatch, the WIP-HELD rule for uncommitted worktree edits, the harvest step
(seat pane scrollback is NOT retained), the doc-consolidation order, the
self-contained brief shape, and the guardrails. First executed 2026-09-03,
recorded as R-495.

---

## 6. Standing prohibitions (coordinator)

- **Never** run `da_midnight_verify.sh` in production mode; never set
  `DA_MIDNIGHT_MODE`; never start, install or pin a unit or timer from a seat or
  from here.
- **Never** write under `data/pm_5min/derived/` — the coordinator is read-only
  under `data/`.
- **Never** run the Phase-4 runner's `--run` against the declared OUTDIR
  `data/pm_5min/derived/phase4_diag_r459`, which must not exist until the ruled
  run.
- **Never** re-point BE's selftest control day (`21000101`) at a real closed day
  — that performs the closed-day scoring run R-486 (6) reserves for the USER.
- **Never** edit BE's `be_forward_day.py`, any seat's file, a landed or frozen
  artifact, or the addendum v2 DRAFT (it is with the USER).
- **Never** run a git command inside a seat's worktree, and do not `git add`
  MEM's or the reviewer's in-flight files in the main tree (` M` on STATUS.yml /
  HANDOFF.md means MEM is mid-round).
- Any full-day BE driver run goes under
  `systemd-run --user --scope -p MemoryMax=12G` into a **new** outdir — never
  `fwd4/`, `fwd5/`, `fwd6/`.
- **Models estimate; the USER decides** (rule 14). Freezes, admissibility, race
  admission, winner rulings, new numbers or thresholds, the addendum package,
  and the accrual call are escalated with facts and a recommendation — never
  decided here.

---

## 7. State at this writing (2026-09-05T11:38Z) — verify, don't trust

**Programme: RESUMED by the USER at 2026-09-05T14:54Z (R-534), WITHIN THE V2
SCOPE** — verification, closure and the ceiling's missing null; NO model fit,
broad replay, survey, grid, cache rebuild or Gate-2. Seats: all five reloaded
(R-533) and ALL carrying batches — USER: "make all modules work" (R-539), no
standby seats while the loop runs (BE 43 cancel-axis null; DA 51 attainable companion + 22 legs + mutation audit;
DE 64 THE RULED GATE-1e RUN at E0/E−R under the reviewer's declared bar (01edfd2);
REV self-attacking its spec + adjudicating DA's straddle; MEM 104 next).
The seat monitor (`seatwatch.sh`) is **RE-ARMED**. **The V2 Gate-1f blocker is DISSOLVED (R-535/R-536): the maker fee is
published at zero by the venue and confirmed on-chain. USER RULING R-537: Gate 1e
is RE-RUN at three fee endpoints (0, +0.07·p(1−p), −rebate) reporting invariance;
the reviewer specifies, DE runs, DA verifies. Gate 1's three sampler refusals
stand.** R-536's header time is wrong (16:04Z; true 15:51Z) — see R-537(A).

- **Tip:** `917d743` (R-532). Next register entry after R-533: **R-534**.
- **V2 line** (`live/pm_research/plans/HARMFUL_FILL_HAZARD_TOXICITY_PLAN_V2.md`):
  USER-authorised 2026-09-04T15:27:56Z, landed by the coordinator at `9b37088`
  + `120a9b3`, **TERMINALLY STOPPED AT 1/7 GATES** on a data-acquisition
  blocker (Gate 1f: no owned execution export; no public substitute satisfies
  the owned-order join). Further work needs a NEW USER RULING (plan §6.7).
  Receipts self-report NOT_FROZEN; nothing V2 produced is citable as frozen.
- **Race:** G = 4 (09-01..09-04 accrued; 09-04 by the scheduled unit at
  00:06:01Z). 09-05 open. **DIRECTIONAL, NOT SIGNIFICANCE-BEARING** (R-529(A));
  and per V2's HANDOFF the old race cannot validate the changed pipeline.
- **Economics:** RESULTS.md §0 — read its opening box first. The 701% ceiling
  and `V_oracle` are **PENDING A NULL** (two seats, independently, R-531(C)).
- **Seat worktrees:** all clean; BE/DA/REV detached HEADs sit at their own
  last landings BEHIND the tip (expected; each reported it, none fixed it under
  DO NOT START WORK — the first act on resume is `checkout --detach mm-research`).
  **DE holds `6003f40` unpushed** (+122 to `de_phase4_diag_runner.py`); V2's
  `9b37088` touched the same file at non-overlapping hunks — **predicted clean
  rebase to 226 checks, to be verified by execution first thing (R-532(F)).**
- **Worktree count at quiescence: 35 with 0–1 prunable** (the BE fixture churn
  comes and goes). Collectors: 4 alive (10d / 4.5d uptimes).
- **Open USER items:** the V2 blocker ruling; the Phase-2 winner; the causal
  incumbent operating point; G-counting (USER said manual).

