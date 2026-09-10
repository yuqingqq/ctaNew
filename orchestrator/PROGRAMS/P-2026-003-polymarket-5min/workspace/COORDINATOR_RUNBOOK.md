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
- **BEFORE every register insertion**, `git status --short -- $R` must be EMPTY (R-661, MEM 187's landing note): the coordinator commits the register by pathspec, so a seat's uncommitted row in the shared tree lands INSIDE the coordinator's commit — rule 21's third landing form, the silent one; nothing is lost but attribution is wrong and a half-written row could be committed. A dirty register HOLDS the entry in the scratchpad until the seat's commit lands; never `git add` a file another seat has open. THE HOLD IS ONLY A HOLD IF THE COMMIT IS CHAINED ON THE INSERTION'S EXIT (`python3 … && git add … && git commit …`, R-662): at 14:46Z the insertion refused correctly and the shell went on to add and commit anyway — `f0cee29` carries MEM's row under an R-662 message with no R-662 in it.
  THE CHAINED FORM IS NECESSARY AND NOT SUFFICIENT (REV 73 §3.1, R-665; three cells driven in a scratch repo): (1) commit by FILE pathspec only — a directory or glob pathspec sweeps another seat's file; (2) the clean-check must cover EVERY path in the commit (the runbook and the register together are two paths); (3) the check→insert→commit window is a TOCTOU the pre-check cannot close — a seat's append between the check and the commit lands inside it — so the landing carries a POST-CONDITION: after the commit, the register's diff in that commit is exactly the inserted bytes (added-line count and content), the commit touches ONE path, and no foreign `| Q-` row is in it; on a mismatch, `git revert` and re-land — never rewrite history. The runbook's `Next register entry` line lands as its OWN commit with its own post-condition (REV 74 §4, R-667: a second path in the same commit got no assertion).
- **A DRIVE'S RESULT IS READ BEFORE THE SENTENCE IS WRITTEN (R-682).** An exception is a refusal only when it is the guard's exception by name; a `TypeError`/`AttributeError` from the caller is the probe. The coordinator wrote five refusals into R-681 from a probe that had called the function without its arguments. Paste the drive's output into the entry verbatim; never paraphrase a result you have not read.
  ENFORCED AT THE LANDING (R-690): the coordinator's landing script refuses an entry that claims a coordinator drive ("driven by the coordinator", "re-drove", "the coordinator drove/ran") without a fenced output block. Two prose claims of drives that had not run landed in one day (R-681, R-689); the check is in the tool now, not in memory.
- **After every register commit**, check:
  - placeholders on the NEW entry only —
    `sed -n '/^### R-NNN/,/^## 6/p' $R | grep -c 'SSTAMP\|xZ\|TBD'` → 0
  - exactly TWO ratification fences (R-419's block and the USER's 08-29 ratification at R-502; the count moves only with a new ratification, and the entry adding one updates this line) — `grep -c '^\x60\x60\x60ratification' $R` → 2 (was written as 1 before R-502; corrected 2026-09-06 at R-643's check)
  - the ratification check passes:
    ```
    sys.path.insert(0,"/home/yuqing/ctaNew")
    from live.pm_research import de_ratification_check as C, de_admissible_windows as daw
    mask=daw.load_mask("20260901")
    sup=daw.supply("20260901", {c: list(daw._grid("20260901")) for c in mask["coins"]}, mask)
    C.check(sup,"R-419",s)["verified_for_new_run"]   # must be True
    ```
- **BEFORE ANY REVERT, RE-LAND OR PUSH IN THE SHARED TREE, RE-READ THE TIP (R-760):** `git fetch` and
  `git log --oneline <the sha you last read>..origin/mm-research`; if it is non-empty, read those commits'
  titles before acting. The coordinator reverted a seat's commit three minutes after reading the tip while
  the tree had moved twice — another seat had already repaired the same defect — and had to revert its own
  revert (06:54–06:56Z 09-07). A state read once is not a state.
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

**A HEAVY RUN IS NEVER A CHILD OF A TOOL SHELL (R-628).** `systemd-run --scope` from a Claude Code Bash task dies when the harness stops that task (the same mechanism that killed the coordinator's shell waiters earlier today). Heavy runs launch as transient SERVICES (`systemd-run --user --unit=… -- flock -n <lock> <cmd>`, no `--scope`); a GO names that form. **The OUTSIDE diagnostic is the unit's `MainPID`'s PPid, measured never argued (REV harvest, R-642; corrected at R-646 per REV 67 §3.2): under a transient service `MainPID` is `flock` and its parent is `systemd --user` (pid 1004 here — a literal that a user-manager restart would change); the PAYLOAD is a grandchild whose PPid is `flock`, so the test applied to the python process reads wrong. The DECIDABLE property needs no parent: the process's own cgroup leaf suffix from `/proc/self/cgroup` — `.service` admits a real day, `.scope` refuses it (DE 96's `assert_launch_form_at_runtime`; the bare tool shell sits in the harness's `run-u*.scope` and refuses). Never demonstrate any of it with `kill -TERM -<pgid>` — that reached the harness's own tree twice.** **The journal is NOT the record (R-641):** copy a journal number into an artifact at the moment of reading with the retention state named; a receipt carries its own journal lines at emit.
- **Exit codes are read through `producer_exit_maps_v<N>.json` (R-709).** 75 = refusal; a declared code = a verdict by name; anything else = UNMAPPED, and an UNMAPPED code does not satisfy a GO condition -- wait for the producer's block by pair. A relaunch under one GO needs four conjuncts (rule 20): nothing written or superseded by pair; lock free + not-found; a unique unit name per launch with the id in a record; code byte-identical or the change reviewed first (a measurement-only addition: the coordinator reads the diff; anything touching a check or a verdict: REV first).
- **A landed declaration version is immutable (R-711).** Chains are versioned by ONE new file per landing; a diff showing `M` on an existing `_v<N>.json` is a refusal. `scripts/declaration_immutability.sh <dir>` flags FORKED_BY_EDIT; run it in the landing battery. A seat versioning a shared chain re-reads the head AFTER its rebase/copy and before its commit. The measurement-only fast path is a PREDICATE (REV 80 §2.2): the diff touches no line containing `ok(`, `refuses(`, `raise`, `assert`, or a function in a check's call path -- state that it was evaluated; otherwise REV first.
- **The status you act on must come from the command you mean (REV 82 §4, R-714).** `cmd | tail -1 && next` tests `tail`; `rc=$?` after an `if` tests the `if`; `HEAD..origin` is not a membership test. In a landing chain never put a pipe before `&&`: capture, test, then trim -- `out=$(git pull --rebase origin mm-research 2>&1) || { echo "$out"; exit 1; }` and print or trim `$out` afterwards. Not `set -o pipefail` (it changes every pipeline in a file that also greps).

**NEVER TOUCH A SEAT'S WORKTREE WHILE THE SEAT WORKS (R-627).** Symlink restores, refreshes and file restores in a seat worktree happen only when that seat is idle and told; read every script's output before reporting it done (two coordinator errors on wt-da, 12:03–12:04Z).

**EVERY LAUNCH IS A SEPARATE GO (R-620).** After any refusal of a real run, the fix is reviewed BEFORE the next launch; a seat does not fix-and-relaunch inside one batch. The coordinator issues GO per launch, naming the commit the run executes from. A launch made without it stays up only by the coordinator's explicit ruling, and a NO-GO from the reviewer stops it. **A GO CONDITION ON ANOTHER RUN'S COMPLETION IS SATISFIED BY ITS ARTIFACT AND ITS JOURNAL BY InvocationID, NEVER BY A UNIT READING THAT MAY BE VOID (R-659, REV 71 §4.2):** a unit launched without `RemainAfterExit` is collected on success, so "finished successfully" and "never existed" read the same; from declaration v3's form onward the five fields plus a non-empty InvocationID make the unit reading admissible again. One GO naming a condition in advance is one launch (R-656(C)); the hazard R-620 names is the fix-and-relaunch inside a batch. **A PHASED GO ACKNOWLEDGED AND NOT STARTED (R-675):** a seat may report "I'll take phase N next" and end its turn (DE, twice on 2026-09-06); the seat monitor shows it idle with the GO consumed. A one-line "proceed now with phase N as issued" is not new work and does not break the one-batch rule; send it, count it in the entry.

**THE THREE-PATH READ CONSUMES (reviewer's harvest, R-600).** `be_race_reader.read()` handed the three REAL feed paths parses them and consumes the race days; only the five-path call refuses (on the absent pins). Nobody "just checks the reader works" on real paths — synthetic feeds in scratch only. BE 59 gates `--open` on the coordinator's explicit GO so this cannot happen by accident.

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

## 7. State at this writing (2026-09-06T02:2xZ, R-541) — verify, don't trust

**⚠ STRUCTURAL FACT, learned three times — WITH A REMEDY SINCE R-552: the seat
monitor's notifications reach the coordinator ONLY inside a running turn, but a
Background Bash (`run_in_background`) that EXITS when a seat idles or origin moves
RE-INVOKES the coordinator — BUT the harness stops such shells within seconds to a
minute of arming (five kills, R-566..R-569). THE MECHANISM THAT SURVIVES is the
harness-native `Monitor` tool with `persistent: true`, a poll loop emitting one
line per seat transition or new origin commit (R-569(B)); its events wake the
coordinator between turns. Arm it once per session; re-arm only if reported stopped. The
loop stalls at the coordinator whenever the USER is not prompting** — 40 min on
09-05, then TEN HOURS overnight with five finished seats idle and G reached. A
coordinator session must be prompted, or the standing duty is a fiction.

**Race: G = 5 REACHED** (09-05 accrued at 00:06:01Z, four conjuncts, CONTENT_LIVE,
mask WRITTEN). DIRECTIONAL by USER ruling. **All five race days carry a sealed score (BE 44/45, R-549(F)); 09-01 and 09-02
were OPENED earlier under the interim read and are CONSUMED (RESULTS §3).** The
scorer is manual. A declared read (`be_read_declaration.py`) precedes any opening — AND THE READER MUST
BE REVIEWED ON SYNTHETIC FILES FIRST: the first reader (BE 50) computed a degenerate
statistic that was not the declared estimand (R-581); opening with it would have
consumed all five days. **UNSEAL
HELD until all five are sealed, then opened in one act** (coordinator's or USER's
act; R-544(B)).

**The midnight unit is RED EVERY NIGHT by design collision**: the open day's mask
refuses (correctly) for want of windows and the script classifies that as
INSTRUMENT FAILURE, rc=4. Its exit status is meaningless as a health signal until
DA reclassified it (DA 52) and the coordinator INSTALLED the fixed unit at 02:24Z 09-06 (R-542). Read the unit's own log
(`data/pm_5min/derived/.da_midnight_verify.log`), not `systemctl`.

**The ruled Gate-1e run (R-537) is DONE and verified at the receipt: INVARIANT
true, MATERIAL false** — the fee moves nothing; the treatment is worse than 94% of
its controls at both endpoints; Gate 1's three sampler refusals stand.

- **Tip:** see `git log`. Next register entry after R-889: **R-890**.

**THE GATE-1 READ ORDER (R-591, from DA 64's finding).** DE's sealed day receipts strip every economic field (D_E0, Z, p, null mean/sd, draws summary) until the seal opens at the ruled `read_not_before` 2026-09-09T00:06Z (G = 6). Before that, DA's verifier (`live/pm_research/da_gate1_day_verdict.py`) can verify population, statuses, seed and provenance only, and says so (`IS_A_VERIFICATION_OF_THE_ECONOMICS: false`). At the read: (1) DA's verifier on each OPENED day receipt against the day book, EXACT comparison, before any number is quoted; (2) the runner's own read; (3) the reviewer's filing; (4) the coordinator reports the direction. A verdict quoted before step 1 is unverified by construction. Four limits (REV 45 §4): the verifier never verifies D_E_MINUS_R (not on DE's surface), the book's construction (only its bytes), an error in the declaration itself (two implementations of a wrong spec agree — R-235's known limit), or a sealed receipt's economics. Its real-day path is NOT BUILT as of R-594 (DA 67).

**CODE FROZEN DURING A RUN (rule 22, R-603).** Never dispatch a code change to a module while a run that imports it is in progress from the same worktree; check `ps` for the run's cwd before dispatching. A run's receipt that misnames its producer is superseded in band with the digest that ran, attested by independent records.

**THE SEAL-OPEN BAR IS A PREDICATE (R-602).** The Gate-1 aggregate read opens when BOTH hold: clock ≥ 2026-09-09T00:06Z AND all six ruled days' sealed day receipts exist (verified by digest). Declared by DE 86 (params v6 / design v14) and completed by DE 87 (params v7 / design v15, R-604): eight conjuncts — the clock; the six RULED days; each receipt at its LANDING digest (DA's pre-read artifact is the record); ≥ 1 admissible arm per day; the ledger verdict; locatable producing code; the HORIZON 2026-09-09T12:00:00Z → G = 5 directional; the params field required by both implementations. The pipeline runs CONTINUOUSLY — one heavy run at a time — through 09-09 ≈ 03:00–04:00Z; the coordinator's overnight duty is to keep the lock busy: inputs → book → smoke per day as each completes. The user may overrule (the alternative is G = 5, directional only).

**NO FIXTURE INSIDE A REAL DAY (R-609, REV 53 §0).** The first smoke died after 84 minutes because `_main_day` ran the in-run battery's fixture day inside the real day's process and compared the fixture's 700 MB budget against the process-wide `ru_maxrss` the real day had set to 2,426 MB. Before any GO: confirm (at the code, not the report) that no fixture check runs inside a real day's process, or that every fixture budget is a highwater DELTA; and that the receipt's battery field says what actually ran. A traceback printed after a mid-run landing shows the wrong source lines — trust the message only.

**THE PEAK-STAGE REFUSAL (REV 47 §2.3, written before DE 84's receipt exists).** The runner's peak-stage predicate (the argmax over per-stage highwater deltas vs the `[DECLARED PEAK]` marker in `DAY_STAGES`) is a real test that can refuse a correct-looking day. If a real day refuses on it, the artifact stays unwritten and the response is a DECLARATION ACT: re-declare the peak stage from the measured deltas (moving the marker moves declaration and predicate together), record why, re-run. Never widen the predicate, never raise the cap (R-174). The refusal costs one run; a predicate loosened after seeing costs the claim the ceiling rests on. If the outcome is close, the headroom is S0's ~290 MB double read (hash from the same buffer), not the predicate.


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

## 7b. State at this writing (2026-09-06T20:59Z, R-733) — verify, don't trust; supersedes §7 for routing

- **Tip:** see `git log`; the register's next entry number is the line in §7 that `scripts/land_register_entry.sh` maintains.
- **Coordinator tools, TRACKED (the scratchpad dies with the session):** `scripts/land_register_entry.sh <entry.md> <N> <msg-file>` (R-entries: dirty hold, drive-claim fence guard, file pathspec, post-condition, the runbook's next-entry line, capture-test-trim push, `Landed-By` trailer); `scripts/land_register_row.sh '<ids-regex>' <msg-file> [--dry]` (every seat's rows); `scripts/declaration_immutability.sh <dir> [--base] [--falsify]`; `scripts/wt_refresh.sh <worktree>`.
- **Seats:** BE %3, DA %4, DE %7, REV %6, MEM %5 (derive from `tmux list-panes -a`, never trust this). DE reset 18:56Z and MEM reset 19:56Z on 2026-09-06 (both reloaded from files; the briefs follow the seat-reset skill). REV about 70 % context after REV 87 — a reset is due within a few reviews.
- **Chain heads (resolve through `live/pm_research/declaration_chain.py`, never a filename):** design v25 (the merge of v24 + five tips), params v15, `heavy_run_form` v4 (`memory_peak_reading`), `producer_exit_maps` v4 (seven producers declared), `be_daybook_structure` v3 (three books), `be_race_read_declaration` v5 (the SECOND read: 09-06..09-09, G = 4, floor 0.125, result family `be_race_read2_result`), `be_race_read_feed_pins` v1 (v2..v5 due one per close), `da_anti_echo` v1.
- **Sealed days and heads:** 09-03 `…140155Z.v2.json`; 09-04 `…171144Z.v3.json`; 09-05 `…180043Z.v3.json` (DE 109's corrections: `n_days_complete_TRUE_AT_EMIT_RECONSTRUCTED` 1/2/3, the emitted `n_days_complete` untouched). The read gate: `n_days_complete` at least 6 at 2026-09-09; three of six.
- **The first race read (09-03..05, G = 3):** `be_race_read_result_v2.json` (the `.v2`, BE 73); only `day_signs` (09-03 negative, 09-04 positive, 09-05 negative) and `permutation_floors.neither_clears_0_05` = true are quotable; the days are CONSUMED; the reader must never be `--open`ed for them again.
- **Tonight's chain (after 00:00:00Z 09-07) — THE DAY'S FIRST ARTIFACT IS DA'S BLACKOUT MASK (`da_blackout_mask_<day>.json`, `da_blackout_mask.py --day <day>`, light, at the close; BE's fragment refuses by name without it — learned at 00:02Z 09-07, R-734):** then BE 72 — fragment → tape → book → structure v4, each a separate enumerated launch via `be_heavy_run.sh` (the peak of record SAMPLED WHILE ALIVE — `--capture` cannot read a released leaf), one commit; THEN the day's SEALED FEED for the race read is a step of its own — `be_forward_day` for the day (heavy, on the lock; the pins vN+1 carry ITS output's {path, sha256}, never the book's — learned 01:33Z 09-07, R-736; v5's `pins.taken_by` misnamed the producer, v6 corrects it) (a draft GO may exist in the coordinator scratchpad as `be72_go_draft.txt`); then DE's 09-06 day run as ONE GO (DE 112 rehearsed it: `rehearse_smoke('2026-09-06')` blocked only on the book and its receipt); then DA's 09-06 pre-read (light, then the heavy open-book half as its own GO); then REV 88 on the chain; MEM sweeps each entry.
- **THE DAY-CLOSE CHAIN AS A PREDICATE (REV 88 §4, R-741) — each step: artifact / produced-by → consumed-by / refusal code when absent.** (0) the deploy pin is CURRENT for the nightly units — `da_midnight_deploy_pin_v<N>.json` / DA → the drift guard / rc 7 `DEPLOY_DRIFT` (nothing ran; catch-up day). (1) the blackout mask — `da_blackout_mask_<day>.json` / DA's nightly unit at D+1 00:06Z (or by hand) → BE's fragment / `AdmissibleWindowsRefused` (rc 1). (2) the closed-day verdict — `da_dayverdict_<day>.json` with `day_closed_calendar: true` / the same unit → BE's forward day / `day_closed_calendar=False` refusal. (3) BE fragment → tape → book → structure v<N+1> — `harmful_exposure_rows_v3_gate1_<day>_btc.json`, `phase2_state_tape_gate1_<day>_btc.json`, `be_daybook_<day>_btc.pkl` + receipt, `be_daybook_structure_verification_<day>_btc.json` / BE → DE's day run and DA's open-book half / each a named refusal in BE's exit-map block. THEN THE TAIL BRANCHES: (4a) the race-read branch — `be_forward_day` → the SEALED FEED (`be_forward_day_receipt_<day>.json` + the feed) → `be_race_read_feed_pins_v<N+1>` / BE → the second read at its horizon (2026-09-10T01:00Z, user-gated); (4b) the Gate-1 branch — DE's day run → `p003_de_gate1_day_run_<day>_SEALED__<stamp>.json` / DE → DA's pre-read + open-book half → REV. "The chain completed" means ONE branch completed; say which. A missing artifact reads as "the step was never reached" only if the step before it has its artifact.
- **The former open USER items — RULED by the coordinator at R-745 (2026-09-07T03:3xZ) on the user's instruction "clear the issues on my side first", each disclosed for overrule by one line:** (1) the SECOND race read IS authorised under R-531 (G = 4 directional is the race's standing design, not a one-shot; each read its own test, never pooled) — pins at each close, the read opens at its horizon 2026-09-10T01:00Z as a separate GO; (2) the overnight cadence stays SESSION-DRIVEN (every launch a separate GO; this session holds the day-close wake) — a timer-driven chain is re-proposed only after two clean session-driven closes (09-07, 09-08), because tonight's close met four upstream refusals a timer would have hit unattended; (3) the V2 line stays PARKED at 1/7 gates — no proxy for an owned execution export (rule 3), it resumes only when the user supplies one; (4) E2-A's admission bar is REPLACED by an OUTAGE predicate (a day inadmissible only if any bookTicker gap run ≥ 60 s; DA measured the old bar as measuring quietness) PRE-DECLARED for days from 2026-09-06 onward and never re-applied to the consumed 08-20..09-05 window; decision-time quote age is REPORTED beside every result, never an admission criterion; E2-A's 14-day window therefore starts 09-06; (5) the XS rebalance notional is NOT invented: the size-aware E2-A arm stays REFUSED for want of a user-declared notional; the min-size arm is the only arm read.
- **Rules landed 2026-09-06 (SEAT_PROTOCOL rule 20 and its neighbours):** transient services with five fields + InvocationID; producer exit maps (75 reserved; UNMAPPED does not satisfy a GO); the four relaunch conjuncts; declaration versions immutable with the CAS at the write; chain link fields in the shared contract; register rows through the shared script (post-condition + trailer); known-bads assert a delta from their own baseline (DISARMED a third outcome, `n_disarmed` fatal); importer cells test the seam; a cited artifact is locatable; readers of history resolve by the pair, never the head; the two-scoped seal census (inherited under v1's scope, added under the union); the register at HEAD is self-describing.
- **Wakes armed in THIS session (die with it):** the seat monitor; a persistent 00:00:30Z wake for BE 72. A cold-start coordinator re-arms both (§0, §5).

## 7c. COLD-START HANDOFF at the second reset (2026-09-07T03:44Z, R-746) — read this FIRST; it supersedes §7 and §7b where they differ

**What died with the previous coordinator session:** its `/loop` wake, its seat monitor, its unit monitors, its scratchpad (all dispatch drafts; the entry-landing script is TRACKED as `scripts/land_register_entry.sh`; the next DA dispatch is tracked as `workspace/coordinator_next_dispatch_DA117.txt`). **What survives:** git (tip = `git log`), `data/`, every seat worktree, the seats' live tmux sessions (cleared and re-briefed at this reset), the systemd timers (`da-midnight-verify.timer` at 00:06Z) and the four collectors, and any transient service still loaded.

**Cold start, in order:** (1) `date -u`; `git -C /home/yuqing/ctaNew status --short` (must be clean) and `git log --oneline -5`; (2) `tmux list-panes -a -F '#{pane_id} #{session_name}'` — derive BE/DA/DE/REV(pm-codex)/MEM(pm-memory) panes, never trust §7's ids; (3) read `workspace/RESULTS.md` §0a, this runbook §0–§6 and §7b (the chain as a predicate), `SEAT_PROTOCOL.md` rule 20 and its neighbours, the register from R-734 to the end, and `HANDOFF.md`'s top block; (4) re-arm the seat monitor (§5) and a unit monitor for any loaded transient service (`systemctl --user list-units --all 'de115*' 'be8*' 'da1*'`); (5) read each seat's last row and its pane's last message before dispatching anything.

**CORRECTION 2026-09-07T03:49:44Z (R-747): GO #6 was REFUSED at its own emit at 03:48:11Z — `assert_source_unchanged`: the worktree's HEAD moved under the run (04d3eb3 → 2503162) because THIS reset's brief told DE to refresh wt-de while the unit ran from it; no receipt; GO #7 = `de115day06_3` LAUNCHED 03:51:22Z, InvocationID `c3f28dfb22584efca4d37a10a72240e8`, from wt-de at 2503162 with the no-move rule (DE holds with a waiter; the receipt lands by itself; DA 117 follows it). The block below is history.** **LIVE AT THIS WRITING — verify, do not assume:** GO #6 = `de115day06_2.service`, InvocationID `17a0320936bd40929474a21e85f0b87c`, launched 02:27:53Z (the 09-06 Gate-1 day run; the 09-05 day took ≈ 70 min). DE's detached helper `wait117.sh` (pid 440727) will capture the five fields + id + journal counts at exit AND STOP the unit, writing `/tmp/claude-1001/-home-yuqing-ctaNew/d0eb65aa-2836-4cbc-8691-f7fc7cce2da9/scratchpad/wait117.out`. If the receipt `data/pm_5min/derived/p003_de_gate1_day_run_20260906_SEALED__*.json` exists: copy `wait117.out` into the ledger as DE's capture record (attributed) before the journal window (≈ 15 min per 18) moves, then dispatch DA 117 (the tracked draft; substitute the receipt name + sha and the book digest `ac2ac952…`), then REV 89 on both branches of tonight's chain. If the unit is still running: wait on it (a monitor), nothing else needs the lock.

**DISPATCHED at 04:51Z 09-07 while GO #7 runs (do not re-dispatch; verify their rows landed):** MEM 245 (sweep R-745..R-747, fold the coordinator's blocks), BE 90 (the mode-0600 fix in the CAS writer + re-mode of landed 0600 versions, digests unchanged), DA 118 (E2-A v8 from v7). **Next dispatches after those, in order:** DA 117 (above) → REV 89 (BE 88/89, DE 117's receipt, DA 117; the mode-0600 finding; `_LAST_PROOF`) → DE 118 (`n_skipped` beside `n_disarmed`; the `_LAST_PROOF` ambient; the Q-DE-116 typo as an in-band note) → DA 118 (P-2026-002: the E2-A admission declaration as **v8 from v7** (57c92c9e…) — R-745 (4)'s outage predicate forward-only from 09-06; the DA 116 dispatch said "v6 from v5" and was WRONG; the earliest read date is 2026-09-20; the consumed set's source is `data/mm_hf/e1/p002_e2a_census8__20260906T055936Z.json`, days 08-19..09-06) → BE (the mode-0600 fix in `declaration_chain.write_next_version`; the pins v3 at the 09-07 close) → MEM (sweep R-745, R-746 and everything after; its round-233 "four items" flag superseded by R-745's five, the enumeration drift explained in RESULTS §0a).

**Fixed events:** 2026-09-08T00:00Z — the 09-07 day closes: run the chain as the predicate in §7b (step 0 the deploy pin is current — DA's landings re-pin in the same round; the nightly unit writes the mask + closed-day verdict at 00:06Z — verify at the artifacts, not the timer; then BE's chain; then `be_forward_day`; then the pins v3; then DE's day run as one GO; then DA). 2026-09-10T01:00Z — the second race read's horizon (authorised at R-745 (4)... no: R-745 (1); the read is its own GO after all four pins exist; the reader's `PREVIOUS_READ_RULE` clause renders the first read's existence). 2026-09-09 — the Gate-1 read gate at six days.

**A refresh under a running unit (learned at this reset):** the reload brief told DE to `wt_refresh` its worktree while `de115day06_2` (WorkingDirectory = that worktree) was still running; the files under the running process moved from 04d3eb3 to the tip. DE assessed it harmless (the runner imports nothing further from the tree after start), and it was NOT: the runner's own `assert_source_unchanged` guard refused the emit at 03:48:11Z (04d3eb3 → 2503162), 80 minutes lost. The rule: NEVER refresh, check out, edit or land from a worktree named as a running unit's `--working-directory` until the unit has exited — and a seat reset waits for a running unit's receipt before it briefs that seat to refresh — reload the seat, let it read, and refresh after the exit.

**The coordinator's own error classes (the pattern file in memory carries the instances):** a headline looser than its artifact; a path typed rather than resolved (the `.v2` suffix; "v6 from v5" for E2-A; the missing `.json`); a commit message read as a landing; a probe reading keys a dict does not carry; a duration estimated rather than read from the clock; a placeholder ("`…`") left in a landed entry; an enumeration that grew without an entry announcing it. The remedies are in §3 and §6 and in the landing script's guards.

## 7d. STATE at 2026-09-07T09:09Z (R-774) — read this AFTER §7c; it supersedes §7c where they differ

**Two USER rulings today, both landed and in the code:** (1) R-754 (06:18:16Z): read the four sealed Gate-1 days NOW — routed as an EXPLORATORY early read (params v16/v17 with the ruling by pair; a new artifact family `p003_de_early_read_day_<D>__<stamp>.json`; the four days CONSUMED; the six-day population no longer confirmatory; G = 4, point estimates, no interval). (2) R-765 (07:47:01Z): "the sealing mechanism is stupid, store the numbers after each run, record everything to avoid rerun" — the seal is RETIRED for every run from now on (params v18 → v19, design v26 → v27, ONE path ONE emission, `PARAMS_REL` → v19) and every day run persists a DECISION LEDGER beside its receipt (`de_decision_ledger.py`, schema v2 with BE 96's inventory fields, ≈ 2 MB gzipped per two-arm day; D_E0, Z, one- and two-sided p, rho, the fills and inventory legs recompute from it without a replay).

**The early read:** E1 (09-03) DONE — artifact `…20260903__20260907T085436Z.json` `5c8a58f5…`, tracked; the table is at R-773 (both arms below the 0-cancel baseline and below the matched random-cancel null; Z −2.30 / −5.30). E2 (09-04), E3 (09-05), E4 (09-06) NOT yet launched: each is its own GO from wt-de refreshed to EXACTLY the commit REV 95 clears (`scripts/wt_refresh.sh /home/yuqing/ctaNew-wt-de <ref>`), unit `deEARLY<D>`, the command in Q-DE-120 with the day substituted, ≈ 100 / 70 / 80 min, one at a time on the lock unless the user rules two at a time (offered at R-762, not taken). After each: DA reads it with `da_early_read_verify.py --print-under-ruling R-764` (Q-DA-351's form) and the coordinator quotes the numbers ONLY from DA's print. 09-03 and 09-04 print under R-764's materiality ruling (their sealed runs used params v14; the delta is the design pointer — seal scope and closure naming only).

**Tonight's close (2026-09-08T00:00Z):** the §7b predicate unchanged (DA's mask + closed-day verdict from the re-pinned nightly unit at 00:06Z — `da_midnight_deploy_pin_v3`; BE's fragment → tape → book → structure v5 as separate GOs; `be_forward_day` → the sealed feed → pins v3 THROUGH `be_race_feed_pins.py` then `--verify`); then GO #8 = the 09-07 day run, the FIRST emitted UNSEALED, from wt-de at the commit REV 95 clears (frozen from E2 on; do NOT refresh wt-de for GO #8 — a refresh would need re-clearance). GO #8's REV 94 clearance is STALE (the runner moved to `ccc4108d…` at `edb9dee`); REV 95 re-clears both E2 and GO #8.

**Chain heads (resolve through `declaration_chain.resolve_head`, never a filename):** params v19 `dd8db7de…`; design v27 `3bcdf3c2…` (v26–v30 DRAFTS of 07:50–08:03Z were WITHDRAWN before landing under R-767's bounded precedent; the surviving v26 `7de8906e…` and v27 are the landed ones); producer_exit_maps v9 (v7 is a superseded fork — `FORKED_BY_EDIT_AND_SUPERSEDED`, R-760); E2-A (P-002) v8; `da_midnight_deploy_pin` v3; `be_race_read_feed_pins` v2 (v3–v5 due at each close through the emitter); `be_race_read_declaration` v6.

**Rules landed today (SEAT_PROTOCOL rules 20/21 and this runbook §3):** a scratch drive is a SCRIPT (`set -u`, `cd || exit`), never an inline chain (R-756); re-read the tip before any revert/re-land/push (R-760); a superseded fork is repaired FORWARD, never by reverting (R-761); the CAS is per-worktree — every version landing re-reads the head after the copy (R-761); compose declarations from a schema, never a deepcopy, and emit through the family's emitter (R-767); a declared block never changes, a derived block only to what its measurement re-derives (R-769); permitted additions classified by MEASURED change cadence (R-770); the constant is never adjusted to the observation — `n_run + n_conditional` (R-771); a GO does not outlive its bytes — name the digests and re-clear when they move (REV 91 §C, R-774).

**Coordinator tools:** `scripts/land_register_row.sh` (exclusive lock; `--row` mode — seats still on the legacy form until told); `scripts/declaration_immutability.sh` (statuses `FORKED_BY_EDIT` rc 1 / `FORKED_BY_EDIT_AND_SUPERSEDED` rc 2, commit names, `pre_edit_digest_pinned_by` with JSON paths, scope line; `--falsify` ≈ 2.6 min). **Wakes armed in this session:** the seat monitor (survived the clear), the 2026-09-08 day-close wake (00:00:30Z / 00:07Z). Re-arm a unit watch per launch (`systemctl --user show <unit> -p SubState` poll; the GO E1 form in R-763).

**Open with the seats:** DE 127 (the phase4 suite per branch, ~37 unreached checks, with the cascade re-point) AFTER REV 95 clears — landing it earlier moves the cascade digests under the gate; BE on standby for the close; DA reads each early-read artifact as its own GO; MEM sweeps each entry; REV 95 in flight.

## 7e. STATE at 2026-09-07T13:11Z (R-795) — read this AFTER §7d; it supersedes §7d where they differ

**Three USER rulings stand (R-754 early read, R-765 seal retired + per-day DECISION LEDGER, R-782 absolutes recorded). A FOURTH decision is OPEN with the user (R-795): whether an INVENTORY LEG is added to the day value and under which rule (which residual / marked at which price / per slug or per day).** Until ruled, every D_E0 is quoted with the label **"fills leg only"** — BE 97 measured at the 09-05 ledger that `_value_cents` sums a per-fill markout term with no position term, `inventory_leg` is NOT a field of the ledger (the artifact's "computed in the ledger" is FALSE — DE 135 Part B corrects the wording), and end-of-day residual positions are non-zero on ~288/288 slugs on every path with the arms holding different terminal inventory. The coordinator's recommendation (disclosed, not adopted): keep D_E0 on the fills leg as the decision metric; a Chainlink-settlement-marked residual as a separate exploratory column once joined; no fill-mark/flat-mark rule.

**The early read (all quoted ONLY from DA's `da_early_read_verify --print-under-ruling R-764` print; RESULTS §0b holds the tables):** E1 09-03 and E2 09-04 both arms below the 0-cancel baseline and the matched null. **E3 09-05 (R-791..R-794): CONDVALUE −29,585.13 Z −3.47; HAZARD +6,540.20 Z +4.33 p 0.002 — the first day a cancel arm ended ABOVE the baseline; 0-cancel baseline absolute 88,698.17 cents (fills leg); arms 59,113.05 / 95,238.37; DA's ledger recompute 0 mismatches over 196,898 rows.** The 09-05 artifact `…20260905__20260907T112912Z.json` `e8f88dc4…` is the FIRST with a real ledger (`p003_de_decision_ledger_20260905__20260907T124104Z.jsonl.gz` `5a2032b5…`, at `day_run.decision_ledger` — NOT top level). Landed in `1bd6b63`. **E4 09-06 RUNNING** (unit `deEARLY20260906`, InvocationID `045ec231…`, from wt-de at `6c3a121`, started 12:47:27Z, ≈ 80 min) — on exit: DE lands (artifact + ledger + run journal by pathspec, `add -f`) → DA 128 (the 09-06 read + the FOUR-DAY table, fills leg only, per arm sign counts, 2^-4 floor; each day's label: computed-under/sealed-stamped params, absolutes yes/no, ledger real/explicit-null/no-key — MEM 272's "four shapes") → the table to the user. Params-stamp spans: 09-03/04 sealed under v14, 09-05 under **v15**, 09-06 read at THAT receipt (a fourth span if it differs — MEASURE key-by-key before quoting, as R-794 did for v15→v19: no estimand/bar/pin key; design pointer v23 in both).

**Compositions (rule 22; branch `origin/mm-research-e3-composition`, one ref per attempt):** `fe76d83` (E2 ran) → `6c3a121` = fe76d83 + DE 132/133's two files (E3, E4 run from it; REV 102 cleared) → **`5020f96`** = 6c3a121 + DE 134's runner (`883b5f3a…`: the REAL day path's ledger anchor + the post-emit census's removed-key read that would have DELETED tonight's receipt; **REV 103: GO #8 MAY PROCEED at 5020f96**). Digests at 5020f96: runner `883b5f3a811e9576…`, early read `5aa544ef8d594efd…`, ledger `d78c370151cea431…`, params v19 `dd8db7ded9e6ed97…`, design v27 `3bcdf3c234cb7d4e…`. **After E4's receipt lands: DE 135 Part A refreshes wt-de to EXACTLY 5020f96** (`scripts/wt_refresh.sh /home/yuqing/ctaNew-wt-de 5020f96`) with REV 103 §5's two BLOCKING conditions checked IN wt-de: P10 clean beyond the `?? data` symlink; cascade pins 10/10; plus the five digests and rehearse_smoke('2026-09-07') NOT_READY on the two P2s ONLY. Then wt-de is FROZEN for GO #8. NOT composed (ride after GO #8): a71b714 (the ABSOLUTES_DO_NOT_RECONCILE cell) and DE 135 Part B (the inventory wording).

**Tonight's close (2026-09-08T00:00Z; day-close wake armed in the coordinator session — re-arm after a clear):** §7b chain unchanged — DA's mask + closed-day verdict at 00:06Z via the re-pinned nightly unit (`da_midnight_deploy_pin_v3`); BE fragment → tape → book + builder receipt → structure v5; `be_forward_day` → pins v3 via `be_race_feed_pins.py` then `--verify`. **GO #8 = the 09-07 day run from wt-de at 5020f96, the FIRST real day emitted UNSEALED; its receipt is read at `day_run.decision_ledger` FIRST** (REV 103 §1: the real-day `day_run_ledger_anchor → write` join runs for the first time; no test reaches it). Then: the FREEZE (R-775) lifts → DE writes params v20 / design v28 by RE-MEASUREMENT (they never existed as files — R-787; REV 96 §5 checklist) → REV reads → GO #9 (absolutes in the receipt per R-782). **Replay debt:** 09-03 and 09-04 each need one replay under the recording code (ledgers + absolutes); the user knows.

**Mechanics landed today:** every register row via `scripts/land_register_row.sh --row <rowfile> '<id>' <msgfile>` (lock; hand-editing RETIRED, R-784); entries via `land_register_entry.sh`; post-checks after each landing (placeholders 0, ratification fences 2, `de_ratification_check` True). **New-commits check anchors on the LAST SEEN origin tip, never on my own landing** (R-793: rebased seat commits hide behind it). Coordinator misreads today, both caught in-pane: a top-level `.get("decision_ledger")` (the key is under `day_run`); naming the row commit as the landing commit. Pane map at this writing: %2 pm-co, %3 pm-be, %4 pm-da, %5 pm-memory, %6 pm-codex (REV), %7 pm-de — DERIVE it, never trust it. Seat rounds at this writing: BE 97, DA 127, DE 134 (135 prepared), REV 103, MEM 273; next entry R-796.

**§7e addendum (2026-09-07T15:15Z, R-800..R-802) — THE ESTIMAND CHANGED BY USER RULING.** R-801 (verbatim): "the pnls are from trades and remaining position's settlement p&l, need to calculate this correctly." A path's P&L on a slug = its trades' cash flow + (net shares at the window's close × settlement, 100 c iff Up won); the day = Σ slugs; the arm's excess = arm − baseline. **The 5-s markout D_E0 is now a DIAGNOSTIC, never "the result".** BE 98 showed why: at settlement the baseline is worth 81,238 / 46,562 c on 09-05 / 09-06 against 88,698 / 90,153 at 5 s, and both arms flip from below to above the baseline on 09-06. No null exists yet under the new valuation (draw fills were discarded). In flight: BE 99 (the ruled P&L at the two ledgers, per slug, winner verified against the Chainlink TWAP stream — `exp_m6_settlement.py` states the rule), DE 136 (the second valuation in the runner, the null re-valued, falsifiers, a rule-11 guard, the DRAFT of params v20 / design v28 naming the new primary endpoint), then REV 104. **Rule 11 guard: 09-03..09-06 are DESIGN data for the new endpoint; NOBODY values 09-07 or later at settlement before v20/v28 land.** GO #8 tonight is UNCHANGED (its ledger is the input). Still OPEN with the user: a position cap in the replay and the quoter's placement (R-800). The review page for the user: https://claude.ai/code/artifact/3214dc3d-ee88-49f5-aa25-d196061baad5

**§7e addendum 2 (2026-09-07T15:38Z, R-803..R-806) — THE USER'S WORK ORDER (R-804) and where it stands.** The user: "make these few days data correct … Check above issues, fix then review" (five critical issues, verbatim in R-804). Three tracks: **A** (data correct under the ruled P&L) — 09-05/09-06 DONE at the ledgers (R-803: trades + residual, Chainlink-verified winner, convention S60(T) ≥ S60(t0) reproduces the venue 576/576); 09-03/09-04 have NO ledgers → GO E1r/E2r re-runs from wt-de at `5020f96`, **NO-GO at REVIEW 104A until the early-read family has a SUPERSESSION RULE** (`EARLY_READ_ALREADY_EMITTED` is right; DE 138 Part A adds `supersedes: {path, sha256}` + the precondition; DA 130 adds head resolution) → then E1r, E2r (after GO #8 if the clock is tight) → BE values them. The misnamed `inventory_leg` (= −trades cash flow) → DE 137 renames. **B** (the settlement null + declaration) — DE 136 LANDED `1f171e8` (estimator, inline-valued null, rule-11 guard, four falsifiers, draft `live/pm_research/drafts/R801_params_v20_design_v28_DRAFT.md`; winner NOT_VERIFIED_AGAINST_CHAINLINK until DE 137 wires BE 99's convention) → REV 104 second half → GO #8 (unchanged) → v20/v28 LAND → settlement-null runs 09-03..09-06 (design) + 09-07 (validation). **C** (design fixes, each a declared parameter + falsifier, validated from the first day AFTER landing) — DE 138 Part B draft: position cap, hold_side_after_cancel, (side, hour)-stratified null, thresholds re-fit on the consumed days, **placement_latency_ms** (BE 100: the reference spec has NO placement latency; 48–56 % of fills land within 250 ms of placement and carry 98 % / 55 % of the baseline's settlement P&L — the largest assumption in the numbers); the quoter's settlement view is DEFERRED — DA 129: the fair-value identity needs a **sigma producer that does not exist** (owner DA). Capacity is NOT a weakness (our fills 2–8 % of tape volume per slug). Coordinator claim corrected in band at R-805 (the "zero placement latency" wording). Seat rounds at this writing: BE 100, DA 130 (in flight), DE 137 (in flight; 138 queued in the scratchpad — re-create from R-806 if lost), REV 104 (second half in flight), MEM 279 (in flight); next entry R-807.

## 7f. STATE at 2026-09-09T05:43Z (R-835) — SUPERSEDED BY §7h; kept as provenance for the retraction

**EVERY ARM RESULT IS RETRACTED. The latency finding survives.** Two USER-found scoring
defects compound: the **value head was never loaded** (thresholds fit on
`p_fill * conditional_value`, scoring on hazard alone — R-833, `1d309bc`) and
**LOOK-AHEAD in the decision** (generation MAX stamped at the GENERATION START, so later
information triggered an earlier cancel — R-834, `c501824`; the rule is now each row at
its own `t_start`, FIRST crossing cancels). The **0-cancel baseline makes no decisions**
and is untouched by either, so the placement-latency result stands and is the programme's
one robust finding: at L=250 the never-cancel maker loses **53.9 / 62.4 / 97.6 / 55.4 %**
of its settled money on 09-03/04/05/06. RESULTS.md carries the retraction notice.

**ALL SEVEN USER DEFECTS CLOSED (R-835), plus a blocker none of them named.** The repairs
moved four pinned cascade modules while params v20 pinned pre-fix digests, so **no day
could run**, and it took THREE pin pairs to close (v21+v29, v22+v30, **v23+v31** —
v22 missed a second site). **REV 109 (`839c072`): a day CAN now run. The pair of record is
params v23 + design v31.** REV also refuted DE's claim that defect (5) was outside its
surface, and demonstrated a non-finite SCORE from a finite feature vector that (3)'s first
fix had missed.

**TWO RULINGS OPEN WITH THE USER — nothing corrected is quoted until the first lands.**
(a) **The null's sampling unit** (`live/pm_research/drafts/DE158_null_sampling_unit_QUESTION.md`):
per-row rows mean "matched on the action count" no longer names one quantity (39.7 % of
rows begin after their generation's start). **A** match on rows — biased in the arm's
favour; **B** match on cancels — honest but breaks seed reproducibility; **C** sample
generations — reproducible but the control no longer samples what the arm decides on.
Comparability with landed days is NOT a consideration: none survives. (b) **`research.slice`
is capped at 200 %**, so the parallel null's 12.7× becomes ~2× until it is raised
(`DE158_parallel_null_slice_PROPOSAL.md`); machine-wide, so proposed not changed.

**WHAT IS NOT DONE.** The cancel-count measurement (old aggregation vs first-crossing, per
arm per day — the evidence a theta re-fit is ruled on; theta is NOT re-fitted, by ruling,
and the consequence is recorded in the receipt instead). Corrected point estimates. Four of
five corrected books (09-03 EV20 was building at this writing; the others queued, ~40 min
each, concurrency ruled out on memory by BE 106). The parallel null's deployment (built and
verified, 403 green, element-by-element reproduction with a can-fail control).

**COORDINATOR DISCIPLINE, learned expensively this session and now standing:** ~47 minutes
were lost because seats finished a unit of work, reported, and waited for a go that needed
no decision. **When a seat is given a numbered queue it works it END TO END and reports
ONCE at the end, or the moment something BLOCKS it or a falsifier FAILS.** Issued at DE 156;
six defects landed in one 17-minute batch afterwards. **Arm a Monitor on any heavy unit and
prompt the seat at its exit — seat in-pane watchers are DEAD (R-821) and every seat idles
waiting otherwise.** All four working seats now carry procedure files
(`BE`/`DA_PROCEDURE.md`/`MEM_PROCEDURE.md`/`DE_PROCEDURE.md`); a seat at the end of its
context WRITES its file rather than reporting it (R-831).

## 7g. ON RESUME — DO THESE FIRST (2026-09-09T05:46Z)

**A cleared coordinator loses three things and NOTHING ELSE: the live Monitors, the
`/loop` duty, and the scratchpad. Everything else is in git.** Re-establish in this order.

**CORRECTED 2026-09-09T07:04Z (R-836): a clear does NOT kill a Monitor.** Three Monitors
armed before the 06:58Z clear survived it and delivered events into the fresh context. What a
clear loses is the `/loop` duty and the scratchpad; a Monitor still ends at its own TIMEOUT
(the build watch expired ~07:02Z and was re-armed `persistent`). So on resume: CHECK what is
still alive before re-arming, and re-arm only what has expired.

**STANDING (R-836): while a build holds the heavy lock, ITS WORKTREE IS FROZEN.** The owning
seat lands from the SHARED tree or it does not land. `p003ev200903` assembled for 74 minutes
and was refused at the write because BE's own BE 109 commit went into wt-be at 05:58:16Z under
a run that had started from `c9f8b31` at 05:09:52Z. Second loss of this class.

1. **Re-arm the unit watch immediately.** Seat in-pane watchers are DEAD (R-821), so a
   heavy unit that exits with no coordinator Monitor leaves its seat idling indefinitely —
   this cost ~47 minutes in one session. Check for a running unit first:
   `systemctl --user list-units --type=service --no-legend | grep -E 'p003ev|be10|dePE|deRR'`
   and `fuser data/.heavy_run.lock`.
2. **NOTHING IS BUILDING, BY USER RULING (R-839, 2026-09-09T07:15:47Z): "dont have to
   build now, we clear issues first". `p003ev200903b` was stopped clean — nothing written,
   lock free — and the five-day queue is STOOD DOWN. Do not launch a build, and let no seat
   take the heavy lock, until the issue-clearing rounds land and the user or you dispatch
   one explicitly. The envelope raise is RATIFIED for the EV20 queue only (R-837) and the
   slice clamp is OWED BACK at 200 % when the null is done (R-838).**
   HISTORICAL, both dead: `p003ev200903b` (06:59:59Z, stopped 07:15:47Z) and before it
   `p003ev200903` — BE's **09-03 book at revision EV20**,
   the FIRST built through the repaired scorer, started 05:09:52Z, ~35 min expected.
   **At its exit: prompt BE** for the book path+sha, wall, peak, and the two things that
   pass only this build can show — (a) the receipt's manifest including `linear_{coin}.json`
   now under the digest set, (b) whether the corrected `asm` DIFFERS from the pre-fix
   book's (identical scores would mean the fix did not reach the book), and (c) whether the
   rebuilt `fr` is byte-identical to the existing book's — **if it is, the remaining four
   days are minutes rather than ~40 each and the night changes.** Then **prompt DE** to run
   the point estimate on it (~2 min).
3. **BE's queue:** 09-04, 09-05, 09-06, 09-07 at L=250 revision EV20, one at a time under
   the lock, ~40 min each, concurrency ruled out on memory (BE 106).
4. **DE's queue:** the cancel-count measurement (old aggregation vs first-crossing, per arm
   per day — the evidence a theta re-fit would be ruled on, still not produced); corrected
   point estimates as books land; the parallel null's deployment.
5. **MEM 289 was mid-sweep when this was written** — STATUS.yml and HANDOFF.md were last
   written at round 288 (`0542820`, 09-08 01:50Z) and are STALE by two retractions until it
   lands. Read §7f and RESULTS.md's retraction notice instead, and check MEM's newest
   commit before trusting the state files.
6. **Do not quote any corrected number** until the user rules the null's sampling unit.

**Read order for a cold start:** **§7h FIRST**, then §7g (this) for the resume mechanics, then §7f as provenance, then RESULTS.md's retraction notice,
then the last five register entries, then the four seat procedure files if you need to
know how a seat works.

## 7h. STATE at 2026-09-09T07:35Z (R-843) — READ THIS FIRST; it supersedes §7f and §7g

**NOTHING IS BUILDING, BY USER RULING (R-839): "dont have to build now, we clear issues
first".** `p003ev200903b` stopped 07:15:47Z, clean, nothing written. **No seat takes the
heavy lock and no day is built until the build gate is cleared and a build is dispatched.**

**THE RULING WAS RIGHT AND THE EVIDENCE IS SIX DEFECTS IN THIRTY MINUTES.** Three seats were
told to hunt with NO TARGET NAMED, and every one of them found something on the path the
rebuild would have run. The build gate, in severity order, with owners:

1. **The builder never asks the day its era** (REV 112, R-843) — `_era_or_refuse(fi, None, …)`
   resolves a module default `clob_v3_1` while every September day is `clob_v4_1`; the gap
   tables are DISJOINT, so `build_reference` gets `gaps=[]` on every window of every day.
   09-03: **160 of 287 windows, 2,294.7 s built as if continuous.** → BE.
2. **The book's `producing_code.import_closure` is recorded and read by nobody** (REV 111,
   R-840) — the only SILENT pin site; true of every book on disk. → DE.
3. **The score key drops rows** (DA 138, R-842) — `(slug, side, t)` lost the generation. → DE.
4. **The engine ignores the score event's generation** (DA 139, R-842) — and a misattributed
   cancel is INVISIBLE in every artifact. Same root as 3; one site (`t1` bound) closes both. → DE.
5. **The builder's coverage evidence uses the pre-fix membership test** (BE 111, R-841) — it
   goes into every receipt and `assert_coverage` only refuses at zero. → BE.
6. **Mode C blames the wrong subsystem** (DA 138) — sends the next debugger into the feature
   pass. → DE.
7. **`de_section81_cache_12.pkl` has no code pin at all** (REV 111, site 4). → DE.

**THE ERA DEFECT REACHES BACKWARD, and this is the thing a resuming coordinator must not
lose:** every book on disk was built through it, including those the retracted arm results
came from AND the reference the 0-cancel baseline runs over. **The placement-latency finding
is triple-derived but all three derivations sit on those books, and NOBODY HAS ESTABLISHED
ITS INDEPENDENCE FROM THE ERA DEFECT.** DA 137's survival argument bounds the SCORING defects
out of the baseline and says nothing about this one, which enters upstream of the arm/baseline
split. The retraction has NOT been extended; the exposure is named and routed. **QUEUED as
DA 141, bounded and read-only: what share of the baseline's settled money falls inside 09-03's
160 gapped windows.**

**GATE STATUS at 07:52Z (R-846) — FIVE CLOSED IN CODE, ONE VERIFIED, AND AN EIGHTH ITEM:**
item 1 (era) OPEN with BE 113 — the largest, and its fix must REFUSE by name rather than
default when a day's era will not resolve; items 2/3/4/6 closed by DE 162 (`060409d`),
unverified, DA 143 driving 3 and 4 now; item 5 closed by BE 112 (`cc148d5`) and **VERIFIED
by DA 142** — new equals old exactly on a PER_GENERATION reference (24/24, 14/14, 0/0), so
nothing correct was narrowed; item 7 (`de_section81_cache_12.pkl`, no code pin) with DE 163;
**item 8, NEW — `da_book_verify.py:683` and `:906` hold only under PER_GENERATION, so DA's
verifier would flag EVERY corrected book as a population defect** (found by BE on DA's
surface, the seam again), open with DA. Also with DE 163: REV 113's sealed-value guard hole
and DA 140's expiring `require_book_declares_L`. **"Landed" is not "reviewed" — DE 161
landed green and REV 113 found it holed two hours later.**

**PAIR OF RECORD: params v24 + design v32** (advanced from v23 + v31 at 07:52Z). DE 162 moved
`de_phase4_diag_runner.py` and turned the pin crank in the SAME commit, so the R-835 blocker
did NOT recur — driven by the coordinator: **v23 is 9/10 against disk (`de_phase4_diag_runner`
pinned `c979fda590ec`, disk `437babc07590`), v24 is 10/10 with zero mismatches.** Design v32
was stranded under rule 21 and was rebased and pushed at 07:47:52Z; origin carries both halves.

**THREE USER RULINGS ARE IN (R-837, R-838):** the null's sampling unit is **B, match on
cancels**; `research.slice` is raised to the ruled **1000 %** (applied and verified at the
running manager — and the declared file says 1200 %, a dead value since 09-06, so never
"reload to make the config take effect"); the build unit's memory envelope is **ratified for
the EV20 queue only**. **The slice clamp is OWED BACK at 200 % when the null is done.**

**SEATS at this writing:** BE was RESET from 97 % (its harvest is R-841, its method
`BE_PROCEDURE.md`) and is on BE 112 (coverage fix against a synthetic PER_ROW fixture, site 3
in its guard); the era fix is its next round. DA 140 — a THIRD untargeted area. DE 162 — gate
items 2, 3, 4, 6, 7. REV 113 — verifying DE 161's rewritten sealed-value guard, because it was
LOOSENED on the strength of a false positive, which is where a real leak walks through.
MEM 293 — the sweep.

**WHAT WORKED, and it is now the programme's most productive instrument: a prompt that names
no target.** Six defects came from three such prompts in thirty minutes; the seats working to
specified briefs had found none of them in the preceding hours. See also DA's own rule, earned
at R-842: **a claim of unchangedness made at the moment its input changed shape is where you
look.**

## Worktree data rule (R-553)

**Refresh a seat worktree ONLY with `bash scripts/wt_refresh.sh <wt> [ref]` (R-625).** A bare `checkout --detach` re-materialises `data/` whenever a landed commit adds a tracked data path (REV 59 §8); the script drops the symlink, checks out, sweeps skip-worktree over every tracked data file, re-links and verifies. Sparse-checkout was tried and rejected: git ignores the skip-worktree bits under it.

A seat worktree's `data/` MUST be a top-level symlink to `/home/yuqing/ctaNew/data`:
after `git worktree add`, `rm -rf <wt>/data && ln -s /home/yuqing/ctaNew/data <wt>/data`;
then `git -C <wt> ls-files data | xargs git -C <wt> update-index --skip-worktree` so git stops
reporting the tracked data files as deleted (R-554); artifacts under data/ are landed from
the MAIN tree by pathspec. Check `readlink -f <wt>/data`. A materialised `data/` directory (git tracks ~135 files
under it) is a partial SHELL — every uncommitted artifact is absent from it, and two
seats reported shell facts as ledger facts on 2026-09-06.

## Shared-tree prohibitions (R-557)

Never `git reset --hard`, `git checkout -- <file>`, `git clean`, or `git stash` in
`/home/yuqing/ctaNew` — seats keep uncommitted work there (MEM's state files all
night). Probes and experiments live in a scratch worktree; a wrong commit in the
shared tree is REVERTED, never reset. Commit messages by heredoc or `-F`, never a backtick inside `-m` (R-567: one
expanded into `git checkout --detach` in the shared tree). Never `--autostash` over another seat's dirty
file — wait for its commit. Every coordinator commit is `git commit -- <paths>` (R-562: a commit without a
pathspec swept a seat's staged file). `PM_DATA_ROOT` names the REPO root
(`/home/yuqing/ctaNew`), not the data directory (R-562). When seats' landing commits are STRANDED in the shared tree (origin moved while
another seat held files dirty, so rule 21 made them wait), the coordinator rebases them
onto origin at the first CLEAN-tree moment (`git status --short` empty but for
untracked files) and pushes — never with a dirty tree, never `--autostash` (R-586, 07:08Z:
DA's and MEM's commits, ahead 2 / behind 2, landed this way). Seat refresh is ONE command:
`git checkout --detach mm-research && git ls-files data | xargs git update-index --skip-worktree`.

## Seat context (R-571)

Read each seat's context from its pane status line (`tmux capture-pane … | grep -o '[0-9]*% context used'`),
never from the seat's own estimate (MEM said 14% at 99%). Reset at 80%; three seats reached 98–100%
on 2026-09-06 before anyone read the line.
