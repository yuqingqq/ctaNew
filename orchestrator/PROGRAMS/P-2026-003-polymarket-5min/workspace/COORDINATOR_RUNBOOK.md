# COORDINATOR RUNBOOK — P-2026-003

Written 2026-09-03T03:40Z, R-495's consolidation. **Single writer: the
coordinator.** Purpose: everything a coordinator session needs that lives
nowhere else, so that clearing or losing a coordinator session costs nothing but
the re-read. `COORDINATOR_DISPATCH.md` is the 2026-08-26 phase dispatch and is
**stale** (it names a coordinator session that no longer exists and a phase that
has been superseded); it is kept as provenance. This file supersedes it for
operations.

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

## 7. State at this writing (2026-09-03T04:00Z) — verify, don't trust

Superseded whenever a round lands; the artifacts are the authority.

- Branch tip after R-496 and the doc corrections. Register: **487** `### R-`
  headers, one ratification fence. **Next entry R-497.** Next Q numbers:
  **Q-BE-239, Q-DA-218, Q-DE-62, Q-MEM-61** (238/217/61/60 are in flight).
- **Round dispatched 2026-09-03 ~04:00Z, all five seats working**: BE 13 (seal
  relocation + the 08-29 free read + the 08-30 labelled secondary + the 09-02
  sealed accrual run), DE 44 (Phase-4 producer half, split now declared), DA 21
  (low-content-without-gap-rows detector + the read-only 09-04 preflight; its
  round-20 chain stays HELD), MEM 72 (resumes sole writership), reviewer on
  BE round 12.
- Held and unpushed: **DA** `3c49cb7` → `a36db71` (round 20; all three files sit
  on the path the 00:06Z unit executes, so it lands only AFTER the 09-04 run);
  **DE** `0d03902` (round 43 WIP, RED by design) — being built on in round 44.
- **G = 2 of 5.** The USER ruled the 09-02 accrual at R-496.
- **ONE USER decision open**: the Phase-2 winner, which the race decides. See
  `RESULTS.md` §7 — three were ruled at R-496 and one was never open.
- **A sealed race artifact was found living only in a dead session's `/tmp`**
  (R-496 (B)). Backup at `~/ctaNew_sealed_backup/`; BE is relocating the
  authoritative copy under `data/` with a superseding receipt. **Check this
  class whenever a sealed artifact is produced: a receipt whose `sealed_file.path`
  starts with `/tmp` is one sweep from voiding a race day.**
- Sister program **P-2026-002's E2.0/E2-A gate opened 2026-09-03** and nothing
  is dispatched against it.
