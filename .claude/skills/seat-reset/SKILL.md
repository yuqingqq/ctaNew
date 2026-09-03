---
name: seat-reset
description: Halt every seat of a multi-seat program, harvest what only their contexts hold, consolidate the results into the repo docs, then clear and re-brief every seat (and optionally the coordinator). Use when contexts are long, before a hand-off, or whenever the user asks to "stop everything and reload".
---

# Seat reset — stop, consolidate, clear, reload

A context reset is **safe only if nothing that matters lives solely in a
context.** This skill is the order of operations that makes that true. Executed
first on 2026-09-03 for P-2026-003 (register entry R-495); every guardrail below
is there because it cost something or nearly did.

**Default target:** every seat pane of the active program. `args` may name a
subset (`be da`), or `+coordinator` to include the coordinator's own reset at the
end. Derive panes, never hardcode them:

```
tmux list-panes -a -F '#{pane_id} #{session_name}'
```

Sessions follow `<program-prefix>-<seat>` (e.g. `pm-co`, `pm-be`, `pm-da`,
`pm-memory`, `pm-codex`, `pm-de`). Pane ids change when a session is recreated —
a stale map sends a dispatch to the wrong seat.

---

## Phase 1 — Halt

1. Read every pane's state; a pane showing `esc to interrupt` is working:
   `tmux capture-pane -p -J -t %N -S -3 | grep -oE 'esc to( interrupt)?|done [0-9:]+ [AP]M' | tail -1`
2. Interrupt the working ones: `tmux send-keys -t %N Escape` (repeat once if the
   first does not take; confirm before dispatching).
3. Send the **stop dispatch** to every seat. It asks for exactly four things and
   forbids tidying:

   > STOP — coordinator, on the USER's instruction. Halt what you are doing NOW.
   > Do not start anything new. Do not commit, do not push, do not run anything
   > further, do not edit any file. If you are mid-edit, leave the edit exactly
   > where it is — a half-finished change left uncommitted is safe; a
   > half-finished change committed is not. THEN, in ONE message and nothing
   > more, report exactly four things, each verified at the artifact (git / the
   > filesystem), never from memory: (1) UNCOMMITTED — `git status --short` in
   > YOUR OWN worktree, verbatim; (2) UNPUSHED — `git log --oneline
   > origin/<branch>..HEAD` in your own worktree (NONE if none); (3) WHAT YOU
   > WERE DOING when you stopped — the round, the item, how far it got; (4) WHAT
   > WOULD BE LOST if your context were erased right now and only the repo's
   > files remained. Be complete on (4): your context IS about to be cleared,
   > and anything you know that is not in a file or in this message ceases to
   > exist. Do not tidy, do not finish "just one thing", do not summarise your
   > round for the record — the register and the docs are the record and the
   > coordinator is writing them. After that one message, stand by.

**Why (4) is the whole point:** seat pane scrollback is not retained. A seat's
narration cannot be recovered after the clear — only what reaches a file or that
one message survives.

---

## Phase 2 — Preserve fragile work

An **uncommitted worktree edit is the only form of work a reset can actually
lose** (any later checkout or rebase clobbers it, and nobody will know what the
bytes were). For each seat that reported uncommitted work, reverse the
"commit nothing" instruction **for that seat only**:

> Make exactly ONE commit in your own worktree, DO NOT PUSH, subject beginning
> `WIP HELD (red): <round> — <one line>`. Body: what is done, what is RED and
> why, the measured facts you hold, and any boundary you did NOT rule, written
> as an OPEN QUESTION rather than a decision. Then reply with the sha, the file
> list, and `git status --short` (must be empty).

A red WIP commit is correct here — it is a preservation act, not a claim of
completion, and its message carries the facts the context held. Verify it at the
object store from the main tree (`git log --format=%B -1 <sha>`), never from the
pane.

---

## Phase 3 — Harvest and verify

1. **Harvest** every fact from the seats' (4) answers that is not yet in a file:
   row corrections, unratified decisions, boundaries left unruled, measurements
   not reproducible from the repo. These go into the register entry in Phase 4 —
   in-band, quoting the artifact, never by editing a landed row (rule 13).
2. **Verify at the artifacts**, not from the reports: each seat's tip and held
   commits (`git -C <worktree> log --oneline origin/<branch>..HEAD` — read from
   the main tree with `git -C`, never run git inside another seat's worktree),
   the working tree clean, worktree count at quiescence, tracked data directory
   count unchanged, collectors alive.
3. Finish or **honestly abandon** any verification battery in flight. If a check
   did not run, the docs say it did not run and name it — an unrun check
   recorded as done is worse than no check.

---

## Phase 4 — Consolidate the docs

The reset is the moment the docs earn their keep. Write, in this order:

1. **A compact results file** (`workspace/RESULTS.md` or the program's
   equivalent) — the artifact-anchored answer to "what has been tested and what
   came out of it", short enough to read in five minutes. Every number read from
   the artifact **during this consolidation**, not copied from an earlier
   summary. Where a previous doc disagrees, say so — that is a correction, not a
   restatement.
2. **The running state files** (`STATUS.yml`, `HANDOFF.md`): a dated
   consolidation block, task statuses that have moved, and flags for what the
   stop found. If those files have a single designated writer who is halted,
   **log the writer exception in the register and end it explicitly** ("X
   resumes as sole writer next round").
3. **The register entry**: what was verified, what each seat holds, the harvested
   corrections, what the stop itself found, and the routing. Append-only; run the
   register's own checks after committing (placeholders on the new entry only,
   fence count, ratification check).
4. Commit **by pathspec**, push, and update any memory/index file that points at
   these docs.

**Reading the artifacts a doc names is where the errors are.** The first run of
this skill found two wrong numbers in the program's own handoff — counts quoted
from a superseded artifact, and a survivor count that the artifact's own
`distinct_results` field contradicted.

---

## Phase 5 — Clear

`tmux send-keys -t %N C-u`, then `/clear`, then `Enter`, per pane. Confirm each
pane returned to an empty prompt before briefing.

---

## Phase 6 — Reload

Send each seat a **self-contained** brief (it has no memory of anything):

- **Common preamble**: which program, repo, branch, current tip; the read order
  (protocol → results file → the last few register entries and its own rows →
  state files); the standing rules that bind every seat (batching, own-worktree
  execution, resource limits, falsifiers, verify-at-the-artifact, compute
  predicates, models-estimate-the-user-decides, times from a clock read).
- **Per-seat section**: what it owns, **its held state named by sha**, what was
  verified about it and by whom, what is explicitly NOT done, any correction it
  still owes, and any standing prohibition specific to it.
- **End with**: DO NOT START WORK. Confirm the tip, `git status --short`,
  `origin/<branch>..HEAD`, and one line on the held state. Then stand by.

Then **wait for every confirmation and read it** — the reset is not complete
until each seat has re-derived its own state from the files:

```
until [ "$(for p in <panes>; do tmux capture-pane -p -J -t %$p -S -3 | grep -oE 'esc to( interrupt)?' ; done | wc -l)" = "0" ]; do sleep 5; done
```

---

## Phase 7 — The coordinator's own reset (only with `+coordinator`)

Before the coordinator session is cleared, make sure a runbook exists holding
what only that session knows: cold-start read order, the pane map's derivation,
dispatch and register mechanics, the verification-battery pattern, standing
prohibitions, and the current routing (next entry number, next filing numbers,
held chains, queued rounds).

State plainly **what dies with the session and what does not**:

| dies | survives |
|---|---|
| the `/loop` wakeup (the standing duty stops) | git, `data/`, every seat worktree and held commit |
| the commit Monitor | the seats' own live sessions |
| background tasks | systemd timers and collectors |
| the scratchpad path | everything already committed |

After the clear, the coordinator resumes by reading the runbook and re-arming
the loop and monitor — those two are the only things a re-read cannot restore.

---

## Guardrails

- **Never** let a seat "finish just one thing" during a halt. The point of a
  stop is a known state, not a tidy one.
- **Never** clear a seat before its (4) answer has been read and anything
  durable has reached a file or a commit.
- **Never** silently rewrite a wrong number in a landed doc — supersede in band,
  quoting what it said and what the artifact says.
- **Never** claim a seat reloaded until it has confirmed from the files. A brief
  that was sent is not a brief that was read.
- A halted seat with a designated-writer role does not lose that role; the
  exception is temporary and must be closed in writing.
- Report what actually happened: which checks ran, which did not, and what is
  still held unpushed and by whom.
