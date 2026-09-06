# Seat protocol — P-2026-003 (consolidated roles & rules)

**Status:** consolidation, not new law. The register (`COORDINATION.md`) remains
the authority; every rule here cites its R-entry. On any conflict, the register
wins. Committed 2026-08-28 by the coordinator; amendments by coordinator commit,
except where marked USER-ONLY.

## Seats

| seat | session (tmux) | owns | never touches |
|---|---|---|---|
| **Coordinator** (pm-co) | this session | the register (R-entries, append-only); rulings; dispatch; verification of every result-bearing claim at its artifact; collector surface changes (R-110); driving the Codex review rounds; boundary deploys | seats' in-flight files; frozen artifacts (rule 13) |
| **BE** (pm-be) | ctanew-fe | phase2/011 model code, fit/score runs, receipts, freeze receipts | STATUS/HANDOFF (R-233); collectors; DA's instruments |
| **DA** (pm-da) | ctanew-1e | independent verification stack, tape gate, day-verdict tool, its Q-filings, peer annotations (sidecar owner) | STATUS/HANDOFF (R-233); collectors; BE's generators (reads, never edits — separate implementations are the point, R-235: do-not-harmonize) |
| **MEM** (pm-memory) | ctanew-ba | STATUS.yml, HANDOFF.md, memory docs, TODO checkbox true-ups (tick-with-citation only) | prose of plans; register; receipts; any result-bearing artifact |
| **DE** (pm-de) | USER-staffed 2026-09-01 (R-379; plane previously "UNSTAFFED, held by the coordinator", R-165 handoff clause) | `harmful_stateful_policy.py`, `de_actionspace.py`, `de_constraints.py`, seven-arm parity battery, Phase-4 grid protocol drafts, EV-Replay seam drafts | BE's generators; DA's instruments; collectors; STATUS/HANDOFF; registry amendments land by coordinator/USER act, never DE self-edit |
| **Reviewer** (pm-codex) | Claude session (was the user's Codex session until 2026-09-01 — quota exhausted; seat continued as Claude by USER order, R-375. Same-model caveat and mitigations in R-375; new filings `REVIEW_*`, legacy `CODEX_*` untouched) | review filings under `workspace/reviews/` (commit); holds and releases | fixing code itself; state files (first filing predates this rule) |
| **USER** | — | freezes (rule 12); frozen-doc amendments; CLAUDE.md; collector deploy approval; race admission; anything marked USER-ONLY | — |

## Standing rules (with their register cites)

1. **Verify at the artifact** (CLAUDE.md rule 16): no seat's claim — including
   the reviewer's — is accepted from a report. A claim is a reproduced defect
   or a review error, established by execution, filed either way (R-238).
2. **Red-first**: every fix ships with a known-bad that FAILS on the pre-fix
   code, plus a positive control. A fixture must never supply what the code
   under test should produce (R-229 class); quiet and empty are different
   (R-236); run the entry point the way the launcher runs it (R-240 cycle).
3. **Corrections supersede in-band** (rule 13): frozen artifacts are never
   edited; superseding versions carry a supersedes block; a citation
   correction never restarts a clock (freeze v2, R-236/R-238-adjacent).
4. **Frozen docs are amended only by the USER** — seats draft
   DRAFT-FOR-USER-FREEZE; nobody amends a design after seeing it (R-237).
5. **Review protocol** (R-239, refined R-240, completed R-377 — USER ruling):
   build → commit+push → **one review round per COMPLETED batch** (never
   piecemeal) → the reviewer files **ONCE, one complete filing per round,
   never a stream of findings**, commits+pushes it → coordinator verifies
   claims → **ALL fixes are applied as one batch — never sent or landed one
   at a time — committed AND pushed together, and the reviewer then gets
   exactly ONE notification naming the pinned tip** → re-review executes
   that exact batch commit. A hold releases only on the reviewer's explicit
   HOLD RELEASED.
6. **State-file ownership**: MEM writes STATUS/HANDOFF; BE/DA commit artifacts
   and file facts (R-233). CAVEAT: CLAUDE.md still instructs every session to
   update these files — a USER-ONLY amendment is pending; until it lands, a
   fresh seat following CLAUDE.md is behaving correctly and MEM sequences
   around it.
7. **Collector surface** (R-110/R-181/R-182): coordinator-owned; changes need
   a USER ruling; deploys only at a UTC day boundary with a `collector_runs`
   era stamp; verification is structural, never a throughput A/B.
8. **Caps are never raised** (R-174). Lowering a unit's cap for
   attributability is permitted (R-238 cycle).
9. **Models estimate; policy decides** (rule 14): no worker boolean encodes an
   entitlement; advancement rules live with the USER.
10. **Numbers of record**: declared before results (rule 6); determinism
    repairs need pre-committed sight-unseen acceptance (R-234); the seed must
    pin the data the RNG is applied to, not just the RNG.
11. **Silent success is failure**: a run that writes nothing must not exit 0
    (R-238 cycle); absence must never read as a pass — expected sets,
    coverage evidence, and optional members are producer-recorded facts,
    never checker assumptions (R-230).
12. **Timestamps**: clock read in a separate call BEFORE composing any entry
    (four slips on record); every population carries n and as-of.
13. **Escalation**: a seat that cannot rule (frozen numbers, another seat's
    surface, USER-ONLY matter) escalates to the coordinator rather than
    acting; refusing to act on ambiguity is the correct move on record
    (BE's canon refusal; DA's sorted-wins refusal). **USER-escalations route
    through the coordinator in the same breath** (R-273): a question parked
    in a seat's own terminal waits for a user who isn't looking there —
    flag it ESCALATION-FOR-USER to the coordinator, who surfaces it where
    the user is active; the answer returns with a register cite. Substance
    unchanged: the user decides, never the dispatching seat.
14. **Peer messages**: relayed authority is verified at the user's own
    committed text when a ruling expands a seat's surface (DA's d506a06
    check — the model).
15. **A register entry citing a property of code carries a check behind it**
    in that code's suite (R-247; DA's standard, MEM's proposal): an unpinned
    claim about code behaviour drifts from the code without either the entry
    or the code noticing — rule 15 of CLAUDE.md applied to the register
    itself.
16. **A control that cannot fail must never be mistaken for a control that
    passed** (R-249; MEM's consolidation of four named instances: a fixture
    supplying what the code should produce; a guard shown only to refuse —
    boundary positive controls must ADMIT; an anchor that includes the arm
    name — it passes nothing and fails nothing; a falsifier that enshrines
    the defect as spec). Phrased on the control side deliberately: the next
    instance will wear a shape none of these four had. Every control ships
    both directions — it fires on the bad case AND admits the good one.
17. **Suite-green is not pipeline-wired** (R-251 finding, R-252 closure;
    MEM's class): a control that cannot RUN is distinct from rule 16's
    control that cannot FAIL — green suites prove the unit, not the wiring,
    and test-counting coverage cannot tell them apart (I11-2: six evaluator
    functions, all falsifier-proven, zero call sites in the runner; DB2:
    both suites green while the integration always refused). Closure needs
    BOTH halves: the wiring, AND an artifact-level guard that refuses output
    produced without it; plus a seam test that runs the integration the way
    the launcher runs it, on the producer's real emitted rows.

18. **Dispatch is batched** (R-378, USER ruling; the coordinator-side twin of
    R-377): the coordinator commits a seat's COMPLETE work batch in ONE
    dispatch — never a trickle of follow-on tasks. While a seat's batch is in
    flight the coordinator sends that seat nothing further (stop-the-line
    hazards excepted); coordination resumes only when the seat reports its
    batch done, and the next dispatch is again a complete batch. New work
    that arises mid-flight queues in the coordinator's own notes for the
    NEXT round, exactly as a reviewer's post-filing discovery waits for the
    next round. **Batching is about completeness, not idleness (R-381, USER
    directive): a closed round is followed promptly by the seat's next
    complete batch — a seat waiting between rounds is a coordination miss,
    not a discipline.**

19. **Per-seat worktrees for execution** (R-397, USER ruling on the review's
    scope-4 recommendation): each build-heavy seat has a detached worktree
    (`~/ctaNew-wt-be`, `-da`, `-de`, `-rev`, each with a `data/` symlink) for
    builds, suite runs, mutation audits and reviews — its own index, so
    staged state can never be swept by another seat's commit. Refresh with
    `git -C <wt> checkout --detach mm-research` after a fetch. LIMITATION
    STATED, not hidden: git refuses the same branch in two worktrees, and
    per-seat branches would conflict daily on the append-shared register —
    so LANDING commits stay in the shared tree under the R-387 discipline
    (explicit pathspec + `carrying_commit` in every result-bearing receipt).
    Isolation covers execution; the ledger keeps one writer path.

20. **One heavy run at a time, one CPU each** (R-551, USER review 2026-09-06):
    the scorer peaked at 6.95 GiB of its 8 GiB cap on one core, so two heavy
    runs cannot share the box. Every heavy step runs as
    (SUPERSEDED FORM, kept for the record: `flock -n <lock> systemd-run --user --scope …` — the R-628 hazard) —
    the lock REFUSES with the DECLARED conflict code **75** (`flock -n -E 75`; R-646, REV 67 §2.1 — without `-E` a refusal and a payload crash are both `ExecMainStatus=1`, measured) if another heavy run holds
    it, say so in the report; never wait on it silently, never raise either cap. The slice itself is
    capped at CPUQuota=200% so light suites can overlap a heavy run. "Heavy" =
    anything expected over 60 s wall or 1 GiB RSS. **A heavy run is never a child
    of a tool shell (R-628):** `systemd-run --scope` registers processes the CALLER
    forks, so the run sits in the launching shell's process group and dies when the
    harness stops that shell's background task — the 09-03 re-run was killed at 35
    minutes that way, nothing written. Launch as a transient SERVICE the manager
    forks, the lock held inside it:
    `systemd-run --user --unit=<name> --slice=research.slice -p MemoryMax=8G -p CPUQuota=100% --setenv=PM_DATA_ROOT=/home/yuqing/ctaNew -- flock -n -E 75 /home/yuqing/ctaNew/data/.heavy_run.lock <cmd>`
    with `-p RemainAfterExit=yes` from DE 98 on (R-648: a succeeded transient unit is otherwise COLLECTED at
    exit — `LoadState=not-found`, and `systemctl show` then returns DEFAULTS inactive/0/success — leaving only
    its Started line; a failed one stays loaded until `reset-failed`; measured on four scratch units 13:48Z).
    (no `--scope`; `flock` inside the unit holds the lock for the run's life; a held
    lock refuses with **75**, the declared conflict code, never the payload's own 1 — read the unit's
    outcome as FIVE fields (`LoadState`, `ActiveState`, `SubState`, `ExecMainStatus`, `Result`) WHILE
    `LoadState=loaded`, copied into the record at once WITH the unit's `InvocationID` (R-653, REV 69 §3.3/§3.5:
    under `RemainAfterExit` a finished unit is loaded/active/EXITED and a running one loaded/active/RUNNING,
    both `ExecMainStatus=0` — `SubState` discriminates; a killed unit reports `Result=signal` with the signal
    number as the status; `InvocationID` is non-empty only while the unit exists, so a copy without one is
    VOID whatever its `LoadState` string says); `not-found` makes the reading VOID, never "success"; after
    the receipt lands and the five fields + id are copied, stop the unit so the name is free — a loaded name,
    exited or failed, makes the next `systemd-run` FAIL). **The form's constants are declared ONCE** in
    `live/pm_research/declarations/heavy_run_form_v*.json` — readers resolve THE CHAIN HEAD (the newest version
    whose `supersedes` pair verifies), never a filename literal (v3 supersedes v2 supersedes v1); every literal
    in code reads it or asserts equality with it in its selftest, and no runner or producer exits 75 for any
    other reason. THREE GUARDS on those words (R-649, REV 68 §3.1–§3.2): a check that depends on the declaration
    FAILS when the file is absent — never skips (a skipped check reads as a passed one); the check on a
    launcher reads the LAUNCHER'S BYTES (the shell literal in `be_heavy_run.sh` is the number the running unit
    uses; a Python constant agreeing with the declaration proves nothing about it); and the declaration's CONFLICT CODE is
    grounded by exactly ONE DRIVE — a unit launched against a HELD scratch lock with a payload whose exit map
    excludes 75, its five fields and `InvocationID` read while loaded and copied into the launcher owner's
    receipt (the drive grounds the code's behaviour; the declared lock PATH is grounded separately by inode) —
    because literals agreeing with a literal is not a measurement; and "the check on a launcher reads the
    launcher's bytes" means the bytes the RUNNING unit executed — the unit's own `ExecStart`, or
    `/proc/<PPid>/cmdline` from inside (four worktrees share one filename; a check reading its own tree's
    launcher satisfies the words, not the property — REV 69 §4). 75 is
    `EX_TEMPFAIL` (sysexits.h), the code a well-behaved program would CHOOSE for "try again": from outside a
    unit a 75 reads "refused OR a producer that broke the declaration"; the only enforcement is inside each
    producer (`75 not in <its declared exit codes>`, asserted in its selftest, the map published in its receipt).
    Poll the UNIT, not a
    child PID; a run's survival of `kill -TERM` on the launching shell's process
    group is a battery falsifier. **The journal is NOT the record (R-641, REV 66 §3.1):**
    it rotates within hours (DE 84's Started line was gone four hours later). A
    number read from the journal is copied into an artifact at the moment it is
    read, with the source's retention state named — the retention state is a MEASUREMENT
    (the oldest entry the journal holds, the query that produced it, and its own as-of; the
    window's start advanced ~15 min in 18 min on 09-06, so a state named once and re-quoted
    later is stale; a typed string satisfies the words and not the property — REV 67 §3.1a);
    a receipt or record carries its own journal lines at emit, filtered on the run's
    **InvocationID** with BOTH fields (`_SYSTEMD_INVOCATION_ID=<id> + USER_INVOCATION_ID=<id>`: the
    payload's lines carry the first, the USER manager's Started/Consumed lines the second —
    `INVOCATION_ID` is the SYSTEM manager's field and matches nothing here; measured at R-646 —
    a unit NAME names every run ever launched under it, 99 manager lines for be64book by 13:37Z),
    cross-checked against `-u <unit>`'s count, and a check compares the id to the unit's; a copy
    returning 0 lines where `-u` has lines is a REFUSAL of the copy, never a record; no control's
    verdict may depend on journal retention.

21. **Landing in the shared tree is add, commit, push — nothing else** (R-576):
    a seat that lands an artifact from `/home/yuqing/ctaNew` runs exactly
    `git -C /home/yuqing/ctaNew add -f <paths> && git -C /home/yuqing/ctaNew commit -F <msgfile> -- <paths> && git -C /home/yuqing/ctaNew push origin mm-research`.
    Never `checkout`, `switch`, `reset`, `rebase`, `stash` or `pull --rebase` in
    the shared tree (two detachments and one orphaned commit on 2026-09-06 — the
    orphan's real mechanism, from both reflogs: a bare `git` inheriting a `cd`
    into a worktree earlier in the SAME compound command, so the commit landed on
    the worktree's detached HEAD while `push` pushed the shared tree's branch;
    therefore every git call names its tree with `-C`, never a bare `git` after a
    `cd`). A
    refused push means another seat landed first: `git -C … fetch` and retry the
    push only if `git -C … status --short` is EMPTY (in a seat worktree, empty except
    the single `?? data` line that is the ledger symlink, R-625); otherwise LEAVE the commit,
    REPORT it as stranded, and continue — the coordinator rebases stranded
    commits onto origin at the first clean-tree moment (R-586/R-588; a retry can
    never fast-forward once a local commit exists, and the shared tree is
    dirty for the length of every MEM batch). **Landing CODE from a worktree
    (R-623):** the worktree's exact bytes are copied into the shared tree and
    committed by pathspec in the SAME compound command — nothing is edited, no
    battery is run, and nothing waits in the shared tree; a modified source file
    seen in the shared tree outside that command is a breach (BE 60, corrected
    at 11:43Z). A seat that prefers not to copy may push its worktree's commit
    directly (`git -C <wt> push origin HEAD:mm-research`) when origin has not
    moved; a refused push is stranded and reported as above.

22. **A heavy run's code is frozen until its receipt lands** (R-603, REV 49 §0):
    a run executes from a worktree whose HEAD is not moved and whose files are
    not edited until the run's receipt has landed — the Gate-1 runner stamped
    its own provenance by re-reading `__file__` at EMIT time, so a landing to
    the runner 13 minutes into a 1.5-hour run would have named code that did
    not execute, and the committed-bytes guard PASSED because the replacement
    was committed. A seat that must land code during its own run lands from a
    SECOND worktree. Runners AND every heavy producer (the fragment, tape and
    book builders included) capture at IMPORT the digest of every module in
    their import closure under `live/` plus the worktree's HEAD sha, stamp
    those into the receipt, and refuse the emit by name if any moved (REV 51
    §3, R-605: a digest of one file closes a third of the class; the closure
    and HEAD close it; a practice that depends on noticing is not a control).

## Cadences

- Day verdicts: 00:06Z per coin; 08-28 under the OLD count bar; 08-29+ under
  day-bar v2 ONLY after the reviewer's HOLD RELEASED (R-238, Codex filing
  7954585).
- Race accrual: freeze-commit epoch (1787897340); accrual ≠ day quality
  (split_verdict, R-240).
- Boundary deploys: 00:00:00Z exactly, per runbook, era-stamped.
