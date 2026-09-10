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
    **Producer exit codes and relaunches (R-709, REV 79 §2.1/§2.2).** A non-zero `ExecMainStatus` is one of THREE
    kinds: 75 = the wrapper's refusal; a code declared in the producer's block of
    `live/pm_research/declarations/producer_exit_maps_v<N>.json` (chain head by the pair; one producer's block per
    version, by that producer; 75 may not be declared) = a VERDICT by name; any other non-zero = UNMAPPED, recorded
    verbatim, NOT a refusal, NOT a failure, and it does NOT satisfy a GO conditioned on that run -- the GO waits until
    the producer declares the code by pair and the record is re-read. A capture record names the producer module and
    the resolved kind beside the verbatim code. A RELAUNCH under one GO needs all four: the earlier launch wrote
    nothing or its record is superseded by the pair; the lock free and the unit `not-found` before each; every
    launch's five fields + InvocationID in a resolvable RECORD (a Q-row is prose) -- a UNIQUE unit name per launch
    (`<name>_1/_2` or the stamp) so `-u` is a per-run query again; and the producing CODE byte-identical between
    launches, or the change reviewed before the next launch (R-620's hazard is an unreviewed fix going live inside
    one batch, not the second launch).
    **Declaration chains: a landed version is IMMUTABLE and the write is a compare-and-swap (R-711, REV 81 §1.4).**
    Chains are versioned by ONE new `<family>_v<N>.json` per landing, never by editing a landed version (the exit-map
    chain lost two seats' blocks to in-place landings on 2026-09-06). The CLOSURE is at the moment of the write: the
    emitter REFUSES to write `<family>_v<N>.json` if a file already exists at that path, or if the head's digest at
    write time differs from the head it read -- one shared implementation, imported, never re-typed. A landing whose
    diff shows M on an existing version (instead of A on a new one) is a refusal; the landing-time head re-read is a
    discipline and `scripts/declaration_immutability.sh` is the detection -- neither is the rule.
    **A superseded fork is repaired FORWARD, never by reverting the fork (R-760, R-761; REV 90 §B6/§B7):** when a
    landed version is found edited in place, the repair is the NEXT version superseding the EDITED bytes by the pair
    (as DA 121 did with `producer_exit_maps_v8`); reverting the edit moves the bytes under any later version that
    already names them and breaks its pair (the coordinator did exactly that at `31c5208` and had to undo it). The
    fork stays in history as `FORKED_BY_EDIT`; the checker names whether a later version supersedes it. **And the
    CAS is per-worktree:** two seats can each write the same `_v<N>` from the same head in separate worktrees, so every
    version LANDING re-reads the head after the copy and before the commit (rule 21's pre-copy check) and re-versions
    from the new head if it moved.
    **A field that changes what a chain resolves to is declared where both readers resolve it (REV 83 §1.2, R-717):**
    `supersedes` and `also_supersedes` (a merge version naming every tip it closes, each by pair) are the chain's link
    fields and live in `live/pm_research/declaration_chain.py`'s contract; no seat adds a link field its own resolver
    reads and the shared one does not -- the design family diverged within one round that way.
    **Register rows land through `scripts/land_register_row.sh` (REV 83 §4, R-717):** the dirty-register hold, the file
    pathspec, the post-condition on the commit's OWN diff (exactly the caller's row ids added, zero foreign rows, no
    landed line changed), the capture-test-trim push, and the trailer `Landed-By: land_register_row.sh <sha256>` -- so
    "was this row landed by the script" is a query over the history; a row that appears without the trailer is visible,
    which is the closure (the hold alone has a race and can be forgotten).
    **A known-bad establishes its own baseline inside the cell and asserts a delta -- never a level read from a counter
    the process shares (REV 83 §5, R-717; third instance: REV 53 §0, REV 59 §3, DE 110).** A battery whose verdict
    changes when its cells are reordered is measuring history, not the property: run the cell alone and run the battery
    shuffled -- if either verdict differs, the cell is measuring the process.
    **A shared implementation's falsifier is a shared cell; an importer's independent cell tests the importer's OWN
    behaviour at the seam (REV 84 §3.2 as amended by REV 85 §3, R-726).** Every importer RUNS `declaration_chain --falsify`
    (as a subprocess) as one cell of its own battery, so a regression in the module fails every importer at once; an
    importer keeps independent cells only for what its own verdicts rest on AT THE SEAM (its translation of a refusal
    into a named per-family status, its scope resolution) -- never a re-test of the module's invariant, which is the
    unreachable duplicate DA retired at DA 110.
    **A cited artifact is locatable (R-601, REV 85 §4, R-726):** a row naming a receipt or record by digest names its
    PATH in the ledger, or states it is scratch-only with the absolute path -- a digest without a resolvable location is
    a pin to nothing.
    **An instrument verifying a PAST act resolves every declaration by the pair the act recorded -- never by the head
    (REV 86 §8, R-729; third instance: the seal scope from the carrying commit, REV 75 §2; a receipt's design pin by its
    own pair, REV 73; the read's declaration by `pre_state.declaration_sha256`, DA 112).** The head is for writers; the
    pair is for readers of history.
    **Nightly units run from a PINNED deploy, and a landing that touches a pinned file re-pins in the SAME round (REV 88
    §3, R-741; both halves, neither alone).** The pin is a declaration by pair (`{unit, commit, files: [{path, sha256}],
    deployed_at}`) written through the CAS at every deploy; the unit's drift guard compares against the head pin and REFUSES
    (a declared code -- rc 7 `DEPLOY_DRIFT` for `da-midnight-verify`), writing nothing, the night a catch-up; the non-head
    census marks a pinned file whose digest moved without a new pin. A refusal there is the guard working.
    **A halted seat's worktree (R-627's clause, REV 81 §4).** "Never touch a seat's worktree while the seat works" --
    *idle* includes *halted for a reset*: a preservation-only commit in a halted worktree is permitted, UNPUSHED, the
    bytes unaltered, the act disclosed in the register by commit id; the rows then land in the shared register attributed.

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

    **The ONE sanctioned `rebase` in the shared tree (R-755, R-756; REV 89 §5.1):** the two
    coordinator landing scripts (`scripts/land_register_entry.sh`, `scripts/land_register_row.sh`)
    may rebase THE COMMIT THEY HAVE JUST MADE onto origin when origin moved during the landing --
    under the register lock, only when no OTHER path is dirty, aborting on failure; otherwise the
    exit says STRANDED and the commit is reported, never rebased over another seat's files. No
    other rebase in the shared tree exists. **Before a copy-land, the guard runs INSIDE the act
    (BE 91's disclosure, REV 89 §5.1 (3)):** the copier verifies that none of the paths it is
    about to overwrite has moved in the shared tree since the worktree read them
    (`git -C /home/yuqing/ctaNew diff --name-only <base>..HEAD -- <paths>` EMPTY) and refuses by
    name otherwise; an afterwards-diff is how you learn you were lucky. **A scratch drive is a
    SCRIPT with `set -u` and `cd <scratch> || exit`, never an inline chain in a fresh shell
    (R-756):** an unset scratch variable sent a coordinator's known-bad drive -- `git config`, a
    `printf` over the register, `git add -A`, `commit`, `push` -- into the shared tree; reverted
    at 0539a36, ninety seconds of a seven-line register at the tip.

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

23. **A numbered queue is worked END TO END** (R-835, coordinator failure
    2026-09-09; MEM 289 proposed the wording): when a seat is given a numbered
    queue, it works every item and reports ONCE at the end — or the moment
    something BLOCKS it or a falsifier FAILS. **It does not stop after each
    item to be told to continue.** The evidence is expensive and specific:
    on 2026-09-09 a seat finished one item, reported, and waited 14 minutes
    after ending with "unless you redirect", then **33 minutes after landing
    a critical fix** — ~47 minutes lost on a queue where nothing needed a
    decision from anyone. The user asked directly: *"Why you become idle?
    Can you just proceed to fix the issues"*. The standing instruction was
    issued and **six defects landed in one 17-minute batch afterwards.**
    The coordinator's half of this rule: **arm a Monitor on every heavy unit
    and prompt the seat at its exit** — seat in-pane watchers are DEAD
    (R-821), so a unit that exits unwatched leaves its seat idling
    indefinitely.

24. **Every seat keeps a procedure file** (R-831, R-835): `<SEAT>_PROCEDURE.md`
    in `workspace/`, written and maintained BY that seat, holding what a fresh
    occupant would otherwise rediscover — the launch forms, the surfaces owned,
    the refusal names, the contracts, and what has cost the seat a round.
    **A seat at the end of its context WRITES the file rather than reporting
    it** (R-831: a stop answer scrolled out of the pane within four minutes;
    one file write costs the seat almost nothing and the knowledge survives).
    A fresh seat that faults its own file within an hour is the file working,
    not failing — record the correction and land it.

25. **After a repair lands, at least one seat per round hunts with NO TARGET
    NAMED** (R-839 through R-844, 2026-09-09). On the night the corrected books
    were about to be built, three seats were told to go find what was wrong and
    to choose where to look; **they found six defects in thirty minutes, all on
    the path the rebuild would have run, and the seats working to specified
    briefs had found none of them in the preceding hours.** The record is
    unambiguous about why: every defect that has actually mattered in this
    programme — the unloaded value head, the look-ahead, the non-finite score
    inside a fix declared complete, the pin blocker none of seven defects named,
    the era label, the score key, the engine's routing — was found by someone
    who was NOT told where to look. A brief tells a seat what to check; it also
    tells it what not to. **"I looked in these six places, drove these controls,
    and found nothing" is an accepted and valuable answer** — DA 140's clean
    round on the placement latency was the second most useful of that night.
    The coordinator's half: name what is already covered, to prevent duplication,
    and NOTHING ELSE.

26. **A claim of unchangedness made at the moment its input changed shape is
    where you look** (DA, R-842). DA 139 found that the policy engine ignores a
    score event's generation by reading the commit message that said *"THE ENGINE
    WAS ALREADY CORRECT … Nothing in the policy is changed"* — written while
    handing that engine multiple events per generation for the first time. The
    engine was unchanged; what it was fed was not. Treat every "this did not need
    to change" written beside a shape change as an unverified claim with a known
    place to check.

27. **Verifying a fix outranks making the next one** (R-844). Twice in one night
    a review of a landed repair was worth more than the repair: REV 110 proved
    the cancel-count instrument sound by MUTATION rather than by reading, and
    REV 113 found that the sealed-value guard rewritten that morning had a hole
    on the commonest form in English prose — a reason ending a sentence with the
    value — because it drove its own corpus against BOTH versions instead of
    trusting four green cells. **A guard LOOSENED on the strength of a false
    positive is the highest-priority thing to re-drive**, and the question to ask
    is always "is there a form the OLD one caught that the NEW one does not" —
    the regression a green battery cannot show. REV's diagnosis generalises:
    when a guard false-positives, find out whether the defect is in the
    COMPARISON or in the MATCHING, and change only the one that is wrong.

28. **THE EVIDENCE IS RECORDED AND THE CHECK IS OFF** (REV, R-848; found four times in
    one night and named by the seat that found them all). *This pipeline repeatedly
    records or publishes the right thing and leaves the check that would make it
    load-bearing switched off.* The instances: **REVIEW 111** — the book's
    `producing_code.import_closure` records the 49 modules that computed its scores and
    no consumer reads it back; **REVIEW 115** — the writer pads a short settlement list
    and only the reader refuses, long after the run that produced the file;
    **REVIEW 116** — the receipt publishes `read_ledger(path, expect_sha256=…)` and the
    parameter **defaults to `None`, so the unsafe call is the shorter one**;
    **REVIEW 117** — `exp_m6_settlement.read_at` returns `(value, sample_time)` and the
    consumer writes `x0, _ = read_at(...)`, so **the staleness of the sample that decides
    a winner is computed, returned, and thrown away on the same line.** Two more of the
    same shape: `require_book_declares_L` was never armable in any landed params version
    (DA 140), and the era default answered for a day it was never given (REV 112).
    **THE RULE, in REV's words: when a producer already returns the evidence, the consumer
    must CARRY it or REFUSE on it — discarding it into `_` is the defect, not the absence
    of the evidence.** And its sibling, which is gate item 4's root in one sentence:
    **identity that exists in the data and is not load-bearing is a defect waiting for a
    boundary.** When you add a producer that returns evidence, add the consumer's refusal
    in the same change, and default the safe call, not the short one.

29. **A PARAMS LANDING NEEDS ITS DA-SIDE RE-RUN IN THE SAME ROUND** (DA, R-850).
    `da_land_gate` refused with two unexpected reds — `da_gate1_day_verdict` and
    `da_accrual_report` — both green in the same seat's runs an hour earlier and both
    **outside the changed module's import closure, established by AST rather than by
    reading.** The failing cell named its own input: *"THE BARS COME FROM THE NEWEST
    PARAMS PRESENT … `de_multiday_gate1_params_v25.json` … versions present [1..25],
    newest chosen"* — and v25 had landed eight minutes earlier, mid-round, from another
    seat. **A checker that resolves "the newest params present" changes verdict the
    moment anyone lands a params version. That is CORRECT behaviour, not a defect** —
    and it means the seat landing a params version owes the DA-side re-run in the same
    round, and the seat that finds the red owes the AST check before blaming its own
    change. **A seat that lands with a gate refusing SAYS SO PLAINLY** — DA did, by
    pathspec, naming the reds and why they were not its doing. A seat quietly landing
    past a red gate is how a programme stops being able to trust its own checkers, and
    it is worse than the red.

30. **A COMMIT MESSAGE CAN DESCRIBE A CHANGE THE COMMIT DOES NOT CONTAIN** (DA, R-852).
    DE 162's message said mode C's message "stops blaming the wrong subsystem"; DA drove
    the three cases and found the message **byte-identical to before** — the phrase
    appears **zero times in the diff, neither added nor removed** — and confirmed no
    later commit fixed it before calling the item open. Worse, the same round's new
    `ROW_AFTER_GENERATION_END` exclusion **added a SECOND route to the misattributing
    message.** *"It is worth catching precisely because a green battery doesn't test it
    and a careful reader agrees with it."* **A battery tests behaviour; nothing tests
    prose. So a verifier checks the claim against the DIFF, not against the message** —
    `git show <commit> | grep` for the string the message says changed, and confirm no
    later commit did it, before either accepting or reopening. This is the same shape as
    rule 10 (compute predicates, never print conclusions) moved up one level: a
    hardcoded verdict beside a table has contradicted the table three times, and now a
    commit message has contradicted its own diff.

31. **ON THE SHARED TREE, COMMIT EACH FILE AS SOON AS IT PARSES; LET THE BATTERY GATE
    THE PUSH, NOT THE COMMIT** (DA, R-857). DA completed a full round — builder,
    wirings, cells, battery green at 38 — then found all four files clean with every
    edit gone, **and the reflog showed nothing, which is consistent with
    `git checkout -- <path>`: it does not move HEAD and leaves no reflog entry.**
    R-557 already forbids that command here. DA's own share, in its words: *"I held
    four edited files uncommitted while a battery ran. Test-then-land is right, but the
    gap between them is my exposure to shorten."* **A local commit is recoverable from
    the reflog; an uncommitted edit is recoverable from nothing.** This binds every
    seat and the coordinator.

32. **ENUMERATE BY THE OPERATION THAT CONSTITUTES MEMBERSHIP, NEVER BY A SPELLING —
    AND PREFER MEASUREMENT TO ENUMERATION** (REV, R-857). Three times in one night a
    set everyone treated as complete was a sample, and **every time the missed member
    was the consequential one**: the pin-site list was two by inspection and three by
    construction (the missed one SILENT — REVIEW 111); the consumer list was three by
    the spelling `in gs` and five by operation (the missed two COMPUTE EXCLUSION
    COUNTS — REVIEW 122); and `ECONOMIC_FIELDS` was built by inspection before R-801
    made settlement P&L the primary endpoint and never revisited, **so the seal
    protects the diagnostic and not the result** (REVIEW 123). Also REVIEW 123:
    `SCORING_PATH_MODULES` is a **hand-typed 5 of a RECORDED 49**, not even a superset
    of the ten-module cascade — so a closure predicate checks a sample of the closure
    the book already records. **The test to apply to any set: what operation would
    constitute membership; was the set built that way; would a missed member be LOUD or
    SILENT. Only the silent ones matter.** And the model for repair is already in the
    codebase: **the heavy-run lock decides heaviness BY MEASUREMENT, not by
    enumeration** — where a set can be derived from what the artifact already records,
    derive it.

33. **A VERIFICATION IS ONLY AS GOOD AS THE IDENTITY BETWEEN WHAT IT EXAMINES AND
    WHAT IT IS CITED FOR** (USER review at `7d374b4`, R-859). The programme's
    verification standard — *closed AND verified by a seat other than the implementer* —
    caught two bad closures in one night and still passed three defects the user found
    in minutes. Each failure had the same shape: **the check was TRUE of the artifact it
    examined and FALSE of the claim it was cited for.**
    - **The wrong id space.** The "one cancel per generation" invariant passes because
      it checks POLICY-generation ids; production repost cancels one REFERENCE
      generation twice (`7`, `7.r1`). REV cited it as proof the cancel is the action
      and the coordinator carried that into a USER RULING.
    - **A subset accepted as the set.** The book-code predicate was verified to be
      capable of passing; nobody tested that it passes on a SUBSET, and it returns
      `BOOK_SCORING_CODE_MATCHES, n_checked: 1`.
    - **A branch nobody reached.** The cancel-matched null was built, reviewed and
      recorded as implemented in FOUR register entries; `run_day()` never passes its
      arguments, so the historical row-matched null ran.
    - **A probe that could not have found what it looked for.** A `Path` passed where a
      parsed receipt was wanted (REVIEW 105) — *the conclusion survived and the evidence
      did not*.
    **So every green carries three answers, not one: what did the check ACTUALLY examine;
    what claim is it cited for; are those the same thing.** And for any predicate:
    **drive it to PASS on the real thing, to FAIL on a known-bad, AND to REFUSE a
    partial input** — a checker that matches on whatever it happens to find is not a
    checker. Where a fix is recorded as landed, **drive the path that reaches it from
    the entry point**, not the unit alone.

## Cadences

- Day verdicts: 00:06Z per coin; 08-28 under the OLD count bar; 08-29+ under
  day-bar v2 ONLY after the reviewer's HOLD RELEASED (R-238, Codex filing
  7954585).
- Race accrual: freeze-commit epoch (1787897340); accrual ≠ day quality
  (split_verdict, R-240).
- Boundary deploys: 00:00:00Z exactly, per runbook, era-stamped.

34. **Days from 2026-09-08 onward are PROTECTED — do not touch them.** No probe,
    diagnostic, smoke test, quick look or sanity check against any day from
    2026-09-08 on. Rule 11 does not care about intent: seen days are consumed,
    and a five-minute curiosity on 09-08 spends a validation day that cannot be
    bought back. **Every day this programme currently holds is already spent** —
    09-03..09-06 are the latency days, and 09-07 carries a sealed run with 500
    drawn nulls — so the untouched set starts at 09-08 and reaches five (the
    rule-8 minimum) around **2026-09-13**. Exercise corrected pipelines on the
    days already consumed; they are free to re-run precisely because they are
    spent. If a fix genuinely cannot be demonstrated on a consumed day, ASK THE
    USER before the day is touched, never after. (R-873)

    **34a. AMENDMENT — WRITING DOES NOT CONSUME A DAY; READING DOES (USER ruling,
    2026-09-10).** `pm-evaluation-pipeline.timer` auto-processes one day every six
    hours and has already written Tier-2 artifacts for **09-08**, a protected day.
    That does NOT consume it. Rule 11's hazard is **selection on seen data**, and
    the pipeline fits nothing, picks no threshold, computes no interval and
    authorizes nothing — it writes `maker_edge_per_share = q_up * (price_up −
    outcome_up)` mechanically. **A day is consumed when a PERSON OR SEAT LOOKS at
    it.** So the timer keeps running and the artifacts accumulate UNREAD.
    **The prohibition therefore extends to the OUTPUTS: do not read, summarise,
    plot, aggregate or quote `data/pm_5min/tier2/**/day=2026-09-08/` or any later
    day.** Their existence is not permission. If you need to know whether such a
    day QUALIFIES, read completeness/coverage/quality metadata only — DA 182
    demonstrated that boundary in practice. **The eye is the thing that spends the
    day, so guard the eye.**

35. **A limit that lives only in a declaration does not bind the result.** If a
    document states a constraint on how its output may be read — "this cannot
    validate", "matched only on X", "this set is superseded" — the OUTPUT must
    carry that limit as a REQUIRED FIELD, not the declaration as prose. A reader
    resolves fields; nobody reads the design doc beside the number. **REV has now
    found this same shape four times** (`MEMBERSHIP_LIMIT`, `matched_on`, the
    superseded-set, and the null's cannot-validate clause), which makes it a class
    and not an instance. **And the guard on such a clause must test the PROPERTY,
    never the TEXT** — the null declaration's five-day check was `"5" in <prose>`,
    which "we have 5 llamas" satisfies. That is rule 15's silent-checker failure
    wearing a different hat: an instrument that cannot fail is not an instrument.
    The fix is always one required field on the output plus a predicate that
    evaluates the thing, never more prose. (REV 149, R-874)

36. **A derived ratio carries its caveat out of the artifact, or it does not
    leave.** REV 151 established that the 09-04 artifact is structurally SOUND —
    `DROPPED.total_cents` and `ALL_TRANCHES.total_cents` sit in the same block as
    `UPPER_BOUND`, `SCOPE` and `HOW_IT_MUST_BE_SAID`, so the caveat is a SIBLING
    of the number. **The detachment happened in PROSE**: the coordinator computed
    62.37 % into a summary and left the upper-bound qualifier behind. Rules 34 and
    35 govern artifacts; this one governs REPORTING, and it binds the coordinator
    first because that is where it broke. **A ratio that is not IN the artifact is
    a claim you are making, not a number you are reading** — so it travels with
    the qualifier that sits beside its inputs, in the same sentence, every time it
    is written. "62.37 %" alone is a different claim from "62.37 % as an upper
    bound on one BTC-only day with no null drawn." (REV 151, R-878)

37. **A hand-off must never rest on a seat's turn staying alive.** "Launch when
    X finishes" is not a hand-off if the seat's turn ends before X finishes — the
    instruction simply evaporates and the resource sits idle. **This cost the
    programme three times in one day: fourteen minutes after the 09-04 retry (BE
    told to launch 09-06 "when the retry finishes", turn ended first), roughly six
    hours after 09-06 (nothing armed at all), and an inverted queue when BE and DE
    were dispatched in the SAME round with BE told to wait for DE — BE launched
    immediately and a 74-minute build went in front of two 4-minute runs.**
    **The rule: a trigger belongs to the coordinator or to a Monitor, never to a
    seat's good intentions. Do not dispatch two seats that contend for the same
    resource in one round.** If a seat must wait on a resource, it polls in a loop
    that keeps its turn alive, or it does lock-free work and the coordinator hands
    the resource over. The coordinator absorbs the collision when it happens —
    killing a long-running unit to restore an order costs more than the delay.
    (coordinator, R-879)

38. **Convergence is evidence only if the instruments differed.** Two seats given
    a question separately, reaching the same verdict, is **one observation** if
    they used the same instrument. R-878 recorded DA 180 and REV 151 agreeing and
    called it *"why the withdrawal is safe rather than merely cautious"* — but
    both compared `TERMINAL_MARK_ENDED_IN_GAP` against `n_gap_bearing_windows`,
    so they could only ever agree, and both were wrong (R-883). **Independence is
    a property of the INSTRUMENT, not of the dispatch.** Before citing agreement
    as corroboration, name what each seat actually measured; if it is the same
    field, the same query or the same helper, you have correlation and not
    confirmation — the same class as reading an exact reproduction as proof.
    (coordinator, R-883)
