# Review — DA round 20 HELD at `d37c3d9` (+ the row-only `0cd18ba`): four checks that could not fire, and one wiring that still cannot
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tips read from the shared object store**, never from `~/ctaNew-wt-da`: chain
**`d37c3d9`** and its row-only child **`0cd18ba`**. Verified at the blobs:
`da_blackout_mask.py` **`10d02c092939be48`**, `da_forward_day_verify.py`
**`bb7213d6e4f78dee`** (`EXPECTED_CHECKS = 254`), `pm_host_load_join.py`
**`8689db8b34697d4b`** (`EXPECTED_CHECKS = 39`), and the new
`fixtures/sysstat_parser_control.sa` **`d663ce6b5d333ee6`**, **31,428 B**.
**Request of record:** `REQUEST_DA_ROUND_20_2026-09-03.md`. **Composed 2026-09-03T02:15:42Z.** One filing, per R-377.

**Constraints observed.** Executed in `~/ctaNew-wt-rev` at `--detach d37c3d9` and
`--detach 0cd18ba`; **no `git` command run in any seat worktree and nothing pushed** — the tips
stay HELD. `~/ctaNew-wt-da`, `~/ctaNew-wt-be`, `~/ctaNew-wt-de` never read;
`be_forward_day.py` never run; `da_midnight_verify.sh` never run; `DA_MIDNIGHT_MODE` never set;
no `--out` run of my own. Seven mutants applied to my worktree copies and **restored
byte-identical** (`git status --short` **0** after each). Nothing written under `data/`:
`derived/` **178 before and after**. `git worktree list`: **34 at quiescence** — it read 35 while
BE round 12 held a transient `be-r10-c3-stale` entry (BE's, as the request notes), and my own probe
worktree was created and removed.

## 1. Counts — CONFIRMED at both tips (item 1)

Ten modules at `d37c3d9`, **both launchers**, rc 0, **stderr empty**:
**254** (0 skipped; `ran+skipped=254`) · **43** · **39** · **53** · **24** · **21** · **19** · **9**
· **39** · **5**. At `0cd18ba`: `da_blackout_mask` **43** and `da_forward_day_verify` **254**,
both launchers. Every number matches DA's and the coordinator's.

## 2. DA18-R1 — **CLOSED**, and the request's premise about LANDED is too strong (item 2)

**(a) The defect is real at the row-only arrangement — replayed outside the module.** A probe
worktree at `d37c3d9`, the four DA files copied in from my checkout of `0cd18ba` (identical bytes),
staged and committed: **rc 1, `nothing to commit, working tree clean`, HEAD stays `d37c3d9`**. The
same replay with `FIXTURE_PLANT` appended to `pm_tape_density.py`: **rc 0, HEAD moves to
`b60fd92`**. So the plant is what makes the fixture commit possible, and the empty commit was not
hypothetical.

**(b) and (c) driven, both red by name, zero tracebacks:**

| mutant | dies at |
|---|---|
| `FIXTURE_PLANT = ""` | *"DA18-R1: and the **PLANTED file is in the commit** …"* |
| `_commit_refusal` always `None` | *"DA18-R1 **KNOWN-BAD**: a commit with nothing staged is REFUSED and named EMPTY …"* |

**The premise that LANDED has no in-module falsifier is not right, and I drove the counter-example.**
It is true that mutating the *content* of any of the four copied files cannot make the commit empty
(the mask is one of them). But LANDED is about the **staging step**, and that is mutable: replacing
`git add` with `git status` (the plant still happens, nothing is staged) makes the commit empty and
**LANDED goes red by name**, quoting `_commit_refusal`'s EMPTY text — rc 1, zero tracebacks. So the
three checks have **three distinct falsifiers**, one per failure mode: the plant stops working
(PLANTED), the staging stops working (LANDED), an empty commit stops being named (KNOWN-BAD).
**Sufficient — and better than claimed.**

**Is the run-twice known-bad the failing arrangement or a proxy? It is the arrangement.** The
defect's state is "the staged tree equals HEAD"; running the same commit twice produces exactly that
state. Only the *reason* the files match differs (the first commit wrote them, rather than the
parent having left them unchanged), and no check depends on the reason.

## 3. DA17-R1 — **CLOSED**, with one claim corrected (item 3)

The predicate is now identity alone and the three falsifiers are fed **through** it. Driven, both
red by name, zero tracebacks: `return … != _here` → *"CO-10 FALSIFIER: a producer reporting HEAD~1
(`d37c3d9`) is REFUSED"*; `return True` → *"a producer reporting the PARENT's HEAD (`0cd18ba`)
… is REFUSED"*.

**But the comparative claim does not reproduce — DA20-R1.** DA says the CO-10-returning mutant
"was green before and is red now". I ran the **pre-round module** (`5a11ee9`'s blob, in my own
worktree) twice: unmutated → **38 checks, rc 0**; with the same `!= _here`-alone mutation →
**rc 1, red at "DA16-R1 FALSIFIER: a producer answering HEAD~2 … is REFUSED by the identity
control"**. The round-17 falsifier I reviewed already killed that mutant, because it feeds a
HEAD~2 producer **through the predicate** and asserts `not`. The code change stands on its own
merits — one predicate instead of two, three falsifiers instead of one, and the `!= _there`
assertion retired — but "green before, red now" is not what the artifacts say.

## 4. The host-load split — the FORM, weighed (item 4)

**The parser fixture: right form, and small enough.** 31,428 B is well inside "small" — two orders
below the 1,087,654 B text report the split rejects, and a rounding error against the repo's
large-file rule. Binary is not a preference but a constraint the module states: `sar -f` refuses the
text form, so a text fixture could not be a parser control at all. Content-addressing by **sha256
and bytes and rows**, with the column binding checked by a second independent read agreeing
row-for-row, is the shape I asked for in R-489 (C).

**One property to make louder rather than fix:** the module's own note says the sysstat binary
format is **versioned** (`sysstat 12.6.1`). So the control proves *this host's `sar` reads a known
12.6.1 archive and our regex binds its columns* — the right property, since the deployed reader is
what the join depends on — but a sysstat upgrade will turn the control red for a reason that is not
the parser's. It fails **loud**, which is correct; the refusal should name both versions (the
fixture's and the host's) so the next reader is not left guessing. Not a finding.

**`sar -f` shelling out: acceptable, and preferable.** A pure-Python sysstat parser would be a
second implementation to maintain and would not prove what the production path actually does.

**`YEAR, MONTH = 2026, 8` hardcoded (`:61`): acceptable **now**, and the next DA17-class item.**
The failure mode has moved from a **wrong number** to a **named refusal**, which is the right
direction, and I confirmed it on this host with the module's own read-only run: `sa01`/`sa02`/
`sa03` are refused as *"sa01 was last written 2026-09-01, so it holds September data, not
2026-08-01"*, nine rows printed, and the trailer computed (*"NO ROW FOR 08-25 in this run (6 day(s)
joined)"*). But correctness is still a function of the calendar: on 2026-10-01 the same constants
refuse every October archive and two literals must be edited. **The derivation already exists at the
refusal site** — the true month is read from the archive's mtime — so lifting it to the constant is
a small round-21 item, not a redesign.

## 5. R-486 `governs` — the wiring has no falsifier: **DA20-R2 (MEDIUM)** (item 5)

**Reproduced independently.** Removing the **per-coin** production call
(`annotate_governance(cp, _reg)`, `:2337`) → suite **green, 254**. Removing the **day-level** call
(`annotate_governance(preds, regime)`, `:2403`) → suite **green, 254**. The suite exercises
`annotate_governance` on synthetic tables and asserts nothing about an **emitted** verdict, so both
production wirings can be deleted without a single check noticing. (For completeness: three of the
suite's own fixture calls go red when removed, and one — `_fail_tbl` at `:4252` — does not.)

This is rule 17's own shape — *suite green is not pipeline wired* — in the feature R-486 asked for,
and DA's rule-17 evidence is a hand `--out` run in DA's worktree, which is a measurement of that
run, not a property of the code. **It is a finding, and it belongs in the HELD chain before the
landing**: nothing has been pushed, so amending the unpushed commit engages no rule-13 question (the
request says so), and the landing is precisely what makes the wiring live on artifacts a reader will
resolve.

**The closure, in this module's own idiom:** assert on an **emitted** verdict, not a synthetic
table — build a small verdict through the production path into a temp dir (the `--out` path the
coordinator already used) and assert that every day row and both `per_coin` blocks carry
`governs` and `governs_why`. A parse-level "both call sites are present" check would be the class
this programme keeps closing (a call asserted by its text) and should not stand alone.

## 6. The two unrepaired observations — both reproduced (item 6)

**(i) Silent shrink under a tapeless root — round 21, LOW-MEDIUM.** With an empty directory as
`PM_DATA_ROOT`: `da_governed_verdict_preflight` **39 → 38** and `da_verdict_check` **21 → 19**,
both **exit 0 with no named skip**, while `pm_tape_density` under the same root does it right
(**8 ran + 1 SKIPPED, named, `ran+skipped=9`**). A suite that runs fewer checks and still says
"passed" reports green for checks nobody ran. Not a landing blocker — the unit path runs canonical
roots, where the counts are full — so round 21.

**(ii) `pm_host_load_join` has no `PM_DATA_ROOT` branch — round 21, LOW.** Zero occurrences of
`PM_DATA_ROOT`; `REPO = Path(__file__).resolve().parents[2]` (`:57`) and `GAP_LEDGER = REPO /
"data/pm_5min/collector_gaps.jsonl"` (`:58`). RR12-1's class, and it bites only someone running the
module against a scratch root — which is what a reviewer does. Low, and worth doing when (i) is
done, since both are the same discipline.

## 7. The class (item 7)

**(a) Other suites that shrink silently under an absent input: none.** Method — run all ten under an
empty `PM_DATA_ROOT` and compare the printed count to the full-root count. Result: the two DA named
are the only ones that shrink; `da_hf_pm_alignment`, `da_content_liveness_v2_check`,
`pm_host_load_join` and `v5_deploy_gates` hold their counts, and `da_forward_day_verify`,
`da_blackout_mask` and `da_cross_venue_forensics` **fail loudly** (rc 1) or skip **by name**
rather than shrinking.

**(b) Other production wiring whose removal the suite would not notice: none beyond the two.**
Method — an AST census of every call whose value is discarded (`ast.Expr` wrapping a `Call`)
outside `selftest` in the three changed modules, then reading each. `da_blackout_mask` has nine
(`sys.path.insert`, `mkdir`, `write_text`, `add_argument`, `sort`) and `pm_host_load_join`
five (`add_argument`, `sys.exit`) — all library mutations or CLI plumbing, none a stamp on an
artifact. In `da_forward_day_verify` the discarded-value calls are output helpers (`p(...)`), list
operations and the suite's own `ok(...)` used from helpers — **the two `annotate_governance` calls
are the only ones of the item-5 shape.**

## Findings

| id | severity | where | one line |
|---|---|---|---|
| DA20-R2 | **MEDIUM** | `da_forward_day_verify.py:2337`, `:2403` | removing **either** production `annotate_governance` call leaves the suite green at 254 — the R-486 wiring has no falsifier |
| DA20-R1 | LOW | Q-DA-216's DA17-R1 claim | "the CO-10-returning mutant was green before" does not reproduce: at `5a11ee9` it is **red** at the DA16-R1 falsifier |
| DA20-R3 | LOW-MEDIUM | `da_governed_verdict_preflight`, `da_verdict_check` | 39 → 38 and 21 → 19 under a tapeless root, exit 0, no named skip |
| DA20-R4 | LOW | `pm_host_load_join.py:57-58` | no `PM_DATA_ROOT` branch; `data/` resolves from `__file__` |

**DA18-R1: CLOSED** (three falsifiers, one more than the round claims). **DA17-R1: CLOSED**, with
DA20-R1 correcting the comparative claim. **The host-load split: CLOSED in form**, with the month
constants named as the next item. **R-486 `governs`: NOT closed as wiring** — DA20-R2.

## 8. Disposition (item 8)

**HOLD `d37c3d9` (+ `0cd18ba`) for ONE item — DA20-R2 — and RELEASE everything else.**

Three of the round's four items are closed and each is driven red by name with zero tracebacks; the
fixture-commit defect is replayed outside the module and repaired at the act; the identity predicate
is simpler and better falsified than the one I released at `e353119`; the host-load split is the
form I proposed, with a fixture small enough and a control that can always fire. None of that should
wait.

What should not land as it stands is the one thing the round added and did not defend: **a
governance stamp whose production wiring no check would miss.** The fix is small, it is in the
module already exercised by the suite, and there are roughly twenty-two hours before the 09-04
00:06Z run — so this is a HOLD measured in one edit, not a round. Nothing is pushed, so DA rebuilds
the unpushed commit and the chain lands whole.

**Order:** DA20-R2 into the HELD chain now; **DA20-R1** as a row correction in the same rebuild (the
claim, not the code); **DA20-R3** and **DA20-R4** to round 21 after the landing. The landing call and
its date remain the coordinator's.
