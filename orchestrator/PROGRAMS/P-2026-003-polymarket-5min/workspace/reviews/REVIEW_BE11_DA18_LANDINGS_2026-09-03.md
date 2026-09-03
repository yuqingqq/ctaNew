# Review — BE round 11 + DA round 18 at `c511750` (the post-verdict landings: the fwd5 receipts of record, the held DA chain rebased and landed)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `c511750`** (Q-DA-214) on `7792fb5 … 21de639` on **`fc70b17`** (Q-BE-236).
**Request of record:** `REQUEST_BE11_DA18_LANDINGS_2026-09-03.md`. **Composed 2026-09-03T00:44:11Z.**
One filing, per R-377. Times from `date -u`.

**Constraints observed.** Executed in `~/ctaNew-wt-rev` at `--detach c511750`; `~/ctaNew-wt-da`,
`~/ctaNew-wt-be`, `~/ctaNew-wt-de` never read; `be_forward_day.py` never run (item 1 is a hash);
**`DA_MIDNIGHT_MODE` never set**; `da_midnight_verify.sh` **never run in production mode** — the
two legs below are the refusal leg and a both-overrides rehearsal into my own scratch; the installed
unit never touched; the declared Phase-4 OUTDIR never passed to `--run`. Nothing written under
`data/`: the main tree's `derived/` is **178 before and after**. `git worktree list` **34 at
quiescence**; `git status --short` **0**; no mutant applied to any file this round (two scratch
worktrees created for a probe and removed).

## 1. The receipts at what git serves — CONFIRMED (item 1)

| path at `fc70b17` | sha256 | bytes |
|---|---|---|
| `…/be_forward_day_receipt_20260901.json` | **`4000106752f816e4…`** | **14,022** |
| `…/be_forward_day_receipt_20260902.json` | **`0907b0369e14d77b…`** | **1,123** |

Both match R-442 §2 and the row. Read from the 09-02 blob: `as_of_utc` **2026-09-02T13:50:22Z**,
`outcome` **REFUSED**, `refused_at` **`day_closed_and_attributed`**, with the refusal naming the
reason (*"20260902 is not closed by calendar (day_closed_calendar=False). Scoring an OPEN day scores
a populati…"*). The 09-01 receipt identifies the sealed file **by content** — `sha256`
`aca22317ab06adbf…`, `bytes` 54,213,086, `contents`, and `not_in_receipt` ("no metric, rho, net
value or sign appears outside this file"). **No sealed or `fwd5` path is tracked in git** (0 hits in
`git ls-tree -r fc70b17`). The commit's file list is **exactly** the two receipts and the row
(379 + 27 + 1 = 407 insertions). Rule 11 is kept at the artifact, not asserted.

One observation, filed as **BE11-R1**: that sealed file — the score of record for the 09-01 race —
lives at
`/tmp/claude-1001/…/32b9d1f8-…/scratchpad/fwd5/be_forward_day_SEALED_scores_20260901.json`, a
**session scratchpad path**. The receipt binds it by sha, which is the right form; the location is
volatile, and 54 MB is not a git artifact. Whether the race's score of record needs a durable home
is a decision, not a defect — I name it because the receipt is now in git and its referent is not.

## 2. The rebase equivalence — CONFIRMED; my RELEASE transfers (item 2)

`git diff a8165d8 7792fb5 --stat` is **exactly `fc70b17`'s three files** (379 / 27 / 1, 407
insertions) — i.e. the only difference between the held chain's tip and the rebased chain's tip is
BE round 11's receipts commit. **The rebase moved no chain content.**

Per-commit, by `git range-diff 3a89e6c^..e353119 21de639^..7792fb5` — **all eight pairs report `=`**
(identical patches):

| held | landed | patch |
|---|---|---|
| `3a89e6c` | `21de639` | **=** |
| `e292439` | `470273a` | **=** |
| `636a455` | `5d0eca4` | **=** |
| `e384792` | `180a298` | **=** |
| `801eb31` | `5c00c75` | **=** |
| `8910701` | `286a1d7` | **=** |
| `3b7e10a` | `bed178d` | **=** |
| `e353119` | `7792fb5` | **=** |

And the constant tree offset between each held commit and its landed twin (12 lines under `live/`,
94 whole-tree, identical for all eight) is **entirely BE's and DE's** files —
`be_forward_day.py`, the seven `de_*` modules and the two plan documents. **No DA file appears in
it.**

**So my RELEASE of `e353119` transfers to `7792fb5` unchanged.** The one qualification worth
stating: a release is of content, and the landed content sits on a different base — anything the DA
modules *invoke* from that base could behave differently. The only such surface is
`v5_deploy_gates`, which runs other seats' selftests as gates; it is not on the unit path (§6), and
I re-ran the nine DA suites at the landed tip below rather than infer.

## 3. The landed tip's nine selftests, executed (item 3)

Both launchers, `__pycache__` cleared before each, from my worktree at `--detach c511750`:

| module | result |
|---|---|
| `da_forward_day_verify` | **247** ran, **0 skipped**, `ran+skipped=247`, rc 0 |
| `da_governed_verdict_preflight` | **39**, rc 0 |
| `da_hf_pm_alignment` | **53**, rc 0 |
| `da_cross_venue_forensics` | **24**, rc 0 |
| `da_verdict_check` | **21**, rc 0 |
| `da_content_liveness_v2_check` | **19**, rc 0 |
| `pm_tape_density` | **9** (8 + 1 named skip, `ran+skipped=9`), rc 0 |
| `v5_deploy_gates` | **5**, rc 0 |
| **`da_blackout_mask`** | **rc 1 — RED**, both launchers, both roots (see **DA18-R1**) |

Two of those needed my mirror corrected before they were honest, and I record it because the first
readings were mine, not the tip's: `da_forward_day_verify` first read **241 + 6 named SKIPs**
because my mirror lacked `derived/.da_midnight_verify.log` (a dotfile), and
`da_cross_venue_forensics` first read **red** because my mirror lacked `data/mm_hf/` — its inputs
resolve tree-relative (`REPO = Path(__file__)…`) while the log lives in the main tree. With both
linked, 247/0 and 24. I also re-ran with a scratch `PM_DATA_ROOT`: the preflight and
`da_verdict_check` read 39 and 21 there, identical to the worktree root — an earlier 38/19 was my
malformed root (one level short of `<root>/data/pm_5min`), not the code.

**DA18-R1 (MEDIUM) — the mask suite is red at the landed tip, and for a reason that is a function of
the branch's shape.** `da_blackout_mask.py:865-878` copies four DA files into the scratch worktree
and commits them; the commit's return code is **not checked** (`capture_output=True`, no rc). When
the executing tree's DA files are byte-identical to the child's checkout commit, that commit is
**empty** and the child's HEAD never moves — and the failure surfaces two checks later as
*"CO-10 precondition: after the fixture commit the child's HEAD (7792fb5d498d) is a THIRD value,
distinct from this tree's (c511750ca6c9) and from HEAD~1 (**7792fb5d498d**)"*. Measured by hand:
copy-then-commit at `HEAD~1` returns **rc 1, "nothing to commit, working tree clean"**. It is red at
this tip precisely because `c511750` is a **row-only** commit on `7792fb5`, so HEAD and HEAD~1
carry the same DA files. This is BE9-C1's class, in DA's suite, and BE round 10's fix applies
verbatim: **plant a difference** (append a marker line to one copied file) so the commit cannot be
empty, and check the commit's rc and refuse by name. **It is not on the unit path** — the verifier
*imports* the mask (`da_forward_day_verify.py:748`, `import da_blackout_mask as BM`), it does not run its selftest — so it is not a before-00:06Z item.

## 4. The launcher path at the landed tip, NON-production (item 4)

Module shas confirmed: `da_midnight_verify.sh` **`4d79d79a2afc8346`**, `da_forward_day_verify.py`
**`9e042ec942af6f07`**, `da_blackout_mask.py` **`15ea6dcb8c97c72d`**.

**Driven, both legs, in my worktree:**

- **Leg 1 — the known-bad** (no `DA_MIDNIGHT_MODE`, no overrides, not the unit's cgroup):
  **exit 6**, *"REFUSED: this run would write CANONICAL verdicts into
  /home/yuqing/ctaNew/data/pm_5min/derived but identifies itself as neither the scheduled unit (by
  cgroup) nor an explicit DA_MIDNIGHT_MODE=production hand run…"* — **nothing written**. The mode
  gate refuses what it should.
- **Leg 2 — the rehearsal** (`DA_MIDNIGHT_OUTDIR` **and** `DA_MIDNIGHT_LOG` both set, plus
  `PM_DATA_ROOT`, into my scratch): wrapper **exit 0**, `worst_instrument_rc=0`, the day computed
  and **verdict artifacts written into the scratch outdir** —
  `da_dayverdict_20260903.json`, `da_dayverdict_20260902.json`, `da_blackout_mask_20260902.json` —
  with `exit=1 for 20260903` recorded as a **failing day, not an instrument failure**, and the
  verdict's own reason reading **`UNATTRIBUTED hand run of da_midnight…`**. The main tree's
  `derived/` stayed at **178**.

So: **the path reaches a verdict write in the scratch root**; the wrapper distinguishes a failing
day from an instrument failure (my first rehearsal, against a malformed root, produced exactly the
other outcome — *"INSTRUMENT FAILURE verifying 20260903: NOTHING WAS VERIFIED. This is exit 4, NOT a
failing day"* — which is the distinction R-402's history is about, driven by accident and worth
recording); and the artifact **labels itself unattributed**, so a rehearsal cannot be mistaken for
the unit's verdict.

**The verifier pin resolves as DA says**, statically and unambiguously:
`V="${DA_MIDNIGHT_VERIFY_BIN:-$SELFDIR/da_forward_day_verify.py}"` with
`SELFDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"`, and the script `cd`s to
`/home/yuqing/ctaNew/live/pm_research`. With the variable **absent as installed**, the unit's
ExecStart (the main tree's script) yields **the main tree's verifier**; the repo's uninstalled unit
file names the same file explicitly.

**What was NOT driven:** the canonical path itself (no overrides, unit cgroup) — that is a
production act and is out of scope by discipline. Everything above is the same code with the input
root and the output directory redirected; the only untested difference is the canonical roots.

## 5. The mask's producer sha — CONFIRMED, and no consumer refuses it (item 5)

`da_blackout_mask_20260902.json` records `producer.module_sha256_prefix` **`d191695dcff0546e`**
with `carrying_commit` **`3eabeeb2cabf6a80…`**, and
`git show 3eabeeb:live/pm_research/da_blackout_mask.py | sha256sum` = **`d191695dcff0546e`** — the
artifact **resolves at its carrying commit**. The working file now hashes `15ea6dcb8c97c72d`.

**No consumer at the tip compares the recorded prefix against the working file.** The prefix is
*written* at `da_blackout_mask.py:253`, `:275` and `da_forward_day_verify.py:655`; the only
read-side check is `da_blackout_mask.py:541`, which asserts the field's **shape**
(`len(...) == 16`), not its value. So a correct artifact carried from `3eabeeb` is not refused,
and the binding is to the commit — which is the right design and the one my DA-17 round argued for.

## 6. The host-load red (item 6)

`pm_host_load_join.py` is **`4ec5b69319f22307`** — byte-identical across the landing, as stated.
**The unit path never invokes `v5_deploy_gates`**: zero references in `da_midnight_verify.sh`,
`da_forward_day_verify.py` and `da_blackout_mask.py`. The failing check is at `:263-269` — the
**real-archive parse** (`sar_cpu(25)` must yield > 100 samples), inside a `try/except Refused` that
turns the refusal into `ok(False, "sa25 must parse: …")`. It is **not** a positive control of the
join arithmetic; it is the guard against a silently-empty parse reading as a clean host.

**My reading of the durable fix.** The check is doing two jobs, and that is why the calendar broke
it: it proves *the parser reads a real sysstat archive* **and** *this host has recent data*. `sa25`
is a day-of-month name that sysstat recycles monthly, so job one is currently a function of the
calendar. Split them:

1. **the parser's control becomes content-addressed** — a small committed fixture (a trimmed `sar`
   text extract of a few hundred lines is enough for a > 100-sample assertion; the 1,087,654 B
   `sar25` is more than the repo's "no large data files" rule wants). This control can **always**
   fire, which is what rule 15 asks;
2. **a separate, labelled liveness check** over whatever sysstat currently holds — day-relative,
   and permitted to report a named status when no archive is present, because that is a statement
   about the host and not about the parser.

A day-relative pin **alone** leaves the parser without a control the day the host has no archive; a
snapshot **alone** stops saying anything about the live host. Neither is a named SKIP of the
existing check, which the coordinator has already ruled out.

## Findings

| id | severity | where | one line |
|---|---|---|---|
| DA18-R1 | **MEDIUM** | `da_blackout_mask.py:865-878`, precondition `:906` | the fixture commit is empty whenever the executing tree's DA files equal the child's checkout commit; its rc is unchecked, and the suite is **red at the landed tip** |
| BE11-R1 | LOW | 09-01 receipt, `sealed_file.path` | the score of record is bound by sha but lives on a session scratchpad path |

Neither is on the unit path.

## 7. Disposition (item 7)

**Nothing must move before Fri 2026-09-04 00:06:00 UTC.** I drove the exact path the unit will
execute — `da_midnight_verify.sh` (`4d79d79a2afc8346`) → `da_forward_day_verify.py`
(`9e042ec942af6f07`) → `da_blackout_mask.py` (`15ea6dcb8c97c72d`) — end to end in a scratch root:
it computes the day, writes the verdict, labels an unattributed run as such, distinguishes an
instrument failure from a failing day, and refuses the canonical write without attribution
(exit 6). The two reds at the tip are **off that path**: the mask's is its *selftest* (the verifier imports the module at
`da_forward_day_verify.py:748`, DA18-R1), and `v5_deploy_gates` is invoked by nothing on it (§6).

**RELEASE `c511750`** as BE round 12's base (BE8-R1/R2 + BE10-R1..R4) and DA round 20's base
(DA17-R1, the `predicates[].governs` observation, the host-load fix after DA's round-19 proposal).
My DA-17 RELEASE transfers to `7792fb5` unchanged, on identical patches and a base offset that
touches no DA file.

**For DA round 19/20, in order:** (1) **DA18-R1** — plant the difference and check the commit's rc;
it is a red suite in every worktree whose tip is a row-only commit, which is now the normal shape
after a landing; (2) the host-load split of §6, after DA's proposal; (3) DA17-R1 as already routed.
**BE11-R1** is the USER's or the coordinator's call, not a code change: it asks where the sealed
score of record should live.
