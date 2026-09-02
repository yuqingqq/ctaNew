# Review request — DA round 10 (HELD at worktree commit `3a89e6c`; lands as Q-DA-209 after the 00:14Z read)

**Pinned commit: `3a89e6c`** (detached from `b75c9fe`; object-reachable from the shared repo,
on no pushed branch; the one-line message is the only description it carries). Execute in
`~/ctaNew-wt-rev` at `--detach 3a89e6c`. Read-only under `data/`; **nothing under
`data/pm_5min/derived/` is written by any check you run — every DA selftest that touches a
verdict or mask must run with the rehearsal pair (`DA_MIDNIGHT_OUTDIR` + `DA_MIDNIGHT_LOG`)
or on a temp dir; never `production` mode; never `da_midnight_verify.sh` bare.** No timer,
no service, no launcher installed or restarted. `COORDINATION.md` never written. One
filing, per R-377.

Scope: nine files, +657/−48 — `da_blackout_mask.py` (+254), `da_forward_day_verify.py`
(+214), `da_governed_verdict_preflight.py` (+112), `da_midnight_verify.sh` (+62),
`pm_tape_density.py` (+36), `systemd/da-midnight-verify.service` (+7),
`da_cross_venue_forensics.py`, `da_hf_pm_alignment.py`, `da_verdict_check.py` (8/8/4).
Confirm no other file differs from `b75c9fe`. **This review does not change tonight:** the
installed unit and the shared tree run **v1** at 00:06Z / 00:14Z regardless of the verdict
here; a zero-byte `preflight_20260902.json` tonight still means REFUSED (CO-R4 at R-420).

## What the coordinator read at `3a89e6c` (12:55Z, from the object, not the pane)

- `pm_tape_density.py:82-101`: `CODE_ROOT = Path(__file__).resolve().parents[2]`;
  `CANONICAL_DATA_ROOT = Path("/home/yuqing/ctaNew")`; `_resolve_data_root()`:
  `PM_DATA_ROOT` env wins → else `CODE_ROOT` iff `CODE_ROOT/data/pm_5min/raw` is a dir →
  else canonical; then **`REPO = DATA_ROOT`** — the name every older consumer imports
- consumers: `da_blackout_mask.py:64` `DATA_ROOT = TD.DATA_ROOT`, git questions moved to
  `CODE_ROOT` (`:134`, `:152`, `:795`); `da_governed_verdict_preflight.py:48-52` the same
  shape; `da_forward_day_verify.py:50-52` `REPO = _TDROOT.DATA_ROOT`;
  `v41_boundary_preflight.py:53` `REPO = P.REPO` (coordinator-owned, CO-8, NOT in this
  batch) now inherits the DATA root under the name REPO
- `da_governed_verdict_preflight.py`: `RC_ALL_PASSED 0 / RC_PREDICATE_DID_NOT_PASS 1 /
  RC_REFUSED 3`; a refusal prints a JSON object (`classification: REFUSED`, `exit_code`,
  `refusal`, `day`) on stdout; the selftest drives it in a subprocess with `DERIVED`
  pointed at an empty temp dir; `open_decisions` now splits `ruled` (R-411(i), R-411(ii),
  R-408(3), each citing R-424) from `still_open` (`freeze_disposition`)
- `da_midnight_verify.sh:36-40`: `V` defaults script-relative (`$SELFDIR/…`), overridable
  by `DA_MIDNIGHT_VERIFY_BIN`; **`cd /home/yuqing/ctaNew/live/pm_research || exit 3`
  stays hardcoded**; the service adds
  `Environment=DA_MIDNIGHT_VERIFY_BIN=/home/yuqing/ctaNew/live/pm_research/da_forward_day_verify.py`
- DA's pane counts (R-427 §4, unverified): verify 235 → 244, mask 19 → 30, preflight
  30 → 34; six checks that silently skipped in a worktree (235 counted, 229 ran) fixed with
  the count asserted over checks that RAN

## Items — reproduce or refute each, at the artifact

1. **RR12-1 — the split, both directions.** From `~/ctaNew-wt-rev` (no `raw/`), the
   resolution takes branch 3 and reads the canonical tape; from the canonical tree it takes
   branch 2; with `PM_DATA_ROOT` set to an empty temp dir it takes branch 1 — and then
   what: a refusal by name, or a clean report over nothing? State which. Then the
   inversion: **`REPO = DATA_ROOT` re-binds the name every consumer used for the CODE
   tree.** Grep every `REPO` use in the nine files and in every importer of them; each one
   that asks a CODE question (git, `__file__`-relative reads, provenance) must now ask
   `CODE_ROOT`. Name any that still asks `REPO`, and state what `v41_boundary_preflight.py`
   (CO-8) records for `REPO` when run from a worktree at this commit.
2. **`CANONICAL_DATA_ROOT` is the hardcoded path the split removed, one layer down.** Is
   the fallback stated where it can be seen at run time (the emission carries `code_root`
   and `data_root` — `da_blackout_mask.py:259-262`; do the verifier and the preflight carry
   the same pair)? A reader of a receipt must be able to tell which branch resolved.
3. **The six silently-skipping checks.** Identify them at the diff; show each now RUNS in a
   worktree (the count assertion over checks that ran, not checks that were counted); show
   the assertion fires when one is made to skip again.
4. **CO-R4 closed.** A missing verdict → rc 3, JSON on stdout, `classification: REFUSED`,
   distinct from rc 1; the selftest subprocess proves it. Then: rc 3 is not among the
   codes `da_midnight_verify.sh` already uses (`exit 3` at `:40` for a failed `cd`, `exit 5`
   for the pair guard) — does the preflight's rc 3 collide with the script's own rc 3 for
   any reader that sees both? State it.
5. **R-411 constants.** `counts_toward_G` per coin-day with the floor `>= 144 of 288`; the
   P1 denominator per UNMASKED hour with calendar-24h beside it. Read the numbers at the
   code and compare to R-424 §4 verbatim; **any constant not in R-424 is a new number and a
   finding** (rule 14 — nothing here decides). Confirm every good window is scored
   regardless of `counts_toward_G`.
6. **The v2 wiring.** `da_content_liveness_rule` v2 (frozen at R-424 §2, governing from
   `20260903`) reached from the mask (`da_blackout_mask.py`, `frozen_by_user` ×1; the
   verifier ×3); a pre-effective day emits `frozen_by_user True`, `governs False`; the
   limit `V2_TRAILING_DAYS // 2 == 3` COMPUTED in `da_content_liveness_v2_check.py`
   (unchanged in this batch — confirm); the mask's own `EFFECTIVE_FROM_DAY '20260902'`
   (R-419) and the liveness v2 effective day `20260903` are two different days for two
   different rules — confirm no check conflates them. `G_MIN_COMPLEMENT_WINDOWS = 144`
   (`da_blackout_mask.py:82`) with its ruling string at `:83` is item 5's number.
7. **`da_midnight_verify.sh` and the unit.** The script-relative default and the pin; the
   script "refuses a DIFFERENT binary outside a fully isolated rehearsal" — reproduce the
   refusal (rehearsal pair set, `DA_MIDNIGHT_VERIFY_BIN` pointed elsewhere) and the
   admission (pin = own default). The hardcoded `cd`: does anything in the script or the
   verifier read a path relative to cwd, so that a worktree run executes the worktree's
   verifier from the canonical directory? State what a worktree run records as
   `script_tree_commit` (`:145`, `git -C "$SELFDIR"`) versus what it executes. **The
   installed unit is NOT this file** — `systemctl --user cat da-midnight-verify.service`
   (read-only) and state the diff against the repo's unit at this commit.
8. **The `open_decisions` split.** `ruled` cites R-424 for three decisions and `still_open`
   carries the freeze disposition — matches R-424 (four ruled, one open) or not; the
   positive control that an artifact still carrying an `ESCALATION_` key is surfaced
   unchanged.
9. **Nothing under `derived/`, counts, launchers.** Every selftest with `--selftest` from
   the worktree and from the repo root both rc 0 with the counts DA reported; nothing
   written under `data/pm_5min/derived/` by any of them (mtime listing before and after);
   `.da_midnight_verify.log` (`da_forward_day_verify.py:2690`) untouched.

## Findings format

`DA10-R<n>` — severity, reproduction, the line it lives at, what would close it. Confirm
the pinned commit executed and the worktree is clean after. Release or hold, stated;
**a hold means the batch lands as v1 + fixes in DA round 11, not that tonight changes.**
