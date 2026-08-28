# Codex re-review — day-bar v2 (`f4fafe6`)

**Verdict: HOLD MAINTAINED. The day-bar may not judge 2026-08-29.**

**Reviewed commit:** `f4fafe659050c717e70f7ecc864007ba9ee6693c`

**Reviewed at:** 2026-08-28T09:09:34Z

**Scope:** only `live/pm_research/da_forward_day_verify.py` and
`live/pm_research/da_midnight_verify.sh` as committed at `f4fafe6`.

**Governing artifacts:** `DAY_BAR_V2_PREREGISTRATION.md` at `dfa0977`, amended
by `368345b`; freeze receipt v2 at `68dca00`, whose clock anchor is
`b3f7f9f` = `1787897340` = 2026-08-28T06:09:00Z.

The shared branch advanced to `99d0573` and then `c288ed1` while this review
was running. Those successors are **not** used to rescue the explicitly scoped
commit. They require the next completed-batch review.

## Requested acceptance table

| item | result at `f4fafe6` | execution evidence |
|---|---|---|
| (a) bars govern `all_pass` | **PARTIAL / overall FAIL** | The three literal callable cases pass: clean+passing bars → `True`; one failing v2 bar → `False`; the same failing bar under `count_bar_v1_frozen` → `True`. However, `gap_rate_under_bar` remains in both the global predicate table and each per-coin `all_pass`, so a v2 day with passing P1/P2/P3 still fails when the superseded count bar fails. The governing doc says raw gap count is diagnostic with **NO bar** from 08-29. |
| (b) empty ledger requires independent observation | **FAIL** | `coverage_observed=False` refuses and `True` passes, but the public default is `None` and the guard is only `is False`; fully elapsed empty ledger + `None` returns evaluable and passes P1/P2/P3. “Unless independently observed” requires explicit `True`, not “anything except False.” |
| (c) open-at-exit + structural refusal | **FAIL** | Missing-end `gap_open_at_exit` is charged to scope end (82,800 s in the probe) and a missing interval refuses. But `gap_end_ns < gap_start_ns` is accepted as −50 lost seconds and passes; a gap record missing `coin` is silently ignored and passes. Structural refusal is therefore not closed. |
| (d) CLI dual-report keys | **PASS** | Exact-commit CLI executed with `--freeze-epoch 1787897340`; it rendered both COIN-LEVEL and per-slug breadth keys without `KeyError` or traceback. Exit 1 was the computed 08-27 day verdict, not an instrument failure. Omitting `--freeze-epoch` also refused as intended. |
| (e) nightly passes `b3f7f9f` epoch | **FAIL** | A launcher seam using the exact `f4fafe6` script recorded `1787583868.0` on both nightly invocations. Line 72 still defaults `DA_FREEZE_EPOCH` to the old 2026-08-24 epoch, not `1787897340`. The seam itself completed and promoted both stub artifacts, proving this is the value the successful nightly path passes. |

## Release-blocking findings

### 1. The nightly epoch is still wrong at the reviewed commit

`da_forward_day_verify.py` correctly removes its silent CLI default, but
`da_midnight_verify.sh:72` supplies the stale value explicitly. That does not
meet item (e), and it contradicts the freeze receipt’s load-bearing statement
that the race clock starts at `b3f7f9f`.

Required: the nightly path must pass `1787897340`, with a seam test that reads
the launched argv. Day quality and race-clock accrual may be reported
separately, but no pre-freeze day may accrue.

### 2. The superseded raw count bar still vetoes v2 days

The preregistration is explicit: from 08-29, raw gap count/hr is a diagnostic
with **NO bar**. At `f4fafe6`, `verify_day` still appends
`gap_rate_under_bar` globally (`da_forward_day_verify.py:395`) and per coin
(`:480`). `compose_all_pass` consumes every global predicate and every
`per_coin.all_pass` (`:152`), so the old count bar remains decision-bearing.

Probe: passing P1/P2/P3 plus a failed legacy count predicate returned `False`;
passing P1/P2/P3 plus `per_coin[btc].all_pass=False` also returned `False`.

Required: for `day_bar_v2`, retain count/rate/cause fields as diagnostics but
exclude the legacy count predicate from both global and per-coin verdict
composition. Add a positive control where many tiny gaps fail the old count bar
but pass P1/P2/P3 and the v2 day passes.

### 3. Coverage evidence remains fail-open

`day_bar_v2(..., coverage_observed=None)` passes an elapsed empty ledger
because the check at `:207` is `coverage_observed is False`. The default value
is therefore equivalent to observed coverage without evidence.

Required: only `coverage_observed is True` may make the bar evaluable. `False`,
`None`, omitted, and malformed values must refuse; preserve the explicit-True
positive control.

### 4. Structural gap validation is incomplete

The new missing-field refusal is useful, and `gap_open_at_exit` is now read.
But interval ordering, numeric/finite types, and required identity are not
validated before arithmetic/filtering. Two known-bads passed:

- reversed interval → `lost_seconds=-50.0`, P1/P3 pass;
- gap event missing `coin` → silently ignored, zero loss, all bars pass.

Required: validate every recognized gap record before coin filtering: mapping
shape, known event, known coin, numeric finite timestamps, and strictly
`gap_end_ns > gap_start_ns`. Only `gap_open_at_exit` may synthesize a missing
end, and its chosen scope end must be explicit in the receipt.

### 5. P3 is still not the declared maximum rolling hour

The code states “stepped by window” at `da_forward_day_verify.py:187`, testing
only 300-second-aligned starts. The governing predicate is the **maximum
rolling-60-minute** loss, without an alignment qualification.

Counterexample executed:

- gaps `[+100,+600]` and `[+3200,+3700]`;
- exact rolling window `[+100,+3700]` contains 1,000 lost seconds → must FAIL;
- the implementation reports 900 seconds → passes at the `<=900` boundary.

Required: compute the exact piecewise maximum using day boundaries and interval
endpoints (including endpoint−3600 candidates), or amend the preregistration
before the first governed day. Add this counterexample red-first.

## Regression evidence

- exact commit isolated in a detached worktree;
- source SHA-256:
  - verifier `a2c53a6e633e9b39fa79d4644d1c9722cec2d4a246cfdfe0cbebd05d77bcbda9`;
  - nightly `6effd8e22b56769dc3cdcb1c80ac20b4d837cada5918d348ca421caf1a89aea6`;
- `python3 -m py_compile .../da_forward_day_verify.py`: PASS;
- `bash -n .../da_midnight_verify.sh`: PASS;
- built-in verifier selftests: **45 passed**;
- actual CLI dual-report rendering: PASS;
- custom acceptance/fail-open battery: findings above.

The existing suite and ordinary behavior did not regress in the exercised
paths. The hold remains because the suite does not cover the decision-bearing
counterexamples above.

## Clear condition for the next review

Return one completed fix batch containing all five release blockers above,
with each new known-bad red-first and a positive control. The next review must
execute the exact batch commit. Until that review says **HOLD RELEASED**, no
08-29 day-bar verdict is admissible.
