# The 09-03 smoke — playbook

**For whoever runs it, not for whoever wrote it.** Every claim below that
code can check names the check. Nothing here authorises the run: the
preconditions in §1 include two that are **not yet met**, and each says so.

**Authority:** R-547 (USER), R-555 (the ruled day set).
**Design:** `p003_de_multiday_gate1_design_v6__20260906T043936Z.json`,
sha256 `966ca76d2803fa5aa45cb5b15c3b6eff498c7888e9d9bb33b53612effcbd39d2`.
**Before-picture:** `p003_de_gate1_dry_run_ledger__20260906T044319Z.json`,
sha256 `56882c75a14322e5ee816e1e033dcde9ae371d229824b225b76c4f5c15cd7912`.

---

## 1. Preconditions

| # | precondition | state | checked by |
|---|---|---|---|
| P1 | The reviewer's runner-approval filing exists under `workspace/reviews/` | **NOT MET** — `REVIEW_RUNNER_2026-09-06.md` says NOT YET APPROVED; a superseding filing is required | human; cite the filing by path in the run receipt |
| P2 | BE's 09-03 reference book, by path + sha256, **carrying `asm`** for both pinned heads, plus BE's builder receipt | **NOT MET** — not built | `verify_day_inputs()` (book digest); design v6 `R1_asm_the_scored_book` states the required content |
| P3 | Params file `live/pm_research/declarations/de_multiday_gate1_params_v1.json` reads sha256 `b06125e9e10b9652ebb5130fe0f08e5d7b1d312828dac219b7e7ac0496130138` | met | `load_params()` — refuses an empty set, a duplicate day, a set whose size ≠ `expected_G`, or a day whose `previously_opened_for` ≠ `none` |
| P4 | `PM_DATA_ROOT` exported as **the repo root** `/home/yuqing/ctaNew` | met in the DE tmux session | `de_data_root.require_canonical()` — refuses a result-bearing emission off the canonical root |
| P5 | `/home/yuqing/ctaNew/data/.heavy_run.lock` free | check at run time | `flock -n` in the wrapper refuses if held |
| P6 | The three pinned model files and both thetas match their pins | met as of this writing | `verify_pinned_models()`, `verify_pinned_thetas()` — a mismatch refuses **the run** |
| P7 | BE's cascade module digest is `67fc7b6c0150d3f9c933d9000481e19f7b8b90088dee1d7e6f4b7836662dd657` | met | `verify_be_module()` — a different digest refuses |

**P1 and P2 are blocking.** The runner will start without them and fail at
P2's digest check, which is the intended order: the refusal is the guard,
not the reminder.

## 2. The one command

```
flock -n /home/yuqing/ctaNew/data/.heavy_run.lock \
  systemd-run --user --scope --slice=research.slice \
    -p MemoryMax=8G -p CPUQuota=100% \
    --setenv=PM_DATA_ROOT=/home/yuqing/ctaNew \
    python3 live/pm_research/de_multiday_gate1_runner.py \
      --day 2026-09-03 --output <receipt path>
```

`--day` does not exist yet: the runner today has `--selftest`,
`--fixture-run` and `--dry-run-ledger`. **Building the real per-day entry
point is the next round's work and is named here so the gap is visible
rather than discovered at the console.**

Caps are never raised (R-174). If the day exceeds the cap or the declared
deadline, **the day refuses** — never a lower draw count, never a bigger
cap (design v6 `R8_time_overrun`; `arm_day()` deadline branch).

## 3. What the run writes

| artifact | where | contains |
|---|---|---|
| sealed per-day artifact | `data/pm_5min/derived/` | population counts, refusal statuses, `admissibility`, `draw_provenance`, `seal_status` — **and no economic field** |
| resource observation | inside the run receipt | wall, CPU, max RSS, the wrapper used |
| run receipt | `data/pm_5min/derived/` | the battery (run + skipped, with the computed equality), the resolved data root and branch, the BE citation, the design digest |

## 4. Published after day 1 — and what is not

**Published:** resource observations, population counts, refusal statuses,
the day's admissibility resolution.

**SEALED until G = 6 days are complete:** `D_E0`, `D_E_MINUS_R`, `Z`,
`p_location`, `null_mean`, `null_sd`, `null_draws_summary` — **absent from
the artifact, not present-and-ignored**, at every depth.

Checked by `seal()` and `assert_no_economic_leak()`, which read **one
shared name list** (`ECONOMIC_FIELDS`) and **one traversal**, so the
emitter and the guard cannot disagree. Both directions are driven in the
battery, including a leak planted inside a nested list.

All six days run **regardless of interim results** (design v6
`R5_the_smoke_is_sealed.all_G_days_run_regardless_of_interim_results`).

## 5. The refusal exits, and what each means

| refusal | meaning |
|---|---|
| `ruled day set is EMPTY` | the params file has no days; the USER's ruling is not applied |
| `against the declared expected_G` | the set is not size 6 — a day was chosen after the fact |
| `previously_opened_for is not 'none'` | a touched day entered the ruled set |
| `reference-book digest mismatch` | **the day** refuses; the book is not the one declared |
| `theta is …` / `model … digest` | **the run** refuses; a refitted model makes every day's object different |
| `cascade module digest differs` | BE's cascade is not the cited one |
| `draw provenance does not bind` | the draws were not produced by the verified module, this book, this arm and the recomputed seed |
| `DEGENERATE_ARM_DAY_REFUSED` | < 30 decisions, or null sd < 0.25·\|mean\| — a **status**, and G does not shrink |
| `below the declared minimum` | fewer than 500 draws |
| `exceeds the declared deadline` | R8: the day refuses |
| `economic fields leaked into a SEALED artifact` | the seal was violated; the artifact is not written |
| `not /home/yuqing/ctaNew/data` | a result-bearing emission off the canonical root |
| lock held (`flock` rc 1) | another heavy run owns the slice |

## 6. Post-run checks the coordinator performs

1. **Digest at both copies** — the receipt's sha256 in the DE worktree and
   in `/home/yuqing/ctaNew/data/...` are equal. *(A single `sha256sum` of
   both paths; they are the same file only when `data/` is a symlink,
   which it no longer is.)*
2. **No economic field present** — walk the per-day artifact for the seven
   names in `ECONOMIC_FIELDS`. The runner asserts this itself
   (`assert_no_economic_leak`); the coordinator's copy is the independent
   one.
3. **The battery equality** — `n_checks_run + n_checks_skipped_offline ==
   expected_checks_in_the_source`, and every skipped check **named**.
4. **The resolved root and branch** — `data_root.data_root_resolved ==
   /home/yuqing/ctaNew/data`, `branch == 1_env_PM_DATA_ROOT`.
5. **The day's admissibility** against the before-picture in
   `p003_de_gate1_dry_run_ledger__20260906T044319Z.json`: 09-03 read
   `CLOSED_AND_QUALIFIES`, all four conjuncts true, `untouched: true`.

## 7. What this run cannot produce

A significance-bearing verdict. At G = 6 the smallest attainable one-sided
sign-test p is 2⁻⁶ = 0.015625 against a Holm threshold of 0.025 at m = 2,
so **G = 6 does clear** — but only if **all six days run and every day
sign agrees**. Day 1 alone establishes nothing, which is why its economics
are sealed. **A pass is a pass on the declared rule and nothing more; a
fail needs no significance at all.**
