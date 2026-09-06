# The 09-03 smoke — playbook

**For whoever runs it, not for whoever wrote it.** Every claim below that
code can check names the check. Nothing here authorises the run: the
preconditions in §1 include three that are **not yet met**, and each says so.

**Authority:** R-547 (USER), R-555 (the ruled day set, G = 6),
R-572(B) (the draws source, the two clocks, the seal layout).
**Design:** `p003_de_multiday_gate1_design_v9__20260906T062115Z.json`,
sha256 `fe4b0db4ddfdcc3a942fef9a35cd702646a9c020704ca81b5f28ff175f7aff2e`.
**Parameters:** `live/pm_research/declarations/de_multiday_gate1_params_v2.json`,
sha256 `ce46b5775d8f17a4b417640b620db275868a9efec19abdde6caaecd4dbd405dd`.
**Before-picture:** `p003_de_gate1_dry_run_ledger__20260906T044319Z.json`,
sha256 `56882c75a14322e5ee816e1e033dcde9ae371d229824b225b76c4f5c15cd7912`.
**Fixture receipt:** `p003_de_multiday_gate1_fixture_run_v8__20260906T062115Z.json`,
sha256 `f6950e9e5dd6e25875bc01b48038fea01384d1965fdc1d1a4fbc961d6c9d4bb2`.

---

## 1. Preconditions

| # | precondition | state | checked by |
|---|---|---|---|
| P1 | The reviewer's runner-approval filing covers the runner **as it now stands** | **NOT MET** — REV 37 (`REVIEW_RUNNER_REDRIVE_DE77_2026-09-06.md`) approved the FIXTURE path and stated the smoke cannot be approved while `--day` did not exist. `--day` exists as of DE 78 and has not been reviewed; the two admission-layer findings from that filing are fixed here and need re-driving | human; cite the filing by path in the run receipt |
| P2 | BE's 09-03 reference book, by path + sha256, **carrying `asm`** for both pinned heads, plus BE's builder receipt | **NOT MET and further away than it looked** — R-573: the build is BLOCKED on a feature fragment that covers only the consumed 08-24/25 era. A per-day top-up feature pass and an assembly that fits 8 GB come first (BE 49) | `verify_day_inputs()` (book digest); design v9 `R1_asm_the_scored_book` |
| P3 | Params file `…/de_multiday_gate1_params_v2.json` reads sha256 `ce46b577…` | met | `load_params()` — refuses an empty set, a duplicate day, a set whose size ≠ `expected_G`, or a day whose `previously_opened_for` ≠ `none` |
| P4 | `PM_DATA_ROOT` exported as **the repo root** `/home/yuqing/ctaNew` | met in the DE tmux session (set 2026-09-06T05:36Z; it was **unset** at reload) | `de_data_root.require_canonical()` |
| P5 | `/home/yuqing/ctaNew/data/.heavy_run.lock` free | check at run time | `flock -n` in the wrapper refuses if held |
| P6 | The three pinned model files and both thetas match their pins | met as of this writing | `verify_pinned_models()`, `verify_pinned_thetas()` — a mismatch refuses **the run** |
| P7 | BE's cascade module digest is `2b164df2ec0653a51c6db71fd69db052564b1dcc6c8956ea576d9840d89d9274` | met — **re-pointed this round**, see below | `verify_be_module()` and `import_be_cascade()` — a different digest refuses |

**P1 and P2 are blocking.** The runner will start without them and fail at
P2's digest check, which is the intended order: the refusal is the guard,
not the reminder.

**P7 was NOT met at the start of this round and the guard is what said so.**
BE's round 47 (`ab75b41`) changed `be_cancel_axis_null.py`, the pinned digest
went stale, and the runner's battery refused on reload. The re-point is
recorded in params v2 `be_module_repoint` and was justified by a **computed**
per-definition diff of the two blobs, not by BE's commit message: three
top-level definitions changed (`load`, `run`, `main`), seventeen did not, and
**none of the nine draw-path functions changed**. What that does *not* cover —
module-level constants, including the read root — is stated there too, and is
closed at the book digest instead.

## 2. The one command

```
flock -n /home/yuqing/ctaNew/data/.heavy_run.lock \
  systemd-run --user --scope --slice=research.slice \
    -p MemoryMax=8G -p CPUQuota=100% \
    --setenv=PM_DATA_ROOT=/home/yuqing/ctaNew \
    python3 live/pm_research/de_multiday_gate1_runner.py \
      --day 2026-09-03 --output <receipt path>
```

`--day` **exists** as of DE 78, and so does `--synthetic-day <DAY>`,
which builds a day book of BE's declared shape, runs the same path on it
through BE's **real** cascade, and emits the receipt. What the synthetic
run proves is the WIRING; it proves nothing about BE's data.

**A real day takes the lock FIRST or does not start.** `--day` refuses
before any work if this process does not hold
`/home/yuqing/ctaNew/data/.heavy_run.lock` — a real day is heavy by
construction (BE projects ~2.3 h for both arms), and R-575(C) records two
heavy scopes running concurrently because a wrapper string in a file cannot
say what launched a process. The receipt's `wrapper` block is now READ FROM
`/proc/self/fd`: `flock` passes the lock's fd through the exec (fd 3,
measured), so the artifact states what actually ran.

**A real day also refuses if its producing code is not the bytes HEAD
holds** — a result naming a commit that does not contain the code that made
it is provenance theatre. A fixture RECORDS that field instead of refusing
(ruled, DE 78), because a hard refusal there would block every pre-commit
emission.

Caps are never raised (R-174). If the day exceeds the cap or the declared
deadline, **the day refuses** — never a lower draw count, never a bigger
cap (design v9 `R8_time_overrun`; `arm_day()` deadline branch).

**What `--day` holds, and which index splits it needs: NONE.**
Six declared stages (design v9 `R11_memory_and_index_residency`), a
high-water recorded at each boundary, and a budget that REFUSES rather than
reports a number over the line. `--day` consumes BE's book — `fr.reference`
plus `asm` — and values fills from the replay's own records, so no tape
index or feature fragment is resident at any stage. That is MEASURED: the
whole day path runs under instrumentation and the receipt records that not
one tape, index or fragment artifact was opened, with a non-vacuity control
that the instrument DID observe the book being read. **Consequence for BE:
the index need not be alive when the reference and `asm` are, so the
whole-day peak is max(index stage, assembly stage) rather than their sum.**
Which split BE must build to produce a September day's `asm` is BE's
measurement and R-496(E)'s ruling, not DE's.

**R8's budget is DE's arm-day, not the whole pipeline.** The 12-hour
`per_day_deadline_s` and R8's CPU-hour table cover the null and the replay.
They do **not** cover BE's feature pass, reference build, tape index or
assembly, which R-573(C) measured at 18 CPU-minutes and 6.4 GB — 80% of the
cap — *before* the slice. Do not read R8 as a whole-day budget.

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

**The layout is symmetric across the two states (R-572(B)(3)).** `sealed`,
`seal_status`, `sealed_at_every_depth` and `sealed_field_names` are present
in BOTH states with explicit values; `economic` is the one key that is
present iff unsealed, and that is a declared rule rather than something a
consumer discovers as a `None`. `assert_seal_layout_symmetric()` is the
consumer falsifier and its known-bad **is the pre-fix layout**.

**A per-day run may happen NOW (R-572(B)(2)).** `read_not_before_utc`
(2026-09-09T00:06Z) governs the **aggregate read** — the unseal and the
section-7 verdict — and not the runs. A closed, qualifying, ruled day may be
run sealed today: `may_run_day()` admits it, `may_read_aggregate()` refuses
the read until BOTH the date has passed AND all six days are complete.

All six days run **regardless of interim results** (design v9
`R5_the_smoke_is_sealed.all_G_days_run_regardless_of_interim_results`).

## 5. The refusal exits, and what each means

| refusal | meaning |
|---|---|
| `ruled day set is EMPTY` | the params file has no days; the USER's ruling is not applied |
| `against the declared expected_G` | the set is not size 6 — a day was chosen after the fact |
| `previously_opened_for is not 'none'` | a touched day entered the ruled set |
| `not in the ruled day set` | a day outside the population, however healthy |
| `not a CLOSED calendar day` / `conjuncts and day-quality` | R9: the day may not run yet |
| `the aggregate read is not before …` / `of 6 days are complete` | R9: the read is held on BOTH conditions |
| `reference-book digest mismatch` | **the day** refuses; the book is not the one declared |
| `the cascade loaded a book whose own digest is …` | BE's loader unpickled bytes that are not the declared day book |
| `theta is …` / `model … digest` | **the run** refuses; a refitted model makes every day's object different |
| `cascade module digest differs` | BE's cascade is not the cited one |
| `the import loaded … but the declaration cites …` | a same-named module earlier on `sys.path` |
| `draws were SUPPLIED on a ruled day` | R-572(B)(1): the runner generates them in process or not at all |
| `fixture draws were claimed for <day>, which IS in the ruled day set` | the fixture door, shut by the day rather than by the caller's word |
| `draw provenance does not bind` | wrong module, seed, book, arm — or a GENERATED claim from another pid |
| `DEGENERATE_ARM_DAY_REFUSED` | < 30 decisions, or null sd < 0.25·\|mean\| — a **status**, and G does not shrink |
| `below the declared minimum` | fewer than 500 draws |
| `exceeds the declared deadline` | R8: the day refuses |
| `economic fields leaked into a SEALED artifact` | the seal was violated; the artifact is not written |
| `the sealed/unsealed layout is ASYMMETRIC` | R10: a consumer would read `None` from a missing key |
| `not /home/yuqing/ctaNew/data` | a result-bearing emission off the canonical root |
| lock held (`flock` rc 1) | another heavy run owns the slice |

## 6. Post-run checks the coordinator performs

1. **Digest at both copies** — the receipt's sha256 in the DE worktree and
   in `/home/yuqing/ctaNew/data/...` are equal.
2. **No economic field present** — walk the per-day artifact for the seven
   names in `ECONOMIC_FIELDS`. The runner asserts this itself
   (`assert_no_economic_leak`); the coordinator's copy is the independent
   one.
3. **The seal layout** — all four layout keys present, `economic` absent
   while sealed.
4. **The draw provenance** — `draw_source == "GENERATED_IN_PROCESS"`,
   `pid` equal to the run's, `module_sha256` equal to P7's digest, and the
   seed reproducible from the book digest and the arm by design v9's
   `seed_convention` fields.
5. **The battery equality** — `n_checks_run + n_checks_skipped_offline ==
   expected_checks_in_the_source`, and every skipped check **named**.
6. **The resolved root and branch** — `data_root.data_root_resolved ==
   /home/yuqing/ctaNew/data`, `branch == 1_env_PM_DATA_ROOT`.
7. **The day's admissibility** against the before-picture in
   `p003_de_gate1_dry_run_ledger__20260906T044319Z.json`: 09-03 read
   `CLOSED_AND_QUALIFIES`, all four conjuncts true, `untouched: true`.

**A green `--dry-run-ledger` is not a preflight.** It reads day verdicts and
the read-state table and nothing else — not BE's book, not the pinned models
or thetas, not BE's cascade digest. It exits 0 and prints a receipt while P2,
P6 and P7 are entirely unexamined. Design v8 carries that scope as fields
(`dry_run_ledger_scope`) precisely because the gap is invisible at the
console.

## 7. What this run cannot produce

A significance-bearing verdict. At G = 6 the smallest attainable one-sided
sign-test p is 2⁻⁶ = 0.015625 against a Holm threshold of 0.025 at m = 2,
so **G = 6 does clear** — but only if **all six days run and every day
sign agrees**. Day 1 alone establishes nothing, which is why its economics
are sealed. **A pass is a pass on the declared rule and nothing more; a
fail needs no significance at all.**
