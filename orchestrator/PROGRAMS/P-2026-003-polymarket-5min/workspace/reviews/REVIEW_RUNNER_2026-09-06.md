# REVIEW — the Gate-1 runner: NOT YET APPROVED for the 09-03 smoke. One blocker breaks it within 20 hours, and the digest that cites BE's cascade never touches the draws

**Filed** 2026-09-06T04:22Z (clock read before composing) · reviewer seat
(pm-codex) · tip `f8108dd` · no code fixed · no write under `data/` · nothing
sealed opened · **every `data/` path read at the absolute ledger path** (§0).

**ROUTING — CHECKED**, everything below driven by me under the rule-20 wrapper.

## VERDICT

**NOT YET APPROVED to run the 09-03 smoke.** The runner itself is the best-built
instrument this programme has produced — 29 checks pass, the R5 guard caught DE's
own emitter and I reproduced the catch, and every refusal I drove fired correctly.
**Three things must change first**, and one of them is on a clock:

1. **(BLOCKER) The design module's R7 assertion hardcodes today's day counts and
   will fail at 00:06Z on 09-07** — about 20 hours from filing. The runner calls
   that selftest on every fixture run, so **the runner breaks with it.**
2. **The digest that cites BE's cascade never touches the draws.** The runner
   verifies BE's module and then receives `null_draws` as an argument. Nothing
   binds the verified module to the numbers.
3. **`--fixture-run` is not data-free** despite `FIXTURE_RUN_NO_DATA`; it reads the
   ledger and cannot be driven from a shell worktree.

---

## 0. Execution surface, stated because it shaped the round

My worktree's `data/` re-materialised as a shell for the **third time in two
rounds** — a `git checkout` recreates it whenever the tip adds newly-tracked
`data/` paths, which every DE/BE round does. I restored the symlink, re-armed
`skip-worktree` (153 flags), and **read every `data/` artifact at
`/home/yuqing/ctaNew/data/...`**. The fixture receipt reads `aa9a51356804d9e8`,
matching the coordinator's digest.

## 1. **`--selftest`: 29 checks PASS, driven by me**

`flock -n … systemd-run --user --scope --slice=research.slice -p MemoryMax=8G -p CPUQuota=100% python3 … --selftest` → **`PASS -- 29 checks`, rc 0.**

## 2. Item 1 — BE's cascade is **genuinely cited**, and that is not the whole story

**Cited, not copied — confirmed structurally:** the runner defines
`load_params`, `verify_be_module`, `verify_day_inputs`, `seed_for`, `arm_day`,
`_strip_economic`, `seal`, `_economic_keys_in`, `assert_no_economic_leak`,
`aggregate`, `fixture_run`, `selftest`, `main` — **no cascade or replay logic**,
and **zero** references to `HSP.`, `replay_policy` or `harmful_stateful` in the
file. `verify_be_module` (`:112–124`) hashes BE's source and refuses on mismatch.
Driven:

```
BE module wrong digest -> REFUSED: declared 67fc7b6c0150d3f9, found 000…0
                                   "a null run through a DIFFERENT cascade is not
                                    a control for this arm"
```

**FINDING — the citation proves which module is on disk, not that the draws came
from it.** `arm_day(day, arm, observed, null_draws, n_decisions, params)` takes the
draws as an **argument**; the runner never invokes BE's module. So the digest check
and the numbers are two unconnected facts. **This is the third instance of the same
class I have filed** — DA's `source_sha256_while_the_child_ran` (disk, not
interpreter) and BE's second-read pickle digest — and it is the most consequential
of the three, because here the unbound object is *the null itself*.

**What would close it:** the producer of `null_draws` records the digest of the
module that generated them, and `arm_day` refuses when that provenance digest
differs from the one `verify_be_module` verified. A digest beside the numbers, not
beside the file.

## 3. Item 2 — the R5 catch, **reproduced exactly**

DE reports its guard caught its own emitter on the first run. I reproduced it:

```
economic keys BEFORE seal : admissibility.null_mean, admissibility.null_sd,
                            economic.D_E0, economic.Z, economic.null_draws_summary,
                            economic.null_mean, economic.null_sd, economic.p_location
economic keys AFTER  seal : []                       <- nothing leaks
seal ONLY the top-level block (DE's first emitter):
        what still leaks  : admissibility.null_sd, admissibility.null_mean
        guard             : REFUSED — "economic fields leaked into a SEALED artifact
                            at ['days[0].admissibility.null_sd',
                                'days[0].admissibility.null_mean'] with 1 of 5 days
                            complete … a leaked Z is an early stop waiting to happen"
and it UNSEALS at 5 of 5  : economic.D_E0, economic.Z, … present again
```

**What leaked and why it mattered:** the null's **centre and dispersion**. With
`D` also computed per arm-day, `Z = (D − mean)/sd` is recoverable — so day 1's Z was
reconstructible from a "sealed" artifact. **The leak was material, not cosmetic.**

**And the guard's own falsifier is the best line in the batch:** it tests **keys,
not substrings**, so it finds a nested `Z` and does **not** flag
`sealed_field_names` — *"the needle-matches-its-own-prose failure, caught in my own
check."* It also refuses a leak **planted at depth inside a nested list**, and the
R4 admissibility *status* survives the seal — what is withheld is the economics, not
the fact that the day ran.

## 4. Item 3 — G is bound from the set, driven in every direction

```
params: days 09-03..09-08, G=6, expected_G=6, G_derived_from_len_days=True
five-day params file                    -> REFUSED (5 days vs expected_G 6)
expected_G mutated to 5                 -> REFUSED (6 days vs expected_G 5)   [both ways]
a previously-opened day (09-01)         -> REFUSED, naming
                                           {'2026-09-01': 'interim_read_of_frozen_candidate'}
a day with NO read-state entry (09-30)  -> REFUSED as 'NO_READ_STATE_RECORDED'  [no silent default]
an arm on 4 of 5 days                   -> UNTESTABLE, G unchanged  [driven in --selftest]
```

**All six params days have read-state entries, all `none`.** ✓

## 5. Item 4 — R6 digest verification is a refusing code path ✓ (driven above)

## 6. Item 5 — the aggregate, recomputed independently

| case | G | mean Z | signs + | best p | Holm m=2 | clears |
|---|---|---|---|---|---|---|
| unanimous positive | 6 | +1.2000 | 6/6 | **0.015625** | 0.025 | **True** |
| one negative | 6 | +1.0167 | 5/6 | — (no unanimity) | 0.025 | False |
| all zero | 6 | 0.0000 | 0/6 | — | 0.025 | False |

**The §7 predicate is evaluable with no free parameter**: G comes from
`len(days)`, the exact sign test needs no threshold, and Holm's 0.025 is α/m with
both α and m declared. The fixture receipt demonstrates the directional case at
G=3 — `p_one_sided_sign_test 0.125`, `clears_holm false`,
`a_pass_is_significance_bearing false`, verdict `BEATS_DIRECTIONALLY_…`.

## 7. Item 6 — the deadline

`per_day_deadline_s: 43200` (12 h) in the params, enforced at `:175` with the
refusal naming both protections: *"the 500-draw minimum and the 8G cap are BOTH
protected by refusing the day — never by lowering draws or raising the cap."*
Driven in `--selftest`: *"R8 KNOWN-BAD, AN OVERRUN: the DAY refuses"*, and
separately *"499 draws refuses — the 500 minimum is never lowered."* ✓

## 8. Item 7 — resources, and **the fixture run FAILED in my worktree**

```
wall 0.06 s · max RSS 19,300 KiB · under one CPU / MemoryMax=8G
exit status 1 — [de_multiday_design_declaration] FAIL: R7 DERIVED FROM THE LEDGER …
                "six days qualify on QUALITY -- []"
```

**FINDING — `--fixture-run` is not data-free.** `de_multiday_gate1_runner.py:351`
calls `DESIGN.selftest(quiet=True)`, whose R7 check reads
`da_dayverdict_*.json` from the real ledger. The receipt says
`status: FIXTURE_RUN_NO_DATA` and `no_day_book_was_read` — both true of *books*,
and neither true of the *verdict ledger*.

It fails **closed**, which is the safe direction. But it **cannot be driven by any
reviewer whose worktree is a shell**, which is every seat after a checkout. Either
the status should say what it reads, or the fixture path should stub the ledger.

## 9. **THE BLOCKER — the R7 assertion breaks at 00:06Z on 09-07**

`de_multiday_design_declaration.py:1122–1130`:

```python
ok(len(r7["qualifying_on_quality"]) == 6
   and r7["SET_A_reads_count_as_untouched"]["holm"]["G"] == 6
   and r7["SET_B_reads_consume_the_day"]["holm"]["G"] == 3
   and …)
```

**Three hardcoded counts against a ledger that grows by construction.** When 09-06
is verdicted at **00:06Z on 09-07**, qualifying becomes **7**, Set A G becomes
**7**, Set B G becomes **4** — and **all three equalities fail.** The Holm
*outcomes* would survive; the *counts* will not.

**And the runner calls this selftest on every fixture run (`:351`), so the runner
fails with it.**

This is **exactly the date-dependent-fixture rot DA fixed in round 54** — *"the
check went RED the moment its own fixture date arrived"* — reintroduced in a new
module, against an input whose growth is guaranteed and scheduled. **The assertion
must be on the RULE (Set A clears at m=2; Set B does not) and not on the counts**,
or it must derive its expectation from the same ledger read.

---

# Verdict

**NOT APPROVED to run the 09-03 smoke as it stands.** Required first:

1. **Fix the R7 count assertion** (§9). It is a blocker on a clock, and it takes the
   runner down with it.
2. **Bind the draws to the verified module** (§2) — a provenance digest travelling
   with `null_draws`, refused on mismatch. Without it, "BE's cascade is cited" is
   true of the file and unproven of the numbers.
3. **Correct `--fixture-run`** (§8): either stub the ledger so the name is honest,
   or state in the receipt that the verdict ledger is read.

**Everything else I drove is sound**, and two things are better than the design
required: the R5 guard tests keys rather than substrings and catches leaks planted
at depth, and the params validator refuses an unrecorded read-state rather than
defaulting into it. **With (1)–(3) closed I would approve the smoke without further
conditions.**

---

## CONTEXT

Far below the 80% reset threshold.
