# REVIEW 94 — **GO #8 MAY PROCEED. GO E2: NO-GO** — REV 93's block is cleared, and a different one opened while I read

**Reviewer (pm-codex), 2026-09-07T09:0xZ. Read at `bc805a6` in `~/ctaNew-wt-rev`. Read-only:
no heavy unit, no lock, nothing written under `data/`, never `--open`, no feed opened, and no
economic value read from the E1 artifact. CHECKED = I went to the artifact or ran the code;
AGREED = I read the same summary.**

---

# PART A — a split verdict

> **GO #8 MAY PROCEED** at runner `c8be65b2c1ce4ea9…`, params **v19** `dd8db7ded9e6ed97…`,
> design **v27** `3bcdf3c234cb7d4e…`. REV 93's NO-GO is cleared and cleared well — I drove the
> widening rather than reading it.
>
> **GO E2: NO-GO on one artifact — `live/pm_research/de_early_read.py`. Its battery ABORTS at
> cell 4d with an uncaught `KeyError`, 13 of 18 checks run and five cells never reached.** The
> cause is that **GO E1 succeeded**: the cell hardcodes `rehearse("2026-09-03")` and expects
> `READY`, and 09-03 now correctly rehearses `NOT_READY / EARLY_READ_ALREADY_EMITTED`. The
> guard is right; the cell assumed a state the programme's own progress consumes.
>
> **The two are independent: the runner does not import `de_early_read` (0 references), so
> nothing in this NO-GO reaches tonight's day run.**

## §A1 REV 93's NO-GO is cleared, and the ten pins are real

**The question the dispatch asked me to answer: do v19's ten digests equal the blobs at
`9e29af7`?** **Yes — all ten, and they equal the disk too:**

```
module                          v19 pin           blob@9e29af7      disk              PIN==BLOB  BLOB==DISK
be_cancel_axis_null.py          5607bfbfe1b4ef89  5607bfbfe1b4ef89  5607bfbfe1b4ef89     True       True
be_data_root.py                 c82451a61758f2b1  c82451a61758f2b1  c82451a61758f2b1     True       True
de_head_scoring.py              60ef48fea69e83f1  60ef48fea69e83f1  60ef48fea69e83f1     True       True
de_matched_random_control.py    f77aaf2bd2f21988  f77aaf2bd2f21988  f77aaf2bd2f21988     True       True
de_phase4_diag_runner.py        ee4034c15c274982  ee4034c15c274982  ee4034c15c274982     True       True
de_rho_estimator.py             048b8e077c3d37e8  048b8e077c3d37e8  048b8e077c3d37e8     True       True
de_score_stream.py              f85be3354610e2ce  f85be3354610e2ce  f85be3354610e2ce     True       True
harmful_stateful_policy.py      14e669b10ff6115b  14e669b10ff6115b  14e669b10ff6115b     True       True
phase4_generation_tables.py     22ac4efe2014ac95  22ac4efe2014ac95  22ac4efe2014ac95     True       True
pm_tape_density.py              41b22727aa76ed26  41b22727aa76ed26  41b22727aa76ed26     True       True
```

(**CHECKED**, each pin recomputed against `git show 9e29af7:<path>` and against the file.)
**`de_phase4_diag_runner.py` — the one where DE caught its own uncommitted DE 125 bytes — is
pinned at `ee4034c1…`, which is the committed blob.** The restore held and no working-tree
byte was pinned. That is the case that would have been invisible in a report and is the reason
the question was worth asking.

**The re-point is deliberate** (`be_module.sha256` now `5607bfbfe1b4ef89…`, the post-BE-96
bytes) and the pairs verify: params **v19 from v18 `cfc2b06fe2c9e792…`** and design **v27 from
v26 `7de8906e607a66d4…`**, both recomputed from disk. `PARAMS_REL` → v19, and
**`P3_design` HOLDS** (head v27 pins `dd8db7ded9e6ed97…` = v19's digest) — the condition I
named at REV 92 §D, met for the second round running (**CHECKED**).

## §A2 The widening is real — I drove it on the modules the old guard would have missed

My REV 93 §A6 said the guard *"fired on the file that changed cosmetically and would NOT have
fired had only the behavioural file changed."* In a scratch copy of `live/`, perturbing
**non-entry** modules:

```
0 unperturbed (control)               ADMITTED
1 harmful_stateful_policy perturbed   REFUSED BE_CASCADE_DIFFERS   <- the exact case §A6 named
2 de_rho_estimator perturbed          REFUSED BE_CASCADE_DIFFERS
3 pm_tape_density ABSENT              REFUSED BE_CASCADE_DIFFERS
4 restored (control again)            ADMITTED
```

(**CHECKED**, driven by me.) An absent module is refused as well as a changed one, which is the
half a digest comparison alone would skip. The code carries the reason where a reader meets it:
*"a digest on the entry point alone covered a tenth of the code that produces the numbers."*

*One observation, not a hold:* `verify_be_module`'s **return value is unchanged** —
`{path, sha256, cited_not_copied, verified_at_run_time}` — so a receipt cannot distinguish
"verified one" from "verified ten". The widening is real in behaviour and invisible in the
artifact. Adding `n_modules_verified` costs a line and makes the receipt say what was checked.

## §A3 The runner's battery, and tonight's rehearsal

```
de_multiday_gate1_runner --selftest   PASS -- 360 checks, n_disarmed 0, n_skipped 0   rc 0
                                      (32.2 s, 869,128 KiB)      -- it refused at check 7 last round
rehearse_smoke('2026-09-07')          NOT_READY, blocking ['P2_book_exists',
                                                          'P2_builder_receipt_exists']
```

(**CHECKED**, both run by me.) **Blocked only on the book**, exactly as expected for a day that
has not closed. Nothing else in the 09-07 chain is outstanding at the runner's end.

## §A4 GO E2's NO-GO, stated precisely

```
de_early_read --selftest   rc 1
  Traceback … line 661, in selftest
    dc = reh["preconditions"]["digest_comparison"]
KeyError: 'preconditions'
  -- 13 of 18 checks ran; 5 ok() cells after line 661 were never reached
```

(**CHECKED**.) Cell **4d, "THE REAL DAY, against the REAL ledger"**, hardcodes
`rehearse("2026-09-03")` and asserts `status == "READY"`. **GO E1 emitted
`p003_de_early_read_day_20260903__20260907T085436Z.json` at 08:54:36Z**, so 09-03 now returns:

```
status NOT_READY | blocking ['EARLY_READ_ALREADY_EMITTED'] | 'preconditions' present: False
```

**The guard is right and the cell is wrong.** `EARLY_READ_ALREADY_EMITTED` is the precondition
I drove at REV 91 and it is doing its job — *"a second emission would leave two answers to one
question with no rule saying which is newest"*. What broke is a battery cell **whose green
depended on a state the programme's own progress consumes**, and it will break again for 09-04
the moment GO E2 runs, then 09-05, then 09-06.

**Why this is a NO-GO and not a routed item.** Not because the production path is broken — the
09-04 rehearsal is `READY, G 4, EXPLORATORY, NONE_BELOW_FIVE_DAYS, full pair true`, and the
preconditions function itself is the one I drove five ways at REV 91. **It is a NO-GO because
the suite stops partway: five of eighteen cells never run, so I have no evidence about them.**
I cannot clear a module for a ~90-minute run on the lock on the strength of a battery that
aborts, and the abort is an **uncaught KeyError** rather than a named verdict, so the module
cannot even report its own state.

**And the crash is the class I keep catching in myself, in a battery cell:** `reh["preconditions"]`
is an unguarded read of a key that a legitimate `NOT_READY` result does not carry. Three rounds
running I have caught myself guessing a nested key and reading `None` as a real negative; here
the same shape raises instead, which at least fails loudly.

**Closure (DE's call, either shape):** derive the rehearsal day from the ruling's days minus
those already emitted — so the cell can never assume a consumed state — or accept both
`READY` and `ALREADY_EMITTED` as the two legitimate outcomes and assert on whichever it got.
Either way the key read must be guarded, so a future unexpected status is a failed check and
not a traceback.

---

# PART B

## §B1 The E1 artifact — censused by KEYS, no economic value read

`p003_de_early_read_day_20260903__20260907T085436Z.json`, 44,037 B, sha256 `5c8a58f501d3b61b…`.
**Scope: I walked 491 leaves and read booleans, counts and identity strings only. 89 floats and
72 ints exist; I printed none and name none — DA prints the numbers.**

```
is_a_validation  false      G 4      interval NONE_BELOW_FIVE_DAYS      verdict_class EXPLORATORY
days_consumed    2026-09-03, 2026-09-04, 2026-09-05, 2026-09-06
seal_standing.line
  "UNSEALED under the USER's ruling R-754: 4 of 6 ruled days, read early on the user's
   instruction. NOT all days complete, NOT a validation, no interval."
computation_params  params v15  92858fc7f9493f8e…   ("the COMPUTATION is the sealed runs' -- v15")
ruling              params v17  81b2c2910b3c4799…
```

**The sentence I checked at REV 91 rendered correctly on the real run** — I had evaluated its
emit-time expression against the ruling precisely because a `KeyError` there would have cost 70
minutes; it produced the line above (**CHECKED**). `computation_params` names v15, the sealed
runs' params, with the bar coming from v17 — the split R-757 declared, holding in the artifact.

*Routed, minor:* `ruling.path` is an **absolute path into `/home/yuqing/ctaNew-wt-de/`** — a
seat's worktree — while `computation_params.path` beside it is repo-relative. The digest makes
it resolvable, but under R-601 a cited artifact should be locatable, and a path naming a
worktree resolves nowhere once that worktree moves. Two adjacent fields, two conventions.

## §B2 REV 93's routed items — not yet landed

DE 126 **phase 2** had not landed when Part A was decided (`bc805a6`): the R-id existence check,
the shared-falsifier cell in `de_early_read.py`, BE 96's fields into the ledger, and the phase4
suite under R-771's *"the constant is never adjusted to the observation"* are all unread by me.
**AGREED nothing about them; they are REV 95's.** My REV 93 items 1 and 3 (the cascade pinned by
pair; the `de_phase4_diag_runner` FAIL) — item 1 is **closed** by §A1/§A2 above.

---

# §C HOLDS AND ROUTING

| id | artifact | what |
|---|---|---|
| **NO-GO** | `live/pm_research/de_early_read.py`, cell 4d at :661 | **GO E2 blocked.** Battery aborts on an uncaught `KeyError`; 13 of 18 run, 5 unreached. Cause: GO E1's success. Not a production defect — a cell that assumed a consumed state |

**GO #8: MAY PROCEED.** The runner does not import `de_early_read` (0 references), so this
NO-GO does not reach it.

| # | to | finding | kind |
|---|---|---|---|
| 1 | DE | cell 4d must derive its day (or accept both states) and guard the key read — it will break again on 09-04, 09-05, 09-06 | routed (§A4) |
| 2 | DE | `verify_be_module` returns the same shape after the widening; a receipt cannot tell one module from ten. `n_modules_verified` | minor (§A2) |
| 3 | DE | the early-read artifact's `ruling.path` is absolute into `wt-de`; `computation_params.path` beside it is repo-relative | minor (§B1) |
| 4 | DE | `de_early_read.py` still ships no shared-falsifier cell (REV 91 §C1) — expected in phase 2 | carried |
| 5 | coordinator | the checker's scope (≤5 MB, `-maxdepth 1`) still unprinted (REV 92 §A6 #4) | carried |

**Closed this round:** REV 93's NO-GO, and its routed item 1 — the cascade is now pinned by pair
across ten modules, each equal to its committed blob, with the guard driven on three non-entry
perturbations.

# §D WHAT I DID NOT ESTABLISH

- **Not established:** DE 126 phase 2 (unlanded); the five unrun cells of the early-read battery
  — that is the point of the NO-GO; the numbers in the E1 artifact (deliberately unread).
- **AGREED:** DE's own re-measurement of the two `be_module_repoint` axes across `c707eb8` (I
  measured them independently at REV 93 §A6 and got draw-path NONE / constants NONE, which is
  what DE reports).
