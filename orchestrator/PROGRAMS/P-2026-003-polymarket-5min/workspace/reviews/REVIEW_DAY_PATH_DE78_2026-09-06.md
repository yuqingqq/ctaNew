# REVIEW — DE 78's `--day`: every declared behaviour reproduces and both my R-577 findings are CLOSED — but **the fixture/real lock is not on the day path, the lock instrument is defeated by `open()`, and a day receipt's battery skips all 23 checks written for the day path**

**Filed** 2026-09-06T06:33Z (clock read before composing) · reviewer seat (pm-codex)
· tip `b2626b4` · **no data touched** — every run was a fixture or a synthetic book;
my only writes were to the scratchpad · no code fixed · nothing sealed opened.

**ROUTING — CHECKED.** Every drive is mine. Two of my probes were inconclusive
before they were conclusive and I say which (§3.3).

## VERDICT

**`--day` is APPROVED for the 09-03 smoke once BE's book exists**, conditional on
three items closed **before** the smoke runs (§5). All four batteries reproduce at
the coordinator's counts under my runs — design **71**, runner **100**, data root
**18**, supersession diff **26** — and the runner suite is still light by
measurement (8.11 s, 44 MB).

**Everything the dispatch asked me to reproduce, reproduces**, and all five declared
attacks fire, plus one of my own. **Both R-577 findings are closed at the artifact.**

**Three findings, in the order I would fix them:**

1. **`--synthetic-day 2026-09-03` emits a SEALED day artifact stamped with the smoke
   day.** `resolve_draws()` — the structural fixture/real lock DE built in round 77
   and which I verified holds under a caller-rewrite attack — **is not on the
   `--day` path at all.** Third recurrence of DE's own round-77 finding (d). **§1.4.**
2. **The lock instrument is defeated by `open()`.** Reading `/proc/self/fd` cannot
   distinguish *holding the flock* from *having the file open*. Two lines make
   `heavy_run_lock_held` report `True`, and `assert_rule20` then **admits a
   1-hour / 6.84 GiB run**. When I drove it, another process genuinely held the
   flock — so the instrument certified my process in exactly the 05:54Z condition it
   was built to expose. **§2.3.**
3. **A `--day` receipt's `battery: PASS` skips 35 checks — including all four R6
   controls and ALL 23 "DE 78 day-path" checks.** `_main_day` calls
   `selftest(offline=True)` unconditionally. Disclosed in the fields, and for a
   *fixture* it is correct; for a **real day** it has no justification, because the
   run is already reading `data/`. **§3.4.**

Plus one smaller: **the per-stage memory instrument is a highwater, so it cannot
falsify the memory plan's central claim** (§4.3).

---

# 1. THE DAY PATH ON A SYNTHETIC BOOK — driven, then attacked

## 1.1 The happy path, and the seven things reproduced

```
python3 de_multiday_gate1_runner.py --synthetic-day FIXTURE-DAY-1 --output <scratch>
 -> status FIXTURE_DAY_RUN_NO_REAL_DATA · arms {CONDVALUE: OK, HAZARD: OK}
    decisions {48, 48} · peak 42.3 MB · wall 1.7 s · lock_held false
    heavy_by_measurement false · no_tape_artifact_opened true · battery PASS
    [my measurement: wall 1.87 s, maxRSS 44 MB]
```

| the dispatch's item | reproduced at the emitted artifact |
|---|---|
| **R6 book digest vs the builder receipt** | `digest_source: "BE's builder receipt, not a DE constant"`, `digest_recomputed_at_read_time: true`, sha `25eed2c1…` |
| **decision population per arm at the pinned theta from `asm`** | 48 / 48, `definition: "above-threshold generations at the arm's FIXED theta — the set a cancel decision is drawn from"` |
| **R4 refusals** | driven in §1.3(b) |
| **the null GENERATED IN-PROCESS at the seed from the book digest** | `draw_provenance.draw_source GENERATED_IN_PROCESS`, `generated_in_process true`, pid, and `arm_day` calls `verify_draw_provenance`, which **recomputes** the seed |
| **D(E0) per arm** | computed and **withheld** — see the seal |
| **the SEALED per-day artifact in the symmetric layout** | `sealed true`, `sealed_at_every_depth true`, `sealed_field_names` 7 names, **`economic` key absent**; all four layout keys present |
| **the run receipt: battery equality + wrapper** | `65 run + 35 skipped = 100`, `run_plus_skipped_equals_source_expected true`; wrapper read from `/proc/self/cgroup` and `/proc/self/fd` |

**And the residency proof is a genuine measurement with a non-vacuity check I would
not have thought to demand:** it refuses unless the instrument observed *the book*
being read — *"an instrument that missed the one read this path certainly makes
cannot testify about the reads it did not see."* My run: `n_paths_opened 6`,
`n_distinct_paths 5`, `tape_artifacts_opened []`, `instrument_observed_the_book_read
true`.

## 1.2 `D(E−R)` is refused rather than approximated — the right call

`what_this_is_not.D_E_MINUS_R_is_UNBOUND`: the robustness endpoint needs the
rebate's identity value, which is not on DE's surface, so **D(E0) is computed from
`fill_value_cents` (level-to-markout, no fee term — genuinely the E0 endpoint) and
D(E−R) is not computed and not approximated.** A missing endpoint named rather than
filled in.

## 1.3 The five declared attacks — all fire, plus one of mine

```
(a) the builder receipt's sha256 mutated to 0*64     REFUSES  "THE DAY refuses"
    POSITIVE: the unmutated pair                     ADMITS
    MINE: the BOOK BYTES mutated, receipt untouched  REFUSES  (declares a2d19418, disk 26f03ef1)
(b) an R4-THIN arm-day (head_policy CONDVALUE=thin)
      CONDVALUE  DEGENERATE_ARM_DAY_REFUSED_TOO_FEW_DECISIONS  n_dec 0  economic_key False  sealed True
      HAZARD     OK                                            n_dec 48 economic_key False  sealed True
      receipt G 6 == params G 6      <- a refused arm-day does NOT shrink G
(c) null_sd planted at depth in a SEALED artifact     REFUSES  'nested.deep[0].admissibility.null_sd'
    POSITIVE: unplanted at 1 of 6                     ADMITS
    POSITIVE: the SAME planted artifact at 6 of 6     ADMITS   <- the guard is about the SEAL, not the field
(d) a five-day params against expected_G 6            REFUSES  "a set that shrank is a day chosen after the fact"
(e) may_run_day('2026-09-02'), previously opened      REFUSES
```

**(a) is the one worth calling out.** My own known-bad — mutating the **book bytes**
and leaving the receipt alone — is the direction a receipt-only check would miss,
and it refuses, because the digest is recomputed from the bytes at read time. The
seam is right in both directions.

**(b)** is the R4 behaviour exactly as declared: a status, not a silent drop, with
no economic field, and G untouched.

## 1.4 **FINDING — `--synthetic-day` runs a FIXTURE on a RULED day, and emits an artifact stamped with it**

`resolve_draws()` refuses precisely this:

> *"REFUSED: fixture draws were claimed for 2026-09-03, which IS in the ruled day
> set. A fixture run on a ruled day is not a fixture run, and only its author would
> know."*

**`resolve_draws()` is not on the `--day` path.** Direct callee inspection (not a
transitive closure — see the caveat below): `_main_day` calls `load_params`,
`write_synthetic_day`, `day_split_residency_proof`, `carrying_commit_block`,
`selftest`; `day_split_residency_proof` calls `run_day` through the instrument;
`run_day` calls `verify_book_against_builder_receipt`, `import_be_cascade`,
`day_decision_population`, `null_draws_valued`, `arm_day`, `seal`,
`assert_no_economic_leak`, `assert_rule20`, `wrapper_observed`. **Neither
`resolve_draws` nor `may_run_day` appears anywhere on it.** `run_day` builds the
provenance dict inline instead.

**Driven:**

```
python3 de_multiday_gate1_runner.py --synthetic-day 2026-09-03 --output <scratch>
 -> rc 0.  status FIXTURE_DAY_RUN_NO_REAL_DATA · day "2026-09-03" · fixture true
    per_day_sealed_artifacts: both arms OK, sealed true
```

**A sealed day artifact stamped `2026-09-03` — the smoke day — from a synthetic
book.** It is *disclosed* (`status`, `fixture: true`), not hidden, and the inline
CLI check works in the other direction (`--day 2026-08-29` refuses on the committed
ruled set). But the lock DE built exists precisely because *"a fixture run on a
ruled day is not a fixture run, and only its author would know"* — and once a real
09-03 artifact exists, a synthetic one carrying the same `day` is exactly the
collision the lock forbids.

**This is the third instance of one class.** DE's own round-77 finding (d):
*"`resolve_draws()` shipped with eleven falsifiers and the fixture run went AROUND
it."* My R-577 finding: `may_run_day` hardened, unwired. Now: the CLI built one
round later goes around both. **Rule 17 — the wiring, not the unit.**

**Also unwired: `assert_seal_layout_symmetric`** — three call sites, all in the
battery (`:2340`, `:2355`, `:2362`), none on the day path. Mitigated in fact,
because `seal()` writes all four keys unconditionally, so the emitted layout *is*
symmetric — I checked it on my own artifact. But the consumer falsifier never meets
a real emission.

> **A caveat on my own method, recorded because it nearly cost a false finding.** My
> first pass used a transitive AST closure and reported *everything* — including
> `wrapper_observed` and `verify_book_against_builder_receipt`, which I had read in
> `run_day` — as unreachable. `day_split_residency_proof` invokes `run_day` through
> an instrument (`DR.instrumented(run_day, …)`), so the static closure breaks there.
> Conversely, a closure that *includes* `selftest()` reports everything as called,
> because the battery calls everything — which is the very confusion rule 17 names.
> **The finding above rests on direct callee inspection of the three functions, not
> on either closure.**

---

# 2. THE LOCK AT THE ARTIFACT

## 2.1 What DE built is the right idea, and it answers my own §A.6 finding

`_lock_fd_held()` reads `/proc/self/fd` and matches on `realpath`, because
`flock -n <lock> systemd-run --scope <cmd>` keeps the lock's fd open across the exec
and the scope inherits it. The docstring cites the reason exactly: *"R-575(C): at
05:54Z two heavy scopes ran concurrently, one holding the lock and one not, and no
artifact could tell them apart."* **That is my REVIEW_BE48 §A.6 measurement, turned
into an instrument.** The intent is right and `declared_wrapper_is_not_evidence` is
the correct sentence.

## 2.2 Three ways, driven

```
(a) BASELINE, no fd on the lock          heavy_run_lock_held False   fds []
(b) an UNRELATED fd (the dispatch's forge: /etc/hostname, landed on fd 3)
                                          heavy_run_lock_held False   fds []   CORRECT
(c) a heavy run without the lock          assert_rule20 REFUSES
      "HEAVY by measurement (3600.0s wall, 6.84 GiB peak, against rule 20's
       60.0s / 1.0 GiB bar) and did NOT hold … The artifact is not written."
```

**(b) is correct and worth crediting**: the match is on `realpath`, not on the fd
number, so an unrelated fd landing on 3 is *not* counted. The dispatch's forge fails
as it should.

## 2.3 **FINDING — but `open()` on the lock file forges it**

The instrument asks *"is one of my fds pointing at this file"*. Holding a flock is a
different fact.

```
g = open("/home/yuqing/ctaNew/data/.heavy_run.lock")     # NO fcntl.flock at all
wrapper_observed()  ->  heavy_run_lock_held: True   lock_fds: [3]
assert_rule20(that, wall_s=3600, peak_rss_mb=7000)  ->  ADMITS
                       {'heavy_by_measurement': True, 'lock_held': True}
```

**Two lines of Python and a 1-hour, 6.84 GiB run certifies itself as lock-holding.**

**And the circumstance sharpens it.** At the moment I drove this, a fresh-fd
`LOCK_EX|LOCK_NB` attempt **failed** — the flock was genuinely held by another
process. So: my process held no lock, another seat's did, and the instrument would
have written `heavy_run_lock_held: true` into my artifact. **That is the 05:54Z
condition, reproduced by the instrument built to prevent it.**

**`/proc/locks` is the authoritative surface and I verified it separates them:**

```
lock inode 1053378
FLOCK entries on that inode: 1
   6: FLOCK ADVISORY WRITE 2916372 103:01:1053378 0 EOF     <- another seat's flock wrapper
after opening the lock file WITHOUT flock: still 1 entry, this pid NOT among them
```

**The fix, in the order I would take it:** at minimum, an inherited fd **plus** a
fresh-fd `LOCK_EX|LOCK_NB` that FAILS (proving somebody holds it) — that alone kills
the accidental case. Exactly, an fd **plus** a `/proc/locks` FLOCK entry on that
inode whose pid is this process or an ancestor of it: note the holder is the `flock`
wrapper (pid 2916372 above), not the python child, so a bare "is my pid in
/proc/locks" test would reject a legitimately-wrapped run. Both surfaces are already
readable from inside the process.

## 2.4 R12 — a real day refuses before any work, verified two ways

```
run_day("2026-09-03", <synthetic book>, fixture=False)   ->  REFUSES
   "a REAL day is heavy by construction (BE projects ~2.3 h for both arms) and
    this process does not hold …/.heavy_run.lock"
structurally: the lock check is at char 869 of run_day; the first work
   (verify_book_against_builder_receipt) at 1358  ->  lock is FIRST: True
```

**Refused before any work, and nothing written.** Behaviour and structure agree.

---

# 3. MY TWO R-577 FINDINGS — both CLOSED

## 3.1 `may_run_day` reads the committed file — attack re-driven

```
caller's params['days'] rewritten to begin with 2026-08-29
   may_run_day('2026-08-29')            REFUSES  "not in the ruled day set [09-03 … 09-08]"
   POSITIVE: may_run_day('2026-09-03')  ADMITS
MINE, the other direction: params['days'] emptied ENTIRELY, ask for 09-03
                                        ADMITS   <- correct: the committed file governs
```

**Independent of the caller's dict in both directions.** The function that admitted
2026-08-29 last round now refuses it. ✓ (It remains unwired — §1.4.)

## 3.2 `the_committed_day_set_is_empty` is COMPUTED, and there is a standing sweep

```
the_committed_day_set_is_empty = False    (the committed set has 6 days)
_literal_audit(): rule "10 -- compute predicates, never print conclusions"
   why: "a receipt of bare booleans cannot tell a reader which were computed.
         `the_committed_day_set_is_empty` was a hardcoded True, asserted by
         nothing, and FALSE"
classified COMPUTED or INTENT: no_day_book_was_read, no_path_under_data_was_opened,
   runnable_from_a_shell_worktree, the_committed_day_set_is_empty,
   declared_before_any_draw
```

The sweep is one-directional — every *emitted* boolean must be classified, which is
the direction that catches a new one. ✓

## 3.3 The plant — caught, and my first two attempts were inconclusive

**Recorded plainly because it is the point of §3.4.** I planted
`reviewer_planted_boolean: True` and the battery **passed** twice. Neither result
was a finding: the first plant was on `fixture_run`, which the check does not call;
the second was on the right function but I ran the battery **offline**, and the
check is one of the 35 offline skips — the skip list says so verbatim: *"(3) the
literal sweep over the emitted receipt (it calls `fixture_run_proven`, which would
recurse)"*.

**Driven against the ONLINE battery, the plant is caught:**

```
[de_multiday_gate1_runner] FAIL: THE LITERAL SWEEP IS EXHAUSTIVE: every top-level
boolean in the EMITTED fixture receipt (['no_day_book_was_read',
'no_path_under_data_was_opened', 'reviewer_planted_boolean',
'runnable_from_a_shell_worktree', 'the_committed_day_set_is_empty']) is classified
COMPUTED or INTENT. A new bare boolean with no entry REFUSES here …
CONTROL unplanted, ONLINE -> rc 0
```

**The sweep works.** It refuses a new unclassified boolean by name, and admits the
unplanted receipt.

## 3.4 **FINDING — but a `--day` receipt carries the OFFLINE battery, which skips every check written for the day path**

`_main_day` calls `selftest(quiet=True, offline=True)` and stamps the result into the
day artifact. The 35 checks that skips:

```
 1-4    ALL FOUR R6 controls  (positive control; a planted model byte;
                               an absent pinned model; a theta disagreeing with its pin)
 5-11   the fixture data-freeness probe, its non-vacuity check, the laundering
        known-bad, THE LITERAL SWEEP, the committed-day-set computation check,
        the derived runnable-from-a-shell-worktree check
12-34   "DE 78 day-path check 1/23" … "23/23"   <- every check written FOR THE DAY PATH
35      the day-path check-count agreement
```

**A day artifact says `battery: PASS` having skipped all 23 day-path checks and all
four R6 controls.** It is honestly disclosed — `n_checks_run 65`,
`n_checks_skipped_offline 35`, `run_plus_skipped_equals_source_expected true`, and
the skip list names each one — so this is not a hidden defect. It is a *meaning*
defect: the field a consumer resolves on a real-day artifact would be a pass that
excluded the path being run.

**Stated fairly: the day path itself still ENFORCES R6.** `run_day` calls
`verify_pinned_models()` and `verify_pinned_thetas()` for real when `fixture` is
false — I read them in its callees. What is skipped is the battery's *controls* on
those verifiers, not the verifiers.

**The offline choice is right for `--synthetic-day`** — it preserves the data-free
property that makes a fixture a fixture. **It has no justification for `--day`,
which is already reading the ledger.** One expression:
`selftest(quiet=True, offline=fixture)`.

---

# 4. DESIGN v9 R11 AND R12

## 4.1 R11 — `index_splits_needed_by_day = NONE at any stage` is right at the code, and it is MEASURED

Three independent confirmations:

* **Measured, in my own run:** `tape_artifacts_opened: []`, `n_distinct_paths: 5`,
  `non_vacuous: true`, and the proof **refuses** unless the instrument observed the
  book read — so the zero is a measurement, not a silent no-op.
* **Structurally:** the only door from the day path into `de_phase4_diag_runner` is
  BE's `load()`, which calls `R.generations_with_fills(reference)`. I read its
  source: no `TAPE_PATH`, no `tape_index`, no `build_tape_index`, no `FRAGMENT`, no
  `_stream_tape_rows` — **it is a pure function over the reference.**
* **The consequence DE draws is the right one:** *"the index is needed only to
  PRODUCE `asm`. It does not have to be resident alongside the reference and `asm`
  at all, because the consumer never wants it."* That is the answer to my
  REVIEW_BE48 §A.6 question about BE's builder passing a pre-built combined tape,
  and it is stronger than I put it: not merely *partition* the index, but **release
  it before the book is handed on.**

## 4.2 R12 — verified in §2.4

## 4.3 **FINDING — the per-stage memory instrument cannot falsify the memory plan**

`DAY_STAGES` declares S1_load as *"THIS IS THE PEAK of the day path: everything after
it is derived and bounded"* — the claim the 8 GiB real-day ceiling rests on. The
instrument beside it is `_peak_rss_mb() = ru_maxrss`, a **process highwater**.

```
observed series: [37.93, 39.18, 39.18, 39.20, 42.27, 42.27]
non-decreasing?  True      <- by construction; ru_maxrss never falls
declared peak stage: S1_load      measured argmax: S4_null
any predicate asserting the peak occurs at S1?  False
```

**The series cannot show S1 as the peak unless nothing after S1 allocates at all**,
and in my run the argmax is S4_null. `within_budget` compares one scalar to one
ceiling and would not notice if the plan's shape were wrong. On a real day, "S1 is
the peak" is the whole basis of the ceiling holding — so it should be a predicate.

**Two small changes:** record a per-stage *delta* (or sample `/proc/self/statm`,
which does fall), and assert `peak_stage == "S1_load"` with the observed stage names.

---

# 5. VERDICT

**`--day` is APPROVED for the 09-03 smoke once BE's book exists.** The path is
sound, every declared behaviour reproduces, the attacks fire in both directions, and
the two findings I filed last round are closed at the artifact.

**Three items to close BEFORE the smoke runs.** All are small, and each is a
*wiring* rather than a design fault:

1. **Put the fixture/real day lock on the day path** (§1.4) — `run_day` should refuse
   a fixture whose day is in the committed ruled set, and refuse a real run whose day
   is not, on the day rather than on the caller's flag. Wiring
   `assert_seal_layout_symmetric` onto the emitted artifact belongs in the same edit.
2. **Make the lock instrument test the LOCK, not the fd** (§2.3) — a fresh-fd
   `LOCK_EX|LOCK_NB` that fails, or a `/proc/locks` FLOCK entry on the inode whose
   pid is this process or an ancestor.
3. **`selftest(offline=fixture)`** (§3.4), so a real day's receipt carries a battery
   that ran the 23 checks written for the day path and the four R6 controls.

**Riding along:** the peak-stage predicate (§4.3).

**Nothing here touched data**, and the smoke's remaining dependency is unchanged and
outside DE: BE's book, which is still blocked on the two missing inputs of
REVIEW_BE48 §A.3.

---

## CONTEXT

Approximately 78% — close to the threshold. **I will report the 80% crossing on my
next round, and I expect it to come early in that round.**
