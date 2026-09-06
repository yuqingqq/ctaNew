# REVIEW — **GO, CONDITIONAL ON ONE THING: settle the output's NAME first.** The seam that cost 85 minutes is structurally closed and driven; (a), (b) and (c) all pass. But (d) has a **certain** end-of-run refusal left: a launch-stamped output name refuses at the emit by **−5,074 s**, and the stamp-free alternative is invisible to the read gate's own glob

**Filed** 2026-09-06T10:48Z (clock read before composing) · reviewer seat (pm-codex)
· tip `353c3ad` (DE 89 `8c20f1b`/`f9a03dc`/`e66861d`; DE 90 `31790ef`/`63e2a54`/`353c3ad` —
verified ancestors) · **LIGHT AND LOCK-FREE.**

**Rule 20.** BE 59 holds the lock (`be59book.scope`, pid 3221533, 09:39 elapsed, peak
3.99 GB) and **I did not take it.** Heaviest steps: runner battery **23.58 s / 850 MB**,
one **6.05 s / 944 MB** growth-budget drive sized under the 1 GiB bar, design **1.06 s**,
DA verdict **2.36 s**. Batteries at the tip under my run: **runner 220, design 100, DA 52,
`de_r608_resolver_agreement` 6** — all 0 failures.

**ROUTING — CHECKED unless a line says AGREED.**

## THE GO ANSWER

| # | question | answer |
|---|---|---|
| **(a)** | does `--day 2026-09-03` reach the emit without the fixture-vs-process-peak seam **under any configuration**? | **YES — structurally, not by ordering.** §1.1 |
| **(b)** | will `producing_code` name the bytes that run? | **YES.** `producing_code_sha256 == LAUNCH_SOURCE_SHA256`, `digest_taken_at: MODULE IMPORT, before any work`. |
| **(c)** | `rehearse_smoke('2026-09-03')` | **READY, `blocking: []`.** (09-04 still blocks on its two book preconditions, correctly.) |
| **(d)** | anything else that refuses at the END? | **YES — one CERTAIN, one OPERATIONAL, one KNOWN-AND-RULED. §2** |

**My recommendation: GO once the output-naming rule in §2.1 is declared.** Nothing else I
can find would spend 85 minutes and then refuse. The naming issue will, with certainty, on
the command as published.

---

# 1. What is closed, driven

## 1.1 The 85-minute seam — closed structurally, and the control still fires

The budget is now **this run's growth in CURRENT RSS from its own `S_start` baseline**,
evaluated at **every stage mark**, refusing at the first crossing. Driven, as the exact
inverse of my REV 53 §0.3 reproduction:

```
A. clean process (ru_maxrss 32 MB): a fixture day ADMITS -- growth 11.4 MB vs budget 700 MB
B. THE REV 53 CASE: allocate 900 MB, free it (ru_maxrss 944 MB, never falls)
   the SAME fixture day ADMITS -- growth 0.0 MB at a 944 MB process peak   ** SEAM CLOSED **
C. clean process, budget forced to 1 MB:
   REFUSED DAY … AT STAGE S0_verify: this run has GREWN 13 MB over its baseline …
   -- refused HERE, at the first stage that crossed it, not at the emit
```

**C is the half that matters and I got it wrong the first time**: run in the process left
over from B, C *admitted*, and I nearly filed that the control could not fire. The cause was
my own probe — after freeing 900 MB the allocator had not returned it, so the baseline was
high and the fixture's work fitted inside the free arena, reading 0.0 MB of growth. Re-driven
in a **clean** process it refuses at `S0_verify`. Recorded because it is also a real property
of the instrument: **growth-on-current-RSS under-reports in a process that already holds
freed-but-unreturned memory.** Harmless for a real day — a fresh process with a small
baseline — but it is why B's number reads 0.0 rather than ~11.

**And the battery no longer sits on the far side of 85 minutes.** It is a `before_work` hook
called after the book digest and before S1. Driven with a spy module and a hook that refuses:

```
a refusing battery -> book loads 0, replay/draw calls 0, wall 0.03 s
```

**Zero draws, zero book loads.** A battery finding can no longer cost a day.

*One consequence DE handled rather than hoped, and I checked the arithmetic:* the battery's
memory is now inside the day's budget, so `REAL_DAY_BUDGET_DERIVATION` carries it as a term —
2008 + 26.1 + 1500 = 3534.1 against a **4000 MB** budget under the 8192 MB cap. **The cap was
not raised** (R-174).

## 1.2 Is the real day's budget a declaration or a measurement dressed as one?

**It is a derivation from measurements, and it says so — but the headroom term is a choice,
not a measurement.** 2008 MB is BE's measured reference; 26.1 MB is DE's measured retained
battery; **~1500 MB is headroom** — a judgement about what else the day might need. The
result (4000) is then rounded up from 3534.1. So: three of four terms are measured, the
fourth is a declared allowance, and the rounding is a declared allowance too.

That is the honest reading and it is fine — provided nobody later cites 4000 MB as
*measured*. What would make it fully honest is one field naming the 1500 as a **declared
allowance with its reason**, beside the three measured terms. As it stands a reader
resolving `REAL_DAY_BUDGET_DERIVATION` sees an arithmetic chain and may take all of it as
measurement.

## 1.3 The pin chain, both ways, at the tip — my REV 52 §0 fully closed

```
design v18 -> parameters.path  live/pm_research/declarations/de_multiday_gate1_params_v11.json
              exists: True   on-disk a82a6320af4eb868…  declared a82a6320af4eb868…  MATCH
params v11 -> design.path      data/pm_5min/derived/p003_de_multiday_gate1_design_v18.json
              exists: True, and it is v18
```

Both directions resolve, by path **and** by digest.

## 1.4 The eight shapes — and two more I added

At DE 90 / DA 76, **ten of ten agree in verdict**, none raising:

| shape | DE | DA |
|---|---|---|
| single / correct pair | `PRESENT` / `PRESENT_CHAIN_HEAD` | `ONE` / `CHAIN_HEAD` |
| sha256-only / path-only | `LINK_NOT_A_PAIR` | `SUPERSESSION_LINK_INCOMPLETE` |
| named file, wrong digest | **`SUPERSEDES_TARGET_DIGEST_MISMATCH`** | `SUPERSESSION_TARGET_DIGEST_MISMATCH` |
| digest under another name | `SUPERSEDES_MOVED` | `SUPERSESSION_TARGET_MOVED` |
| two, no link | `AMBIGUOUS` | `AMBIGUOUS` |
| **bare string** | **`SUPERSEDES_MALFORMED`** | `SUPERSESSION_LINK_INCOMPLETE` |
| **a list** / **an int** *(mine, unasked)* | `SUPERSEDES_MALFORMED` | `SUPERSESSION_BLOCK_MALFORMED` |

My §0.1 crash is closed, and row 5's message — which I flagged as *accurate in verdict,
wrong in cause* — was corrected without being routed.

## 1.5 DE 90's own finds, and what they were

* **Two seat-divergences at the landing seam.** (i) DE never checked
  `is_the_declared_LANDING_RECORD`, so the seats could disagree about which artifacts *are*
  landing records; (ii) DE's landing side did not follow `supersedes` **at all**, so **DA's
  own declared correction path — a `.v2` carrying the pair — read AMBIGUOUS on DE's side**
  and would have refused a day whose landing record had been corrected exactly as DA
  declares corrections. Both closed through one resolver. Driven:

  ```
  agreeing copies            -> PRESENT
  DISAGREEING copies         -> LANDING_RECORD_FIELD_COPIES_DISAGREE, naming the field
  v1 + a chained .v2         -> PRESENT_CHAIN_HEAD, using the v2's digest
  an UNDECLARED record       -> NO_LANDING_RECORD
  ```

  REV 54 §1.3 is closed, and closed wider than I filed it: **every** twice-written field is
  compared, not only the digest.

* **Two files, one version — and it is my line that caused it.** Verified at the artifacts:

  ```
  p003_de_multiday_gate1_design_v16.json   protocol …_V16   pins params_v2.json
  p003_de_multiday_gate1_design_v17.json   protocol …_V16   pins params_v10.json
  ```

  Two substantively different declarations under one version identity. The cause is exactly
  what DE says: **version-only naming removed the stamp that made filenames unique and
  nothing replaced it with a check** — and version-only naming is the remedy I verified and
  approved in REV 53 §1.3 without asking what had been making the names unique. My miss,
  recorded. `assert_output_name_matches_version` now refuses an emission whose filename,
  protocol and chain length disagree.

---

# 2. (d) — what can still refuse at the END

## 2.1 **CERTAIN, and it is the GO condition: the output's NAME**

`_main_day` calls `assert_name_stamp_is_the_clock(a.output, payload["as_of"])` at the emit,
tolerance **300 s**, and `payload["as_of"]` is taken when the day finishes. The operator
supplies `--output`. Driven on the last smoke's **own** name:

```
p003_de_gate1_day_run_20260903_SEALED__20260906T082155Z.json   emitted 09:46:29Z
  -> REFUSED: … stamped 20260906T082155Z while the artifact was written at 09:46:29 -- -5074s
the same name, emitted 2 minutes later -> ADMITS
```

**A launch-stamped name — the natural choice, and exactly how the last run was launched —
refuses at the emit after 85 minutes.** And the obvious escape is closed by the read gate's
own glob:

```
sealed glob: p003_de_gate1_day_run_20260903_SEALED__*.json
  a stamp-free name …_SEALED.json           matches: False
  find_sealed_day_receipt on such a file -> MISSING
```

So a stamp-free name passes the emit and is then **invisible to the gate it exists for**.
And the published command carries no rule: `THE_ONE_COMMAND` ends `--output <receipt path>`.

**This must be settled before GO, and it is a declaration act.** The cleanest resolution, and
the one I would recommend: on the day path compare the stamp against **`LAUNCH_TIME_UTC`**,
not `as_of`. DE's original defect was names stamped for a moment that **had not yet
occurred** (+12 and +17 minutes); a launch stamp is honest about when the run began, is the
only stamp knowable when the name is chosen, and the check still refuses a future stamp and
a typed one. The alternative — the runner deriving the final name from `emission_stamp()` at
write time — also works and needs `--output` to become a directory or a template.

## 2.2 OPERATIONAL: `assert_source_unchanged` at the emit

The emit still refuses on closure drift, a moved HEAD, or a worktree dirty at import. That is
rule 22 working — and it means **any landing into the run's worktree during the 85 minutes
refuses the receipt**. BE 59 is landing now; MEM and DA land every few minutes. The
mitigation is entirely operational: the run must execute from a worktree that is frozen for
its whole life, and DE must land from a second one. Nothing in the code can distinguish "a
seat landed" from "the code moved", nor should it.

## 2.3 KNOWN AND RULED: the peak-stage predicate

`assert_peak_stage` is still evaluated after S5 and can refuse a correct-looking day. That is
REV 47 §2.3 and the runbook already carries the response — a **declaration act**, never a
widened predicate. DE 90 reduced the risk: the argmax now excludes the `before_work` hook's
delta, and a hook high-watering 9 GB still leaves the peak at `S1_load`.

## 2.4 Not a refusal risk, but worth one line before an 85-minute run

The runner's battery is now **850 MB / 23.6 s** (52 MB three rounds ago). It runs *inside*
the day now, so its memory is budgeted — but as a standalone command every seat runs several
times a round, it is within 17 % of rule 20's 1 GiB heavy bar.

---

# 3. VERDICT

**GO for the 09-03 re-run, conditional on §2.1 being declared first.** (a), (b) and (c) all
pass, driven. The seam that cost 85 minutes is closed structurally — not by moving the
battery, but by making the budget a per-run delta — and the battery's move is what makes a
battery finding cost 0.03 s instead of a day. §2.2 is a discipline, not a defect, and §2.3 is
already ruled.

**What I did not establish.** I did not run the day — no reviewer should, and the lock is
BE 59's. Every drive is on synthetic days and scratch roots; **the real book has never been
through this code path end to end**, and the only real-day evidence anyone has is a run that
died at the emit. My growth-budget drives are on a 24-slug fixture, not on a 2 GB book, so I
have verified the *rule* and not the *headroom*: whether the real day's growth actually stays
under 4000 MB is unknown until it runs, and if it does not, the per-stage check will refuse
early — which is the correct outcome and the point of the fix. I did not drive
`assert_peak_stage` on real stage deltas. And §2.1's remedy is my recommendation, not a
verified fix: I drove the failure, not the repair.

**Context: ≈33%** — 330k tokens of the 1M window by my own count; this build's pane status
line carries no `% context used` field, so it is my count, not the pane's.
