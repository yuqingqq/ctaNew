# Review — BE rounds 3–4 (`be_forward_day.py`: the frozen bytes execute; 09-01 scored, sealed)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `248e99f`**, `be_forward_day.py` sha256 `4a2c5e667b9765c0…` (confirmed
against `git show 248e99f:…`). Filing row **Q-BE-229** at `0f45921`.
**Request of record:** `REQUEST_BE_ROUNDS_3-4_2026-09-02.md`.
**Composed 2026-09-02T14:27:24Z.** One filing, per R-377.

**Constraints observed and checked.** BE's artifacts under `…/32b9d1f8-…/scratchpad/fwd5/` were
read only — the five entries are unchanged after this review. Nothing was re-run into `fwd4/` or
`fwd5/`; my three driver runs went to new outdirs of my own under this session's scratchpad,
each under `systemd-run --user --scope -p MemoryMax=12G`. Read-only under `data/`; the
`derived/` listing (184 entries) is **identical** before and after, including after the killed
run. `da_midnight_verify.sh` never run. Reviewed the pinned commit, not BE's working tree —
which matters here (BE34-R3).

Suites at the tip: `be_forward_day --selftest` **46 checks OK, rc 0 under both launchers**;
`v5_deploy_gates.py` roster **21** gates at `248e99f`.

---

## 1. What "the frozen bytes execute" covers — the fact, not the disposition

*Stated for the USER's open freeze-disposition decision. I do not rule it.*

**What is the freeze's bytes.** The receipt binds a manifest (`harmful_candidate_manifest_v1.json`,
sha `eb8733da…`) and names **8 anchors** sha-checked before import against `1b53929`. **7 are code
modules imported from the run dir** with `__file__` asserted under it — `flow_fill_development`,
`flow_intensity`, `harmful_action_eval`, `harmful_exposure_rows`, `harmful_hazard_model`,
`harmful_rows_loader`, `policy_bounds_v1`. The 8th is the **data** anchor
(`harmful_exposure_rows_v3_eraB.json`), gitignored, verified by content, not copied (item 2). So
the scoring path's own code is the freeze's bytes, materialised and hash-checked.

**What runs at HEAD.** The anchors' static import closure is **26 modules**; the 19 non-anchors
execute at HEAD. The receipt names the two whose bytes differ between `1b53929` and HEAD —
`tier1_pipeline.py` and `warning_window.py` — with both shas, and says plainly that the run "is
not claiming they were frozen".

**Where they sit, measured rather than inferred.** I traced the import graph from the seven
imported anchors and then reproduced the driver's own import environment (the frozen run dir
first on `sys.path`, HEAD behind it):

| module | reached via | in this run |
|---|---|---|
| `warning_window` | `policy_bounds_v1` → `warning_window` (module level) | **imported, executes at HEAD** |
| `tier1_pipeline` | `policy_bounds_v1` → `layer2_v1` → `tier1_pipeline` | **never imported** — the import is deferred inside `layer2_v1.load_winners()`, which nothing on this path calls |

And whether either is **on the scoring path**:

- `warning_window`'s only use anywhere in the closure is `ww.select_by_day(...)` inside
  `policy_bounds_v1.run()` (`:643`) and `layer2_v1.run()` — CLI entry points the driver never
  calls; `be_forward_day.py` does not reference `policy_bounds_v1` or `layer2_v1` at all beyond
  importing the anchor. So its **module-level code runs at HEAD; none of its functions do**.
- Its diff `1b53929 → 248e99f` is +29/−5 and **every changed hunk is inside `select_holdout()`**;
  `select_by_day` — the only symbol the closure uses — is **byte-identical at both commits**.
- `tier1_pipeline`'s diff is +109/−3 (`normalize_clob`, `ParseStats`, `selftest`, one module
  constant) and **none of it executes**, because the module is not imported.

**What a reader of R-424 §6's "race on frozen bytes at `1b53929`" should understand at this
driver:** the candidate's own code and its data anchor are the freeze's bytes, sha-verified
before import and imported from the run dir; everything else in the closure is HEAD; of the two
HEAD modules whose bytes differ from the freeze, one executes only its module level with the
symbol the closure uses unchanged, and the other does not execute at all. No frozen artefact was
edited, and the receipt discloses the gap rather than papering it.

One correction to the receipt's wording, which matters because the decision turns on what
*executed*: `not_frozen_in_closure` is computed **statically**, and the `why` says the two
"execute at HEAD". At this run **one of the two does not execute** — `tier1_pipeline` is in the
static closure only. That is in BE's favour (less HEAD exposure than declared) and it is still a
claim in a receipt that does not reproduce; **BE34-R5** below.

---

## Verdict

### RELEASE for `248e99f` as the confirming driver of record for the 09-01 score — with two findings I would close before this driver runs a second day.

The frozen contract, the population, the ratification pair and the sealing are all checked at the
artifact and the 46-check suite is strong where it reaches. What it does not reach is the change
this round is *about*: **the streaming scoring loop has no falsifier — three of my mutants,
including one that disables the day-level reconciliation refusal, leave the suite green** — and
**the driver still silently overwrites its own receipt**, which is how the 12:49 artefact was
lost last round.

---

## 2. The data-root discovery — three refusals, each driven by the suite

The selftest carries all three, by name: *"KNOWN-BAD: without the data-root symlink the frozen
modules index ZERO archive slugs"*; *"a DATA anchor is verified by content and NOT copied into the
run dir — copying it creates a shadow"*; *"KNOWN-BAD: a CODE anchor absent from the freeze commit
REFUSES"*. The run dir's `data` → the repo's `data/` by symlink is asserted, and the receipt
carries the probe: `data_root_probe {frozen_flow_intensity_PM: /home/yuqing/ctaNew/data/pm_5min,
n_archive_slugs: 28031, resolves_to_repo_data: true}` — a positive number, not an absence.

**Is "verified by content" a freeze for the gitignored anchor?** It is a **disclosure**, and the
receipt is right not to call it a freeze. Rule 12 says a freeze is a commit; `data/` is gitignored,
so `harmful_exposure_rows_v3_eraB.json` has no commit to be frozen at. What the receipt binds is
the sha the frozen manifest recorded for it plus the source it was read from — that makes the
input *identified and checkable*, which is the strongest claim available for a file git never
held, and weaker than the code anchors' claim. A reader should read the eight anchors as **seven
frozen-by-commit plus one identified-by-content**.

## 3. The streaming pass — the strict condition is preserved in the code and unfalsified in the suite

**Structure:** `build_and_score()` (`:622`) replays, labels, scores and drops per window, and
accumulates `reconciliation_failures`; the caller refuses the whole day on any non-zero count
(`:869`, *"a mismatch fails the DAY and is never absorbed"*). So a failure in window k after k−1
windows were dropped **does** fail the day — the counter survives the drops and the refusal is a
post-condition over the day.

**Coverage:** none. The held-in-memory path `score_rows()` (`:707`) is **defined and never
called**, so "streamed == held" cannot be observed by running both, and no check in the 46 drives
the streaming loop at all. My three mutants — **BE34-R1**:

| mutant | suite |
|---|---|
| a streamed score perturbed by `+1e-9` | **green, rc 0** |
| every second streamed row dropped | **green, rc 0** |
| **the day-level reconciliation refusal disabled** (`if False and built[…]`) | **green, rc 0** |

## 4. Receipts flushed after every gate — verified by killing the driver

I ran 09-01 into my own outdir under a 12 G scope, let it reach the early gates, and sent
**SIGKILL** at ~75 s. The partial receipt is on disk and carries exactly what was established:

```
gates: day_closed_and_attributed, population_supply_and_bridge, materialise_frozen_bytes,
       import_closure_disclosure, import_anchors_from_run_dir, selection_from_specs
outcome: None          (not SCORED)      as_of 2026-09-02T14:21:47Z
```

`derived/` identical afterwards. The guarantee itself is stated in the code (`:805-807`: *"an OOM
kill bypasses every `finally` … the receipt is written after each gate so a killed run still says
how far it got"*) — **but not in the receipt's own words**. A receipt-only reader meeting a
partial file gets `gates` and a missing `outcome` and must infer the rest. The module already has
the idiom one field over (`sealing_note`); a one-line `durability` note would put the guarantee
where the reader is. Recommendation, not a finding.

## 5. Outdir-per-run — it does not refuse, and it does not supersede

Two runs into the same outdir (09-02, which refuses at gate 1, so this is cheap):

| | receipt sha | `as_of_utc` | files in the outdir |
|---|---|---|---|
| run 1 | `b6c47b1f5f6f7f67` | 14:21:24Z | one |
| run 2 | `1f40ca270749fc85` | 14:21:27Z | **one** |

No refusal, no numbered successor, no preserved copy: `outdir.mkdir(parents=True, exist_ok=True)`
(`:783`) and fixed-name `write_text` for both the receipt (`:772`) and the 54 MB sealed scores
(`:758`). **BE34-R2.**

## 6. The two disclosures a receipt-only reader needs — both present and computed

- `masking`: `n_masked_at_scoring {btc: 0, eth: 0}` **beside** `n_masked_at_supply` per coin
  (22/23/22/22/9/23/20) and `n_masked_total_at_supply: 141`, with `applied_at: "supply
  (de_admissible_windows), before rows"` and a `why`. The zero is explained where it is read.
- `coin_coverage`: `coins_with_a_frozen_fit ['btc','eth']`, `coins_supplied_without_a_fit` (five),
  `n_windows_supplied_without_a_fit: 1344`, with a `why` naming the cause. `n_actions_scored` is
  **per coin** (`btc 610064`, `eth 441409`), never presented as a day total.

Both are computed from the run (counts carried out of the loop, not restated), and the labelling
is honest: `n_windows`, `n_windows_with_rows`, `n_windows_supplied_without_a_fit` each say what
they count. The R-433-era mislabel does not recur.

## 7. The three fixes, and my two additions

- **The §10(1) control that accepted either answer** now asserts against an **independent
  re-reading**: *"the gate REFUSES and an INDEPENDENT re-reading agrees that at least one bound
  input has moved"*, with a positive control that a matching contract HOLDS — it discriminates
  rather than refusing universally.
- **The import control catching bare `Exception`** now catches `ForwardDayRefused` specifically
  and asserts on the wording (`"does not hold" in str(e)`), so any other exception fails the check
  instead of satisfying it.
- **The forgeable `pair_asserted: True`** is now the checker's own emission: three CO-5 known-bads
  (verified-with-unbindable, not-verified, neither) plus *"the assertion returns WHAT IT SAW"* and
  *"the block in the receipt is the CHECKER's own emission"*. The receipt carries
  `{asserted, ref_seen, unverifiable_seen, verified_seen}` — values from `de_ratification_check`,
  not a boolean BE computed.

**My additions, as asked:** the streaming mutants (§3, BE34-R1) and the flush (§4 — the SIGKILL
run, which the driver survives correctly).

## 8. Rule 10 / rule 14, and the gate count

`sealed: true`; the receipt carries **no metric** — the only occurrences of "rho"/"metric" are
inside the `not_in_receipt` sentence, and "median" appears in the liveness rule's *definition*.
`sealed_file` carries path, sha256, bytes and a description only. The 09-02 receipt is `REFUSED`
at `day_closed_and_attributed`. Nothing admits a day or a candidate; no new constant; the frozen
artefacts are unedited (the anchors' shas equal the blobs at `1b53929`, and the manifest sha is
the one the candidate binds).

**Gate count reconciled:** `v5_deploy_gates.py` carries **21** gates at `248e99f`; DA's held
`e292439` adds `DA hf/pm window alignment` for **22**. BE's 21/21 and DA's 22-gate roster are the
same file at two different commits — no disagreement. (My own run of the 22-gate roster at
`e292439` had all 22 pass.)

**Register note, recorded not acted on:** the Q-BE-229 disposition column's "USER-pending
unchanged (R-408(2), R-408(3), R-411(i), R-411(ii))" is stale — all four are RULED at R-424 §7
and only the freeze disposition is open. The coordinator records it in R-436; BE supersedes in
band (rule 13).

---

## Findings

### BE34-R1 — MEDIUM — the streaming scoring path has no falsifier, and neither does the day-level refusal

The round's substance is the shape change forced by the OOM: `build_and_score()` (`:622`) replays,
scores and drops each window. Nothing in the 46-check suite drives it. Measured, each mutant run
against the full selftest under the path launcher:

- a streamed score perturbed by `+1e-9` → **rc 0, green**
- every second streamed row dropped (`if len(actions) % 2 == 0: continue`) → **rc 0, green**
- `if built["reconciliation_failures"]:` → `if False and …` — the strict day-level refusal
  disabled → **rc 0, green**

The comparison the request asks for cannot be made by running both paths either: `score_rows()`
(`:707`), the held-in-memory implementation, is **defined and called from nowhere** — dead code
at this tip, so the two implementations can drift with nothing to notice.

The receipts are evidence that a run happened and that its counts reproduce across two passes;
they are not evidence that the loop is right. The coordinator's 68-field cross-check between the
12:49 and 13:50 receipts is real and valuable, but both passes ran the same streaming code, so it
demonstrates determinism, not streamed-versus-held equality.

**Closure:** a small synthetic fixture (a handful of windows with known features and fits) driven
through `build_and_score()` and asserted equal to `score_rows()` on the same input — which also
gives the dead function a reason to exist — plus one fixture window carrying a reconciliation
failure, asserting the caller refuses the day by name. That is the "one fixture, two consumers"
idiom `v5_chain_equivalence_test.py` already uses in this repo.

### BE34-R2 — MEDIUM — the driver still silently overwrites its own receipt and its sealed scores

`outdir.mkdir(parents=True, exist_ok=True)` (`:783`), then fixed-name writes: the receipt at
`:772` and the 54 MB sealed scores at `:758`. Driven above: a second run into the same outdir
replaced the receipt in place (sha `b6c47b1f…` → `1f40ca27…`, `as_of` 14:21:24Z → 14:21:27Z), one
file, no refusal, no successor, no preserved bytes.

This is the defect that destroyed the 12:49 receipt in `fwd4/` — the artefact that today's
cross-pass evidence rests on, and which survived only because the coordinator copied it by hand
into another scratchpad. The driver would do it again on the next re-run, to an artifact of
record.

The repo already carries the idiom: `da_forward_day_verify.preserve_prior_bytes()` copies the
bytes a canonical write is about to replace, beside it, under a name carrying their `as_of`
(R-412 / RR9-3(b)).

**Closure:** refuse an outdir that already carries a receipt for that day, or write an
`as_of`-stamped successor and keep the prior bytes. Either satisfies "never a silent overwrite";
the second matches what DA does.

### BE34-R3 — LOW-MEDIUM — the launch-invariance check runs another tree's file

`_selftest_launch()` spawns `[sys.executable, "-m", "live.pm_research.be_forward_day",
"--selftest"]` with **`cwd=str(REPO)`**, and `REPO = Path("/home/yuqing/ctaNew")` is hardcoded
(`:35`). From the canonical tree — where BE runs it — parent and child are the same file and the
check means what it says. **From any worktree the child is the shared tree's copy, not the file
under test.**

Measured at the pinned tip, in my worktree: the parent reports **46** while the quoted child tail
read **45** in one run and **77** minutes later; the shared tree's `be_forward_day.py` is dirty
(sha `4c0425c5…` vs the pinned `4a2c5e66…`) with BE's in-flight round 5, and running that child
directly gives 77 while the worktree's own gives 45. So the check's result depends on another
seat's uncommitted edits and says nothing about `248e99f`.

Two consequences worth naming: it cannot detect the CO-1-class break it cites when run from a
worktree, and a reviewer's "both launchers green" must come from running the parent twice — which
is what I did — rather than from this check.

**Closure:** spawn the child against the tree of `__file__` (`Path(__file__).resolve().parents[2]`)
rather than the hardcoded `REPO`, and assert the child's own check count equals the parent's minus
one — the parity the check is named for, currently unasserted.

### BE34-R4 — LOW — a usage error exits 0

`main()` prints the usage line and `return 0` when `--forward-day` is absent (`:1437`), while every
real refusal returns 2. I hit it by mistyping the flag as `--day`: rc **0**, nothing written, no
receipt. "A run that writes nothing must not exit 0" is the programme's own rule 11, and an
automated caller cannot tell this from a completed run. **Closure:** `return 2`, the code the
driver already uses for its other refusals.

### BE34-R5 — LOW — the closure disclosure says "execute" and computes "static closure"

`not_frozen_in_closure`'s `why` reads *"every module in their import closure that is NOT an anchor
executes at HEAD"* and names two. Reproducing the driver's own import environment,
`tier1_pipeline` is **never imported** — its only import is deferred inside
`layer2_v1.load_winners()`, which nothing on this path calls — while `warning_window` is imported
and executes its module level. The disclosure over-states HEAD exposure by one module, in the
direction that is against BE's own interest, and the USER's freeze decision turns on exactly this
distinction.

**Closure:** compute the executed set from `sys.modules` after the anchor imports and report
`in_static_closure` and `executed_at_head` as two fields. It is a few lines, and it turns the
sentence into a measurement.

---

## Executed evidence

At `248e99f`, 2026-09-02T14:15–14:27Z:

| check | result |
|---|---|
| driver identity | worktree file sha `4a2c5e66…` = `git show 248e99f:…` = the receipt's `producing_code` |
| selftest | **46 checks, rc 0** under `-m` and by path |
| item 1 import trace | `warning_window` imported (HEAD, module level only); **`tier1_pipeline` not imported**; `select_by_day` byte-identical at both commits; `warning_window` diff confined to `select_holdout()` |
| item 2 | three refusals present by name in the suite; `data_root_probe` records 28,031 slugs and `resolves_to_repo_data: true` |
| item 3 | `:869` refuses the day on any reconciliation failure; **three mutants survive the suite**; `score_rows()` never called |
| item 4 | SIGKILL at ~75 s → partial receipt with **6 gates** and `outcome: None`; `derived/` identical |
| item 5 | two runs, same outdir → receipt **replaced in place**, one file, no refusal |
| item 6 | `n_masked_at_scoring` 0 beside `n_masked_at_supply` 141 per coin; `n_windows_supplied_without_a_fit` 1,344; `n_actions_scored` per coin |
| item 7 | the three fixes assert against an independent re-reading / the module's own refusal type / the checker's own emission |
| item 8 | no metric in either receipt; 09-02 `REFUSED` at gate 1; roster **21** at this tip vs **22** at DA's `e292439` |
| BE's artifacts | the five entries in `fwd5/` unchanged; nothing re-run into `fwd4/`/`fwd5/` |
| my runs | three, all under `systemd-run --scope -p MemoryMax=12G`, into new outdirs of my own |
| worktree | clean at `248e99f` after every mutant |

---

## Disposition

- **RELEASE** for `248e99f` as the confirming driver of record for the 09-01 score. The frozen
  contract, the population, the ratification pair and the sealing are checked at the artifact; the
  09-01 counts reproduce across two passes; nothing is admitted by this receipt.
- **Qualified, precisely:** I would close **BE34-R1** and **BE34-R2** before this driver scores a
  **second** day (BE round 5). The evidence that backs today's numbers is the cross-pass agreement
  — and it exists only because the earlier receipt was preserved by hand, which is exactly what
  BE34-R2 says the driver will not do; and BE34-R1 means the scoring loop and its day-level
  strictness would not tell anyone if they broke.
- **FILED:** BE34-R1 (MEDIUM), BE34-R2 (MEDIUM), BE34-R3 (LOW-MEDIUM), BE34-R4 (LOW),
  BE34-R5 (LOW).
- **Item 1 is stated as a fact in its own section above and is not ruled here.** The one thing a
  reader should carry away: the candidate's own code and its data anchor are the freeze's bytes;
  everything else in the closure is HEAD; and of the two HEAD modules the receipt names, one runs
  only its module level with the used symbol unchanged and the other does not run at all.
