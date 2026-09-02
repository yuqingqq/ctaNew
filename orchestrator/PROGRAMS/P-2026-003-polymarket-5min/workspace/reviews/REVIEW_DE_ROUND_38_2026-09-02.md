# Review — DE round 38 at `dfd4c00` ((γ) built and the seal literalised; on the fixture that proves (γ), the accepted null is the treated arm)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `dfd4c00`** (row Q-DE-56, same commit). Verified at the blob: runner
**`a49458a04253175d`** (2,811 lines, `EXPECTED_CHECKS = 101`), `de_score_stream`
**`4ccdadeafe982b87`** (378, 25), v2 DRAFT **`a45b87624f72b567`** (245); head-scoring
`60ef48fea69e83f1`, protocol check `f20323d02303baf1`, rho `048b8e077c3d37e8`, matched-random
`f77aaf2bd2f21988` — untouched, as stated.
**Request of record:** `REQUEST_DE_ROUND_38_2026-09-02.md`. **Composed 2026-09-02T22:05:52Z.** One filing, per R-377.

**Constraints observed.** Executed in `~/ctaNew-wt-rev` at `--detach dfd4c00` (`data/pm_5min`
mirrored); `~/ctaNew-wt-de` / `~/ctaNew-wt-be` never read. Two mutants (items 2 and 7) applied to
my worktree copy and **restored byte-identical**; `git status --short` **0** after each. The
declared OUTDIR never passed to `--run`; `derived/` **173 before and after**; nothing written under
`data/`; no plan file edited — the v2 DRAFT included; no unit, timer, scope or anchor;
`DA_MIDNIGHT_MODE` never set; `git worktree list` **34 at quiescence**, unchanged.

## 1. Counts, pin statuses and the six literals — CONFIRMED (item 1)

Eight modules, both launchers, PASS = summary = rc 0, zero stderr:
**31 / 101 / 26 / 21 / 25 / 21 / 184 / 92** — R-480 §1 reproduces, and the row's prior counts
(85 / 24) are the `218509e` blob's, which I measured myself last round: **DE37-R4 closed**.

`pin_statuses()`: 12 rows, **11 IDENTICAL + 1 ADDITIVE_DECLARED**, and **every row carries
`comparison` and `reached`** — `harmful_exposure_rows.py` reached **18**, `phase2_arms.py`
reached **1**, with `comparison: "per-function over the reached set"`. **DE37's reached-set
recommendation is closed.** `--run --outdir <scratch>` → **rc 2**, no traceback, nothing created;
`preflight()` refuses in **1.29 s** naming the missing step.

**The six literals, verified by me against both artifacts** (`_fn_asts` of the fit blob at
`fit_code_ref e12e2c70c133` and of the tip):

| function | declared fit / tip | mine | match |
|---|---|---|---|
| `select_v2_era` | `e97a6662273d8abc` / `3b34bdc86b1056ca` | same | **yes** |
| `_era_or_refuse` | `None` / `830c4fa88ba44280` | same | **yes** |
| `_refuse_empty_selection` | `None` / `a6cfb900e1ced0b8` | same | **yes** |

The two `None`s are correct: both functions are **absent** from the fit blob (new at `851edaf`).

## 2. DE37-C2 / R1 — **CONFIRMED CLOSED**, driven by me (item 2)

The same edit that PROCEEDED at `218509e` — an `Assign` inserted after `select_v2_era`'s
signature — now yields **BLOCKING**, with
`undeclared: ["select_v2_era (declaration stale: declared {…'sha_at_declaring_tip': '3b34bdc86b1056ca'}, now {…'9…'})"]`,
and `verify_called_code()` **refuses by name**. `_seal_declarations` and
`DECLARED_ADDITIVE_SHAS` are gone from the source. The suite's falsifier (`:2182-2212`) is now the
edit in a `TemporaryDirectory` copy through `pin_statuses(here=…)`, with two positive controls, and
the `join_fills` pairing (`:2370-2379`) is a source edit too — so both halves of the pairing I
drove in round 37 are shipped.

**What the literal form still leaves open** — three things, none of them a defect at this tip:
(i) the literals live in the file that reads them, so a seat editing code and declaration in one
commit re-seals silently; that is inherent to any in-source pin (a sidecar has the same property)
and is closed by review, not by code — worth one sentence in §6 saying so; (ii) the `sha_at_fit`
half is pinned against an immutable blob and is therefore strong, while the `sha_at_declaring_tip`
half pins the **declaration**, not the behaviour: every future edit to a declared function BLOCKS
until re-declared, which is the intended cost and should be stated as such; (iii) nothing pins the
**reason** — a re-declaration can keep a stale justification while the shas move. (iii) is the only
one I would ask for: carry the reason's own hash, or require the reason to name the commit that
changed the function.

## 3. DE37-C1(a)(b)(c) and R2 — **CONFIRMED closed on the run path** (item 3)

At the blob: `null#3` (`:1146-1156`) refuses a stream whose events do not name `gen`; the demand
is the **above-threshold event count** (`:1179-1182`); `stream_predicates` runs **per draw** at
`:1264` with rejection **under its own reason before the replay** (`:1266-1280`) and P4 after it
(`:1286-1298`); the receipt carries `n_rejected_by_reason`, `first_rejection` and
`predicates_per_draw` (`:1317-1339`) — I read them back from a live `run_cell`:
`{'P1': 0, 'P2': 0, 'P3': 0, 'P4': 19, 'PERM_NOT_OK': 0}` with `first_rejection {'seed': 0,
'reasons': ['P4']}`. `permuted_stream` (`:962-1030`) leaves the below values in place
(`_stay`/`_needs`/`_spare`, `:1012-1016`) — **my item-4 ruling of round 37 adopted**. The
`gen` defect DE reports is real at the `218509e` blob and is what made my round-37 measurement
show P2/P3 failures rather than silent identities. **DE37-R2 is closed by the same change**: the
fixture no longer certifies a state the run path cannot produce, because the demand now equals
`|above|` by construction.

**Is a falsifier input on the run path's signature acceptable in the form it has?** `_known_bad_demand`
and `_draw_log` are keyword-only parameters of `run_cell` with a single call site each
(`:2361-2369`). **Acceptable, with one condition, and it is met:** the parameter must be
*inert by default* and *asserted inert*, so that a production call cannot silently take the bad
path. `_known_bad_demand` defaults to False and the parse check names its one call site. I would
add the smaller guard that costs nothing: assert in the receipt-building path that both are falsy
(a receipt produced under a falsifier flag must say so, or must not exist). That is **DE38-R4**,
LOW.

## 4. DE38-C1 — **CONFIRMED, reproduced independently, and it is the round's finding** (item 4)

Measured by me on DE's own C1 fixture (slug A: gen 1 above 0.9 acting, gen 2 above 0.8 HELD; B and
C below; one stratum), at the tip:

- **(a)** `run_cell(draws=5)` → **24 attempts, 5 accepted**, `null_quantiles {n 5, q50 40.0,
  q95 40.0, max 40.0}`, `net_diff_vs_null_median_cents` **0.0**, treated value **40.0**. Replaying
  the loop draw by draw: **all five accepted draws are `{A-gen1, A-gen2}`** — the above set, the
  **identity** permutation — each valued 40.0. On this fixture the identity is the only draw that
  satisfies P4 (any above value on B-gen1 or C-gen1 realises two cancels against one).
- **(b)** The identity guard is handed the **actions** (`:1184-1186`, `:1233`). Over 200 seeds:
  handed the actions it fired **0 ×**; handed the demand it fired **65 ×** — exactly the 65 identity
  draws. `refuse_if_not_random` opens `if sorted(drawn) != t: return`, and under (γ)
  `|drawn| = |above| ≠ |actions|` whenever a stratum holds a non-acting above event, so the guard
  returns before it can refuse. The parse assertion at `:1713-1719` certifies that call.
- **(c)** `n_distinct_draws` (`:1334`) and `point_mass` (`:1335`) are computed over
  `_seen_draws`, which accumulates **every attempted** draw (`:1225`): **3 / False** on the
  fixture, while the **accepted** set has **one** distinct draw and is a point mass at the treated
  value.

### THE FOUR RULINGS

**(1) The identity draw is ADMITTED and COUNTED; the guard is RETIRED for (γ).** Under (γ) a draw
is a uniform choice of which `|above|` generations carry the above values; the identity is one of
the `C(N, k)` permutations and a permutation null that excludes it is not the permutation
distribution. So it must not be rejected per draw, and the cell-level refusal must not fire on it.
The receipt carries `n_accepted_identity` **per stratum**, so a reader sees how much of the
accepted set is the treated arm. **The guard itself should be retired for (γ), not re-pointed:** it
was written for a draw matched on actions, where equality with the actions meant "the control did
nothing"; handing it the demand would make it refuse the very sample points the null must contain
(the measured 65/200). Retire it with the reason written into the source, and delete the parse
assertion that certifies it — a check that certifies a call that cannot go red is worse than no
check, because it reads as coverage.

**(2) `n_distinct_draws` and `point_mass` move to the ACCEPTED set — and an accepted set of one
distinct draw is a REFUSAL of the cell's interval, not a declared point mass.** The accepted set
*is* the null; a statistic over the attempted set describes the sampler, not the null, and rule 10
says the number must be the one the claim rests on. Report both, labelled
(`n_distinct_accepted` / `n_distinct_attempted`). On degeneracy: rule 6 declares ≥ 200 **draws**,
and 200 copies of one draw is one draw — so the cell must **not** publish `null_quantiles` or
`net_diff_vs_null_median_cents`; it publishes `null: DEGENERATE (n_distinct_accepted = 1)` with
the reason and falls back to the labelled point estimate the addendum already declares for
non-null cells. That is stronger than a "declared point mass" (which invites reading 0.0 as a
result) and weaker than failing the run (rho and retention for that cell remain valid).

**(3) The DRIVEN check must assert what was ACCEPTED.** `:2283-2294` asserts `accepted == 2`,
`P4 > 0` and `first_rejection is not None` — all true of the degenerate case. It must additionally
assert, on a fixture built to allow it, that **at least one accepted draw's control stream differs
from the treated stream** and that `n_distinct_accepted ≥ 2`. Without that, "the run redraws" is
demonstrated while "the null is a null" is not.

**(4) The collapse re-opens the REPORTING, not the frozen matching rule.** `DRAFT:147-156` asks
for matching on action count, side and hour, and the runner does exactly that; the frozen text is
silent on what to do when the matched set degenerates, and silence there is the addendum's to fill.
So: **per stratum, before any §3 number is read**, the receipt must carry the accepted-set size,
its distinct count, `n_accepted_identity`, and whether it collapsed. If the USER later wants a
tolerance around the action count, **that** would change the frozen rule and needs its own ruling —
nothing measured here forces it.

## 5. DE38-C2 and C3 (item 5)

**C2 — CONFIRMED (LOW).** `null#2` (`:1305-1314`) names only P4's reason while `rejected` totals
every reason; on my fixture the totals happened to be all-P4, which is exactly when the wording
looks right. **It should carry `n_rejected_by_reason`** — the runner already computes it, so the
refusal is one interpolation from being accurate.

**C3 — the pool should be the STREAM's support, and there is a second consequence.** The pool is
built from the reference's generations (`:1139-1142`) while the demand is the stream's above
events; a generation with no event can be drawn and is then rejected PERM_NOT_OK/P3, spending
budget. Worse, and not in C3: `_room = pool(stratum) − demand(stratum)` (`:1200`) feeds
`strata_with_room` and `strata_forced` (`:1331-1332`), so **the freedom statistic is computed on
a support the draw cannot legally use** — a stratum whose only spare keys carry no event reads as
having room while every draw touching them is rejected. Filed **DE38-R1**. One change fixes both:
build the pool from `treated_scores`' keys.

## 6. My conditions (i)–(iv) against the DRAFT as landed (item 6)

| condition | verdict at `dfd4c00` |
|---|---|
| (i) §5 says BUILT and names the receipt fields | **MET as to the STREAM and the rejection accounting** — measured: demand over above events, P1–P3 per draw, reasons and `first_rejection` in the receipt. **NOT MET as to the null §5 promises to deliver**: on the seat's own proving fixture the accepted set is the treated arm (item 4) |
| (ii) the below values stated, §2 re-read | **MET** — §5 states they stay; §2's paragraph reads the pair against that stream |
| (iii) §6 states the seal's form as a pin claim | **MET** — with my three notes in §2 above, of which only the unpinned *reason* is worth asking for |
| (iv) the split question is ruling 5, asked WITH 2 and 4 | **MET** — it travels with the two numbers, which is what makes it answerable |

**May the package travel whole with DE38-C1 open? No** — and for a sharper reason than "a finding
is open". §5 asks the USER to adopt a control property whose only demonstrated realisation, on the
seat's own proving fixture, reproduces the treated arm and yields `net_diff = 0.0` **by
construction**. A USER reading §5 today would be adopting the words while the artifact behind them
produces a null that cannot differ. The remedy is small and touches nothing the USER must re-read
except §5's reporting sentence: rulings (1)–(3) of item 4.

## 7. What the coordinator missed — the class (item 7)

- **A check that certifies a call that cannot go red:** item 4(b) is the instance, and the parse
  scan finds **no others** in the two changed modules — the runner is now clean of unfailable
  predicates, docstring predicates, self-greps **and** the last substring check (`:1566` at
  `218509e`) is gone. One prose predicate remains next door, `de_score_stream:342`
  (`"IR-R4" in (__doc__ or "")`): a *declared limit* tested at the document, which I accepted as a
  form in DE round 13 — acceptable, and the stronger version is an AST assertion that the module's
  non-suite functions open no file. **DE38-R2**, LOW, offered not demanded.
- **A receipt statistic computed on the attempted population:** item 4(c), ruled above.
- **A falsifier input on a run-path signature:** ruled in §3 — acceptable, with the receipt
  assertion I ask for as **DE38-R4**.
- **A per-draw refusal that ends the CELL:** `ControlRefused` from the identity guard is the one,
  and it is why the treatment of the identity is an accident of the stratum — on a one-above
  fixture the identity draw refuses the whole cell, and with a held event it is accepted five times
  over. Ruling (1) removes the asymmetry.
- **The ninth mutant.** Driven by me: dropping `gen` from the adapter's required-key tuple
  (`de_score_stream:155`) gives **`KeyError: 'gen'`**, unnamed. **DE declines a second guard and
  is right** — the shape contract belongs to the adapter, and the runner's `null#3` is already the
  defence in depth. The honest closure is not a third guard but **one source**: build the event
  dict from the same tuple the check iterates (`{k: r[k] for k in REQUIRED}`), so removing a key
  from the list removes it from the output and `null#3` refuses **by name**. Filed **DE38-R3**, LOW.

## Findings

| id | severity | where | one line |
|---|---|---|---|
| DE38-R1 | **MEDIUM** | `:1139-1142`, `:1200`, `:1331-1332` | the pool is the reference's generations while the draw is over the stream's above events, so `strata_with_room` counts freedom the draw cannot legally use |
| DE38-R2 | LOW | `de_score_stream:342` | the declared limit is asserted against the module's own docstring |
| DE38-R3 | LOW | `de_score_stream:155`, `:172` | the required-key tuple and the event construction are two sources, so dropping a key dies as `KeyError` |
| DE38-R4 | LOW | `:2361-2369`, receipt | a receipt produced under `_known_bad_demand` / `_draw_log` should say so, or not exist |

**DE38-C1 CONFIRMED** (all three parts, reproduced independently). **DE38-C2 CONFIRMED.**
**DE38-C3 CONFIRMED**, with the second consequence above. **DE37-C1(a)(b)(c), R2 and C2/R1:
CLOSED.**

## Disposition and round 39's order

**RELEASE `dfd4c00` as round 39's base.** The round closed everything I ruled last round and drove
each closure: (γ) is built on the run path with per-draw predicates and reasons, the seal is
literal and the stale-declaration case now BLOCKS (I drove the same edit that used to proceed), the
reached set is in the receipt, the below values stay put, and the last substring check is gone.
Nothing can run — `preflight()` refuses in 1.29 s — so no finding here can reach an artifact.

**Round 39, in this order:**
1. **Item 4's rulings (1)–(3)** — admit and count the identity, retire the guard and its parse
   certificate, move `n_distinct`/`point_mass` to the accepted set with the DEGENERATE refusal, and
   make the DRIVEN check assert what was accepted. **DE38-C1 is the only thing standing between
   this package and the USER.**
2. **DE38-R1 + C3** — the pool becomes the stream's support; `strata_with_room` recomputed on it.
3. **C2** — `null#2` carries `n_rejected_by_reason`.
4. **Item 4's ruling (4)** — the per-stratum accepted-set block in the receipt, before any §3 number.
5. **DE38-R3, R4, R2** — one source for the event contract, the falsifier-flag assertion, the
   docstring limit.
6. The DRAFT's §5 reporting sentence last, so the text the USER reads matches the code.
