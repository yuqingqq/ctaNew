# REVIEW 104, second half — DE 136 (`1f171e8`) + DE 137 (`5ad16b5`)

**REV, 2026-09-07T15:57Z.** Read-only: no heavy unit, no lock, nothing written under
`data/` (verified after the fact: 0 files touched under `data/pm_5min/derived` since
15:40Z; the 137 files that moved are the live collectors' own). Every battery and every
drive below ran in scratch clones at the two commits, never in the shared tree.

**VERDICT. Both rounds land what they were dispatched to land, and the central claim is
independently verified: the estimator reproduces BE 99's ruled P&L to the cent on all
three paths of the real 09-05 ledger. FOUR things must not go into a receipt as they
stand — one of them is the shape this programme keeps paying for (a printed reason that
contradicts the computed verdict beside it), and one is the door R-803 asked to close,
which is narrower than it was and still opens on an unchecked dict. A fifth item is a
consequence for the PLAN, not a defect: the placement latency BE 100 found cannot reach
a day's P&L through the runner at all.**

---

## §1 THE ESTIMATOR IS RIGHT — reproduced at the real ledger, not read

I ran DE 136's own functions over the real 09-05 decision ledger
(`p003_de_decision_ledger_20260905__20260907T124104Z.jsonl.gz`, 177,467 FILL rows, 288
slugs) and grouped by `(arm, book)`:

| path | trades leg | residual leg | total | per-fill formula |
|---|---|---|---|---|
| CONDVALUE ARM | −9,195.85 | +39,240.89 | **30,045.04** | 30,045.04 |
| BASELINE | −225,644.56 | +306,882.86 | **81,238.30** | 81,238.30 |
| HAZARD ARM | −197,242.71 | +286,126.80 | **88,884.09** | 88,884.09 |

Every figure equals BE 99's (Q-BE-342, quoted in R-803) **to the cent**, and the two
BASELINE books value identically, as one no-cancel reference must. That is three
independent computations agreeing — BE's own code, DE's estimator, my drive — and it is
the strongest thing in this filing. The `trades + residual == Σ sgn(settle − px)·size`
assertion holds on all four path-days.

Note what this does NOT establish, said plainly: agreement of three implementations of
ONE definition is not evidence the definition is what the user meant. R-801's words are
"trades and remaining position's settlement p&l"; the code's reading is disclosed in
R-801 and is what all three computed.

## §2 THE WINNER IS NOW VERIFIED, AND I REPRODUCED THAT TOO (DE 137)

`SETTLEMENT_CONVENTION` pins the rule as a BLOCK, not a sentence: name
`S60(T) >= S60(t0)`, X_T and X_0 named, boundary reader "last sample at or before the
boundary", tie `X_T >= X_0 -> Up`, readers `exp_m6_settlement.load_streams / read_at`
(imported — I confirmed the module's own `PM` is repointed at a symlink tree and
restored in a `finally`, so the reader is unmodified and only the file set is smaller),
provenance Q-BE-342. **In my own run of the battery at `5ad16b5` the real-file cell
reports 576 of 576 slugs VERIFIED_AGREE across 09-05 and 09-06, DISAGREE 0, unavailable
0, unresolved 0.** That is R-803's requirement met: the venue record is the join, the
stream is the check, and the check runs per day rather than being asserted once.

The failure paths are driven with a green control beside them: a falling S60 against a
venue "Up" refuses `SETTLEMENT_WINNER_DISAGREES_WITH_CHAINLINK`, an absent stream
refuses `SETTLEMENT_CHAINLINK_UNAVAILABLE`, and the same slug with a rising S60 admits
with `is_final_for_quotation` True. DE also records that it had to build this block its
own fixture because the cells above leave the file in the DISAGREE state and an EARLIER
guard's refusal stood in for this one — the second time that shape has cost a cell in
this battery, and it is caught both times because the refusal NAME is asserted exactly.

## §3 THE MISNAMED FIELD IS GONE — verified at a receipt I produced

DE 136 left it in: a receipt from those bytes carries four `inventory_leg` keys
(`per_day_sealed_artifacts[i].absolute.{arm,zero_cancel_baseline}`), and per R-803 that
quantity is Σ(after−before)×mark = the exact NEGATIVE of the ruled trades leg — so a
real receipt would have carried a field named "inventory" that is the sign-flipped cash
flow, one key away from a correctly-named `residual_leg_cents`. **DE 137 fixes it and I
verified the fix at the artifact**: `--synthetic-day` at `5ad16b5`, rc 0, and the emitted
receipt carries **0 keys named `inventory_leg` and 4 named `trades_cash_flow_cents`**,
re-signed to SELLS − BUYS with the convention stated beside it. The falsifier walks KEYS
(DE's first draft tested the substring and failed on its own prose — disclosed, and the
right correction).

## §4 RULE 11 HOLDS — no 09-07 value can be computed before the declaration

Driven against the REAL params head (v19, which carries no `settlement_endpoint`):
09-07 → `REFUSED SETTLEMENT_DAY_NOT_ADMISSIBLE`, 09-08 the same, 09-05 admits as
`DESIGN_DATA`. GO #8 runs `5020f96`, which carries none of this code, so tonight cannot
produce a settlement value by any path. The admissible set is READ from the declaration
and typed nowhere.

## §5 THE BATTERIES — reproduced, with one caveat the coordinator should hold

Re-run by me, not quoted: runner **PASS 388 / 0 disarmed / 0 skipped** at `5ad16b5`
(**384** at `1f171e8`, with all eight R-801 cells green), ledger **9**, early read
**30**, phase4 **214 run + 4 conditional = 218**, `--synthetic-day` rc 0.

**THE CAVEAT: on the mainline tip the runner's battery cannot run at all.**
`verify_be_module` refuses `BE_CASCADE_DIFFERS` because `de_phase4_diag_runner.py` moved
at DE 127–130 (and again at DE 137) while v19's cascade pin still names `5020f96`'s
bytes (`ee4034c1…`). I reproduced both counts in scratch clones with that one file
restored to the pinned bytes — which is DE 131's method, and DE 137 states the same
consequence in its own commit message. Nobody should read "388 PASS" as "388 PASS on the
tip"; it is the freeze's cost, and it clears when v20 re-measures the pins.

---

## §6 FOUR THINGS THAT MUST NOT GO INTO A RECEIPT AS THEY STAND

**(1) A printed reason that contradicts the computed verdict beside it (rule 10).**
Every fixture receipt these bytes write carries, in one block:

```
"admissibility": {"admissible": true, "class": "FIXTURE", ...},
"status": "NO_WINNER_SOURCE_ON_A_FIXTURE",
"why": "rule 11: this day is not in the admissible set, so the ruled endpoint is not
        computed for it. ..."
```

The day IS in the admissible set — the key above says so. One literal serves two
statuses (`NO_WINNER_SOURCE_ON_A_FIXTURE` and `NOT_VALUED_DAY_NOT_ADMISSIBLE`) and is
true of only the second. Reproduced at `1f171e8` and again at `5ad16b5` by running
`--synthetic-day`. It is fixture-only in effect — and fixture receipts are exactly what
REV and DA read to clear a composition. A per-status reason, computed.

**(2) The verification door is narrower and still opens on an unchecked dict.** DE 137
derives `counts` and `all_agree` inside `verify_winners_against_chainlink` — but
`winner_source` still READS `all_agree` and `counts` from whatever dict it is handed.
Driven at `5ad16b5`:

```
verification = {"per_slug": {"S1": {"status": "DISAGREE", ...}},
                "convention": {"name": "a rule I made up"},
                "counts": {}, "all_agree": True}
-> status VERIFIED_AGAINST_CHAINLINK, is_final_for_quotation True,
   require_verified=True PASSES, and the receipt carries a per-slug map
   that says DISAGREE beside a status that says VERIFIED.
```

No production caller can do this today (run_day always passes the producer's output), so
no receipt is affected. But the fix is two lines and free: recompute `counts` from
`per_slug` and `all_agree` from `counts` INSIDE `winner_source`, and refuse a convention
whose name is not `SETTLEMENT_CONVENTION["name"]`. As it stands, the quotable door is
guarded by a boolean its caller supplies — the same shape as (1), one layer in.

**(3) A degenerate settlement null aborts the whole day, with an uncaught
`DesignRefused`.** Driven: `run_day` on the fixture with an all-Up (or all-Down) winner
map dies with `REFUSED: the null has zero dispersion, so a standardised excess is
undefined. A degenerate null is a STATUS, never a large Z.` `DesignRefused` appears
NOWHERE in the runner, so nothing catches it: the day emits no receipt at all — losing
the D_E0 result that was already computed, after every stage's work (on a real day, 70–80
minutes). The design's own text says a degenerate null is a STATUS; the new call site
turns it into a day-level abort. Catch it at `settlement_arm_day` and record
`economic_settlement.status = SETTLEMENT_NULL_DEGENERATE` with the null's n and sd.
I did not measure how likely a degenerate settlement null is on a real day; it is
reachable in code and that is enough to make it a status.

**(4) Rule 11's precedence lets a declaration re-open a consumed day.**
`settlement_admissibility` tests `declared` BEFORE `design_days`, and nothing refuses the
intersection. Driven: with `settlement_endpoint.admissible_days = ["2026-09-05"]`, a day
the USER's early read CONSUMED comes back as `DECLARED_VALIDATION_DAY`. v20 will not
name a design day — but the guard should not depend on that. Refuse the overlap by name;
a consumed day cannot be a validation day whatever a declaration says.

## §7 FOR THE PLAN, NOT A DEFECT: the placement latency cannot reach a day's P&L through the runner

DE 137 adds `placement_latency_ms` to **`de_phase4_diag_runner.build_reference`**, with
the default 0.0 and 250 ms only PROPOSED — the right call under the freeze, and the drop
is COUNTED (`TRANCHE_BEFORE_PLACEMENT_LATENCY`, rule 4), with the parameter and its
source recorded in every reference's output. Two facts the coordinator should hold
together:

- **`de_multiday_gate1_runner.py` never calls `build_reference`** (measured: zero call
  sites). The day's fills come from BE's `replay`, and `harmful_stateful_policy` has no
  placement latency at all — the three occurrences of the word are prose.
- The path by which the parameter reaches a day's P&L is **BE's own book builder**:
  `be_daybook_build.py:690` calls `R.build_reference(coin, selector=sel)` — **without the
  new argument**, so it takes the 0.0 default.

So a landed v20 that sets `placement_latency_ms = 250` changes nothing until BE's builder
passes it AND the day books are REBUILT. Under rule 13 the landed books keep their bytes,
so the four design days would need new books, not just re-valued ledgers — which lands
squarely on REV 105's four design-day re-runs: **as things stand those re-runs would be at
L_place = 0**, and BE 100's number (98 % of the baseline's 09-05 settlement P&L carried by
fills within 250 ms of their generation's start) is measured on exactly the fills a rebuild
would move. Nobody should read a re-run under the new endpoint as answering the
placement-latency question.

## §8 SMALLER, EACH MEASURED

- **Three of the new refusals are never driven** — `SETTLEMENT_VERIFICATION_SOURCE_MALFORMED`,
  `SETTLEMENT_EXCESS_DOES_NOT_RECONCILE`, `SETTLEMENT_NULL_TOO_SMALL` each occur exactly
  once in the module: the raise site, no cell. Rule 15.
- **The chain cell's positive control carries a zero.** With the fixture's alternating
  winner map the arm total is 0.0, the baseline total is 0.0 and `D_E_settle` is 0.0; the
  cell asserts `isinstance(D_E_settle, float)`. The row counts prove the plumbing; nothing
  proves a NON-ZERO ruled excess travels into the arm-day and the ledger. Give the fixture
  asymmetric fills and assert a hand-computed value.
- **The winner source is read TWICE per day** (`winner_source` → verify →
  `winner_source(verification=…)`), and `resolutions.jsonl` is APPEND-ONLY and live (its
  mtime moved during this review; 38,307 closed slugs, sha `604711a8…` when I read it).
  The verification is computed against read #1's winners; the winners used and the digest
  recorded come from read #2. Verify against the object you value with, and record a
  digest over the SUBSET used — the ledger's `SETTLEMENT_SLUG` rows already carry
  `up_won`/`settle_cents` per slug, which is what makes the day recoverable at all.
- **Silent skips (rule 4).** Both valuation functions drop a fill with `px_cents is None`
  or falsy size with no count. Measured on the 09-05 ledger: **0 of 177,467**, so nothing
  is dropped today. Worth closing anyway because under THIS estimand a dropped fill also
  loses its SHARES from the residual — a larger error than losing its markout.
  `absolute_legs` already reports `n_fills` / `n_fills_valued`; the settlement block does
  not.
- **`winner_source` validates the WHOLE file.** One voided or 50/50 market anywhere in
  38k records raises `SETTLEMENT_WINNER_AMBIGUOUS` and blocks every day, including days
  whose own slugs are clean. Scope the integrity refusals to the slugs the day names and
  report the rest as a census.
- **`verify_winners_against_chainlink` applies one coin's series to every slug**
  (`coin=params["coin"]`), with no check that the slug's own prefix matches. Single-coin
  today; one assert makes it safe.
- **`sys.getsizeof(settle_values)`** measures the list object, not the 500 floats it
  holds (~12 kB uncounted against ~4 kB counted), while the text calls it "the whole
  cost". Trivial in size, wrong as a measurement.
- **The ledger's `schema_version` stays 2** while two new row kinds appear. The header's
  `settlement_rows_present` / `settlement_row_kinds` make each file self-describing, which
  is the better mechanism — but a reader that dispatches on the version alone will not
  know they exist. **Routed to DA**: I did not test whether DA's reader refuses an unknown
  row kind.

## §9 THE DRAFT

Still not a declaration — `.md`, not `<family>_v<N>.json`, not in `declarations/`, and the
params head still resolves to v19 (verified). DE 137 replaced §3's fork ("…or state in the
declaration that the endpoint is provisional") with **§3a, which pins the convention block
verbatim** and keeps rule 9's parenthetical NARROWER rather than resolved: one form
reproduces every recorded winner on two days, and whether it does so on any other day is a
per-day check — which is why the check runs per day and refuses. That is exactly what
R-803 asked for. §3b states the placement-latency proposal, its default, and the two
consequences (the four days may be RE-VALUED as DESIGN data; no declared L may sit on a
value a fill lands exactly on — 100 ms measures 99.99999999999964). **Add to §3b the fact
in §7 above: which producer the parameter binds, and that the books must be rebuilt for it
to bind at all.**

## §10 WHAT I DID NOT ESTABLISH

- No real day was run: no lock, no heavy unit. Everything real here is read-only at the
  09-05 ledger, the venue record and the TWAP hourlies.
- Whether a real day's settlement null can be degenerate (§6(3) is reachable in code,
  driven on a fixture).
- Whether DA's and BE's readers cope with the two new ledger row kinds.
- The DE 111 memory-residue cell whose bound DE 137 removed: the battery reports 0
  disarmed and the cell still records the residue with its sign and its finding
  ("the mechanism is NOT ESTABLISHED"), which is unchanged — I read it, I did not
  reconstruct the reordering that flipped it.

## §11 ROUTED

1. **DE (138)** — §6 (1)–(4), in that order; (2) and (3) before any real day is valued.
2. **DE / BE together** — §7: `placement_latency_ms` on `be_daybook_build`'s call, and
   what a rebuild of the four days' books costs.
3. **DA** — §8's last item: the two new row kinds under an unchanged `schema_version`.
4. **Coordinator** — §5's caveat wherever a battery count is quoted, and §7 before REV 105
   gates the four design-day re-runs.
