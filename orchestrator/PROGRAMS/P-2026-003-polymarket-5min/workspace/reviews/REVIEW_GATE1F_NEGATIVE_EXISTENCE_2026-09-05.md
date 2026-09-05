# REVIEW — Gate 1f's negative existence claim FAILS, and it fails on a quantity this programme measured, bounded and declared discharged sixteen days ago

**Filed** 2026-09-05T15:04Z (clock read before composing) · reviewer seat
(pm-codex) · **tip `1404302`**, worktree `~/ctaNew-wt-rev` clean, refreshed from
`8f08387` at the start of this round · no code fixed · no sealed day opened · no
write under `data/` · no other seat's worktree read · one bounded read of one
pre-freeze raw file under `systemd-run --user --scope --slice=research.slice -p
MemoryMax=8G`.

**ROUTING.** Every finding below is **CHECKED** — I went to the artifact and
computed or read the predicate myself. Nothing here is AGREED. Where a claim is
an inference from artifacts rather than a reading of one, I say so and mark the
premises.

---

## THE ANSWER, FIRST

**The terminal stop is wrong on its stated cause.** Gate 1e refused because
"every arm lacks an owned-order per-fill maker-fee ledger"; Gate 1f then ruled
that no public tape, public trade fee or other maker's on-chain tier can supply
it. **That ruling was reached without ever looking at the artifact that
supplies it.** The Gate-1f audit's entire enumerated surface is **three files in
this repository** (`collect_pm.py`, `tier1_pipeline.py`,
`FLOW_UNCERTAINTY_LOOP.md`), tested by **five substring predicates**, and its
"conclusion" is a hardcoded string that none of those five predicates evaluates.
The word "on-chain" appears in the module exactly once — **inside that hardcoded
conclusion** (`de_v2_owned_execution_input.py:426`). No on-chain artifact is
read.

Meanwhile the quantity itself is in this repository's own established-facts
table, with an instrument, an n, and a signed bound:

> **`FLOW_MODEL_STATE.md:78`** — *"Taker pays the fee, maker does not"* ·
> `0.07·p·(1−p)` $/share to 4 dp · **600/600 taker legs charged, 744/754 maker
> legs zero** · n = 600 transactions.

**A negative existence claim was made about data the programme had already
measured.** That is the fifth instance of this class in two days, and this one
is load-bearing: it stopped the programme.

---

## 1. THE INSTRUMENT — four defects, all CHECKED at `de_v2_owned_execution_input.py`

### 1.1 The conclusion is printed, not computed — CLAUDE.md rule 10, in the module that terminates the programme

`_public_surface_audit()` computes five booleans and then returns a
**`conclusion` key whose value is a constant string literal** (lines 420–429).
No predicate evaluates it. The five that exist test:

| predicate | what it actually tests |
|---|---|
| `collector_uses_public_market_websocket` | a URL substring is in `collect_pm.py` |
| `collector_records_public_last_trade_fee_field` | two substrings are in `collect_pm.py` |
| `tier1_trade_schema_has_public_transaction_hash` | a schema line is in `tier1_pipeline.py` |
| `tier1_trade_schema_has_no_client_order_id` | a substring is absent from an 1,800-char window |
| `historical_audit_says_venue_ack_lag_not_observed_without_orders` | a sentence is in `FLOW_UNCERTAINTY_LOOP.md` |

Every one is a fact about **our own repository's text**. None is a fact about
what the venue publishes. The receipt then carries the sentence *"additional
public tape does not satisfy the owned execution contract"* as though it were an
output. It is an input.

### 1.2 The surface is three files, and it excludes the artifact that refutes it — CHECKED

`source_sha256` in the receipt has **exactly three keys**. The repository's
on-chain fee record lives in `live/pm_research/da_feeds_polygon.py` (a committed,
read-only Polygon decoder whose `OrderFilled` and `OrdersMatched` signatures are
**verified by keccak-256 rather than taken from a lookup**, and whose selftest
decodes a real maker leg and asserts `of.fee == 0` at line 435). **It is not in
the surface.** Neither is `FLOW_MODEL_STATE.md`, the programme's own
established-facts table, which carries the fee row quoted above.

A negative existence claim whose surface omits the file that answers it is not a
negative existence claim. It is a scope statement.

### 1.3 It truncated — and it is my own error class, back in someone else's code

```python
'("client_order_id",' not in tier1_text[
    tier1_text.find("TRADE_SCHEMA") : tier1_text.find("TRADE_SCHEMA") + 1800]
```

CHECKED by execution: `TRADE_SCHEMA` first occurs at index 5,264
(`tier1_pipeline.py:146`); the block from there to its first `])` is **13,046
characters**. The window reads **1,800 of 13,046 — 13.8%** — and ends
mid-schema (the character at +1,800 falls inside `"price_gap_ms"`).

**In fairness, the predicate's VALUE is correct**: `client_order_id` occurs
nowhere in `tier1_pipeline.py` (checked over the whole file). But the predicate
as written would have returned `True` had the field been declared at character
1,801. It is right by luck, not by method — the same shape as the `head -25`
that hid line 61 from me on 09-04, and it is now in a receipt that stopped a
research programme.

### 1.4 No falsifier in the direction that matters — CLAUDE.md rule 15

The module ships **eight** `refuses()` known-bads. **All eight exercise
`validate_export()`** — the path that admits a hypothetical export. The
public-surface half, which carries the existence claim, has **none**: it raises
if any predicate is false, so the selftest's `ok(all(predicates.values()), ...)`
cannot fail for the reason that matters. **There is no input that would make
this audit report "a public source CAN supply the term."** A checker that cannot
fire in the refuting direction has not proved a negative.

---

## 2. THE SUBSTANCE — what the owned-order join needs, and what the repository has

Gate 1e names the missing term precisely:
`FEE_COMPONENT = "VENUE_MAKER_FEE_EXCLUDING_REBATES_REWARDS"` — the per-fill
maker fee, rebates and liquidity rewards explicitly out of scope.

**That term is measured, on-chain, in this repository, twice, at two scales.** All
CHECKED at the named lines:

| fact | where | evidence |
|---|---|---|
| taker fee `= C·0.07·p(1−p)`, maker legs **0** | `PM_SKETCH_REVIEW_ITER1_M.md:20–42` | two decoded Polygon receipts, exact to 6 decimals; docs confirm **"makers never charged"**; `maker_base_fee = 1000` identified as a **legacy signature cap**, not a charged rate |
| **600/600** taker legs charged, **744/754** maker legs zero | `FLOW_MODEL_STATE.md:78`; `STATUS.yml:3780–3800`; `HANDOFF.md:13570–13578` | n = 600 transactions, formula to 4 dp across the full moneyness range |
| second, larger sample | `HANDOFF.md:15159` | **901 taker legs 100% charged; 1,056 maker legs, 10 charged** |
| the term is **bounded and signed** | `HANDOFF.md:14985`, `MEASUREMENT_PLAN.md:959` | **0.0232 ¢/share** at the measured incidence; **1.75 ¢/share** at the absolute bound (every maker leg at max `0.07·p(1−p)`); **"on the maker leg a fee can only SUBTRACT, so net ≤ gross"** |
| the precondition was **DISCHARGED** | `MEASUREMENT_PLAN.md:945–946` | *"IS UNBLOCKED AS OF ITERATION 6 — BOTH PRECONDITIONS ARE DISCHARGED … Fee: … zero WITH ITS EXCEPTION CARRIED"* |

**And the instrument demonstrably fires.** 600/600 taker legs return a non-zero
fee matching the formula to four decimals, in the same transactions where the
maker legs return zero. A zero on a maker leg is therefore a **measured zero, not
a blind spot** — the positive control sits inside the same decoded receipt. That
is rule 15 satisfied by the data, which is exactly what the Gate-1f audit lacks.

### 2.1 Gate 1e's own prohibition does not reach this evidence

Gate 1e forbids two things (plan lines 442–445):

1. *"Do not **assume** a zero maker fee"* — nothing here is assumed; it is decoded
   from `OrderFilled` word[4] at n = 600 and again at n = 1,056.
2. *"…or **infer** an owned maker fee **from a public taker/trade fee field**"* —
   nothing here uses `fee_rate_bps`. Every document in the repository forbids
   that field by name (`HANDOFF.md:14379`: *"Read fees from the CHAIN, never from
   `fee_rate_bps`"*; it is `"0"` on all 446,412 trade events and that is an
   **artefact**). The on-chain `OrderFilled` log is not a public fee *field*; it
   is the settlement record of the charge.

The prohibition was written against the WebSocket artefact and is right about it.
**It does not cover the chain, and the refusal applied it to the chain without
reading the chain.**

### 2.2 Gate 1f's specific rebuttal mischaracterises the evidence

Gate 1f pre-empts this with *"another maker's on-chain tier cannot satisfy the
owned-order join."* That describes **account-specific pricing**. The evidence is
not a tier: it is **incidence by role** over 754 (then 1,056) maker legs across
many addresses, plus a venue-published schedule under which **makers are not
charged at all**. There is no maker tier to be wrong about. Account-specificity
is a live concern for exactly one term — the **maker rebate** — which the V2
scope **excludes**, and whose omission is **conservative** (a rebate can only
help the maker).

### 2.3 The deeper problem: Gate 1e's exit condition is unsatisfiable for its own population

*(Inference; both premises CHECKED.)* Premise one: the V2 route evaluates a
**counterfactual** — simulated fills on a neutral no-cancel reference path, and
Gate 1e states plainly that *"owned-order acknowledgement/fill causality remains
unobservable from this public-market counterfactual replay."* Premise two: Gate
1f's contract requires **≥200 owned orders and ≥200 owned maker fills** joined to
**acknowledged** orders on a real account.

An owned per-fill ledger is defined only on a **realized** path. These fills were
never placed and never will be. **So no data acquisition — not even live trading —
could produce an owned per-fill ledger for *this* population.** Gate 1e's exit
condition is not blocked; it is undefined for the thing it gates.

### 2.4 And Gate 1f cannot be cleared without violating the same paragraph that defines it

The plan requires an owned-account export while, four lines later, ruling: *"This
repo must remain research-only; do not add venue credentials, signing, order
submission or cancellation code here."* CLAUDE.md's scope section says the same.
**"Data-acquisition blocker" is the wrong label**: there is nothing to procure,
and the USER should not be asked to rule on acquiring it as though there were.

---

## 3. WHAT I DO **NOT** CLAIM — the limits, stated because they are real

1. **There is no committed per-fill maker-fee ARTIFACT, and DA said so first.**
   `HANDOFF.md:14998` records DA's own provenance defect against this very
   number: the G-FF1 receipt carries **per-leg side attribution only and no fee
   amounts**, and scanning every artifact in `data/pm_5min/derived/` returns
   **zero maker-fee fields**. So a Gate-1e implementation looking for a fee
   *ledger* in the derived artifacts correctly found none. **My finding is not
   that the ledger exists. It is that the ledger is not the estimand** — a
   bounded, signed term is identified, and the plan's own §8 precedent
   (R-109/Q-DA-48) treats exactly this shape by reporting the interval.
2. **The exception rate is a within-sample rate.** The G-FF1 sample is
   **stratified** (9 per cell against strata of 595–99,172), so 1.3% is not a
   population rate — DA states this at the same line, and
   `FLOW_UNCERTAINTY_LOOP.md:628` independently notes the 600-tx sample cannot
   supply a population share. **The absolute bound (1.75 ¢/share) does not depend
   on the rate; the point estimate does.**
3. **U5 is open.** The 10 fee-bearing maker legs are `UNRESOLVED — mechanism NOT
   identified` (`FLOW_UNCERTAINTY_LOOP.md:703`). "Maker pays no fee" is the right
   headline and the wrong absolute.
4. **Ack/fill causality is genuinely unobservable** and remains so. But the plan
   itself classes it as a **carried limitation, not a refusal condition** — the
   receipt status names the *fees*: `LIFECYCLE_LEDGER_COMPLETE_GATE1_REFUSED_
   REQUIRED_MAKER_FEES_UNAVAILABLE`. Remove the fee ground and the stated cause
   of the stop is gone.
5. **I have not read Gate 1e's gross numbers** and take no position on which way
   the result falls. Gate 1e ruled its gross fields uninterpreted; I honoured
   that. **I am not claiming the programme has a positive result.**
6. **`FLOW_UNCERTAINTY_LOOP.md:632`, the audit's own cited authority, does not say
   what the audit uses it for.** Its not-identifiable list is `own_impact` (#10)
   and `venue_ack_lag` (#11) — **the fee is not on it.**

---

## 4. WHAT WOULD SETTLE IT — cheap, precedented, and unrun

Not a new data source. **The interval.** Report Gate 1e's decision metric at both
ends of a bound the repository already owns — fee = 0 and fee = `0.07·p(1−p)` on
**every** maker leg (1.75 ¢/share, far beyond any measured incidence) — and check
whether the decision is invariant across it. That is precisely the instrument DA
used to close §8 (*"the STOP verdict is INVARIANT to the user's unpinned fee
parameter"*), and its asymmetry does half the work for free: since a maker-leg fee
can only subtract, **if the gross result is already unfavourable the fee cannot
rescue it, and no interval is needed at all.**

**I am not asserting the invariance holds.** I am reporting that the check is
arithmetic on numbers already computed, that the programme has run this exact
check before, and that **nobody has run it here** — because the audit nulled the
metric instead.

---

## 5. ITEM 2 — the L4 citation check: **COMPLETE. The specification survives; three numbers do not.**

`8f08387` / `REVIEW_L4_SPECIFICATION_2026-09-04.md` was labelled
DRAFT-PENDING-CITATION-CHECK. **Checked, all of it, as-of 2026-09-05T15:01:14Z.**

**Load-bearing claim — REPRODUCES at a 6.5% larger population.** Re-ran every
structural predicate on `resolutions.jsonl`:

| predicate | spec (09-04) | now (15:01:14Z) | verdict |
|---|---|---|---|
| records / distinct slugs | 32,150 / 32,139 | **34,224 / 34,213** | **grew** |
| `closed == True` on distinct slugs | 32,139 of 32,139 | **34,213 of 34,213** | holds |
| `outcomePrices` `None` on every closed record | all | **all 34,213** | holds |
| `umaResolutionStatus` `None` | all | **all 34,224** | holds |
| open snapshots carrying `outcomePrices` | 8 | **8** | **identical** |

**So "the repository does not record the realised settlement of any market"
stands, and with it the specification's central move — build L4 against the
terminal mark, not settlement.** Confirmed too: the three contested price
directories exist exactly as cited (`crypto_prices`,
`crypto_prices_twap_thirty`, `crypto_prices_twap_sixty`); `20260826` holds
**1,974** files, exact; the raw schema is `recv_ns \t [ {market, asset_id,
timestamp, hash, bids:[{price,size}], asks:[…]} ]` with full depth ladders per
side, exact.

**Register citations — all four resolve, by identity not vocabulary.** R-253
(17248) is the Chainlink settlement correction; **R-506(E)** exists and its
subject is *"TERMINAL-IN-A-GAP IS `NOT_AVAILABLE`, A STATUS WITH A COUNT"* —
which is the exact shape both citations claimed; R-500 (19452) withdraws 08-29
and **its own block says "development read" twice**, so that citation is sound;
Q-DA-142 and Q-DA-146 both exist. **RESULTS.md §0** carries the ladder at lines
383–389 exactly as cited: base 48.02%, CONDVALUE **+2.2 pp**, HAZARD **+3.1 pp**,
break-even **+3.3 pp**, 10% capture **+10.3 pp**.

**Three numbers fail, and the failure is mine and instructive:**

1. **32,150 / 32,139 has drifted to 34,224 / 34,213** — the tape grew during
   measurement. The document carries a filing timestamp but the population line
   carries no as-of of its own. **CLAUDE.md rule 8, and I broke it.**
2. **"17 day-directories" is now 18.** Same cause.
3. **"99,460 snapshots in one btc window; a bid level of 95,290.49 at price
   0.01" names no file, so it is UNVERIFIABLE AS WRITTEN.** The nearest check —
   the largest btc window in `20260826`, `btc-updown-5m-1787751600.jsonl.gz` —
   gives **228,036 lines** and a maximum size at 0.01 of **98,687.3** (94,999.26
   in the first snapshot). **The order of magnitude is confirmed and the hazard
   is real; the exact figures are not reproducible.** A quoted population needs
   its surface, not just its number.

**None of the three is load-bearing** — all are illustrative of hazards in §4,
and the specification's argument is unchanged. **I have relabelled the spec in
band** (label → `CHECKED`, with a dated citation-check section appended, original
text untouched).

---

## 6. ITEM 3 — the arms diff: **MOOT. CLOSED.**

`arms53.py` **does not exist in the repository** (checked; the 15,727 B file was
a scratch-dir artifact), and `de_section81_arms.py` is now **66,862 B**, not the
23,893 B I diffed against — so a source diff today would compare nothing to
nothing.

More to the point, **the question the diff existed to ask has been answered by a
stronger instrument.** R-526(A) records DE's re-emission **from a clean tree at
`2a3bb30`, on the branch, with 7/7 identity files matching the commit the
artifact names, and 26 of 26 economic quantities BIT-IDENTICAL, delta exactly
0.00e+00.** A source diff would have shown textual differences that say nothing
about output equality; the re-emission shows output equality directly. **The
defect was the record, not the result.** Closed, and I will not reopen it.

---

## 7. WHAT THIS DOES AND DOES NOT PUT TO THE USER

**It does not restart the programme.** Gate 1's other three refusals — the iid
sampler at 1/200 draws in 4,000, the exact-fiber ESS 10.53 < 100, and the
sequential quota at 16/1,000 with 16 distinct states — are untouched by anything
here. They are sampling failures on a consumed window and they still stand.
**Clearing the fee ground does not clear Gate 1.**

**What it changes is what the USER is being asked to rule on.** The blocker was
presented as *acquire an owned-execution export or stop*. That framing does not
survive: the export cannot be produced for a counterfactual population (§2.3),
cannot be produced within this repo's scope (§2.4), and **is not needed for the
term that caused the refusal** (§2). The real question is narrower and cheaper:
**should Gate 1e be re-run reporting the fee interval the repository already
owns, instead of nulling its decision metric?**

That is the USER's call. My part is the finding: **the stop was taken on a
negative existence claim that was never checked against the programme's own
measurement, by an instrument that could not have found it and could not have
reported it if it had.**
