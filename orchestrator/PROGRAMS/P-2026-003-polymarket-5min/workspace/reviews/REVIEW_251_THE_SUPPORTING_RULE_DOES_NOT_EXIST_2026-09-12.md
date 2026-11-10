# REVIEW 251 — the supporting rule does not exist, and the pipeline already says so

REV round 214. Filed 2026-09-12T00:08:26Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled.

## THE ANSWER, PLAINLY

**You are right, and it is worse than "may be counting an unpopulated field."
The lane's own schema already labels every one of those 76,617 rows as
unpopulated, in those words.** `tier1_pipeline.py:1220-1222`:

    "fee_source_status": (
        "UNPOPULATED_WS_ZERO" if fee_raw == 0 else "OBSERVED_NONZERO"
    ),

The pipeline does not treat a zero in that field as an observation. It names it
`UNPOPULATED_WS_ZERO`. **"76,617 of 76,617 at `fee_rate_bps = 0`" is therefore
76,617 rows the pipeline itself classifies as having no fee observation** — the
count erases a distinction the schema draws one line above where the value is
read.

Four measurements, in the order they close the question.

**1. The field is a constant in the source.** Scanned across a full day of the
raw tape, all markets, 2026-09-07:

    "fee_rate_bps":"0"   638,439 occurrences
    any other value            0

Not a sample — every occurrence in the day. A second day's single-market sample
gave 944 occurrences, all `"0"`.

**2. The reader defaults a missing field to zero.**
`tier1_pipeline.py:1198`:

    fee_raw = _finite_float(message.get("fee_rate_bps", 0), "fee_rate_bps")

So a record that omits the field and a record carrying an explicit zero are
**indistinguishable downstream**. Whichever the venue is doing, the pipeline
records 0.

**3. `OBSERVED_NONZERO` is unreachable on this data.** It fires only when
`fee_raw != 0`, and by (1) the field is never nonzero. So the control that
would detect a fee has never fired and cannot fire on this source. **Rule 15:
a zero from an instrument that never proved it can fire is not a result.** That
rule was written for exactly this shape and it applies without modification.

*(One check I could not complete and am not claiming: I tried to count
`OBSERVED_NONZERO` occurrences in the written tier-1/tier-2 artifacts, but those
are binary and a text grep does not read them; the path I globbed for parquet
returned zero files. The unreachability above is an argument from (1) and the
branch at line 1221, not from an artifact scan.)*

**4. The audit says it in DA's own words.** `limits[3]`, verbatim:

> fees are read from the OrderFilled fee word, never from the websocket
> fee_rate_bps field, which is unpopulated

DA's own instrument refuses to use the field that DA's summary then cited as the
supporting rule. Both statements are DA's; they contradict each other; the
audit's is the one with the measurement behind it.

**So the rule you ruled on does not exist.** The only fee observations this
programme holds are the onchain ones: 1,056 maker legs (1,046 zero, 10 charged)
and 901 taker legs (901 charged). There is no order-level corroboration, and
there was never going to be one from a field that is always the same string.

## 2. What `limits[1]` does to the inference — it makes the evidence asymmetric

> the 901 receipts are a SAMPLE of our own recorded trades, not a population;
> incidence here bounds observed volume only

You drew the right conclusion and I want to state its shape precisely, because
it decides what any future query can establish:

- **Presence proves the charged class.** If our address appears among the six,
  we are charged — and charged on 100% of maker legs, since the partition is
  total.
- **Absence proves nothing.** Not appearing among the six is equally consistent
  with being in the zero class and with having had too few maker legs in a
  sample to be charged even if we are in the charged class.

**The qualified zero rests entirely on the second inference**, which the file
forbids. So even a favourable address query cannot make the qualified zero an
identified rule — it can only fail to refute it. The honest ceiling on this
evidence is: *a declared assumption of zero, with the charged class described,
our position in it unestablished, and a sensitivity sized to the charged case.*
That is a legitimate thing for §7 to record. It is not "a supporting rule
identified", which is what §9 asks for.

## 3. My costing error, owned

I wrote that DA could settle our address's class "tonight from this same file."
**That was wrong and you were right to check it.** The file enumerates only the
charged side — `maker_charged_by_address` (6), `maker_charged_address_total_maker_legs`
(the same 6), `maker_charged_detail` (10 legs). **There is no by-address
enumeration of the 1,046 zero-fee maker legs anywhere in it**, and I confirmed
that against the exhaustive top-level key list. The query needs the 901 receipts
under `onchain/receipts/`, which is a larger job than I costed.

I made the same class of error I have been filing against all night: I asserted
what an artifact could answer from the fields I had looked at, without checking
that the field I needed was among them.

## 4. Which basis the sensitivity should use

`min(p, 1−p) ≥ p(1−p)` for every p, with ratio `1/max(p, 1−p)` ∈ [1, 2]:

| p | min(p,1−p) | p(1−p) | ratio |
|---|---|---|---|
| 0.99 | 0.0100 | 0.0099 | 1.01 |
| 0.90 | 0.100 | 0.090 | 1.11 |
| 0.75 | 0.250 | 0.1875 | 1.33 |
| 0.60 | 0.400 | 0.240 | 1.67 |
| **0.50** | **0.500** | **0.250** | **2.00** |

**So DE should build two columns, not one:**

- **Worst case / the bound: `min(p, 1−p)`.** It is the larger of the two at
  every price, so it is the only one of the pair that can be called a bound.
  This is the basis my REVIEW 249 table used, and for a *bound* it was the right
  one — my REVIEW 250 §3.2 over-corrected in calling it an error. 249's
  multipliers (50× at p=0.5 on the 9.9% tier, 250× at 49.5%) are correct **as a
  bound**; 250's (25×, 125×) are correct **as the documented central estimate**.
  Both numbers are right for their own question and I should have said which
  question each answered.
- **Central estimate: `p(1−p)`**, the form the audit documents and the one its
  taker fit was computed against.

Report both, at the **49.5%** tier, with 9.9% beside it. The corpus cannot
choose between the forms — all ten charged legs sit at p = 0.9900, where the two
differ by 1% — so choosing one silently is picking a number the evidence does
not support. Two columns removes the choice, and the gap between them is itself
the honest statement of how little the ten legs constrain the schedule.

## 5. Owed

- REVIEW 247 Part 1 awaits a resolver; 246, 245 and 250's citation-locator
  proposal remain open.
- The `OBSERVED_NONZERO` artifact-level count, which I could not complete
  above — worth someone finishing, because if it has *ever* fired anywhere the
  field is not uniformly dead and (3) would need qualifying.
- Standing: one review per day for the 09-09..09-13 records (REVIEW 228); 09-11
  is closed and still unread by me.
