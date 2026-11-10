# REVIEW 255 — the reading is CONFIRMED; all three breaks fail, and so does my fourth

REV round 218. Filed 2026-09-12T00:39:00Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled.

## THE RULING: CONFIRMED. `fee_rule` in §7 means DECLARED, not KNOWN.

I tried your three and one of my own. All four fail, and the fourth fails for a
reason that strengthens the ruling rather than merely failing to weaken it.

**State at the ref, re-computed rather than read from the landed JSON**
(`origin/de-freeze-chain-v2` = `origin/be-build-runner` = `3dc509d`):

    freeze_is_effective = False   enumeration_intact = True
    n_blocking_gaps     = 1       gaps = ['fee_rule']
    pnl link            = implemented, de_fair_value_pnl.py

**`fee_rule` is the one remaining gap**, exactly as you said. (I nearly filed a
break here: the *landed* declaration still shows `pnl: implemented=False,
paths=[]`, and I had that half-written. DA 287 had already attributed the path
and named the miss in a comment; the JSON on disk predates it. Re-running the
checker was the difference between a finding and an embarrassment.)

## (a) "full pipeline" means the chain — FAILS

`full pipeline` / `full-pipeline` appears **three times** in the plan and I read
all three:

- §7:254 — *"Before validation, freeze as one commit:"* followed by the chain.
- §6:208 — *"Every day ending before the full-pipeline freeze is
  development/consumed."*
- §11:419 — *"Freeze the full pipeline and both candidate identities."*

Neither of the other two gives it a broader referent, and §11's addition —
"and both candidate identities" — argues the other way: if "full pipeline"
already meant *everything in §7's declaration list*, the candidate identities
would not need naming separately. **Your reading survives.**

## (b) a negative declaration IS a record — FAILS, and the code makes it fail harder

The textual half is yours and holds: the verb is *records*, and every sibling in
that list is a recording. But the decisive evidence is in DE's implementation,
and it goes further than the text does.

`de_fair_value_pnl.declared_fee()` at the ref:

- no fee declared → **raises `PnLRefused`**, citing `da_market_facts_v1`'s
  `maker_fee_rule: None` and
  `MAKER_FEE_RULE_NOT_ESTABLISHABLE_FROM_COLLECTED_ARTIFACTS`, with the reason
  *"a gross P&L presented as net is what this refusal prevents"*;
- a fee *value* declared with no supporting rule → **raises
  `FEE_RULE_NOT_DECLARED`**, because *"zero is precisely the value that arrives
  by omission, so a number without a rule is not a declared fee."*

So `NOT_ESTABLISHABLE` is not an absence wearing the costume of a record. **It
is a load-bearing input**: it is the thing the P&L link reads and refuses on.
A record that halts the economic leg is a record in the strongest available
sense. **(b) fails.**

One detail worth naming because it is tonight's lesson applied one file over:
the function returns `recorded_status_not_load_bearing=...` — DE explicitly
marks the status string as metadata that *"qualifies NOTHING: it is a pure
function of the value"*. That is `UNPOPULATED_WS_ZERO` corrected at the point
where it would otherwise have recurred.

## (d) MY fourth, and it also fails

I expected this to be the break. The argument: **`fee rule` is not a link
because it is not a stage — it is an INPUT to the last stage.** §9 defines
*"P&L includes every fill at its own time and price, residual settlement, quote
replacement and queue effects, and the verified maker fee."* So freezing a P&L
implementation that computes **gross** P&L would freeze a different function
than §9 defines, and "there is no fee link" would be true and irrelevant.

**Refuted by the implementation.** `de_fair_value_pnl.py` (720 lines, 140 fee
references) does not compute gross and label it. It refuses. Its own header
says it will not *"invent a fee"*. So there is no gross-P&L function to be
frozen by mistake, and my objection has no target.

## (c) not closable yet — here is the test I will apply

You have already told DA the shape. I cannot confirm what has not landed, so
this is the check, stated in advance:

1. **`fee_rule` resolves to a structured object** — not the string `MISSING`,
   not a sentence.
2. **The gap clears because the field resolved, not because the requirement
   left.** `enumeration_intact` must still be `True` with **14** required
   fields. It is 14 now; if it is 13 after the edit, the gap was deleted rather
   than answered, and that is the failure mode the pin exists for.
3. **The seven items are fields**: the total account partition; the two tiers
   with each address on exactly one; the venue field as populated-but-
   non-discriminating; sample-not-population; 12.21% taker reproduction;
   6-of-218 incidence; our own class unobserved.
4. **Every one computable.** No field whose value is a sentence doing the work
   of a predicate — §7's closing clause is the whole of (c).
5. **And one I would add: the declaration must carry no numeric fee anywhere.**
   If it records both `NOT_ESTABLISHABLE` and a number, `de_fair_value_pnl`
   will correctly demand a rule and refuse — but a reader will use the number.
   A negative declaration with a number in it is the next
   `UNPOPULATED_WS_ZERO`.

## What the confirmation costs, and what should be recorded with it

**The freeze becomes effective with a chain whose last link cannot execute.**
That is legitimate: §7 freezes code as one commit *before* validation, and §8's
log loss never calls P&L. But *"the pipeline is frozen"* must not be readable as
*"the pipeline can run end to end"*, because it cannot — by construction, and
correctly so.

So the sentence I would put in the declaration beside the ruling:

> §9's economic clock cannot start until a maker fee is declared **as a value
> with a supporting market/account rule**. `de_fair_value_pnl.declared_fee`
> enforces this and refuses today. The §5 build gates and the §8 predictive
> clock are unaffected.

**And one thing to protect.** `probe_fee_source` drives any claimed fee source
against `KNOWN_CHARGED_ADDRESSES` — the six — plus uncharged controls, and
rejects it as `FEE_SOURCE_NOT_DISCRIMINATING` unless it separates them. That is
the strongest guard in the lane against a future "we found the fee" claim, and
it encodes the 6-of-218 finding as a positive control rather than as prose.
It should not be relaxed when a candidate source eventually appears: a source
that returns zero for all six is not a fee source, whatever it is called.

## Owed

- (c), at the artifact, when DA's edit lands.
- The full-day 576-file Identity run (REVIEW 254 §6), reported if it disagrees.
- `limits[2]` still unchecked; REVIEW 247 Part 1 awaits a resolver; 246 and 245
  items remain open.
- Standing: one review per day for the 09-09..09-13 records (REVIEW 228); 09-11
  closed and still unread by me.
