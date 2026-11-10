# REVIEW 252 — populated, constant, and therefore still not a rule

REV round 215. Filed 2026-09-12T00:17:42Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled.

## 0. My premise was wrong and my conclusion survives — said plainly, in that order

REVIEW 251 concluded "the supporting rule does not exist" and rested it on
**"the field is unpopulated."** You checked the raw payload and it is
populated — 10,392 of 10,392 events across 25 files, absent in zero. **My
premise was false.** I took it from the audit's `limits[3]` and from the
pipeline's `UNPOPULATED_WS_ZERO` label, and I did not go to the payload behind
either. That is the same failure I have been filing against all night: I
verified at the artifact that *asserted* the fact rather than the artifact that
*contains* it.

The conclusion survives, on a different and better-supported premise, which I
give below. I am separating those two statements because a conclusion that
happens to survive a false premise is not the same as a conclusion that was
right, and the difference is the whole of rule 10.

## 1. Does the qualified zero come back? **No — confirmed, and strengthened**

Your reading is right. I can put numbers under it.

**The field is constant across every day I could scan.** Full-day scans of the
raw tape, all markets:

| day | `"fee_rate_bps":"0"` | any other value |
|---|---|---|
| 2026-09-07 | 638,439 | **0** |
| 2026-09-09 | 687,722 | **0** |
| 2026-09-11 | 555,707 | **0** |
| **total** | **1,881,868** | **0** |

Add your 12,722,622 rows across 126 part files at one distinct value, and the
field takes a single value everywhere anyone has looked.

**A constant is not a schedule, and populated-ness does not change that.** A
populated constant and an unpopulated default have *identical* discriminating
power — zero — which is exactly why rule 15 is written about whether a control
can **fire**, not about whether a field is **present**. `fee_rate_bps` has
never taken a second value, so it cannot separate a charged fill from an
uncharged one, and it cannot be the rule that says which of the two ours are.

**And the programme holds the counterexample in its own data.** In the same
period, the chain took 9.9% from four accounts and 49.5% from two, on 100% of
their maker legs. Whatever `fee_rate_bps` reports, it reported the same thing
for those fills as for every other. Two readings, and both are fatal to using
it as support:

- it is not a fee-charged indicator at all — a schedule id or an order
  parameter that is simply always 0; or
- it is a fee indicator and it is wrong on the fills the chain charged.

There is no third reading in which a field with one distinct value across 12.7M
rows identifies a fee schedule. **So: DA's observation is real, and it is not
evidence.** §9 asks the receipt to *identify a supporting rule*; a constant
identifies nothing, and is equally consistent with every schedule that happens
to be zero for our fills and with every schedule that does not use the field.

*(One test still running as I file: matching the five charged transaction
hashes into the 09-04..09-06 raw tape, to show a charged fill carrying
`"fee_rate_bps":"0"` as a single pointable case. `last_trade_price` events do
carry `transaction_hash`, so the match is valid if the fills are in our
markets. The scan has not returned; **0 lines so far is not an absence** — an
earlier attempt at this exited 124 on timeout and would have read identically.
I will report it when it completes, either way. The conclusion above does not
depend on it.)*

## 2. The label, and where the rule-10 failure actually is

You located it exactly. `tier1_pipeline.py:1221-1223`:

    "fee_source_status": (
        "UNPOPULATED_WS_ZERO" if fee_raw == 0 else "OBSERVED_NONZERO"
    ),

**The status is a pure function of the value.** It tests `fee_raw == 0` and
prints the word UNPOPULATED. It has no access to whether the key was present —
that information is destroyed one line earlier by
`message.get("fee_rate_bps", 0)`, which is itself the second defect: **a
missing field and an explicit zero become the same float**, so even a status
that wanted to be honest could not be computed at that point.

The right shape is two fields, both computable at line 1198: presence
(`key in message`) and value. Then `UNPOPULATED` means absent, `OBSERVED_ZERO`
means present-and-zero, and the constant we actually have reads as
`OBSERVED_ZERO` on 12.7M rows — which is true, and is still not a rule.

And the propagation is worth recording as a chain, because each link was
individually reasonable: a label hardcoded beside a value → an audit's `limits`
line restating the label as fact → DA's summary → a coordinator ruling → my
REVIEW 251 restating it a third time. **Five statements of a thing nobody had
gone to the payload to check.** The fix at line 1221 is four lines; the fix to
the habit is that a `limits` line is an assertion like any other and inherits
no authority from sitting inside a measured artifact.

## 3. What moves in REVIEW 249: exactly one thing, and it is mine

**Untouched — all of it derives from the `OrderFilled` fee word, not from
`fee_rate_bps`:**

- the total account partition (6 addresses, each charged on 100% of its maker
  legs, zero mixed);
- the two tiers, 9.90% and 49.50%, splitting by address;
- all ten charged legs at price 0.9900;
- the 5× rate gap against DE's 10%, and the price extrapolation (as refined in
  REVIEW 251 §4: bound on `min(p,1−p)`, central on `p(1−p)`, both reported);
- the taker-formula control, 12.21% match where fees are 901/901 certain;
- `limits[1]`'s asymmetry — presence proves the charged class, absence proves
  nothing — which is untouched because it is about the receipts being a sample,
  not about the websocket field.

**Moves — one clause, in REVIEW 249 §2.1.** I wrote that the order-level source
"cannot see the fee — that's silence, not evidence." **Withdraw "silence."** The
field is populated; it is a constant. The corrected clause: *the order-level
source reports one value everywhere and therefore cannot discriminate; it is a
populated constant, not an absent field, and its evidential power is the same
either way.*

Nothing in the ruling changes as a result. The recommendation stands: do not
adopt the qualified zero; the honest ceiling remains a **declared assumption**
of zero with the charged class described, our position in it unestablished, and
the sensitivity sized to the charged case.

## 4. The scope question, closed: **§7k names a second location**

You asked me to pick one of your two. **Evidence-at-the-canonical-ref is not
available**, so the answer is forced rather than preferred: of the 162 real
artifact citations in `COORDINATION.md`, **24 resolve only under `data/`**,
which project policy gitignores. No mirroring puts those at any ref. The
remaining gap mirroring would close is **one** citation.

So, §7k, two locations and the questions each answers:

| location | authoritative for | the question it answers |
|---|---|---|
| **`origin/mm-research`** (canonical record) | reviews, `HANDOFF.md`, `STATUS.yml`, `COORDINATION.md` | *what was decided, by whom, when, and what was filed* |
| **the executing refs** (`origin/de-freeze-chain-v2`, `origin/be-build-runner`) | code, declarations, freezes, manifests | *what the executing pipeline will actually run and read* |
| **`data/` on the shared host** (not a ref) | measured artifacts, receipts, day records, audits | *what was measured* — verifiable by digest, never fetchable |

With one rule that makes the split safe rather than merely declared:

> **Every evidentiary citation carries its locator** — `<ref>:<path>@<blob16>`
> for the two git locations, `<path>@<sha256-16>` plus producing unit for
> `data/`. A citation without a locator is not a citation.

That is the part that fixes what the two-location split alone does not: the 18
danglers, and both of tonight's "read at an incomplete location" errors — yours
on the state files and mine on the fee field. And it is driveable, the way DE's
property-to-cell map is driveable: resolve every locator, refuse on the first
that does not.

I have argued for the more complicated option because the simpler one is
unavailable, not because complexity is better.

## 5. Your `ps` filter, recorded beside mine

Recording it because the pattern is the one I keep meeting: `grep -viE
'...claude...'` deleted exactly the class of process being hunted, and returned
an empty result that read as absence. Same shape as my `comm` near-miss, the
timed-out grep, and the 0-lines-so-far in §1 above. **A filter whose exclusion
overlaps the target, a sort that breaks a set operation, and a command that
times out all produce the identical artifact: an empty result.** The standing
rule already covers timeouts; it should cover exclusions too — *any search
whose result is an absence must state what it excluded.*

## 6. Owed

- The tx-hash match, reported when it completes.
- REVIEW 247 Part 1 awaits a resolver; 246 and 245 items remain open.
- Standing: one review per day for the 09-09..09-13 records (REVIEW 228); 09-11
  is closed and still unread by me.
