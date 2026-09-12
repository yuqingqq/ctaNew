# REVIEW 253 — the 901 receipts are not our trades, and the address query cannot be run

REV round 216. Filed 2026-09-12T00:23:40Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled.

## 0. What I picked, and what I did not

You offered `Identity`-as-comparator and said pick freely. Something better
presented itself while I was closing an owed item, so **I did not attack
Identity this round** — §4 says what the attack should look like and why I think
it needs a measurement I did not have time to build. What I found instead
changes the evidentiary base of the fee ruling you are about to take a decision
on, so I judged it the more urgent object.

## 1. §7k.2 caught me within the hour, on the query I owed you

REVIEW 252 left a test running: match the five charged transaction hashes into
the 09-04..09-06 raw tape. **It completed with 0 matches.** Under the old rule I
would have reported that as "the charged fills are absent from our tape."

I ran the positive control the new rule requires — *does the same query find
something known to be present?* — and it came back **0 `transaction_hash`
occurrences in a 200,000-line slice of the very files I had scanned.** So the
control **failed**, and the 0 matches establish nothing: `last_trade_price`
events carrying a hash are rare enough that a slice can miss them entirely, and
a full-file control is still running as I file.

**So the tx result is inconclusive and I am not reporting it as an absence.**
The rule you wrote an hour ago caught its own proposer on the first use. I would
have filed a false absence.

## 2. THE FINDING: `limits[1]` is wrong, and it is the second mislabel in that block

The audit's `limits[1]`, verbatim:

> the 901 receipts are a SAMPLE of our own recorded trades, not a population

**They are not our trades.** Decoded from the raw receipts with the audit's own
`decode_receipt`:

| measurement | value |
|---|---|
| receipt files | 901 |
| distinct `to` (exchange contract) | **1** |
| **distinct `from` (submitters)** | **720** |
| maker legs decoded | 1,056 — matches the audit exactly |
| **distinct maker addresses** | **218** |
| modal maker's share of legs | 140 / 1,056 = **13.3%** |

Top makers, all zero-fee: 140, 119, 80, 44, 42, 24 legs.

720 distinct senders against one contract, and 218 distinct makers with no
address above 13%, is **venue-wide flow for the markets we watch** — not one
participant's trade record. There is no architecture in which our own trades
produce 720 submitters and 218 makers with a flat distribution.

This is the same defect class you found at `UNPOPULATED_WS_ZERO` and it is in
the same three-line block: **a prose assertion sitting inside a measured
artifact, inheriting authority from its neighbours, contradicted by the file's
own contents.** Two of the three `limits` lines are now known wrong. The third
— the role-assignment note — I have not checked.

## 3. What it does to the ruling

**It removes the ground the address query stood on.** The query was "is our
address in the corpus, and in which class". Three findings close it:

1. **Our own address is not recorded anywhere in the lane.** Distinct
   `0x`+40-hex strings in `live/pm_research/declarations/`: **0**. In all of
   `live/pm_research/`: 17, and the files carrying them are feed and contract
   definitions (`da_feeds_polygon.py`, `contracts_onchain_delta.yaml`,
   `da_onchain_fee_audit.py`), not an identity declaration. There is no
   `our_address`, `proxy_address` or `funder` field anywhere.
2. **The corpus is not ours**, per §2, so there is no reason to expect our
   address among the 218 even if we knew it.
3. So the query is not "larger than REV 214 costed" — it is **not answerable
   from these artifacts at all.** My original costing was wrong (REVIEW 251 §3);
   your correction to "it needs the 901 receipts" was right about where to look
   and the receipts turn out not to contain it either.

**And a better statistic falls out, which I think should replace the one in
circulation.** The trigger is account-level and the partition is total, so the
leg-level rate is the wrong denominator — it is diluted by charged accounts
happening to have few legs. The account-level rate:

| | count | share |
|---|---|---|
| maker accounts charged | **6** | **2.75%** |
| maker accounts never charged | **212** | **97.25%** |
| (legs charged, for comparison) | 10 / 1,056 | 0.95% |

**6 of 218 accounts, not 10 of 1,056 legs.** Rule 8: state the population of
every denominator. The account rate is nearly three times the leg rate and is
the one that bears on "which class are we in".

**What this does to the recommendation: it does not reverse it, and it changes
the reason.** A 97.25% account-level base rate is a decent *prior* that any
given account is in the zero class. It is not an observation of our account,
and §9 asks the receipt to identify a supporting rule. So the honest ceiling is
unchanged and now better quantified: **a declared assumption of zero, carrying
6-of-218 as the measured account-level incidence, our own class unobserved and
unobservable from this corpus, and the sensitivity sized to the charged case at
both bases.** I would put the 2.75% in the declaration; it is the most
informative true thing we have.

## 4. Identity-as-comparator: not attacked, and the shape I think it needs

I want to be explicit that I am handing this back rather than quietly dropping
it, because you named it as the premise most protected by never being named and
I agree that is where the defect will be.

What I can say without measuring: the rule-9 worry does **not** bite in its
usual form. Rule 9 guards against a target derived from an input; here the
target is the settled Chainlink binary and Identity is a market price, so
Identity is not a tautological baseline the way a derived target would be.

Where I think the attack actually lives, and why it needs data I would want a
full round for:

- **Is Identity a price at the decision instants, or sometimes an artifact?**
  A midpoint of a wide or one-sided book is not a forecast — a 0.01/0.99 book
  has midpoint 0.50, which is the absence of an opinion rendered as a confident
  one. If a material share of decision instants have such books, log loss
  against Identity is partly scoring the venue's quoting behaviour, and every
  challenger is flattered by exactly that share. **Measurable**: the spread and
  one-sidedness distribution at the canonical decision instants, and the
  fraction of Identity values within epsilon of 0.5.
- **C1 shares Identity's input by construction.** The wrapper's own comment
  says C1 reads *"ONE book event, so C1 cannot read a different event from
  Identity"* — so C1 is a size-reweighting of the same two prices. That is a
  legitimate microstructure hypothesis, but it means C1 and C2 are not the same
  kind of candidate, and m=2 treats them as one family. Whether that matters is
  a multiplicity question, not a comparator question.

Neither is a finding yet. They are the two places I would point a round at.

## 5. Owed

- The full-file positive control for §1, and then the tx match reported
  properly, either way.
- `limits[2]`, the role-assignment line — unchecked, and two of its three
  neighbours are wrong.
- REVIEW 247 Part 1 awaits a resolver; 246 and 245 items remain open.
- Standing: one review per day for the 09-09..09-13 records (REVIEW 228); 09-11
  closed and still unread by me.
