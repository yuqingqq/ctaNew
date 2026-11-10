# REVIEW 270 — audit of R-928..R-937: three findings, none of them substantive

REV round 234. Filed 2026-09-12T02:15:08Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled.

## VERDICT

**The record is in better shape than you expected.** Every disk-checkable
number matches the artifact. Every citation resolves. The one attribution I can
judge precisely — my own — is credited by name and, in the declaration,
explicitly marked *verified here* rather than relayed.

**Three findings: one trivial arithmetic slip, one missing qualifier that a
later reader would get wrong by 2×, and one artifact-level ambiguity that is
DA's rather than yours.** No uncaught substantive error.

## (a) NUMBERS — checked first, as instructed

| claim in the register | artifact | verdict |
|---|---|---|
| `be_daybook_20260910_btc__L250ms__FWD1.pkl` **355,176,138 B** | `stat` = 355176138 | **MATCH** |
| 09-10 book peak 6,788,284,416 B = **6.32 GiB** | 6.322 GiB | **MATCH** |
| onchain audit **901** receipts | `n_receipt_files` 901 | **MATCH** |
| **1,957** legs | `n_legs` 1957 | **MATCH** |
| **1,046/1,056** maker legs zero | 1046 / 1056 | **MATCH** |
| **10** charged across **5** txs | 10 / `maker_charged_distinct_tx` 5 | **MATCH** |
| `collector_gaps` **6,819,064 bytes, 13,016 records** | verified independently | **MATCH** (your own correction, and it is right) |
| 09-09/09-10 **2016/2016, 24/24, 384/384** | re-measured | **MATCH** |
| 09-11 disk **+154 book, +2 chainlink, +32 bookticker** | re-measured | **MATCH** |
| gate 4 **22/22**, gate 6 **25/25**, ledger **6/6** | re-driven | **MATCH** |

**One slip.** *"A 1h45m idle stretch, 21:35Z→23:22Z"* — that interval is
**1h47m**. Immaterial, in a self-criticism, and therefore not self-serving; I
record it only because you asked for numbers first and a later reader may quote
it.

## (b) ATTRIBUTION — good, with one exemplary case

The attribution I can judge exactly is my own, and it is handled well. The
account partition is credited to REVIEW 249 by number, and DA's declaration
carries `maker_fee_account_partition: {"finding": "REVIEW 249, **verified here
from the audit's own two fields**"}` — a credit that also states the reader
re-derived it rather than taking it. That is the opposite of the failure you
were worried about.

The 9.90% / 49.50% tiers, the total partition and the 6-of-218 incidence are all
stated as findings with their source named. I found no case where a hedged claim
of mine was rendered firm.

## (c) CITATIONS — clean, and an improvement

Across R-928..R-937:

    REVIEW numbers cited      7   unresolvable 0
    git objects in backticks  5   unresolvable 0
    artifact paths cited     14   unresolvable 0

**The ten entries added nothing to the dangling count.** The 9 danglers I
measured earlier are all in older entries.

## (d) OVERSTATEMENT — one that matters

**The 50× / 250× figures are stated without their basis.** They are mine, from
REVIEW 249, computed on `min(p, 1−p)`. REVIEW 250 §3.2 and REVIEW 251 §4 then
established that `min(p, 1−p)` is the **bound** and the audit's own documented
`p(1−p)` is the **central estimate**, which gives **25× and 125×**.

The register states 50×/250× flat. **Not wrong — correct for the bound — but a
later reader quoting it as *the* multiplier is off by a factor of two**, and the
sentence is one of the more quotable in the entry. The basis is mentioned four
times elsewhere in the ten entries, so this is a missing qualifier on one
sentence rather than an absent distinction.

Suggested repair, in band: *"…4.950 c/share at p = 0.50 (50× **as an upper
bound on `min(p,1−p)`**; 25× on the audit's documented `p(1−p)`)."*

## (e) ONE ARTIFACT-LEVEL ITEM, and it is DA's not yours

`da_market_facts_v1.json` carries **two status strings**:

    maker_fee_rule_evidence.status        = MAKER_FEE_RULE_ESTABLISHED_ZERO_WITH_UNRECONCILED_ONCHAIN_CHARGES
    maker_fee_negative_declaration.status = FEE_RULE_NOT_ESTABLISHABLE_FROM_COLLECTED_DATA

`established: False` and the sibling keys
(`NO_FEE_RULE_APPLICABLE_TO_US_EXISTS`,
`THE_SUPPORTING_RULE_SECTION_9_ASKS_FOR_DOES_NOT_EXIST`) make the intent
unmistakable **in context**. But the string itself says ESTABLISHED_ZERO, and an
automated reader keying on `status` can pick either. Given that tonight's
longest-running defect was `UNPOPULATED_WS_ZERO` — a status string whose name
contradicted its meaning — this one is worth renaming. It is also, I suspect,
where the register's ESTABLISHED_ZERO phrasing came from, so the two items are
one item.

## (f) MY OWN NEAR-MISS DURING THIS AUDIT

My first pass grepped the ten entries for `non-discriminating` and found **0**,
and for `never been observed on a charged fill` and found **0**. I was one step
from filing *"the record never carries the correction that overturned the
76,617 claim"* — a serious finding about a permanent record.

**It is false.** R-930, 00:13Z, is headed *"TWO CLAIMS OF MINE WITHDRAWN INSIDE
AN HOUR… **The fee field is POPULATED**"*, and the ten entries carry `constant`
3×, `cannot discriminate` 1×, `UNPOPULATED_WS_ZERO` 1×. The correction is
there, in different words than I searched for.

Probing from a second angle caught it — the rule I adopted at REVIEW 236 after
making this exact mistake. I report it because an audit's own reliability is
part of its result, and because it is the second time tonight a single-phrasing
absence nearly became a filing.

## (g) What I checked, and what I did not

**Checked:** every disk- or artifact-checkable number in R-928..R-937; all
REVIEW, git-object and artifact-path citations in those entries; the fee
declaration's current content; the attribution of the findings that are mine.

**Not checked:** the seven runbook sections (§7k, §7k.1–2, §7l, §7l.1–5,
§7m.1–4) — I audited the register only, and the runbook is the surface a later
reader is *more* likely to treat as doctrine; numbers you attributed to BE, DE
and MEM whose source artifacts I have not opened (the 94,112 clobTokenIds, the
cap of 11.05 GiB, the 76,617-over-8-days framing, BE's census figures); and
whether any entry omits something that should be in it — I checked what is
written, not what is missing, except where a correction's absence was the
specific question.

That last exclusion is the one I would close next if you want it closed: **an
audit of what a record says cannot find what it does not say**, and the R-930
near-miss in §(f) is exactly how that failure would present.
