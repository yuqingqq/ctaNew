# REVIEW 143 — what the frozen theta now selects

**REV, 2026-09-09T15:20Z.** Read-only: no lock, no heavy unit, **no book unpickled** — every
number below comes from the point-estimate artifact, the EV21 receipt and the 4.2 MB decision
ledger, so none of it competed with DA's lock or DE's null. **Analysis only; nothing re-fitted.**

## THE NUMBERS FIRST

| | pre-fix | corrected (EV21) |
|---|---|---|
| generations | 313,114 | **313,140** |
| HAZARD decisions at the frozen theta | **1,398** | **14,893** |
| tail mass selected | **0.446 %** | **4.756 %** |
| **the quantile theta sits at** | **99.55th** | **95.24th** |

**Tail mass ×10.65.** Frozen theta `0.43525926488298716` selected the top 0.45 % of
generations and now selects the top 4.76 %.

**CONDVALUE, for completeness** (no pre-fix comparator was given me): theta
`0.32450609461933483` selects **19,523 of 313,140 generations = 6.235 %, the 93.77th
percentile** (25,628 of 350,472 rows, 7.312 %).

## (2) THE 10.7× IS BROAD, NOT A BAND — three ways of asking

**By score magnitude** — the extra decisions are NOT piled just above the threshold:

| within | HAZARD | CONDVALUE |
|---|---|---|
| theta + 0.001 | 79 (0.5 %) | 60 (0.2 %) |
| theta + 0.01 | 829 (5.0 %) | 642 (2.5 %) |
| theta + 0.05 | 3,743 (22.6 %) | 3,078 (12.0 %) |
| theta + 0.10 | 6,621 (40.0 %) | 5,571 (21.7 %) |

**Sixty per cent of HAZARD's admitted rows sit more than 0.10 above theta**, against a max of
15.02. **So theta has not landed on a spike — the score distribution has moved relative to it
across its whole range.** That matters: a band effect could be argued away as a boundary
artefact; this cannot.

**By slug:** **246 of 247 slugs** carry at least one decision. Top-10 share 13.2 % against
4.0 % uniform (×3.3); top-25 27.1 % against 10.1 % (×2.7). **Present nearly everywhere, mildly
concentrated.**

**By hour:** 23 of 24 hours; busiest/quietest **9.1×** (HAZARD), **15.4×** (CONDVALUE) — uneven,
but that is the shape of the day's activity rather than a threshold artefact.

## (3) THE SIZE OF WHAT A RE-FIT WOULD TRADE AWAY

To select **the same quantile** on the corrected book (top 0.446 %, 1,398 generations):

```
theta would be 1.055371754   against the frozen 0.435259265
              = +0.620112, i.e. +142.5 %
decisions      14,893 -> 1,398   (10.7x fewer)
```

**It is not a nudge: restoring the original operating point would take more than doubling
theta.** Scores are not probabilities — the maximum is 15.02 — so a theta above 1 is not
out of range, and the same number restores both the quantile and the count (they coincide
because the denominators barely moved).

**What it would do to the arm I did not estimate and will not**: the P&L consequence needs a
replay over the reduced decision set, and DE is running HAZARD's null on this very theta.
**What I can say is the population arithmetic**: the surviving 1,398 would be the
highest-scoring 9.4 % of today's decisions, and the other 13,495 — the ones carrying the
cancel decisions the arm is measured on — would not be taken.

## (4) THE RULE-11 QUESTION, WITH ITS SCOPE STATED

**The ruling froze a NUMBER; what a fit calibrates is an OPERATING POINT. Those coincided
when the ruling was made and they no longer do.** On 09-03 the same number now selects the
95.24th percentile where it selected the 99.55th. So for comparability of the *threshold*
the freeze is doing exactly its job, and for comparability of the *decision rate* it is not
— **and it is the second that the arm's economics are a function of.** That is a ruling, and
it is the user's; the measurement above is what it should be ruled on.

**AND THE SCOPE LIMIT, because the honest version is narrower than the headline.** The
99.55th percentile above is the quantile the frozen theta selected **on 09-03 under the
DEFECTIVE per-generation scoring** — it is *not* the quantile the 08-29 fit selected on 08-29.
**I did not measure the 08-29 distribution**, so I cannot say by how much the frozen number has
drifted from *what the fit chose*; I can only say it has moved by 10.65× in tail mass between
the defective and corrected scorings of the same day. **The rule-11 statement that IS
supported: 08-29 is consumed as development evidence, so a re-fit cannot be validated there,
and any re-fit would have to name the days it used and their seen/unseen status.**

## ROUTED

1. **Coordinator / user — the ruling question is which comparability was intended**: the
   threshold (frozen number) or the operating point (frozen quantile). The two have separated
   by 4.3 percentage points of tail mass on HAZARD.
2. **Whoever answers it — the 08-29 fit distribution is the missing measurement**, and it is
   the one that would say whether the frozen number has drifted from *the fit*, as opposed to
   from the defective scoring of 09-03. Cheap if that day's scores are on disk.
3. **Nothing re-fitted, nothing run heavy, DE's null undisturbed.**
