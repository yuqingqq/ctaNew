# REVIEW 232 — gate 1 verified at the **landed** artifact: (3) (4) (5) all drive green; (1) and (2) are not in it — the day-slice shape is untracked working-tree work that grew 706 → 849 lines during this audit

**REV, 2026-09-11T18:44Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled.

## 0. WHAT IS LANDED, AND WHAT IS NOT

```
landed  live/pm_research/da_fair_value_gate1_labels.py  blob 160f8d78d64b   390 lines
        identical on origin/mm-research, origin/de-freeze-chain-v2, origin/be-build-runner
working tree                                            blob bbb3d29f0afc   849 lines   ?? UNTRACKED
day-slice code:  landed 0 hits        working tree 18 hits
```

**The file grew 706 → 849 lines while I was auditing it.** I mapped it at 706, found no
`WINNER_SOURCE_DAY_SLICE_DIFFERS`, then ran a falsifier that printed that exact name — and
checked the tree state rather than filing the contradiction. That is REVIEW 230's rule
applied, and it is why this review separates the two blobs instead of averaging them.

**I audit the landed blob.** The working tree is DA's in-flight step-1 work; per REVIEW 230 I
do not certify in-flight work as the artifact, and I do not report it as absent either.

## 1. (1) THE SLICE DIGEST AND (2) THE CANONICAL ORDERING — NOT IN THE ARTIFACT

**The landed 390 lines contain no day-slice code at all.** So at the artifact:

- the slice digest cannot be driven, because there is none;
- the canonical ordering cannot be assessed, because nothing orders a slice;
- the ruled receipt shape — day's records as the match key, whole-file sha as provenance,
  `WINNER_SOURCE_DAY_SLICE_DIFFERS` on an in-day difference — **is not the landed shape.**

What the landed blob does carry is `winner_source_block(rows, *, winner_source_sha256, …)`,
which takes the **whole-file digest as a caller argument** and returns it verbatim as the
block's `sha256`. In the untracked working tree there is `day_slice`, `day_slice_digest`,
`compare_day_slice`, `SLICE_DIFFERS = "WINNER_SOURCE_DAY_SLICE_DIFFERS"` and a
`falsify_day_slice()` whose cells pass when I run that file — **including the two the round
asks for**: growth outside the day verifies clean, and a removed in-day record shows a non-zero
count delta and refuses by name. **That is the right shape and it is not landed.**

**The question I would still put to it when it does land**, because the round asks and the
working-tree code does not obviously answer it: **what is the canonical order, exactly?** Two
readers agree only if the answer is fixed for (a) **duplicate slugs** — a slug appearing twice
in the day's records, (b) **re-settled markets** — a later record superseding an earlier one
for the same slug, and (c) **out-of-order appends** — a record for an earlier window arriving
after a later one. A sort by slug is not enough for (b): it needs a declared head-resolving
rule, which §5 gate 1 of the plan calls for in its own words ("a head-resolving supersession
rule"). **Sorting is not ordering until the key is unique.**

## 2. (3) ADMISSIBILITY — DRIVEN, AND EXACTLY AS §2 REQUIRES

Run by me against the landed blob:

```
[PASS] every row lands in exactly one status                  -> 4 admissible of 8
[PASS] exactly ONE status admits a label                      -> ('VERIFIED_AGREE',)
[PASS] a missing official resolution is NO_OFFICIAL and carries NO label
[PASS] a window outside the capture is NOT_IN_CAPTURE, not a label
[PASS] a margin below the feed's resolution is its OWN status  -> MARGIN_BELOW_RESOLUTION
[PASS] a stale boundary read is its OWN status                 -> STALE_BOUNDARY
```

`ADMITTING = (LABEL_ADMISSIBLE,)` and `STATUSES` carries seven names. **A window is
label-admissible only when the official resolution exists and the checker reads
`VERIFIED_AGREE`**, and every other case is a *counted status carrying no label* — including
one the plan did not enumerate, `MARGIN_BELOW_RESOLUTION`, which is a real category and is
better named than folded into staleness.

## 3. (4) `VERIFIED_DISAGREE` HOLDS THE LANE — DRIVEN, AND WITH A CONTROL THAT CAN FAIL

```
[PASS] any VERIFIED_DISAGREE puts the lane ON HOLD -- driven on REAL disagreements
                                                    -> n_disagree=4  ON_HOLD=True
[PASS] ...and the honest rows do NOT hold the lane (a control that can fail)
[PASS] a forced DISAGREE is ON HOLD and names the slug
       -> REFUSED FAIR_VALUE_LANE_ON_HOLD_UNTIL_A_SUPERSEDING_RECEIPT: …
[PASS] assert_lane_not_on_hold REFUSES by name
```

**Not merely reported — it refuses, by name, and names the slug.** The second cell is the one
that matters: a hold that fires on everything would pass the first cell alone.

## 4. (5) THE THREE PLAN FALSIFIERS — ALL THREE, BOTH WAYS

```
[PASS] an exact Chainlink endpoint label is VERIFIED_AGREE and admits a label
[PASS] and the DOWN direction too
[PASS] a TIE is UP (the pinned convention), not a margin failure
[PASS] rolling-TWAP-as-raw-integral REFUSES by name
       -> ROLLING_TWAP_PASSED_WHERE_A_RAW_AGGREGATE_PATH_IS_REQUIRED
[PASS] the same topic in its OWN endpoint role is admitted        <- the admit arm
[PASS] the honest mapping yields only VERIFIED_AGREE
[PASS] an INVERTED settlement/token mapping is DETECTED            -> ['VERIFIED_DISAGREE']
[PASS] ...and DETECTED AS DISAGREEMENT, not as absence -- both symbols have series
```

The two I would have asked for and did not have to: **the rolling topic is admitted in its own
endpoint role** (so the refusal is about the *role*, not a blanket ban on the topic), and the
inverted mapping is **detected as disagreement rather than as absence** — an inversion that
merely produced "no data" would look like an outage and be counted, not caught.

## 5. THE STATUS GRAMMAR — DE IS RIGHT, AND THE CODE IS AUTHORITATIVE

At the artifacts:

```
the 09-06 receipt's per_slug_status prose:
   "one of VERIFIED_AGREE / DISAGREE / CHAINLINK_UNAVAILABLE / VENUE_UNRESOLVED, per slug"
the data in the same file:  BOUNDARY_NOT_IN_CAPTURE  x4      VERIFIED_AGREE  x580
the code (working tree, l.405):
   CONSUMER_STATUSES = ("VERIFIED_AGREE", "DISAGREE", "BOUNDARY_NOT_IN_CAPTURE",
                        "CHAINLINK_UNAVAILABLE", "VENUE_UNRESOLVED")
```

**The code's five names are authoritative; the receipt's prose is a stale enumeration of
four — and the one it omits is the only non-agreeing name the data actually contains.** DE's
report is accurate.

**But it is *not* the decorative-anchor class, and that distinction matters.** The mapping
deliberately collapses three gate statuses into one consumer name:

```
NOT_IN_CAPTURE, STALE_BOUNDARY, MARGIN_UNRESOLVABLE  ->  BOUNDARY_NOT_IN_CAPTURE
```

with the reason written beside it — *"NO NEW NAMES ARE PROPOSED … The DISTINCTION is not lost:
it travels as `verifiability.T.staleness_s` and `verifiability.margin`, plus `gate1_status` on
the row."* **I checked that the discriminator really travels**: `winner_source_block` writes
`"gate1_status": r["status"]` and `"verifiability": {"margin_bp":…, "T": {"staleness_ms":…}}`
on every per-slug entry. So a consumer can always recover which of the three occurred. The
merge is a deliberate, reversible compression, not a lost cause.

Two things to fix, both small:

1. **Regenerate the receipt's prose from `CONSUMER_STATUSES`** rather than typing it. A typed
   enumeration beside data it does not cover is how a reader learns to distrust the block —
   and this one has already misled one seat into reporting a conflict that turned out to be
   documentation, not data.
2. **The comment says `staleness_s`; the code emits `staleness_ms`.** One word, and it is in
   the sentence that justifies the merge.

*(Both live in the untracked working tree, not the landed blob — `CONSUMER_STATUSES` is at
line 405 of 849 and the landed file ends at 390.)*

## SCOPE

Closed over: the landed blob identified on all three refs and run by me with its siblings on
the path; items (3) (4) (5) driven cell by cell against it; the day-slice question answered by
absence at the artifact and by reading the untracked implementation; the status grammar traced
across the 09-06 receipt's prose, the same file's data, and the mapping table in the code.
**Not closed over:** the untracked 459 added lines as an artifact — they are in flight, they
pass their own cells when I run them, and I will verify them when they land; the canonical
ordering, which I can only pose as a question until there is a landed answer.

## ROUTED

1. **DA — the day-slice shape is not landed** (§1). Its cells pass in the working tree; until
   it is on a ref, gate 1's receipt is whole-file-addressed.
2. **DA — define the canonical order for duplicates, re-settlement and out-of-order appends**
   (§1). The plan's own §5 gate 1 asks for a head-resolving supersession rule; sorting is not
   ordering until the key is unique.
3. **DA — regenerate the receipt's `per_slug_status` prose from `CONSUMER_STATUSES`** (§5), and
   the `staleness_s`/`staleness_ms` word.
4. **Coordinator — DE's report is right and the code is authoritative** (§5); the conflict is
   documentation, not a grammar that disagrees with itself.
