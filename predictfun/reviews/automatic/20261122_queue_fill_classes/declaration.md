# Queue-fill classes — independent review measurement

2026-11-22. USER requests classification of mixed worthwhile exposure,
unprofitable mixed exposure and purely bad fills. This implements the
classification portion of review amendments 3e84ca6d / 6d200317 as independent
review evidence; it does not edit the development session's in-progress code.

Freeze this declaration, classify.py and input_manifest.json in a review-only
commit before `run`. The `prepare` command hashes inputs only, not outcomes.
Pinned source receipt: 85127b3dfbdfb8689159fe8b9e76b447324b7cfc.

Population: every submitted generation of CURRENT_ASYNC_FAST10 and
FRONT_ASYNC_FAST10, separate arms, all 19 previously accepted complete markets
on consumed Oct 24–27. Anchor: first recorded live KEEP command of each gid;
effect = decision + 10 ms. All recorded fills at/after that effect form the
suffix. Earlier fills are a separate sunk/timing-constrained prefix. No KEEP
callback -> NO_OPPORTUNITY, never silently omitted. One row per generation.

Primary value: signed shares times (mid at fill+1s minus fill price), in cents.
Use the saved mid only when fill+1s <= END and the mid is finite. Let G be
positive suffix value, L the positive magnitude of negative suffix value.
The accounting tolerance is 1e-6 c. Mutually exclusive economic classes:

1. NO_OPPORTUNITY, if no live KEEP callback.
2. INCOMPLETE, if any suffix fill lacks its +1s value.
3. NO_SUFFIX_FILLS, if no suffix fills.
4. PURE_BAD, if L>tol and G<=tol.
5. GOOD_ONLY, if G>tol and L<=tol.
6. MIXED_COMPENSATED / MIXED_UNCOMPENSATED / MIXED_TIE, if both>tol,
   according to the sign of G-L relative to +/-tol.
7. NEAR_ZERO, the remaining complete cases.

Separate sequence field: NO_SUFFIX / INCOMPLETE / GOOD_ONLY / BAD_ONLY /
ZERO_ONLY / BAD_THEN_GOOD / GOOD_THEN_BAD / INTERLEAVED_OR_SIMULTANEOUS.
For sequence order use the exact signs of individual finite values and strict
timestamp separation. GOOD_THEN_BAD has an additional separable flag only
when a recorded KEEP's effect is after every good fill and at/before every
bad fill. Do not sum sequence categories with economic categories. Early
prefix and full-generation accounting remain available, including missing
marks. Economic classes describe the first anchor, not an optimal policy.

Require complete membership and per-market reproduction of accepted +1s and
END totals. Refusals, no fills, recorded removal, known closure and unknown
queue observations remain in the rows. Label continuation as recorded/truncated:
PURE_BAD means no positive in the observed suffix, not no future good ever.
Separate current queue and hypothetical front; no matched causal queue claim.

Outputs: one compressed generation CSV, class summary by arm/day/side and
pooled, sequence cross-tab, deterministic two-case-per-arm/class casebook
(smallest SHA256 of arm/day/market/gid), checks.json and result.json. No
interval, null, predictive-skill claim, model or policy selection. Descriptive
classification only; no new market tape, replay, fitting or untouched day.

Synthetic positive/known-bad controls cover all classes, incomplete marks,
request/effect boundaries, good/bad ordering, feasible intervening cuts,
and independent arithmetic/class verification. Match labels and totals at
the actual exported row, not a summary claim.

Bounded run through PF_SLOT=codex run_light.sh, 30-second wall timeout; this
independent classification including checks <=0.01 core-hours. Count against
the pending audit's 0.05h development allowance and cumulative 2.50h ceiling;
do not add an extra round budget. Development should reuse this evidence in
the same combined review. The full headroom/re-entry question is not answered
here. No fixes to another session's implementation.
