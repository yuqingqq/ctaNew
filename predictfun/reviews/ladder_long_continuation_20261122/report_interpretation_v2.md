# LONG-CONTINUATION-PAIRED-VALUE-v1 — superseding interpretation (v2)

2026-11-22, session aea5bfb9.
- **Supersedes:** the wording of `report.md` per review `d799cb77` findings 1–3.
- **Unchanged:** the frozen receipt (`7407bdf1` / `6b806c19`), its 64 Y labels and the computed reading `reliable_and_varied`. `report.md` stays as provenance.
- **Source of every number below:** the frozen saved-data follow-up PAIRED-TURNOVER-ATTRIBUTION-v1 (`../ladder_turnover_attribution_20261122/`, frozen `db044afb`), computed and checked.

## 1. Day signs (replaces `report.md:38, :81`)

Computed (`audit.json` `day_signs`):
- **REVIEW `NEGATIVE_EVERY_DAY`:** −544 / −1,897 / −2,563 / −2,696 c, Oct 24–27.
- **ADMISSION `MIXED`:** +2,491 / +4,547 / −901 / −1,898 c.

The sentence "neither kind has one sign on every day" was wrong.

## 2. "A wash" and "most Y flows through other orders" (replaces `report.md:79–83`)

Equal counts, a median of 0 and a pooled net do not establish equivalence. The sample mixes two actions over overlapping paths, and both per-pair components are large and offset each other: Σ|Y_focal| is 47.6k vs Σ|Y_other| 34.5k for REVIEW (review d799cb77's computed table).

Correct wording: "Coupled, offsetting contributions; on these sampled days REVIEW favours the alternative in aggregate (on every day), ADMISSION favours posting in aggregate (mixed days)."

The attribution explains these aggregates:
- **REVIEW's deficit is completed turnover:** ΔMatched −7.7k c vs ΔResidual +0.01k.
- **ADMISSION's gain is residual exposure:** +4.1k, mostly on Oct 25.
- The top 5 pairs carry 86 % / 97 % of Σ|Y|.

## 3. The sequence axis (replaces the reading of `report.md:40–47, :85`)

The END-signed sequence values every fill of one fixed-price generation at the same END mark, so its sign cannot reverse between that order's fills. Opposite-side spread capture lies outside it by construction.

The 0 count of `early_loss_later_benefit` is correct for the declared predicate. It is **no evidence** about early adverse fills followed by profitable two-sided turnover.

The selftest's positive fixture gave one gid two prices, which no order in this engine can have. The follow-up uses only realizable fixed-price fixtures.

The round-trip view replaces it: many own-END-negative focal fills close profitably, and many own-END-positive ones sit on losing round trips. The queue mechanism stays **unidentified**.

## 4. Replacement delay (replaces `report.md:62`)

`first_rep_delay` skipped a WAIT-arm replacement that reused the factual numeric gid. Corrected from the saved rows:
- 12 ADMISSION alternative arms are at **+50 ms**;
- the alternative-arm mean over the 63 arms with a replacement is **869.8 ms**, not 1,361.

Y, queues, fills and lineage are unaffected. FIFO holding times are now measured from the actual acquisition; carried lots are no longer reset at the fork.

## 5. Unchanged

The reading, every Y, the gates and the "does not show" list stand.
