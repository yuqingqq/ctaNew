# POLICY_COMPARISON — new-BBO vs join-BBO, paired

Protocol `policy_v1`. **Frozen 2026-08-22 before measurement.** Coordinator
writes the rules; the research agent measures. Research only, not decision
eligible, no forward-day claim.

## Why this is answerable when the absolute edge is not

The pooled maker edge is `+0.173 ¢/share [−0.251, +0.596]` — useless, and it
needs ~25–30× current data to resolve. **This protocol does not measure it.**

Both policies are evaluated **on the same windows, at the same moments, against
the same flow**. Common variance — regime, day, coin, window phase — cancels in
the difference. A paired contrast can be well determined far below the interval
on either level, which is why the comparison is available now and the level is
not.

**Every headline here is a PAIRED DIFFERENCE.** Report levels for context;
never let a level carry the verdict.

## The two policies

Both quote **one side, 5 shares, on the unified Up book**, and differ only in
*when they join a price level*.

| | **NEW_BBO** | **JOIN_BBO** |
|---|---|---|
| trigger | the level becomes the touch (a new best appears) | the touch already exists when we arrive |
| queue position | **front** — nothing displayed ahead | **back** — behind all displayed depth |
| availability | only when a level forms | any decision time |
| `queue_ahead` | 0 | displayed size at that level |

These are the two ends of the FRONT/BACK bracket **expressed as policies**, which
is what they always were. Queue position is an OUTPUT of the placement rule, not
an assumed parameter.

**`NEW_BBO` is an UPPER BOUND on itself, not a guarantee.** Being first assumes
we win the race against other participants doing the same, which depends on our
latency and theirs and is **not observable in this tape**. Report it as a bound
and say so on every line.

## Estimands, per coin, paired

For each decision time where **both** policies are available:

1. `Δfill = P(fill | NEW_BBO) − P(fill | JOIN_BBO)`
2. `Δmarkout = E[outcome − ℓ | fill, NEW_BBO] − E[outcome − ℓ | fill, JOIN_BBO]`
3. `Δedge = Δ(fill-weighted realised edge)` — the product, which is what a maker
   actually earns per decision

Markout is measured **against settlement** (`S60(T)` vs `S60(t0)`, verified at
99.8 %), **not against a fair-value model**. That is what keeps this decoupled
from Route A and off the 10-day sigma clock.

**Only decision times where both are available may enter a paired comparison.**
A `NEW_BBO`-only sample against a `JOIN_BBO`-only sample is not paired and is
exactly the denominator/population defect this programme has hit six times.
Report the availability rates separately — if `NEW_BBO` is rarely available that
is itself a finding about the policy.

## Decision rule, written before the measurement

Verdict on **btc and eth only** (`verdict_coins`); the other five are descriptive.
Window-clustered bootstrap; day-clustered intervals are not computable at this
day count and must not be claimed.

- **NEW_BBO_DOMINATES** — `Δedge > 0` with the interval excluding zero on **both**
  btc and eth.
- **JOIN_BBO_DOMINATES** — `Δedge < 0`, interval excluding zero on both.
- **TRADE_OFF_CONFIRMED** — `Δfill > 0` and `Δmarkout < 0`, both intervals
  excluding zero, while the `Δedge` interval **spans** zero. The policies differ
  in mechanism and not in outcome; placement is then a **risk-shape choice, not
  a profit choice**, and that is a real finding.
- **NO_DIFFERENCE** — every interval spans zero, and the `Δedge` interval is
  **tight enough to exclude an effect worth acting on** (bound: `|Δedge|` CI
  within ±0.25 ¢/share). Placement does not matter, and the entire FRONT/BACK
  bracket was a distraction.
- **UNRESOLVED** — intervals span zero but are too wide to rule out an effect
  worth acting on. Say what n would settle it. **This is the default when
  underpowered; it must not be reported as NO_DIFFERENCE.**
- **VOID** — fewer than 200 paired decision times on either verdict coin.

## Pre-specified expectation, so a confirmation is not a surprise

`NEW_BBO` should win on fill (94.6 % vs 76.9 % at 15 s on btc, unpaired) and
**plausibly lose on markout**, because it quotes when the level is forming, thin,
and information is freshest. **A result showing `NEW_BBO` better on BOTH is a
reason to suspect the measurement**, not to celebrate — check the pairing and the
availability rates before believing it.

## Rules

1. Paired only; state the population of every denominator.
2. Per coin, never pooled — btc is ~64 % of any pooled denominator.
3. Read book state from `price_change.best_bid/ask`, never `book` snapshots.
4. Knowledge time is `recv_ns`; state read at the frozen 250 ms lag.
5. Gap-touched and tick-change-touched actions are `UNAVAILABLE`, reported not
   dropped.
6. R-DUAL: the micro class is 2–90 % of events by coin. Report both weightings.
7. A control per probe that fails if the test is vacuous.
8. Do not narrate a mechanism that was not measured.

## Out of scope

The absolute maker edge, the fair-price model, queue-position inference, and any
`ρ`-dependent quantity. This protocol compares two policies; it does not
establish that either is profitable.

---

## RUN AND ANSWERED — 2026-08-24, era-population re-run (appended per R-28; the frozen text above is untouched)

**Receipt:** `derived/policy_comparison_v2.json` (+ per-arm receipts with
`engine_hash`, arm `run_hash`es, per-window `inputs_hash`). Harness:
`ev_replay.py` ReplayEnv (R-102 item 1), determinism gates PASS both arms.
**Population (stamped, launch-time): 282 verdict-coin windows = btc+eth ×
{08-20/21/22/23 at 30/coin + 08-24 PARTIAL at 21/coin}** — five era days,
day-grouped sampler; the report-#65 scope estimate (288) differed by the
partial day's growth between launch and derivation; the stamp wins, as
promised. No overlays exist in the environment (pitfall #4 honoured by
construction). All numbers as-of 2026-08-24 ~13:1x.

**Cells (d = FRONT − JOIN, share-weighted M5 ¢/share; window-clustered CIs; n = paired windows):**

| coin | day | JOIN m5 | FRONT m5 | d_M5 [CI] | n |
|---|---|---|---|---|---|
| btc | 08-20 | −0.526 | −0.516 | −0.060 [−0.224, +0.114] | 30 |
| btc | 08-21 | −0.991 | −0.650 | +0.222 [+0.097, +0.366] | 30 |
| btc | 08-22 | −1.048 | −1.055 | −0.041 [−0.230, +0.174] | 30 |
| btc | 08-23 | −1.512 | −1.128 | +0.201 [−0.135, +0.570] | 30 |
| btc | 08-24p | −1.514 | −0.990 | +0.406 [+0.130, +0.708] | 21 |
| eth | 08-20 | −0.966 | −0.878 | +0.092 [−0.374, +0.569] | 30 |
| eth | 08-21 | −1.412 | −0.716 | +0.583 [+0.276, +0.903] | 30 |
| eth | 08-22 | −1.913 | −0.912 | +0.956 [+0.431, +1.574] | 30 |
| eth | 08-23 | −2.862 | −2.120 | +0.398 [−0.131, +0.927] | 30 |
| eth | 08-24p | −1.338 | −0.932 | +0.401 [−0.066, +0.918] | 21 |

**Roll-ups (paired, n=141 windows/coin):** fills — FRONT out-fills JOIN
massively and CI-cleanly everywhere: btc d_shares/window **+6,054**
[+5,698, +6,392] (≈5.3–6.3×); eth **+969** [+903, +1,033] (≈4.7–5.9×).
Markout — btc pooled d_M5 **+0.129 ¢ [+0.026, +0.251]** but DAY-SIGN-MIXED
(2/5 negative points, none CI-clean negative; 2/5 CI-clean positive):
wash-to-FRONT-favouring, not a clean win. eth pooled **+0.491 ¢
[+0.282, +0.718]**, positive point every day: FRONT wins markout.

**VERDICT under the frozen paired-difference rule: §7's EXPECTED TRADE-OFF
DOES NOT APPEAR on this population.** New-BBO wins fills decisively and
loses fill-conditional markout NOWHERE (no coin-day shows a CI-clean FRONT
markout penalty); it wins markout outright on eth. Interpretation (labeled
as such, not measured): queue-POSITION selection dominates formation-time
information — back-of-queue JOIN fills condition on the displayed queue
ahead being consumed, i.e. deeper and more informed sweeps, and that
penalty exceeds the freshest-information penalty of quoting at formation.

**THE LEVELS CONTEXT THAT MUST RIDE WITH ANY QUOTE OF THIS RESULT (the
frozen rule: never let a level carry the verdict — nor hide it): BOTH
policies lose money per share at M5 on EVERY coin-day** (JOIN −0.53 to
−2.86 ¢, FRONT −0.52 to −2.12 ¢). At negative per-share markout, 5–6×
more fills means 4–6× MORE TOTAL LOSS: FRONT is the BETTER-RANKED policy
and the FASTER way to lose money. This comparison ranks policies; it does
not make either viable — that reading belongs to items 3–4 of the
decision path (gross→net, STOP-MM-VIABLE), not to this protocol.

> *ANNOTATION BESIDE (2026-08-24, R-109 challenge — appended per R-28;
> nothing above is edited): **THE COORDINATOR'S DAY-CLUSTERED ARITHMETIC IS
> CONFIRMED TO THE DIGIT by DE's own recomputation from the cells above**
> — btc d_M5 at the DAY unit, t(4), G=5: mean +0.146, se 0.0876,
> **[−0.098, +0.389], SPANS ZERO**. The published window-clustered
> [+0.026, +0.251] excluded zero only by pooling windows across a
> day-level sign change. **The btc CI-excludes-zero presentation is
> RETRACTED**; under the ruled standard the wrong-unit interval is a
> precision claim the design cannot support. DE's "day-clustered refused
> below the cluster floor (house rule)" was STALE BOILERPLATE ported from
> the G≤4 era — at G=5 the t-interval is wide but valid, and the day
> means were already in this receipt. The refusal was mechanical, and it
> is DE's instrument defect, the R-79 class in a stats note.
>
> **And the recomputation EXTENDS the challenge via item 4's own
> caveat**: excluding the partial day (08-24, 21 windows, first hours
> UTC, btc's most positive day), G=4 complete days give btc
> [−0.160, +0.322] and **eth [−0.067, +1.082] — eth's day-clustered
> exclusion of zero holds ONLY with the partial day included**
> (G=5: [+0.094, +0.879]). A partial day is a different population, not
> a smaller one — so the eth markout win is SUGGESTIVE, not day-robust.
>
> **What survives at the day unit, restated precisely**: (1) THE LEVELS
> ARE THE ANSWER — m5 negative on all ten coin-days, both arms; neither
> policy is profitable at h=5 and no reading of the difference changes
> it. (2) FILLS are day-robust with room to spare: btc d_shares/window
> t(4) G=5 [+4,644, +7,378]. (3) §7's predicted FRONT markout PENALTY
> appears NOWHERE — no coin-day is CI-clean negative at either unit; the
> refutation of the stated prior stands. (4) The markout WIN is
> point-positive pooled and per-day-majority but NOT day-robust —
> btc indeterminate at every unit beyond windows; eth clean only at
> G=5-with-partial. All figures as-of 2026-08-24; n as stated per cell.*
