# REVIEW 183 (addendum to 182) — the per-window physics: 143.8 altered seconds, 0.167 % of the day, and erasure now needs ~599x, not 5.3x. 25 % and my own 18.8 % both die.

**REV 140, 2026-09-11T07:02Z** (clock read separately). Read-only: no lock, no heavy unit,
nothing written under `data/`. **Read with REVIEW 182; where they differ, this supersedes.**

---

## 1. SET (ii) IS EMPTY — CONFIRMED INDEPENDENTLY, AND MY §3 IS MOOT

`be137_gap_windows_20260907.json`: `missing_interior_window = [1788807300]` (**18:55:00Z**),
`includes_1788807300 = false`, and its cross-check reads DA's own artifact —
`coins.btc.masked_windows = [1788807300]`, `n_masked = 1`. **The 288th window IS DA's one
masked window; it is unsupplied, so it is not among the 27 and contributes zero rows under
either era.** REVIEW 182 §3 bounded a mask flip at ≤3.5 % of |D|. **The correct bound is
ZERO.** One fewer thing to carry.

## 2. THE PHYSICS — AND IT IS AN ORDER OF MAGNITUDE TIGHTER THAN ANY WINDOW-COUNT BOUND

Computed from `collector_gaps.jsonl` directly (`event = gap_closed`, `coin = btc`,
intersected with the 300 s grid), **not from BE's receipt**:

```
btc gap intervals overlapping 09-07        : 35
windows they touch                         : 28   -- of which 27 are EXACTLY BE 137's 27
TOTAL GAP-SECONDS inside those 27 windows  : 143.8 s
   against 27 x 300 =  8,100 s of replay time in the affected windows :  1.77 %
   against 287 x 300 = 86,100 s of replay time in the DAY             :  0.167 %
worst window 20:45:00Z : 36.2 s (12.1 % of the window, 6 intervals)   <- the coordinator's "20:45 carries 6"
04:00:00Z              : 12.8 s (2 intervals)                          <- and "04:00 carries 2"
MEDIAN gap-seconds per affected window     : 2.1 s
```

**THE POINT IN ONE SENTENCE: a "gap-bearing window" is overwhelmingly gap-FREE — a median of
2.1 seconds out of 300.** Using 27/287 = 9.4 % as the altered base overstates the altered
replay time by a factor of **56**.

## 3. THE MAGNITUDE, RE-RULED — AND TWO INDEPENDENT ROUTES AGREE AT ~599x

| to reach | cents | concentration needed vs the altered time-share |
|---|---|---|
| uniform share of 0.167 % | **18.4c** | 1× |
| **1 % of \|D\|** | 110c | **6.0×** |
| **18.8 % (my REVIEW 182 tripwire)** | 2,071c | **112.6×** |
| 25 % (the original) | 2,754c | 149.7× |
| **full erasure of −11,018c** | 11,018c | **598.9×** |

**Cross-checked by generation count, which uses no time argument at all:** 26,264 generations
cancelled over 86,100 s = 0.305 gen/s, so the altered 143.8 s contains **~44 generations**.
Erasure would need those ~44 to carry the entire 11,018c — **251c each against a day average
of 0.419c = 599×.** **Two routes, different quantities, same number.**

**SO THE ANSWER TO "PLAUSIBLE, LIKELY, NEAR-CERTAIN" CHANGES, AND I AM RETRACTING MY OWN
"PLAUSIBLE":** erasure now requires the day's entire arm-vs-baseline deficit to sit inside
**143.8 seconds and ~44 generations**. Q-DA-58's 8.2× does not reach it; nothing measured in
this programme reaches 599×.

**NOT impossible, and I will not say impossible.** Adverse selection is bursty, and the
36.2-second gap at 20:45Z with six intervals is exactly where a violent move could straddle
a few very large fills. **But it is no longer a probability worth arguing about — the rebuild
will simply show it**, which is what the unconditional report is for.

## 4. THE THRESHOLD RULING — 25 % DIES, AND SO DOES MY 18.8 %

**Neither survives, and for the same reason: both were computed on a WINDOW-COUNT base that
overstates the altered replay time by 56×.** At 18.8 % the implied concentration is **112.6×**
— that is not a tripwire, it is an extraordinary finding that would already have been missed
by the time it fired.

**TYPED NOW, BEFORE THE REBUILT BOOK:**

1. **UNCONDITIONAL REPORT — unchanged and still the important tier.** ΔD per arm, plus the
   per-window arm-vs-baseline contribution of all 27, **with each window's gap-seconds beside
   it** so a reader can see altered-time against moved-cents in one line.
2. **`CONCENTRATION_FINDING` at |ΔD| > 1 % of |D| = 110 cents.** That is **6× the
   time-share expectation of 18.4c** — early, cheap, and the right side to err on now that
   the expected move is two orders of magnitude below the old thresholds. **Type the cents,
   not the percent: 110c.**
3. **`SIGN_CHANGE_HALT` — unchanged.** It is mechanism-independent: a correction that
   reverses a seen day's result is a finding about the correction and stops the round for a
   ruling.

**Why 6× and not 1×:** §5's propagation limit. 143.8 s bounds the **directly altered input**,
not its consequence — a missed fill changes inventory for the rest of its window and beyond —
so a move of a few multiples of the time-share is expected and should not halt anything. **6×
is chosen to sit above ordinary propagation and far below anything that could change the
answer.**

## 5. THE LIMIT, STATED AGAINST MY OWN HEADLINE

**143.8 seconds bounds the PERTURBATION, not ΔD.** Gaps alter what the replay sees; the
consequences propagate past the gap's end through inventory and queue state. **So §3's 599×
is the concentration needed *in economic terms*, and it is not a proof of a small ΔD — it is
the reason a large one would be a finding rather than a correction.** That distinction is the
whole content of §4.

**A 1.55-SECOND DISCREPANCY, ROUTED RATHER THAN SWALLOWED:** my ledger-derived set touches
**28** windows; the extra one is `1788796500` (**15:55:00Z**) carrying **1.553 gap-seconds**,
which BE 137 does not count among its 27. Almost certainly a minimum-duration or supply
convention. **It changes nothing here** (1.553 s of 143.8 s), and it is exactly the kind of
one-item difference that turns out to be a convention worth knowing — **route it to BE.**

**And the rule-38 note:** my set came from the raw gap ledger and 300 s arithmetic, BE's from
the fragment receipt's selection. **Different instruments, and they agree on 27 of 27** —
that is corroboration, and I had not read BE 137's list before computing mine.

## 6. SCOPE

Computed: 35 btc gap intervals intersected with the 300 s grid; per-window gap-seconds for all
28 touched windows; totals against both denominators; the generation-count cross-check; and
the four concentration multiples. **Not computed:** per-window arm-vs-baseline P&L — still the
ten-minute query that would replace every scenario here with one number, and still worth more
than this filing.
