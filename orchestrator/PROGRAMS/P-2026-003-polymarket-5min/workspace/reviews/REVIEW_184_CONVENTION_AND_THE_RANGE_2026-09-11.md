# REVIEW 184 — your instrument and mine agree exactly; the convention is answered at 35/35; and `gap_overlaps` is a BOOLEAN, which reopens my own 599x as one END of a range

**REV 141, 2026-09-11T07:03:46Z** (clock read separately). Read-only. **Folds into 182/183;
§3 CORRECTS REVIEW 183's headline.**

## 1. YOUR NUMBERS CHECK — EVERY ONE, INDEPENDENTLY DERIVED BEFORE I READ YOURS

35 intervals; **27/27** of BE's windows matched; **one btc gap outside the 27**; **143.8 s**;
**1.77 %** of the 27's time; **0.167 %** of the supplied day; and every per-window figure
(20:45 = 36.2/6, 14:45 = 18.6/2, 06:55 = 12.8/1, 04:00 = 12.8/2, 04:30 = 10.5/1,
16:00 = 9.7/1, 12:15 = 6.3). **I can name the outlier you could not:
`1788796500` = 15:55:00Z, carrying 1.553 gap-seconds** — a sub-2-second graze, which is why
a minimum-duration or supply convention is the likely explanation. Routed to BE.

**One arithmetic difference to reconcile:** you put 25 % · |D| at ~100× by time; I get
**149.7×** (2,754c ÷ 18.4c, where 18.4c = 0.167 % × 11,018c). Yours is the same order; the
exact figure is 149.7 and the denominator is the day's replay seconds.

## 2. THE CONVENTION — ANSWERED, AND IT IS NOT THE INDEPENDENCE YOU HOPED FOR

```
btc gap_closed rows in 09-07                                        : 35
gap_start_ns == the preceding disconnect's last_message_recv_ns     : 35/35   EXACT
gap_end_ns   vs the gap_closed row's own recv_ns                    : differ by <= 0.01 ms
SUM full durations via gap_end_ns : 145.306s   via recv_ns : 145.306s   (0.000s apart)
```

**The ledger ALREADY starts a gap at `last_message_recv_ns`.** `gap_closed.gap_start_ns` *is*
the data-stop time, so your pairing route and my field-read route recover the same two
timestamps. **That is why we agree to 0.1 s — and it is weaker corroboration than it looks:
we did not independently choose a convention, we both inherited the one the producer
encoded.** A convention error in the collector would be invisible to both instruments
equally. **Agreement here corroborates the EXTRACTION, not the CONVENTION.**

Given your instrument's three prior attempts, that still matters: two independent extraction
routes landing on identical per-window seconds makes a schema or filter error in the fourth
very unlikely. It says nothing about whether data-stop is the right start.

## 3. **THE CORRECTION, AND IT IS TO MY OWN 183** — `gap_overlaps` IS A BOOLEAN

```python
def gap_overlaps(gaps, coin, w0, w1) -> bool:
    return any(gs < w1 and ge > w0 for gs, ge in gaps.get(coin, ()))
```

**The pipeline's own reader never uses gap DURATION. It asks gap PRESENCE, per window.**

So the altered quantity may not be 143.8 seconds of replay input at all — it may be a
**per-window boolean**, and the era change flips which gap set `gaps_by_slug(era)` consults.
**If a flipped boolean gates the whole window, the altered base is 27 × 300 s = 8,100 s =
9.4 %, and erasure needs 5.31× — REVIEW 182's number — not 599×.**

> **So 5.31× and 599× are the two ENDS OF A RANGE, not a correction and a truth. REVIEW 183's
> headline over-claimed by treating the duration bound as settled, and I am withdrawing that
> framing.** Which end applies depends on whether the fix's effect is **interval-scoped**
> (input altered only during the gap) or **window-scoped** (a boolean gating the window). **I
> cannot establish which from the artifacts; `gap_overlaps` being boolean is evidence for the
> window-scoped end.**

**The consolation is real: ΔD itself DISCRIMINATES the two worlds.** Tens of cents ⇒
interval-scoped. Thousands ⇒ window-scoped. The rebuild answers the question I cannot.

## 4. THE THRESHOLD RULING — LOW, AND **PER-WINDOW AS WELL AS AGGREGATE**

**Per-window, and you are right to push for it.** The largest window holds **25 % of all gap
time**. An aggregate-only tripwire is defeated by offsetting moves — +2,000c at 20:45 and
−1,900c elsewhere gives ΔD = +100c and fires nothing, while something large has plainly
happened. **Type it as a disjunction:**

1. **UNCONDITIONAL REPORT** — ΔD per arm; each of the 27 windows' Δcontribution **with its
   gap-seconds beside it**; and **both reference levels printed**: interval-scoped
   **18.4c** and window-scoped uniform **1,036.5c**, so ΔD is read against the right one once
   §3's question resolves.
2. **`CONCENTRATION_FINDING` if |ΔD| > 110c (aggregate) OR any single window's
   |Δcontribution| > 110c.** Set LOW deliberately: firing costs a paragraph, missing costs
   the finding. In the window-scoped world 110c will fire routinely and that is acceptable —
   it is a *report* trigger, not a verdict.
3. **`SIGN_CHANGE_HALT`** — unchanged, mechanism-independent.

## 5. Q-DA-58 IS BY FILLS, NOT TIME — AND IT DOES **NOT** MAKE 2,754c PLAUSIBLE AT 20:45

Uniform generation density is 26,264 / 86,100 s = 0.305/s.

| scope | generations | at Q-DA-58's 8.2× fill concentration | needed for 2,754c |
|---|---|---|---|
| the 36.2 s gap at 20:45 | ~11 | ~38c | **~600× (250c per generation vs a 0.419c average)** |
| the whole 300 s window at 20:45 | ~92 | ~316c | ~72× |

**Neither reaches 2,754c.** Fill-based concentration of 8.2× leaves the burst an order of
magnitude short even on the generous window-scoped reading. **The answer to your question is
no** — and the reason to prefer a per-window tripwire is not that 20:45 could carry 25 % of
|D|, but that it is where the ratio Δcents-per-gap-second will be most legible if anything
anomalous happened.

## 6. SCOPE

Driven: the 35/35 convention identity; the two end-field sums to 0.000 s; per-window seconds
for all 28 touched windows; the outlier's 1.553 s; `gap_overlaps` read at source; the
generation-density arithmetic. **Not established, and it now governs the whole bound:**
whether the fix's effect is interval- or window-scoped. **Still not computed:** per-window
arm-vs-baseline P&L from the existing book — which would collapse §3's range to a number.
