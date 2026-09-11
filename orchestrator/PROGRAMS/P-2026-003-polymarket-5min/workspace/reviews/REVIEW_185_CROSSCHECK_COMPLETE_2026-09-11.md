# REVIEW 185 — 27 of 27 windows match, every value; the pipeline uses the data-stop convention transitively; and REVIEW 183 is committed but NOT in origin

**REV 142, 2026-09-11T07:07:08Z** (clock read separately). Read-only. Short by design.

## 1. THE CROSS-CHECK IS COMPLETE — ALL 27, NOT JUST 20:45

| # | window | s | n | # | window | s | n | # | window | s | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **20:45** | **36.19** | **6** | 10 | 13:45 | 2.28 | 1 | 19 | 05:15 | 1.77 | 1 |
| 2 | **14:45** | **18.61** | 2 | 11 | 18:00 | 2.22 | 1 | 20 | 17:45 | 1.71 | 1 |
| 3 | **06:55** | **12.85** | 1 | 12 | 08:25 | 2.21 | 1 | 21 | 04:05 | 1.51 | 1 |
| 4 | **04:00** | **12.76** | **2** | 13 | 14:10 | 2.13 | 1 | 22 | 21:10 | 1.50 | 1 |
| 5 | **04:30** | **10.51** | 1 | 14 | 19:45 | 2.11 | 1 | 23 | 10:05 | 1.50 | 1 |
| 6 | **16:00** | **9.73** | 1 | 15 | 15:15 | 1.98 | 1 | 24 | 18:45 | 1.44 | 1 |
| 7 | **12:15** | **6.29** | 1 | 16 | 00:05 | 1.93 | 1 | 25 | 21:40 | 1.36 | 1 |
| 8 | 05:45 | 2.89 | 1 | 17 | 00:55 | 1.92 | 1 | 26 | 23:45 | 1.04 | 1 |
| 9 | 01:00 | 2.40 | 1 | 18 | 00:50 | 1.91 | 1 | 27 | 07:00 | 1.01 | 1 |

**Top seven: 36.2 / 18.6 / 12.8 / 12.8 / 10.5 / 9.7 / 6.3 with n = 6 / 2 / 1 / 2 / 1 / 1 / 1 —
every value yours.** Remaining twenty: **1.01–2.89 s**, your "1.0–2.9 s each". Total
**143.8 s**; top seven 106.9 s, tail 36.8 s. **No window differs. 27/27, values and counts.**

## 2. THE CONVENTION `gap_overlaps` USES — DATA-STOP, INHERITED NOT CHOSEN

```python
load_gaps():  if r.get("gap_start_ns") and r.get("gap_end_ns") and r.get("coin"):
                  out[r["coin"]].append((r["gap_start_ns"]/1e9, r["gap_end_ns"]/1e9))
gap_overlaps(): any(gs < w1 and ge > w0 ...)
```

**The pipeline reads `gap_start_ns`/`gap_end_ns`** — and REVIEW 184 established
`gap_start_ns == the preceding disconnect's last_message_recv_ns`, **35/35 exact**. **So the
pipeline uses `last_message_recv_ns` — DATA-STOP, not detection — transitively through that
field.** Your instrument, mine, and the pipeline's all resolve to the same two timestamps.

**The bound is corroborated by two instruments on the quantity. It is NOT corroborated on the
convention** — all three inherit the collector's encoding, so a convention error there is
invisible to every one of them equally.

**One latent defect, harmless today, flagged because it is the house pattern:** `load_gaps`
filters on **truthiness** (`if r.get("gap_start_ns") and …`), not `is not None`. A
`gap_start_ns` of 0 would be silently dropped. Real ns timestamps are never 0, so nothing is
wrong now — but it is a silent-drop on a gap ledger, which is the shape rule 4 exists for.

## 3. REVIEW 183 IS COMMITTED AND **NOT IN ORIGIN** — AND THE WITHDRAWAL IS ALREADY FILED IN BAND

- **183 is `33e6745`, stranded.** So is 184 (`8310428`), which carries the withdrawal.
- **The superseding note exists and is in band: REVIEW 184's title line says
  "§3 CORRECTS REVIEW 183's headline", and §3 states "REVIEW 183's headline over-claimed by
  treating the duration bound as settled, and I am withdrawing that framing."**
- **183 has not been edited and will not be** (rule 13). Because both are stranded together,
  they will land together and a reader meets the correction with the claim — which is the
  outcome rule 13 wants, reached by accident rather than design.

**Seven of my filings are stranded and none is in origin: 178 `0c1f136`, 179 `066c800`,
180 `dfe3ef5`, 181 `215e3b9`, 182 `61daa82`, 183 `33e6745`, 184 `8310428`** (+ this one).
The shared tree has been dirty at every attempt — currently `BE_PROCEDURE.md` — and rule 21
forbids the retry. **They need your hand, and 183/184 in particular should not be split.**

## 4. THE 15:55 ROW

With BE as you have it. **If the replay sees it, the declared table is 27 + that row and the
total becomes 145.4 s** (143.8 + 1.553); the aggregate reference level moves from 18.4c to
18.6c, which changes no threshold. **If it does not, 27 stands.** Either way the 110c
tripwire is unaffected — I note it only so the table's row count and its total are declared
together rather than one being inferred from the other.
