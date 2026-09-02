# Cross-venue discrimination of the blackout signature — DESIGN, DECLARED BEFORE THE ANSWER

**Seat:** DA. **Declared:** 2026-09-02T08:06Z. **Status:** pre-registration
(rule 6). **Nothing in this document was written after reading a rate inside
any event window.** What was inspected first, and is not the answer: which
units exist, the shape of their log lines, their heartbeat cadence, and
whether their counters are cumulative — established on the first 2,000 lines
of each log, which predate every event window by days.

## The question

Three events now share one signature: near-total content loss on the
Polymarket tape, **no gap rows**, all governing duration bars passing, and an
offset that lands on a single instant across every coin.

- **E1** 2026-08-26 (R-362, contiguous ~3 h 20 m, all coins)
- **E2** 2026-08-31 (R-368/R-369, ~4 h at 0.51 % of normal rate)
- **E3** 2026-09-02 (Q-DA-202, btc 40 consecutive windows, all seven coins
  ending at 04:55Z) — on the frozen rule's first governed day

Neither E1 nor E2 was ever root-caused. R-365 proposed compute contention and
**R-366 withdrew it by measurement**, so the field is open.

- **H1 — POLYMARKET-SIDE.** Something at the venue, its edge or its CDN.
- **H2 — HOST OR NETWORK-PATH-SIDE.** Something on this box or on its route.

## The discriminator this box uniquely offers

Three collectors run on **one host**, from **one network path**, against
**three different venues**, all inside `collectors.slice`:

| unit | venue | log |
|---|---|---|
| `pm-collector-clob` | Polymarket | `data/pm_5min/collector.log` (+ the raw tape) |
| `collect-hf` | Binance USDM | `data/mm_hf/collector.log` |
| `collect-hl` | Hyperliquid | `data/mm_hf/hl_collector.log` |

**If H2, all three thin together. If H1, only Polymarket thins.** No amount of
Polymarket-only evidence can separate these; two other venues on the same wire
can.

## Measure (declared)

Per-venue **received-message rate per minute**, from each collector's own
heartbeat line. All three emit a 60 s heartbeat with **cumulative** counters
and a **dateless `HH:MM:SSZ`** stamp, so all three are differenced over
consecutive stamps and dated by the backward walk already used and falsified
in `content_liveness_for` (Q-DA-196).

- **PM** — `Δ msgs / Δt`
- **HF** — `Δ (bookTicker + depth20 + trade) / Δt`
- **HL** — `Δ (bbo + l2Book + trades) / Δt`

## Predicate and thresholds (declared BEFORE the answer)

For venue *v* and window *W*:

> `thin_fraction(v, W)` = the share of one-minute intervals inside *W* whose
> rate is **below 10 % of that venue's median rate over the same UTC day**.

The 10 % cut is **not newly chosen**: it is the reporting cut already carried
in `content_liveness_for` (`note_10pct`), reused so this design invents no
threshold after seeing three events.

**Declared outcomes, fixed now:**

| verdict | condition |
|---|---|
| **H1 — Polymarket-side** | PM ≥ 0.50 **and** HF < 0.10 **and** HL < 0.10 |
| **H2 — host / network path** | all three ≥ 0.50 |
| **UNRESOLVED-ASYMMETRIC** | PM ≥ 0.50 and exactly one of HF/HL ≥ 0.50 |
| **UNRESOLVED-OTHER** | anything else, including PM < 0.50 |

**No result is forced into a bucket.** UNRESOLVED is a verdict, not a
measurement failure, and it is reported as one.

## Third leg — the host (R-163 resource monitor)

The 60 s CSV from `journalctl --user -u resource-monitor`:
`mem_avail_mib, swap_used_mib, research_mib, collectors_mib, docker_mib,
load1, research_cpu_pct, collectors_cpu_pct`.

**Declared:** an H2-host mechanism is *supported* only if a resource excursion
**coincides** with *W* — `mem_avail_mib` below the monitor's own 4096 MiB
alert floor, `swap_used_mib > 512`, or `load1` materially above that day's
median. **Absence of an excursion removes one H2 mechanism; it does not prove
H1.** (R-366's lesson: an absence is a claim about the world and needs the
artifact opened; and a null must not be over-read.)

## Fourth signal — the path, free (declared)

`collect-hf` prints `recv-lat~NNms` on every heartbeat. **Declared:** a
degraded network path should raise HF receive latency inside *W*. HF rate
normal **and** HF recv-lat flat is positive evidence the path was healthy;
HF recv-lat elevated while rates hold is evidence of a path problem that did
not reach the point of loss.

## Windows — DERIVED, never hand-picked

For each event day, *W* is the **longest contiguous run of invisible-thin
windows** on that day, taken from the frozen rule's own detector
(`da_content_liveness_rule` / `pm_tape_density`: below 5 % of the (day, coin)
median, and not overlapped by a gap-ledger interval), unioned across coins.
E3's 01:35–04:55Z is quoted from Q-DA-202 and will be **re-derived here**
rather than carried over.

## Controls, both directions (rules 15/16)

1. **Negative control — a quiet day.** The same clock window on the **day
   before** each event. The discriminator must report thinning **nowhere**.
   Thinning in a control window means the measure is broken, not that the box
   was ill.
2. **Positive control — each parser must be able to fire.** Every venue's
   rate series must contain at least one interval below its own 10 % cut
   *somewhere in the record*, otherwise a "0.00 thin fraction" is an
   instrument that has never been shown to fire.
3. **Refusals are statuses.** A venue whose log does not reach a window is
   **UNMEASURED**, never "normal" — the exact shape that would let a missing
   file read as an alibi.

## Reporting discipline

Every population carries **n and as-of**. **No fix is proposed in the filing
that carries this diagnosis** — measurement first, per the dispatch.
