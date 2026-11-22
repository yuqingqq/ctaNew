# FEASIBLE-CANCEL-HEADROOM-v1 — receipt

2026-11-22, session aea5bfb9.
- **Specified by:** review `3e84ca6d`'s ONE next priority, with the USER's amendment `6d200317`.
- **Later directions read before the freeze:** `2a8cde50` / `c72f0d37` (Codex's independent classification), `90e82f8e` (whole-life queue retention), `c590b083` / `60768c83` (long-continuation, two-sided value).
- **Frozen:** `2a2f78da`, before any table.
- **Run:** timing → write-once projection (fits) → run → checks → identity → readout.

**What this is:** an **ORACLE fixed-path deletion accounting diagnostic** on the saved ASYNC records. Each order is either kept, or one recorded live KEEP is replaced by a permanent PULL effective 10 ms later; every other order, the external flow and inventory feedback stay fixed. It is not a signal, a strategy, an upper bound on market-making policies or a profitability claim.

**Scope:** 19 complete markets, consumed Oct 24–27. CURRENT_ASYNC_FAST10 is primary; FRONT_ASYNC_FAST10 is an artificial-priority reference, never pooled. Four days: point estimates only.

## Integrity

| Item | Result |
|---|---|
| Population | 12,135 generations (5,832 CURRENT / 6,303 FRONT), every status kept; 21,150 fills, each counted once; 564,851 decision states (every recorded live KEEP, including 217 with the fallback reason; the oracle chose none of those) |
| Unchanged path | = the ASYNC receipt's committed eligible +1 s and END on **38 / 38** arm-markets |
| Independent checks | `ch_checks.py` A / B / E / O / Q / H / R / S / K / C / P / I / X all pass. **X:** per generation, the Codex classification `c72f0d37` agrees on all 12,135 (class, its finer good-only split, suffix and prefix G / L, whole +1 s and END, end cause, anchor) |
| Falsifier / mutation / rehearsal | 19 / 19 corruptions refused, including a known-bad classification; 38 / 38 planted errors caught; synthetic end-to-end rehearsal passed before the freeze |
| Resources | Round **0.0402 core-hours** (cap 0.05). This includes the Codex classification's 3.01 s, counted here (`logs/counted_codex_queue_fill_classes.log`), and a declared 30 s unlogged allowance. Line **2.4189**; with the 0.02 review reserve, **2.4389 of 2.50**. Peak 826 MiB |

## Result 1 — first feasible callback, every order (anchor a; = the Codex table)

The suffix is the generation's fills at or after first live KEEP + 10 ms; values are eligible +1 s, in c.

| Class | CURRENT generations | CURRENT suffix net | FRONT generations | FRONT suffix net |
|---|---:|---:|---:|---:|
| pure observed loss | 1,114 | −122,122.64 | 1,241 | −134,201.68 |
| good only | 711 | +58,278.41 | 1,370 | +96,368.44 |
| compensated mixed | 68 | +2,025.88 | 121 | +2,996.13 |
| uncompensated mixed | 84 | −6,889.27 | 238 | −18,900.46 |
| mixed tie | 0 | 0 | 2 | 0 |
| near zero | 21 | 0 | 18 | 0 |
| no suffix fills | 3,416 | 0 | 2,877 | 0 |
| incomplete valuation (observed part) | 18 | −258.47 | 17 | −204.87 |
| no feasible callback | 400 | — | 419 | — |

- **Where the losses sit:** pure suffixes carry **92.85 %** (CURRENT) and **85.02 %** (FRONT) of the anchor-(a) suffix losses (131,521.00 / 157,838.23 c).
- **Gross bad +1 s value over all fills:** 148,114.54 / 178,529.60 c.

## Result 2 — the oracle (anchor b)

| (c, eligible +1 s) | CURRENT | FRONT |
|---|---:|---:|
| Recorded KEEP | −78,485.90 | −61,051.63 |
| Cut every order at its first callback | −9,519.81 | −7,109.19 |
| **Oracle** | **+52,682.23** | **+96,455.85** |
| Oracle cuts / feasible generations | 1,254 / 5,432 | 1,578 / 5,884 |
| Bad value removed (L) | 131,384.63 (88.7 % of gross) | 157,604.26 (88.3 %) |
| Good value sacrificed (G) | 216.51 (128.65 positive shares) | 96.78 (104.79) |
| Residual bad value | 16,729.91 | 20,925.34 |
| Chosen-cut classes: pure / uncompensated / incomplete | 1,238 / 14 / 2 | 1,560 / 17 / 1 |
| Removed L: through-price / same-price | 98,698 of 114,947 / 32,687 of 33,167 | 88,651 of 103,820 / 68,953 of 74,710 |
| Top-3 markets' share of removed L | 24 % | 25 % |

**Per day (oracle retained, c) and the computed reading:**
- **CURRENT:** +14,747.84 / +22,166.98 / +16,618.40 / **−850.99** → `local_cancel_headroom` Oct 24–26, **`no_local_cancel_rescue` Oct 27**; pooled `local_cancel_headroom`.
- **FRONT:** +24,535.80 / +27,192.58 / +35,420.49 / +9,306.98 → `local_cancel_headroom` every day.

**Post-run descriptive arithmetic** (undeclared, `tools/post_run_descriptive.py`):
- **1,129 of the 1,254** CURRENT cuts (90 %) fall at the first callback, **40 ms after activation** (age p25 = p50 = p90 = 40 ms). They remove 122,887 of the 131,385 c.
- FRONT: 1,253 of 1,578 (p90 1,155 ms).

So the oracle's headroom is almost entirely a **per-generation** discrimination: this order should not have stayed past its first review. It is not a late-life cut.

## Result 3 — the same cut at longer horizons

| Retained value, c | CURRENT KEEP | CURRENT oracle | FRONT KEEP | FRONT oracle |
|---|---:|---:|---:|---:|
| +1 s | −78,486 | +52,682 | −61,052 | +96,456 |
| 5 s (native) | −101,654 | +42,218 | −98,169 | +94,070 |
| 30 s (native) | −67,838 | +7,903 | −79,223 | +111,179 |
| END | −68,320 | −55,252 | −86,961 | −23,510 |

**Common cohort** of the removed fills (valued at 1 s, 5 s and 30 s):

| Arm | Fills | 1 s | 5 s | 30 s | END | Computed |
|---|---:|---:|---:|---:|---:|---|
| CURRENT | 3,634 | −102,650 | −115,903 | −75,742 | **+19,972** | `loss_recovers_by_end` = true |
| FRONT | 4,932 | −126,521 | −151,554 | −190,402 | −31,884 | no recovery at any horizon |

**What this means for CURRENT:** the +1 s headroom shrinks to +13.1k c at END. On Oct 25 the oracle's END is worse than KEEP's (−60,391 vs −23,128 c). END carries the directional inventory movement of a changed fill set, with everything else fixed. As the USER's direction `c590b083` says, +1 s locates adverse selection; it cannot decide KEEP / PULL.

## Result 4 — sequences and the whole-life question

Generation-level flags overlap; they are never added as disjoint groups.

| Flag | CURRENT | FRONT |
|---|---|---|
| bad then good (with a feasible cut before the first bad fill) | 29 (27): at that cut G 891.57 vs L 629.11; 13 compensated, 14 not | 47 (44): G 1,028.91 vs L 683.35; 24 / 19 |
| good then bad (with a feasible intervening cut) | 133 (133): G 369.87 vs L 8,707.22; 3 / 130 | 343 (343): G 427.44 vs L 23,166.44; 7 / 336 |
| interleaved / simultaneous within one generation | 7 / 0 | 17 / 0 |

**Prefix link (T3):** the first-callback anchor separates the fills before the first review effect as a sunk prefix.
- Of CURRENT's 711 good-only suffixes, **0** carry a prefix loss.
- 43 of the 1,114 pure suffixes do (5,591.83 c), and 64 of the 3,416 no-suffix-fill generations do (10,907.74 c).
- FRONT: 1 of 1,370 good-only suffixes.

**What this does and does not say:** in the recorded paths, an early loss followed by later good fills on the same generation is rare and small (29 generations, L 629 c of 148,115 c). **This does not test the USER's whole-life queue hypothesis (`90e82f8e`).**
- The recorded continuations are truncated by the policy's own cancellations. Of the 1,254 cut generations, 728 ended in a recorded cancellation, 525 filled out and 1 was open at END; 873 had a recorded cancel request (761 structural, 112 policy).
- A cancel / re-entry path is not modelled.

The trade-off is *linked under the recorded keep-versus-permanent-cancel choice* only; queue dependence stays **unidentified**.

## Result 5 — decision states and queue fields

**Decision panel** (every recorded live KEEP is a row; rows are **not independent samples**):

| Suffix class at that callback | CURRENT (288,319 states) | FRONT (276,532 states) |
|---|---:|---:|
| good only or no fill | 228,139 | 210,713 |
| pure observed loss | 49,548 | 50,193 |
| uncompensated mixed | 5,993 | 10,178 |
| compensated mixed | 4,402 | 5,162 |
| incomplete valuation | 237 | 261 |
| mixed tie | 0 | 25 |

**Queue fields** (recorded only; FRONT is 0 by construction):

| CURRENT feasible generations | Oracle-cut | Kept |
|---|---:|---:|
| Median activation venue ahead (count at 0) | 92 (91) | 100 (356) |
| Median first-review venue ahead | 90 | 100 |
| Median size | 84 | 87 |

These are activation / first-review records, not queue trajectories. No queue-causality claim.

## What this shows and does not show

**Under the hindsight oracle,** the ASYNC pressure ladder's +1 s loss sits in a minority of generations:
- 1,254 of 5,432 feasible CURRENT orders are cut;
- most of those (1,238) have only bad fills after their first review, which comes 40 ms after activation;
- one permanent cancel there would remove ~89 % of the gross +1 s bad value and sacrifice ~0.2k c of good value.

The economic discrimination worth learning is therefore **per generation, at admission / first review**. A false cancel costs the good-only value: 58,278 c over 711 CURRENT orders, about 82 c each, and more under the long-horizon view. A true cancel saves about 110 c per pure order.

**It does not show:**
- that any decision-time information identifies those orders (the cut is hindsight; there is no null and no skill claim);
- a realizable gain (fixed path; no re-entry, inventory or other-order response);
- longer-horizon value: at END the CURRENT bound is small and day-dependent;
- anything about the whole-life queue hypothesis.

**Per the declared reading:**
- No cancellation-only variant is started.
- Headroom permits only a later causal-discrimination question, which the USER's long-continuation direction now governs. The design-only contract is `../ladder_long_continuation_contract_20261122/contract.md`.

**Bundled from review `3e84ca6d`:**
- **P2:** the ASYNC report's interpretation is superseded in band by `../ladder_async_fast_20261122/report_interpretation_v2.md`. Its END identity is computed by `tools/async_identity.py` and agrees with the review's `end_decomposition.csv` on 76 / 76 rows.
- **P3** (inherited checker coverage) stays deferred, as allowed.

## Files

- **`outputs/`:**
  - `summary`, `classes`, `markets`, `sequences`, `panel_summary`, `cases`, `generations_core` (.csv.gz; each = the audit's sha256);
  - `audit.json`, `read_headroom.json`, `async_identity.{csv,json}`.
- **Data directory only, hash-bound in the audit:** the full `generations.csv` and `panel.csv.gz`.
- **Checks:** `checks.json`, `checks_falsifier.json`.
- **`logs/`:** the projection, every job's run_light line, the mutation runs and the rehearsal.
