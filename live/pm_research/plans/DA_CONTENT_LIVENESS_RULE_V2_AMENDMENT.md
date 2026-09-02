# Content-liveness rule — v2 AMENDMENT: an absolute floor beside the relative one

**DRAFT FOR USER FREEZE.** Seat: DA. Drafted 2026-09-02T08:20Z.
**It governs nothing and is wired into nothing.** The frozen v1 rule
(`da_content_liveness_rule.py`, USER-frozen R-386) is **not touched by this
draft** — no constant, no status, no line. Wiring follows a freeze; it never
precedes one (rule 14).

Checker: `live/pm_research/da_content_liveness_v2_check.py` (12 checks).

---

## 1. Motivation — the reviewer's RR6-1, verbatim

> ## RR6-1 — HIGH (on the FROZEN RULE, not on this batch) — "L2 cannot shrink" is false where it matters, and the rule is blind to a TOTAL blackout
>
> DA's projection (7): *"L2 is a RUN LENGTH and cannot shrink as the day fills — 40
> windows stays 40 — so 09-02 will fail L2 at close whatever the rest of the day does."*
>
> A run length over a fixed classification cannot shrink. But the classification is not
> fixed: `invis = [w for w in wins if b < median(wins) * thin_frac]`, and the median is
> recomputed over **all** the day's windows. As the day fills, the median moves — and if
> it moves DOWN, windows stop being thin.
>
> **Computed on the real 09-02 btc window sizes, extending to 288 windows:**
>
> | continuation | n_thin | longest run | L1 | L2 passes? |
> |---|---|---|---|---|
> | as measured now (97 windows) | 40 | **40** | 0.4113 | no |
> | rest of day NORMAL | 40 | **40** | **0.1385** | no |
> | remainder 30% dark | 97 | 57 | 0.3360 | no |
> | remainder 50% dark | 135 | 95 | 0.4677 | no |
> | **remainder 60% dark** | **0** | **0** | **0.0000** | **YES** |
> | **rest of day DARK** | **0** | **0** | **0.0000** | **YES** |
>
> DA's **L1** projection is exactly right — 0.1385 against the quoted ~0.14, ~1.7× the
> 0.08 bar. The **L2** claim is right under the benign continuation and **wrong under the
> severe one**: the flip happens precisely when dark windows pass half the day, because
> the median crosses into the dark regime and every dark window stops being "thin
> relative to the median".
>
> **The general property, which is the part that matters beyond tonight: the rule detects
> PARTIAL blackouts and is structurally blind to TOTAL ones.** A day 100% dark has
> `L1 = 0`, `L2 = 0` and reads **CONTENT_LIVE**. The only absolute floor in the rule is
> `median <= 0 → UNJUDGEABLE`; a median of 11,637 bytes (0.2% of normal) is judged
> healthy. This is rule 16's shape on the instrument itself — a control that cannot fire
> in its worst case.

Verified rather than accepted: the checker reproduces the blind spot on a
synthetic 100 % dark day — **v1 reads `CONTENT_LIVE` with run 0**.

## 2. The amendment, in one sentence

Keep v1 exactly as frozen and add **one** predicate that asks v1's own
question against a denominator **the day under test cannot move**.

> **L3 — ABSOLUTE FLOOR.** A window is **DARK** if its bytes fall below
> `0.05 × ref(coin)`, where `ref` is the **median of that coin's daily medians
> over the prior complete days**. The coin-day fails L3 if the longest run of
> consecutive DARK windows exceeds **12 windows (60 min)**.

**IT INTRODUCES NO NEW MEASUREMENT THRESHOLD.** `0.05` is v1's `THIN_FRAC`;
`12` is v1's frozen `L2_RUN_WINDOWS_MAX`. Only the **denominator** changes.
The checker asserts both equalities, so a later edit that quietly forks them
fails.

The two genuinely new numbers are **structural**, not bars:

| constant | value | why |
|---|---|---|
| `V2_TRAILING_DAYS` | 7 | with a median-of-priors reference, the reference only turns dark once **more than half** the trailing window is dark — so v2 is robust to **up to three consecutive fully dark days** and degrades on the fourth |
| `V2_MIN_REFERENCE_DAYS` | 3 | below three priors the reference still moves; at three or more the measured ratio band is stable (§4) |

**Point-in-time by construction:** `trailing_reference` reads only days
*strictly before* the day under test, so a day cannot move its own reference
and a later day cannot move an earlier one. The checker asserts this by adding
a fully dark **later** day and confirming an earlier reference is unchanged.

## 3. Statuses EXTEND; nothing is replaced

v1's vocabulary is untouched. v2 adds two:

| status | meaning |
|---|---|
| `CONTENT_DARK` | the absolute floor is breached (L3 run > 12) |
| `CONTENT_LIVENESS_NO_REFERENCE` | fewer than 3 prior complete days — **not a pass** |

Every coin-day carries **both** `status_v1` and `status_v2`. A day's v1 reading
is never overwritten; v2 reports beside it.

## 4. Calibration — days ≤ 2026-08-31 ONLY (rule 11)

`calibrate()` **REFUSES** any day after `20260831`, by name, with a falsifier.
09-01 and 09-02 are seen days; **09-02 is the event that motivated this
amendment and may be cited, never calibrated on.** The anchors E1 (08-26) and
E2 (08-31) both predate the boundary.

**How far a legitimately quiet coin-day falls.** Over **70 judged coin-days**
(13 days × 7 coins, less the 3 opening days that have no reference), the ratio
`day median / trailing reference` spans:

| n | min | median | max |
|---|---|---|---|
| 70 | **0.3961** | 0.8135 | 1.3596 |

The quietest honest coin-day in the entire record sits at **0.396**, which is
**7.9× above** the 0.05 dark fraction. That is the discrimination the floor
rests on, measured rather than assumed.

## 5. A slow venue hour is not a blackout — the discrimination, stated

Two things are being told apart:

- a **quiet market** — fewer trades, thinner books, but a live socket. Measured
  floor across the record: **40 % of normal** at the day level, and the worst
  window-level run below `0.05 × ref` on any non-event day is **5 windows**,
  against a 12-window bar.
- a **blackout** — near-zero content. Measured in E1/E2 (Q-DA-203): window
  medians of **1.23 and 1.62 msg/s against day medians of 634 and 475**, i.e.
  **0.2–0.3 % of normal**.

Those differ by **more than two orders of magnitude**. A floor at 5 % of a
trailing reference sits in the empty space between them.

## 6. What the amendment changes on the record — measured

On the 13 calibration days: **0 of 13 DAY verdicts move.** Exactly **1 of 70
judged coin-days** changes, and it is a **true positive that v1 under-read**:

| day | coin | v1 | v2 | why |
|---|---|---|---|---|
| 2026-08-26 | hype | `CONTENT_LIVE`, run **3** | `CONTENT_DARK`, run **40** | hype's own median had already collapsed to **0.574** of its reference, so its dark windows stopped being "thin relative to the median" |

**The other six coins all read 40 windows on that same event.** hype is the
smallest-volume coin, so a 3 h 20 m outage moved its median furthest — the
median-collapse blind spot in miniature, on a *partial* blackout. Q-DA-203's
E1 table recorded the anomaly (`hype:3w` beside six coins at `40w`) without
explaining it; this is the explanation.

**08-26's day verdict does not move** — v1 already reads it `CONTENT_THIN` on
the other six coins. Coverage extends; no verdict is overturned.

## 7. The falsifier, both ways — and the difference IS the amendment

| case | v1 | v2 |
|---|---|---|
| synthetic **100 % dark** day | `CONTENT_LIVE`, run 0 | **`CONTENT_DARK`, run 288** |
| genuinely **quiet but honest** day (40 % of normal, harder than any real one) | `CONTENT_LIVE` | **`CONTENT_LIVE`**, run 0 |
| **partial** blackout (30 windows) | `CONTENT_THIN`, run 30 | `CONTENT_DARK`, run 30 — **identical** |
| fewer than 3 prior days | — | `NO_REFERENCE`, never a pass |

Row 1 is the amendment. Row 2 is the guarantee it does not cost. Row 3 is the
proof it extends rather than re-judges.

## 8. Limitations, declared rather than guarded

1. **A sustained multi-day blackout degrades the reference.** With K = 7 and a
   median of priors, the reference turns dark once **4 of the 7** trailing days
   are dark. v2 is robust to three consecutive fully dark days and blind on the
   fourth. A guard that cannot fire is not a guard, so this is stated, not
   patched.
2. **A coin whose true volume steps down permanently** will read DARK until the
   trailing window catches up — at most 7 days. This is a false-positive mode
   and it is the price of a reference the day cannot move.
3. **v2 inherits everything v1 inherits**: gzip-trailer bytes as the content
   proxy, gap-covered windows excluded as accounted loss, and per-(day, coin)
   scaling.

## 9. What the USER decides

This draft answers none of the following; it supplies the numbers for them.

- **(e)** adopt L3 as drafted, adopt it with different structural constants, or
  reject it;
- **(f)** whether `CONTENT_DARK` joins the governing set or is REPORTED beside
  it — the §8(a) question, now with a second predicate attached;
- **(g)** whether the 08-26 hype coin-day is **re-stated** under v2 or left as
  v1 recorded it. This draft leaves it as v1 recorded it and reports the
  difference; re-stating a frozen day is a USER act.
- **(h)** §8's original (a)/(b)/(c) remain open and unaffected.
