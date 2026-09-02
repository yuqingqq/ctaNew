# Review — DA round 4 (cross-venue forensics) + round 5 (content-liveness v2 DRAFT)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `509859f`** (round 4: design `9785e5e`, forensics `4f892de`;
round 5: v2 draft `509859f`).
**Continues** `REVIEW_R402_CONTENT_LIVENESS_WIRING_2026-09-02.md` (`81eea68`).
**Composed 2026-09-02T08:29:56Z.** One filing, per R-377.

Method unchanged: detached worktree at the pinned tip, ledgers symlinked read-only,
verdicts from execution; mutants applied in the lab and reverted; no repository file
modified, nothing written to any production ledger, verdict or tape.

**Disclosure of my own position in this round:** round 5 amends a rule in response to
**RR6-1, which is my finding from the previous round.** Agreeing with a batch built on
my own finding is consistency, not confirmation, so I rebuilt the reproduction from
scratch with my own fixture rather than running DA's, and I tested the amendment's
declared LIMIT as hard as its claim.

---

## Verdict

### Both batches RELEASED. No hold from this seat is open at `509859f`.

The forensics honoured its pre-committed design and I reproduced its central
discrimination independently. The v2 draft is DRAFT in both directions, introduces no
new measurement threshold, calibrates only on days ≤ 08-31, and — the thing that
matters — **the amendment actually closes the case that produced it**, verified on my
own synthetic day.

Two low findings (**RR7-1**, **RR7-2**), neither holding. One property the USER should
see stated plainly before ruling on the amendment: **v2 narrows my RR6-1 blind spot
rather than closing it, and I measured exactly where the residual begins.**

---

## Scope A — the forensics

### The design was committed first, and it did not move

| check | result |
|---|---|
| commit order | design `9785e5e` **08:06:41Z** → forensics `4f892de` **08:14:14Z** → v2 draft `509859f` 08:22:25Z |
| design file identity | **`4e74a8b` at `9785e5e`, `4f892de`, `509859f` and HEAD** — byte-identical; never edited after the answer |
| thresholds in code vs the declared table | `THIN_CUT = 0.10`, `THIN_MAJORITY = 0.50`, `THIN_QUIET = 0.10` — **exactly the declared values** |
| the outcome table | all four declared verdicts have a selftest case, including *"an UNMEASURED venue is NOT an alibi"* |

The 10% cut is reused from `content_liveness_for`'s existing `note_10pct` rather than
invented, which is the right way to avoid choosing a threshold after seeing three
events.

### I recomputed E3's discrimination from the raw logs, with my own parser

My own regexes, my own backward-walk dating for the two dateless logs, my own
differencing, on the 2026-09-02 01:35–04:55Z window:

| venue | day median | window median | window min | n_thin / n | **thin_fraction** |
|---|---|---|---|---|---|
| polymarket | 345.875 /s | 1.758 /s | 0.0333 /s | 200 / 200 | **1.000** |
| binance_hf | 1341.097 /s | 1518.067 /s | 761.05 /s | 0 / 200 | **0.000** |
| hyperliquid | 59.392 /s | 66.150 /s | 44.9167 /s | 0 / 199 | **0.000** |

DA's artifact reports 1.000 / 0.000 / 0.000 with window medians 1.758 / 1518.067 /
66.150 and minima 0.0333 / 761.05 / 44.9167 — **identical to my independent read**;
the day medians differ in the third decimal only because my run is later and sees one
more interval. Under the pre-declared table (PM ≥ 0.50, HF < 0.10, HL < 0.10) this is
**H1-POLYMARKET-SIDE**, and the same holds for E1 and E2.

Two other venues on the same host and the same wire held **full** rate — HF's worst
minute inside the window is 761 msg/s against a 1341 /s day median — while Polymarket
sat at 0.5% of normal. That is the discriminator the design promised, and it delivers.

### The vacuity control fires where it counts

DA's disclosure is that its own Binance parser returned a meaningless zero mid-build —
the "alibi" shape, where a broken parser makes a venue read as *not affected* and
manufactures H1.

**Executed known-bad:** I renamed the HF counter field in the shipped regex
(`bookTicker` → `bookTickerXX`) and re-ran the real event pass. All three events became
**`UNRESOLVED-UNMEASURED`** — not H1. A broken parser cannot produce an alibi; the
verdict degrades instead. The parser also refuses a renamed field by name
(*"FORMAT CHANGE"*) rather than reading it as an absent venue.

### RR7-1 — LOW — the suite does not exercise the real per-venue regexes

The same mutation leaves `--selftest` at **17/17 passing**: its fixtures are built with
the `hyperliquid` spec, so the shipped `binance_hf` regex is never matched against a
real line. The protection is in the product (the verdict refuses), which is the more
important half, but the suite would not tell you the parser broke.

**Closure:** one check per venue that the shipped regex matches a real sample line from
that venue's own log — three assertions.

---

## Scope B — the v2 amendment DRAFT

### DRAFT discipline holds in both directions

| check | result |
|---|---|
| frozen v1 untouched | `da_content_liveness_rule.py` is **`12ae66a` at `3298a1d`, `4f892de`, `509859f` and HEAD** — byte-identical |
| v2 → v1 | the draft does **not** import v1; it reads `pm_tape_density` only |
| v1/verdict path → v2 | grep across `live/pm_research/*.py`: the **only** reference to `da_content_liveness_v2_check` is `v5_deploy_gates.py`'s `--selftest` entry — no verdict path imports it |

### No new measurement threshold — verified at the constants, not the prose

```
V2_DARK_FRAC == da_content_liveness_rule.THIN_FRAC          → True  (0.05)
V2_RUN_MAX   == da_content_liveness_rule.L2_RUN_WINDOWS_MAX → True  (12)
```

Only the **denominator** changes: v1 compares a window to the day's own median, v2
compares it to the median of the coin's prior daily medians. The two genuinely new
constants (`V2_TRAILING_DAYS = 7`, `V2_MIN_REFERENCE_DAYS = 3`) are structural, not
bars, and the checker asserts the two equalities so a later fork fails loudly.

**Point-in-time, tested rather than read:** I built five synthetic days and computed
09-04's reference with and without a **fully dark 09-05** present. Both give
**18800.0** — a later day cannot move an earlier reference.

### Calibration is bounded, and refuses by name

`calibrate(['20260830','20260831','20260902'])` → **REFUSED**, naming 09-02 and the
reason (*"the day that MOTIVATED this amendment … may be cited, never calibrated on"*).
`calibrate([])` → **REFUSED** as the empty-set trap. The CLI's own day list is filtered
to ≤ `20260831` before it ever reaches the function, so the guard is a second line
rather than the only one.

### RR6-1 is carried verbatim and REPRODUCED — with my own fixture

The amendment quotes my finding verbatim; I diffed the quoted block against my own
filing, sentence by sentence, and it is unaltered.

**The reproduction, built independently:** three healthy synthetic days followed by a
day whose 288 windows are all near-empty.

| rule | reading on a 100%-dark day |
|---|---|
| **v1 (frozen)** | **`CONTENT_LIVE`** — L2 run **0**, L1 **0.0**, median 47 bytes |
| **v2 (draft)** | **`CONTENT_DARK`** |

That difference is the amendment, and it is real.

### The record-level claims reproduce exactly

Recomputed over the 13 calibration days (08-19..08-31) with my own loop:

| claim | mine |
|---|---|
| 70 judged coin-days, 21 with no reference | **70 / 21** (3 opening days × 7 coins) |
| exactly 1 coin-day flips v1-LIVE → v2-DARK | **exactly 1: 2026-08-26 hype**, v1 run **3** → v2 `CONTENT_DARK` |
| the other six coins read 40 on that event | **confirmed** — bnb/btc/doge/eth/sol/xrp all v1 `CONTENT_THIN` run 40 |
| ratio band over 70 coin-days | **min 0.3961, median 0.8135, max 1.3596** |
| headroom over the dark fraction | **7.9×** |

The hype case is a true positive that v1 under-read, for the reason the amendment
states: hype is the smallest-volume coin, so the outage moved its own median furthest
and its dark windows stopped being thin relative to it.

### For the USER, before ruling: v2 NARROWS my blind spot; it does not close it — and I measured where the residual begins

DA declares this in §2 (robust to up to three consecutive fully dark days, degrading on
the fourth). I tested it rather than taking it:

| trailing days fully dark | reference | a DARK day under test reads |
|---|---|---|
| 3 of 7 | **18800.0** (healthy) | v1 `CONTENT_LIVE` → **v2 `CONTENT_DARK`** ✓ |
| 4 of 7 | **47.0** (collapsed) | v1 `CONTENT_LIVE` → **v2 `CONTENT_LIVE`**, ratio 1.0 ✗ |

So the amendment converts RR6-1 from *"any total blackout is invisible"* into *"a total
blackout sustained past the fourth day is invisible"*. That is a large narrowing of a
real hole, the limit is declared honestly in the draft rather than discovered by a
reader, and the residual is the price of a self-referential denominator at any window
length. **Worth stating in the ruling so the bound is chosen, not inherited.**

### RR7-2 — LOW — the two status vocabularies extend rather than map

`status_v1` and `status_v2` use different labels for the same reading: on 08-26 all
seven coins differ **string-wise** (`CONTENT_THIN` vs `CONTENT_DARK`) while the
amendment's own — correct — reading is that **one** coin-day changed. A consumer that
diffs the two fields naively would report 7 changes where there is 1.

**Closure:** emit the comparison the amendment actually makes — a `verdict_changed`
boolean, or a named `v1_live_to_v2_dark` flag — so the count is computed rather than
inferred from string inequality.

---

## Executed evidence

At `509859f`, as of 2026-09-02T08:29Z:

| check | result |
|---|---|
| `v5_deploy_gates.py` | **ALL 19 GATES PASS**, exit 0 (the `v4 behaviour` flake did not fire this run) |
| `da_cross_venue_forensics.py --selftest` | 17 checks passed |
| `da_content_liveness_v2_check.py --selftest` | 12 checks passed |
| design file across four refs | byte-identical `4e74a8b` |
| declared thresholds vs code | 0.10 / 0.50 / 0.10 — exact |
| E3 recomputed independently | PM **1.000**, HF **0.000**, HL **0.000**; window medians/minima match to the digit |
| mutant: HF counter field renamed | all three events → **UNRESOLVED-UNMEASURED**, never H1 |
| …the same mutant against the suite | **17/17 still passes** — RR7-1 |
| frozen v1 across the batch | byte-identical `12ae66a` |
| v2 wiring | referenced only by the gates `--selftest` entry |
| `V2_DARK_FRAC` / `V2_RUN_MAX` vs v1 | **equal** (0.05, 12) |
| calibrate on a post-08-31 day / empty set | **REFUSED**, by name |
| 100%-dark synthetic day (my fixture) | **v1 `CONTENT_LIVE` (run 0, L1 0.0) → v2 `CONTENT_DARK`** |
| point-in-time reference | unchanged (18800.0) with a fully dark later day present |
| calibration record | 70 judged / 21 no-reference; **exactly 1 flip: 08-26 hype** |
| ratio band | min 0.3961 / median 0.8135 / max 1.3596; headroom 7.9× |
| residual blind spot | **4 of 7 trailing days dark → v2 reads a dark day as `CONTENT_LIVE`** |
| mutants executed this round | **1** (the HF parser). The rest of this round was recomputation and fixture construction, which is where the claims lived |

---

## Disposition

- **RELEASED:** DA round 4 (the pre-committed design was honoured, the discrimination
  reproduces independently, the vacuity control fires at the verdict) and DA round 5
  (DRAFT in both directions, no new threshold, calibration bounded, RR6-1 reproduced
  and closed for the case that produced it). **No hold from this seat is open.**
- **FILED, not holding:** RR7-1 (no per-venue regex check in the suite), RR7-2 (status
  vocabularies extend rather than map).
- **For the USER's ruling on the amendment:** v2 turns *"any total blackout is
  invisible"* into *"a total blackout past the fourth consecutive day is invisible"* —
  measured, at the exact boundary, above. The draft states that limit itself, which is
  the reason to trust the rest of it.
- **On the forensics result:** three events, one signature, and two independent venues
  on the same host and wire at full rate through all three. **H1-Polymarket-side** is
  the honest reading of the discriminator this box uniquely offers, and it is now the
  first evidence-backed cause statement this programme has had for E1 and E2 after
  R-365 was withdrawn.
