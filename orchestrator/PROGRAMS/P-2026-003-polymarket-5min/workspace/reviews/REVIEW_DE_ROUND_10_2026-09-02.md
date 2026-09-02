# Review — DE round 10 (CO-5, CO-R1, CO-R3 closed) — and a correction to my own released review
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `2282e5c`** (Q-DE-28; `de_ratification_check.py`, 58 checks).
**Request of record:** `REQUEST_DE_ROUND_10_2026-09-02.md`.
**Composed 2026-09-02T11:40:09Z.** One filing, per R-377.

Executed in `~/ctaNew-wt-rev` at `--detach 2282e5c`. Read-only under `data/`; every
register fixture was built on a **copy in a temp dir** — `COORDINATION.md` was never
written. No timer, no service, no launcher.

---

## FIRST: my own error, corrected in band (rule 13; the released review is not edited)

`REVIEW_DE_ROUNDS_7-9_2026-09-02.md` (`b4da910`) stated that R-421 §6's *"the driver
already refuses … ledger-only windows"* does not hold. **That was wrong, and the way it
was wrong is worse than the claim.** I searched three files for `scan_day` / `raw/` /
`TD.RAW`, found none, and concluded no layer reads the tape. **That is a grep
establishing an absence** — the exact failure this programme records at R-365 ("asserted
an absence without opening the artifact that would have carried it"), and I have quoted
that lesson at other seats.

Reproduced here at 11:36–11:38Z:

- `be_forward_day.selected_from_specs` reads the tape through
  `harmful_exposure_rows.qr.base.fi._archive_paths()` and `fi.token_map()` — **27,884
  archive paths, 27,919 tokens** at my read;
- given two real 09-01 specs plus a window with no archive it **REFUSES by name**:
  *"REFUSED: 1 supplied windows have no archive or no token map … R-418 scores the
  complement WHOLE; dropping windows here would silently re-select the population the
  supply already fixed."*

One methodological note that matters for the next such check: the coordinator's own
known-bad — `bnb-updown-5m-1788348600`, the 11:30Z window, ledger-only at 11:32Z — **no
longer reproduces at 11:36Z**, because that window has since gained an archive. I had to
use the current in-flight window and a future one to reproduce the mechanism stably; both
refuse. A fixture whose essential property is "not written yet" expires.

**What my execution did establish, and it stands:** DE's `supply()` and the seam bridge
**do** emit a tape-less window (1,876 specs with the extra window present), so the
refusal lives downstream, in the driver, after the frozen-contract gate. The population
DE hands over is not tape-checked; the driver is what refuses.

**On the disposition (R-422 §4): I agree, and I withdraw the weaker half of my own
recommendation.** I wrote "intersect (with the difference reported) or refuse". Intersect
is wrong for exactly the reason `selected_from_specs` gives — it would silently re-select
the ratified population. **Refuse on any ledger-minus-tape difference, named per coin, at
the population gate** is the right form, and BE round 5 is the right owner.

---

## Verdict

### RELEASED. CO-5, CO-R1's checker half and CO-R3 all close, and CO-R3 shipped the stamp rule I recommended.

Three findings, none holding: **DE10-R1** (timestamps are compared as strings, so a
malformed clock or bound is ordered rather than refused — and lands on the permissive
side), **DE10-R2** (nonsense field *values* still verify clean — the surviving half of
DE-R3), **DE10-R3** (DE-R4 reproduces: 13 raise sites, 12 audited).

---

## 1. Malformed refuses; the seam between "missing" and "undecidable" is not yet clean

**Missing is closed.** Each of the ten `RATIFICATION_FIELDS` — `ref, kind, population,
sampling, present_source, scope_days, scope_from, scope_to, revocable_by, supersedes` —
**refuses by name** when absent. That closes the first half of my DE-R3.

**Present-but-malformed is not.** The seam question has an answer, and two inputs land on
the wrong side:

| input | result | should be |
|---|---|---|
| `scope_to: not-a-date` | **`verified True`, `unverifiable []`** | undecidable, or refused |
| `scope_from: not-a-date` | `verified False`, `unverifiable []` | undecidable, or refused |
| `present_source: /etc/passwd` | `verified True`, `unverifiable []` | — DE10-R2 |
| `present_source:` (empty) | `verified True`, `unverifiable []` | — DE10-R2 |
| `revocable_by: nobody` / `scope_days: WHENEVER` | `verified True`, `unverifiable []` | — DE10-R2 |
| `sampling: none` (lower-case) | **REFUSED** by name | correct |

`scope_to` garbage is read **permissively** (the day is in scope) and `scope_from` garbage
**restrictively**, and neither surfaces as undecidable. Same class of input, opposite
directions, and the reader cannot tell either happened.

## 2. `day_closed` — the boundary is right, the clock is not validated

| `now_utc` | `day_closed` | `verified` |
|---|---|---|
| `2026-09-01T23:59:59Z` | **False** | False |
| `2026-09-02T00:00:00Z` (the boundary) | **True** | True |
| `2026-09-02T00:00:01Z` | True | True |
| **`zzzz`** | **True** | **True** |
| `''` (empty) | True | True — falls back to the wall clock, emitted as `2026-09-02T11:38:40Z` |

The boundary closes with `<=` as specified, and the **default clock is read at call time**
(the emission carries `now_utc: 2026-09-02T11:38:21Z`, moving between calls) — no
module-load time leaks in.

### DE10-R1 — MEDIUM — timestamps are compared as strings, so an unparsable value is ordered, not refused

The request says *"malformed `now_utc` refuses"*. It does not: `zzzz` yields
`day_closed: True, verified: True`, with the emission faithfully echoing `now_utc: zzzz`.
The mechanism is visible in the failure mode of a non-string: passing an `int` raises
`TypeError: '<=' not supported between instances of 'str' and 'int'` — the comparison is
**lexicographic on strings**. `'zzzz'` sorts after any ISO timestamp, so a malformed clock
reads as *the day is closed*; `scope_to: not-a-date` is the same mechanism and the same
permissive direction.

This is one root cause with three symptoms (item 1's two rows and this one). **Closure:**
parse both sides to datetimes and refuse an unparsable value **by name** — the rule the
module already applies to a population it cannot evaluate (*"reported as unknown rather
than assumed to be the full one"*). Empty-means-now is defensible but should say so in the
emission rather than silently substituting.

## 3. Provenance is computed, and the chain behaves

Chain built on a copy: **R-419** (real, 11:03Z) ← superseded by **R-9701** (12:00Z) ←
superseded by **R-9702** (13:00Z), stamped at **12:30Z**:

| ref | stamped 12:30Z | unstamped (new run) |
|---|---|---|
| R-419 | **REFUSED** — *"ALREADY superseded by ['R-9701'] at the stamped instant"* | REFUSED — superseded |
| R-9701 | `verified True`, `verified_for_new_run False`, **`provenance True`**, `superseded_by ['R-9702']`, `superseder_times {'R-9702': '2026-09-02T13:00:00Z'}` | REFUSED — superseded |
| R-9702 | `verified True`, `for_new_run True`, `provenance False` | verified |

`superseder_times` is complete for the queried ref. **The parse is heading-anchored:** a
superseder whose heading carries no timestamp but whose **body** contains one refuses —
*"its heading carries no parsable register timestamp, so WHEN it took force cannot be
computed"* — so timestamp-shaped prose cannot stand in for a heading.

This is the CO-R3 judgement I gave, shipped: the receipt's `stamped_at` against the
superseder's heading time, with the API able to say which question is being asked.

## 4. `require_verified()` — all three conjuncts load-bearing

| mutated result | outcome |
|---|---|
| `verified: False` | `NotVerified` — names it |
| `unverifiable: ['day_in_scope']` | `NotVerified` — *"each is a q…"* |
| `provenance: True` | `NotVerified` — *"this result is PROVENANCE for a run stamped before the supe…"* |
| unmutated | **passes** |

**Can a consumer reading `verified` alone pass a provenance result? Yes** — a provenance
result carries `verified: True`. And it **is** stated where a consumer reads it: the
emission carries `refusal_scope: "a refusal here is about STARTING A RUN; a receipt
already carrying a ref keeps it as provenance"`, plus `verified_for_new_run` and
`provenance` as separate fields. The gate is opt-in by design; the fields that make the
distinction are in the same object, which is the right place for them.

## 6. The suite and the audit

**58 checks under both launchers** from the repo root (`python3 -m live.pm_research.…`
and `python3 live/pm_research/…`), rc 0.

The audit reports **12 guards, `survivors: []`**, each with
`refuses_on_its_known_bad: true` **and** `refuses_on_the_control: false` — live and
control as distinct inputs, both directions, which is what makes "load_bearing" mean
something.

**Deleting each raise site in turn (13 sites at this tip): 12 KILLED, 1 SURVIVED.** The
three new ones all die by name (`already_superseded_at_stamp`,
`malformed_block_missing_field`, `superseder_timestamp_unparsable`). The survivor is line
476 — the `population not in KNOWN_POPULATIONS` branch, which is **DE-R4 reproducing at
this tip**: reachable (`population: SOMETHING_NEW` refuses by name when driven) and proven
by nothing.

**Rule 10 / rule 14.** `decides: "nothing -- this reports; admission is the coordinator's
act and accrual is decided elsewhere"` is in the emission and asserted by a check;
`verified_for_new_run` and `provenance` sit beside `refusal_scope`, which states what a
refusal means. No printed conclusion beside a predicate. `verified` reads as consistency,
not entitlement, and the emission says so.

---

## Executed evidence

At `2282e5c`, 2026-09-02T11:36–11:40Z:

| check | result |
|---|---|
| suite, both launchers | **58 checks**, rc 0 |
| all ten fields missing, in turn | **all REFUSE by name** — CO-5 closed |
| `scope_to` / `scope_from` malformed | verified True / verified False, **neither surfaced** — DE10-R1 |
| `now_utc` boundary | False / **True** / True at −1s / 0 / +1s |
| `now_utc='zzzz'` | **`day_closed True`, `verified True`** — DE10-R1 |
| default clock | read at call time, emitted and moving |
| three-ref chain stamped between B and C | A refuses, B is provenance with `superseder_times`, C verifies |
| superseder heading without a timestamp | **refuses**; body timestamps do not substitute |
| `require_verified` conjuncts | **all three load-bearing**, each named |
| audit | **12 guards, 0 survivors**, control direction included |
| raise-site deletion | **12 killed / 1 survived** (the unknown-population branch) — DE-R4 stands |
| `selected_from_specs` + a tape-less window | **REFUSED by name** — my correction |
| supply + bridge on a tape-less window | **1,876 emitted** — what my finding did establish |
| register file | never written |

---

## Disposition

- **RELEASED:** DE round 10. **No hold.**
- **CORRECTED, in band:** my claim that no layer refuses ledger-only windows. The driver
  does, through `fi._archive_paths()`/`fi.token_map()`; I established an absence with a
  grep and that was the error. What stands is narrower: DE's supply and the seam bridge
  emit tape-less windows, and the refusal is downstream. **I withdraw the "intersect"
  option and agree with refuse-on-difference at the population gate.**
- **FILED:** DE10-R1 (parse timestamps rather than ordering strings — one root cause,
  three symptoms, all permissive-leaning), DE10-R2 (nonsense values still verify clean),
  DE10-R3 (the uncontrolled refusal branch, carried from DE-R4).
