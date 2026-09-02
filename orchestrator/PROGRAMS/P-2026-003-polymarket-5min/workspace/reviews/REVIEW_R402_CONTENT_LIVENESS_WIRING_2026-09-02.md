# Review — R-402: the frozen content-liveness rule wired into the verdict path (Q-DA-202)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `3298a1d`.**
**Continues** `REVIEW_RR4_CLOSURE_ACTION_UNIT_2026-09-02.md` (`cd67c20`).
**Composed 2026-09-02T08:11:00Z.** One filing, per R-377.

Method unchanged: detached worktree at the pinned tip, ledgers symlinked
read-only, verdicts from execution; mutants applied in the lab and reverted; no
repository file modified and nothing written to any production ledger, verdict
or tape. Every re-run of the verdict tool was to stdout or scratch — the canonical
`da_dayverdict_*.json` artifacts were read, never rewritten.

---

## Verdict

### The wiring is REAL and RELEASED. I could not produce a verdict that skipped the frozen rule, and the live finding reproduces independently from the raw bytes with my own code.

### One finding, and it is not about the wiring: **DA's "L2 cannot shrink" projection is false in the case that matters, because the frozen rule measures thinness against the day's OWN median. A day that goes predominantly dark reads CONTENT_LIVE with L1 = 0.0000 and L2 run = 0.** Computed on the real 09-02 btc data. **ESCALATION-FOR-USER** — the rule is USER-frozen and this is a property of its §3 definition, not of DA's batch.

Second, minor: no canonical verdict on disk carries the block yet (RR6-2).

---

## Scope 1 — the wiring, proven by execution

### The guard chain, driven through the REAL launcher

Baseline first: `verify --day 20260829 --freeze-epoch 1787897340` → **rc 0**.

| mutant (production code, applied in the lab) | selftest | real CLI |
|---|---|---|
| **the production call `content_liveness_rule_for(day_token)` DELETED** | **DIES** at *"WIRING END-TO-END: a report from the REAL verify_day CARRIES the frozen rule's block — the call cannot be deleted without this failing"* | **rc 4**, "INSTRUMENT FAILURE … NOTHING WAS VERIFIED" |
| the emitted block reports `governs=True` regardless of the module | **DIES** | **rc 4** |
| the composition adopts the veto (`content_thin_vetoes_HEALTHY=True`) | **DIES** | **rc 4** |

**This is the claim I most wanted to test and it holds: rc 4, never rc 1.** A verdict
produced without consulting the frozen rule is an instrument failure, not a failing
day, and the two do not share an exit code. W1's fix is genuine — the check builds a
report from the REAL `verify_day` and runs the REAL guard on it, so deleting the call
can no longer pass through stubbed seams. That was R-402's own defect surviving its
own fix, DA found it, and the closure is the right shape.

`governs` is checked for **honesty**, not pinned: the guard compares the artifact's
field against `CLR.governs(day)`. I confirmed the failure mode it protects against by
mutation (hardcoding True → refusal), and the reasoning is right — pinning either
value would have been the draft-state trap in the other direction.

The composition is **escalated, not chosen**: `content_thin_vetoes_HEALTHY: false`
in the emitted block, with the frozen §7/§8 citation in `why_not`, the
`ESCALATION-FOR-USER` marker, and decision material computed **both ways**
(`would_flip_HEALTHY_under_worst_coin_composition: true` and the per-coin map, all
seven true). A worker declining to adopt a policy by shipping it is rules 11/14
applied correctly, and the guard makes the decline enforceable.

### The wiring is additive — measured, not accepted

I re-ran 09-01 at the tip and diffed against the stored canonical verdict:

- `all_pass` **True → True**, `race_accrual_eligible` **True → True**;
- **no predicate added, none removed, none changed PASS/FAIL**;
- the only differences are `gap_series` and two `detail` strings
  (`complete_tape`, `entirely_post_freeze`) — all as-of dependent, and the re-run is
  eight hours later than the stored artifact (`as_of` 00:06:01Z vs 08:07:08Z).

The additive claim holds on every governing field.

### RR6-2 — LOW — no canonical verdict on disk carries the block yet

All eight `da_dayverdict_*.json` in `derived/`, **including `20260902`**, were written
at **00:06:0xZ** — before this batch landed at 08:03Z — and none carries
`content_liveness_rule`. The wiring is proven end-to-end, but the artifacts of record
predate it; the first governed artifact is tonight's 00:06Z run (or a re-run with
`--write-reason`). Worth stating so the on-disk 09-02 verdict is not later cited as
the governed one.

---

## Scope 2 — DA's live finding, reproduced independently

I did not read DA's numbers out of the artifact. I measured the raw window files
myself — **compressed** bytes, my own grouping, no gap exclusion — against the rule's
`thin_frac = 0.05`:

| coin | windows | thin | longest run | run span | last thin window ends |
|---|---|---|---|---|---|
| **btc** | 97 | 40 | **40** | **01:35 → 04:55Z** | **04:55Z** |
| eth | 97 | 41 | 40 | 01:35 → 04:55Z | 04:55Z |
| xrp | 97 | 40 | 40 | 01:35 → 04:55Z | 04:55Z |
| sol | 97 | 40 | 40 | 01:35 → 04:55Z | 04:55Z |
| bnb | 97 | 39 | 39 | 01:40 → 04:55Z | 04:55Z |
| doge | 97 | 39 | 39 | 01:40 → 04:55Z | 04:55Z |
| hype | 97 | 25 | 15 | 03:40 → 04:55Z | 04:55Z |

**btc's 40 consecutive windows, 01:35–04:55Z, and all seven coins ending at exactly
04:55Z — both confirmed.** My medians differ from the artifact's by ~9× because
`scan_day` aggregates **uncompressed** bytes and I measured compressed ones; I checked
that before treating it as a discrepancy, and it is not one. The classification agreeing
across a 9× change of scale is itself corroboration.

**The gap ledger's charge across that window, computed by overlap:** btc **32.0 s**,
eth **1.5 s**, sol 33.4 s — against a 12,300 s span. The ledger sees **0.26%** of it.
That is the finding, and it is real: the governing bars pass because they are fed by a
ledger that never recorded the loss.

---

## RR6-1 — HIGH (on the FROZEN RULE, not on this batch) — "L2 cannot shrink" is false where it matters, and the rule is blind to a TOTAL blackout

DA's projection (7): *"L2 is a RUN LENGTH and cannot shrink as the day fills — 40
windows stays 40 — so 09-02 will fail L2 at close whatever the rest of the day does."*

A run length over a fixed classification cannot shrink. But the classification is not
fixed: `invis = [w for w in wins if b < median(wins) * thin_frac]`, and the median is
recomputed over **all** the day's windows. As the day fills, the median moves — and if
it moves DOWN, windows stop being thin.

**Computed on the real 09-02 btc window sizes, extending to 288 windows:**

| continuation | n_thin | longest run | L1 | L2 passes? |
|---|---|---|---|---|
| as measured now (97 windows) | 40 | **40** | 0.4113 | no |
| rest of day NORMAL | 40 | **40** | **0.1385** | no |
| remainder 30% dark | 97 | 57 | 0.3360 | no |
| remainder 50% dark | 135 | 95 | 0.4677 | no |
| **remainder 60% dark** | **0** | **0** | **0.0000** | **YES** |
| **rest of day DARK** | **0** | **0** | **0.0000** | **YES** |

DA's **L1** projection is exactly right — 0.1385 against the quoted ~0.14, ~1.7× the
0.08 bar. The **L2** claim is right under the benign continuation and **wrong under the
severe one**: the flip happens precisely when dark windows pass half the day, because
the median crosses into the dark regime and every dark window stops being "thin
relative to the median".

**The general property, which is the part that matters beyond tonight: the rule detects
PARTIAL blackouts and is structurally blind to TOTAL ones.** A day 100% dark has
`L1 = 0`, `L2 = 0` and reads **CONTENT_LIVE**. The only absolute floor in the rule is
`median <= 0 → UNJUDGEABLE`; a median of 11,637 bytes (0.2% of normal) is judged
healthy. This is rule 16's shape on the instrument itself — a control that cannot fire
in its worst case.

**What I am NOT saying:** this does not affect 09-02's reading (the day is 41% dark and
reads CONTENT_THIN correctly), and it is not a defect in DA's wiring, which is exactly
what it claims to be. The rule is **USER-frozen**, so the remedy — an absolute floor
beside the relative one, or a cross-day median — is a frozen-document amendment and
belongs with the USER, not with a seat.

**Two things to correct in band:** the projection's *"whatever the rest of the day
does"* should read *"unless the remainder is itself predominantly dark"*, and the §8
escalation now carries a fourth question worth asking beside (a)/(b)/(c): **should the
rule carry an absolute floor, given that its relative one cannot see a total blackout?**

---

## Executed evidence

At `3298a1d`, as of 2026-09-02T08:11Z:

| check | result |
|---|---|
| `v5_deploy_gates.py` | **16 of 17 pass**; the single red is `v4 behaviour (git-extracted)` |
| that gate standalone | **10/10, three times** — the documented wall-clock flake, second consecutive round |
| baseline `verify --day 20260829` | **rc 0** |
| mutant: production call deleted | selftest **DIES**; real CLI **rc 4** |
| mutant: lying `governs` | selftest **DIES**; real CLI **rc 4** |
| mutant: veto adopted | selftest **DIES**; real CLI **rc 4** |
| 09-01 re-run vs stored verdict | `all_pass`, `race_accrual_eligible`, every predicate's PASS/FAIL **identical**; only as-of-dependent details differ |
| emitted block | `governs: true`, `effective_from_day: 20260902`, `frozen_by_user: true`, module sha `7196676840304f30`, R-386 authority, `content_thin_vetoes_HEALTHY: false`, both-ways decision material |
| btc blackout, measured independently | **40 consecutive thin windows, 01:35–04:55Z** |
| all seven coins' thin runs | **all end at 04:55Z** |
| gap seconds charged in that span | btc **32.0 s**, eth **1.5 s** of 12,300 s (**0.26%**) |
| median-scale difference vs the artifact | explained: `scan_day` counts **uncompressed** bytes |
| L1 projection to 288 windows | **0.1385** — DA's ~0.14 confirmed |
| L2 under a dark remainder | **run 0, L1 0.0000, L2 PASSES** — RR6-1 |
| canonical verdicts carrying the block | **0 of 8** (all written 00:06Z, before the batch) — RR6-2 |
| mutants executed this round | **3, all killed and all rc 4 through the real launcher** |

---

## Disposition

- **RELEASED:** the R-402 wiring, the rc-4 discipline, the honesty check on `governs`,
  the refusal to adopt the composition, the both-ways decision material, and the
  additive-wiring claim. **No hold from this seat is open at `3298a1d`.**
- **ESCALATION-FOR-USER (RR6-1), routed through the coordinator:** the frozen rule's
  relative-median definition cannot see a total blackout, and the "L2 cannot shrink"
  projection fails in exactly that case. Both the correction and the proposed fourth
  §8 question are above.
- **FILED, not holding:** RR6-2 — no canonical verdict yet carries the block; tonight's
  00:06Z run produces the first one.
- **On the finding itself:** two independent instruments and now a third — my own read
  of the compressed bytes — agree that 09-02 lost roughly 3 h 20 m of content while the
  gap ledger charged 32 seconds. Whatever §8 is ruled, that gap between what the tape
  holds and what the ledger says is the thing worth carrying forward.
