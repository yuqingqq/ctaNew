# REVIEW 245 — DA's blocking list checked at the code; 13 of 15 are economic-only

REV round 208. Filed 2026-09-11T21:14:37Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled. Fetched at 21:08Z. Worktree
detached at `e2863db`, clean. Both executing refs `e2863db`.

**The freeze landed, verified as a count at both fetched refs:**
`da_step6_full_pipeline_freeze_v1.json` — chain 1, runner 1, blob `3afe6c3cb5f8`
on both. `freeze_is_effective: false`, `n_blocking_gaps: 15`.

## THE TWO JUDGEMENTS, FIRST

**(2) Thirteen of the fifteen gaps are economic-only. The predictive half is
blocked by ONE thing, and it is named twice.** §7's chain splits cleanly: links
1–7 (immutable inputs → labels/statuses → actions → sigma → FairPrice →
fallback → score) are what §8's log loss consumes; links 8–10 (quote mapping,
replay, P&L) are the §9 economic leg.

| gap | blocks |
|---|---|
| `chain_link_not_implemented:immutable_inputs` | **PREDICTIVE** |
| `source_manifests` | **PREDICTIVE** (the same gap: nothing identifies the inputs) |
| `chain_link_not_implemented:pnl`, `fee_rule`, `initial_inventory`, `latency:placement_latency_ms_not_bound`, `quote_parameters`, `tick_rounding`, and the 7 `quote_mapping_property:*` | economic only |

Of §7's 14 recorded fields, 9 are present and 5 missing — `fee_rule`,
`initial_inventory`, `quote_parameters`, `source_manifests`, `tick_rounding` —
and exactly one of those five, `source_manifests`, is needed predictively.

**So the user's alternative is available.** Freeze and validate predictively
now, economics later, with one deliverable on the critical path: a manifest
enumerating the immutable inputs with digests, which resolves both the chain's
first link and the `source_manifests` field. That is one seat's work, not four
seats' night. The caveat to state with it: §8 resolves day eligibility from
*"the frozen day/book gate, official resolutions and settlement-verification
coverage"*, which is precisely what that manifest must enumerate — so the one
blocker is load-bearing for eligibility, not incidental, and it must be a real
manifest rather than a field filled in.

**(1) The hardened read is not unspoofable, and it is weaker than it looks in a
specific way: the two fields are one bit.** At the code,
`"freeze_is_effective": not gaps` (line 549) and
`"n_blocking_gaps": len(set(gaps))` (line 566) are both functions of the same
`gaps` list — DA's own cell asserts the identity
(`d["freeze_is_effective"] == (d["n_blocking_gaps"] == 0)`). Requiring both in
the same document is checking one thing twice, not two things.

The gap list has two halves. The **enumerated** half comes from two module
tuples, `REQUIRED_FIELDS` (14) and `CHAIN` (10). I shrank both and re-ran
`build()`:

    as shipped:   freeze_is_effective=False  n_blocking_gaps=15  (14 fields / 10 links)
    after shrink: freeze_is_effective=False  n_blocking_gaps=8   ( 9 fields /  8 links)

Seven gaps removed by editing two tuples, with every underlying gap still real.
Grepped in visible-failure form: **zero** lines in the checker pin
`len(REQUIRED_FIELDS)`, `len(CHAIN)`, or the ten link names. The **driven** half
— the seven quote-mapping properties — survived the shrink, which is genuinely
better than I expected and is the part that does resist editing.

Except that it does not resist a probe error, which is §3 below.

To close (1) properly: assert the two constants against §7's enumeration as
cells (14 fields; the ten link names in that order), and make a probe that
cannot run an explicit gap rather than an absence.

## 1. DA's list, item by item, checked at the code

Nine of eleven claims verified exactly as stated. Two are wrong as worded, and
both are wrong in the direction of **more** work than is needed.

| claim | verdict at the code |
|---|---|
| rounding is symmetric `round(x,12)`, no directional tick | **TRUE** — `quote_from` lines 74–75, both sides `round(·, 12)` |
| ask 1.009 at p=0.999, bid −0.009 at p=0.001 | **TRUE, reproduced exactly**; also 1.01 / −0.01 at p=1.0 / 0.0 |
| `quote_from` takes no side or outcome, so no DOWN | **TRUE** — signature has neither; zero `1 - p` expressions in the seam |
| the seam models no latency | **TRUE** — `placement_latency` in 0 of the 8 frozen-chain files |
| `placement_latency_ms = 250` bound only in the cancellation lane | **TRUE** — bound in `be_daybook_build.py`, `de_phase4_diag_runner.py`, `de_multiday_gate1_runner.py`; 0 in the chain |
| no fee anywhere in the frozen chain | **TRUE** — every `fee` hit in the chain is `feed` / `feeds` / `feeding` |
| no frozen `initial_inventory` | **TRUE** — only `initial_state.get("inventory", 0.0)` in the replay seam |
| `immutable_inputs` and `pnl` unimplemented | **TRUE** — 8 of 10 links carry paths; `pnl`/`profit`/`cash`/`notional` all 0 in the chain |
| the seam satisfies 1 of §7's 8 quote-mapping clauses | **TRUE** — the one satisfied is *"replace only the Identity anchor; on non-OK status use Identity"*, which I drove green at gate 5 (9/9, fallback counted) |
| **`MARKETABLE_CROSS` in ZERO `.py` files lane-wide** | **FALSE as worded** — it is in exactly one: `da_step6_full_pipeline_freeze.py`, DA's own checker. Zero *producers*, which is the substance; the count is wrong |
| **no legal tick declared anywhere in the lane** | **FALSE as worded** — see below |

### 1.1 The tick already exists, and it is not a constant

In the frozen chain DA is right: all 31 `tick` hits are *time-series* ticks (a
price observation), `_tick_lo` / `_tick_hi` are fixture variable names, and no
price tick size is declared. But lane-wide:

    tier1_pipeline.py:781   self.tick_size = 0.01
    tier1_pipeline.py:802   self.tick_size = _finite_float(message["tick_size"], "tick_size")

**The legal tick is a per-market field on the book envelope, already ingested**,
and the lane's own witnesses carry two different values — `tick_size="0.01"`
(`da_duplicate_identity_scan.py:409`, `tier1_pipeline.py:2324`) and
`tick_size="0.001"` (`da_envelope_witness.py:237,243`).

This changes the work. The task is **not** "declare a legal tick" — declaring a
lane constant would be wrong, because two markets already differ. It is: bind
the envelope's per-market `tick_size` into the quote mapping, and **refuse** when
a market supplies none. DE's own note, which I found in the probe's captured
output, already says this: *"a quoter that invents its own tick or latency is
quoting a market it made up."* DE is right, and the envelope is where the tick
comes from.

### 1.2 `PLACE_WITHHELD` exists too

Zero in the frozen chain, but four files lane-wide: `be_inert_arm_run.py`,
`be_trajectory_export.py`, `da_replay_parity_battery.py`, and DA's checker. So
the withhold vocabulary exists to be reused rather than invented; only the
`MARKETABLE_CROSS` reason and the seam's emission of it are new.

## 2. One item DA's list does not contain, and should

**DA's own checker is RED at the ref.** `da_step6_full_pipeline_freeze.py
--falsify` exits **1**, 20 PASS / 1 FAIL, and the failing cell is the
quote-mapping one:

    [FAIL] the quote mapping is DRIVEN, and its unmet clauses are NAMED -> PROBE_ERROR

Two separate faults behind it.

**(a) The probe errors, and the error is permissive.** `quote_mapping(REF)`
returns `{"error": "IndexError: list index out of range", "raw": …}` — the
harness takes the subprocess's last stdout line and JSON-parses it, and the last
line is DE's prose note about the tick. **When that happens the eight
quote-mapping gaps vanish rather than being recorded as unmeasured.** Driven,
reproducibly, twice:

| | n_blocking_gaps | fields present/missing | `quote_parameters` |
|---|---|---|---|
| landed artifact | **15** | 9 / 5 | `"MISSING"` |
| `build()` now, at the ref | **6** | 10 / 4 | dict, `unsatisfied` absent |

The artifact on disk is the good one; the live recomputation is degraded and
reads *closer to effective*. This is the identical shape DA fixed in the ledger
one round ago — an unmeasured thing must not read as satisfied — and the
identical harness bug (last-stdout-line JSON parse) that killed the gate-6
property probe. Third appearance.

**(b) The cell's predicate is inverted.** It passes only when
`quote_parameters` is a string or when `unsatisfied == []`. A cell named *"its
unmet clauses are NAMED"* should assert that the clauses **are** named —
non-empty and each named — not that there are none. As written, the instrument
fails its own falsifier precisely because the lane is in the state the
instrument exists to describe.

Neither fault changes any of §1's findings — I verified those at the code
independently of the probe — but a checker whose failure mode deletes findings
cannot be the thing four seats work against overnight. **Fix this before the
list is worked.**

## 3. What I would put on the critical path

1. **The immutable-inputs manifest** — resolves the chain's first link and
   `source_manifests`, and unblocks the predictive half on its own.
2. **The probe-error path** — a gap that cannot be measured is a named gap
   (`quote_mapping_probe_did_not_run`), never an absence; and fix the
   last-line-JSON harness, which has now cost three probes.
3. **The two constant-pinning cells** — `len(REQUIRED_FIELDS) == 14` and the ten
   link names in order, so the enumerated half of the gap list cannot shrink.
4. Economic leg, unblocked by nothing above and not on the predictive path:
   bind the envelope `tick_size` (refuse when absent), directional rounding,
   range bounds, `PLACE_WITHHELD` / `MARKETABLE_CROSS`, the DOWN complement, a
   latency model in the seam, a fee rule, a frozen initial inventory, and a P&L
   link.

## 4. Owed

- On any re-landing of the freeze: the count at both fetched refs, the six gate
  rows re-driven against the frozen blobs, and the §7 checklist from REVIEW 244
  applied field by field.
- REVIEW 244 §1 stands and is unaffected by this round: under a freeze dated
  today, validation day one is 2026-09-12, which is day six of the cancellation
  lane's live N=7 population.
- Standing: one review per day for the 09-09..09-13 records (REVIEW 228); the
  round-boundary landing sweep as counts at fetched refs.
