# EV-Replay — the replay environment plan

Status: **DESIGN**, not decision-eligible. **Revision 3, 2026-08-23** —
applies `DE_PLAN_REVIEW_LOOP.md` iteration 6: the header's gating claim scoped
to exactly what exists (Rev 2 said "gated green against §4" while §4.3's
must-fail controls did not exist — the claim-stronger-than-artifact defect in
the plan whose §4 exists to prevent it); §2's plugin-path rows marked NOT IN
v1; §4 given an explicit status table with the acceptance DEBT named and made
blocking. **Now Revision 7** — iteration 10, the loop-closing revision: §4.2's
retained "closure completed" reworded (the retired claim form had survived in
the file that retired it), cell vintage brought current, §2's spec-hash row
extended with the iteration-9 `inputs_hash` stamp. Prior — Revision 6,
iteration 9 (streak 1 of 2 under the pinned stop rule): `engine_hash` EXTENDED again — the record shapes (`Fill`,
`WindowFills`), the env's own record mapping, and the parity comparator were
residue (a `Fill` field reorder would have transposed every tuple with the
hash unmoved); the wording "closure completed" is retired — closure is a
property review earns per-iteration, not a state; per-window `inputs_hash`
(gaps + token ids) stamped so a `gaps_by_slug`/`token_map` change can no
longer shift `run_hash` unattributably. Prior: iteration 8: `engine_hash` extended
(the four line-filter marks, `HORIZONS`, `_gz_lines` — each could change
records with the hash unmoved); gate outcomes persisted IN the receipt (they
were stdout-only, leaving PASS cells checkable only by entailment); §2's
spec-hash row corrected (stamps live on the RECEIPT — `RunRecord` stays raw
per §0 — and the receipt carries a protocol NAME, not hash); §4 vintage
labels updated. **Revision history: Revisions 4–1 (all same-day) predate
per-iteration version control of THIS file, which was untracked before
`c0bae24` — iteration 8 corrected iteration 7's blanket claim: two OTHER
corpus files (the placement plan, `EDGE_LAYER1_RESULTS.md`) had 2026-08-22
commits, so their cumulative diffs ARE recoverable. Committed per-iteration
from `c0bae24` onward.**
Written by the DE session under dispatch B3: EV-Replay had no owning plan, is
on the critical path, and DE is today its only consumer.
**v1 of the harness (`ev_replay.py`) is built; its gate status is §4's table —
green on §4.1 (v1 form) / §4.2 / the iteration-6 sensitivity controls; §4.3's
engine-perturbation controls are OPEN DEBT that BLOCKS any engine change.**

> Precedence: `contracts/contracts.yaml` v22 wins on types;
> `PM_ARCHITECTURE.md` §9/§10 contract the environment; `FLOW_MODEL_STATE.md`
> wins on facts. Where this plan conflicts with any of them, they win.

---

## 0. Plane and the boundary that must survive DE authorship

`EV-Replay` is an **EV module**: EV reads all planes and is read by none. DE
writing this plan does not move the module — it records the one boundary that
matters and how it is enforced:

**The harness must never become a channel by which evaluation state reaches a
decision.** Concretely, three rules, each with an enforcement point:

1. **No EV output enters the policy loop.** Markout, calibration, gate
   verdicts and attribution are computed *after* a replay completes, from its
   emitted fill/action record — never inside the event loop, never visible to
   the solver. The replay emits **raw events**; evaluation is a separate pass.
   (Enforcement: the policy-facing interface contains no markout/gate type;
   evaluation runs only on a completed `RunRecord`, after the run loop
   returns — iteration 6 removed a reference to an `env.close()` that does
   not exist.)
2. **No gate state in `DecisionProblem`.** The problem is constructed from
   DA/BE/SP-plane views only; EV-Gates results influence *whether a run
   happens* (programme control, coordinator seat), never *what the policy
   sees*. (Enforcement: the problem constructor takes a `StateView` +
   `SelfState` + specs, and nothing typed from EV.)
3. **Replay is an `Environment` implementation, not a module with outputs.**
   Per architecture §9, Live/Replay/Sim are implementations behind the same
   ports; modules receive narrow ports declared in manifests. The replay
   clock + tape port goes to the replay runner alone.

---

## 1. What exists de facto, and what this plan converges

Five ad-hoc replay dialects already run inside the DE research code, one per
probe: `edge_layer1.replay_window`, `inventory_walk.simulate_window`,
`warning_window.replay_ww`, `placement_skew`/`skew_bound` arms, and
`policy_comparison`. They agree on the load-bearing conventions —

- state applied at the **frozen 250 ms knowledge lag** via a scheduled event
  heap; the lag is an environment constant, never policy-visible;
- book state from `price_change.best_bid/ask`, never `book` snapshots; mid as
  a prevailing step function; knowledge time = `recv_ns`;
- **gap state kill**: a collector gap clears state, retracts resting quotes,
  and marks the interval `UNAVAILABLE`;
- queue accounting via `RestingSide` under two bounds (`FRONT` /
  `BACK_DISPLAYED`), with auto re-post at the back on full lift;
- tick-change intervals marked and excluded from markout comparisons;
- complement trades deduplicated by transaction hash; micro-class flagged;
- window-clustered bootstrap; day-clustered refused below the cluster floor

— and diverge in everything else (loop structure, diagnostics, receipt
shapes). The `warning_window` conformance check proved the risk is real: a
sixth dialect diverging silently is one refactor away. **EV-Replay v1 is the
single environment these five converge into.** Existing receipts stand as
provenance; nothing is re-derived retroactively.

---

## 2. The environment contract (architecture §9, made concrete)

```
ReplayEnv(window_spec, tape, params) -> session
  session.run(policy: ControlSolver plugin | scripted arm) -> RunRecord
  RunRecord: fills, actions, cancels, episodes, mid path, UNAVAILABLE
             intervals, diagnostics — RAW events only, no evaluation
```

| contracted property | v1 mechanism |
|---|---|
| deterministic tie-breaking | total order on `(recv, seq)`; seq assigned in tape order; no dict-order dependence |
| RNG owned by the environment | **plugin-path contract, NOT IN v1** (v1's arms are deterministic; the receipt's `seed` field is stamped-and-reserved so receipts stay comparable when a stochastic arm lands). Then: seeded per `(run_id, window)`; policies receive the env RNG port, never seed themselves |
| knowledge-time reads | the 250 ms lag heap is the only path from tape to `StateView`; `EventTimeView` construction is licensed to the CANARY alone (R-CANARY: direct construction elsewhere is fatal — iteration 5 removed a parenthetical that had silently widened this to "conformance"; the parity gate compares fill records, which needs no event-time view) |
| warm-state snapshot + restart parity | per-window replay is pure: same `(tape, params, seed)` ⇒ byte-identical `RunRecord`; multi-window runs are per-window pure by construction (windows settle independently). **License boundary (iteration 5): this discharge lapses the moment cross-window state enters a replay** — e.g. the Allocator's capital coupling (module plan §5b RESOLUTION: next-window quoting gated on settlement-latency headroom). A capital-coupled replay needs real snapshot/restart machinery; do not extend the purity claim to it |
| artifact resolution | **plugin-path contract, NOT IN v1** (iteration 6: v1 resolves no artifacts and has no refusal to selftest — the row previously claimed "both selftested"). When the plugin path lands: resolve by `artifact_id + fit_data_through`; REFUSE `fit_data_through` postdating the window (**R-WFWD** `no_future_train` / R-REQ); separately refuse `t_known` manufactured from `t_event` (**R-IMPUTE** — `observed_needs_wire`/`strict_delay`/`rule_named`, admission via R-REFUSE `no_peek`). Two classes, two refusals, both selftested THEN |
| spec-hash pinning | every **receipt** stamps (iteration 8: the stamps live on the receipt — `RunRecord` stays raw per §0): `engine_hash`, protocol NAME, `params`/SP set, seed, `provenance(days_sampled)`, collector era, per-record queue bound + content hash, **per-window `inputs_hash` over gaps + token ids (iteration 9)**, lag, gate outcomes |
| actuation latency | contract: cancels/placements effective at `t + lag + τ`; τ an env parameter, fills before effectiveness still happen. **v1 status (iteration 6): the lag lives as the engine's internal frozen constant and no v1 arm cancels, so neither lag nor τ is yet a parameter** — parameterization arrives with the first cancel-capable or perturbation-control work (§4.3), whichever is first |

**Window selection is NOT the environment's job.** The sampler defect
(earliest-first, one-UTC-day samples — `FLOW_MODEL_STATE.md` §1f) is exactly
why: selection is an R-ADMISS decision the coordinator ratifies; the env takes
an **explicit window list** and stamps it, never chooses.

---

## 3. The policy seam — what the cancel grid and promotion actually need

1. **Scripted arms** (static JOIN/FRONT, skew rules, cancel triggers) for
   development grids — cheap, no plugin machinery, matching today's probes.
2. **The registered plugin path** for promotion: the same `RulePolicy_v1`
   registered in the solver registry, driven through `DecisionProblem`-shaped
   inputs constructed at knowledge time. **Promotion parity is a rule of the
   module plan (§6.1): a candidate promotes only as the registered plugin run
   under EV-Replay — a scripted transcription can develop, it cannot
   promote.**
3. **Counterfactual rows**: excised fills (the avoided fill's subsequent mid
   path), partial-fill-then-cancel rows, and `UNAVAILABLE` rows are
   first-class in `RunRecord` — the cancel protocol's accounting (§2.2)
   consumes them, not recomputes them.
4. **Both queue bounds in one run** where the policy permits it; bound is a
   stamped parameter, never inferred.

---

## 4. Acceptance — the harness itself is measured before it measures anything

| # | gate | v1 status (kept current per iteration; claims match artifacts) |
|---|---|---|
| 4.1 | **Golden-window fill parity** — reproduce `edge_layer1.replay_window`'s fill sequence exactly, both arms, the `warning_window::conformant` pattern as the acceptance gate | **PASS in the v1 form only**: the v1 engine IS the reference, so the gate compares two invocations (engine determinism) and **structurally cannot fail**. Honest label, stated here as in the code: parity becomes a real gate at the FIRST non-reference engine, and no engine change may land before it can fail |
| 4.2 | **Determinism** — same `(windows, params, seed)` ⇒ identical `run_hash`, where `run_hash` covers the FULL records via per-record content hashes (fills, mid path, unavailable intervals, diagnostics — everything `evaluate_markout` consumes), plus the `engine_hash` (coverage EXTENDED iterations 7/8/9 — never "completed"; iteration 10 removed this cell's own retained use of the retired claim form) | **PASS against the iteration-9 receipt** (`gates` block: 14/14 parity + determinism true, both arms; per-window `inputs_hash` present) |
| 4.3 | **Engine-perturbation must-fail controls** — a +50 ms lag perturbation must change a golden fill record; a broken tie-break must trip parity | **HALF-BUILT (B2, `7fc3702` + QA fixes).** The lag control is BUILT and PASSES in the strong form (compares FILL records across the smoke set, not hashes — QA F4 caught that hash difference alone is mechanical via mid-path timestamps; data-vs-plumbing failures exit distinctly per QA N4), landing in the same commit as the `lag_s` parameterization it required. **The tie-break control remains OPEN DEBT with a NARROWED BLOCK: no engine change touching event ORDERING may land until it exists and fails on demand.** QA F3 caught this cell claiming "not expressible" after the parameterization made it expressible — claims match artifacts again as of this edit |
| 4.4 | **Boundary checks** | **Partial**: class-namespace scans on `ReplayEnv` AND `RunRecord` (methods included, iteration 6) + raw-fields check + hash-sensitivity must-fail controls. Import-level separation does not exist in v1 (one module); it arrives when the plugin path splits the modules |

Selftest fixture status, stated exactly: the known-fill decomposition fixture
exists; known-`W` and known-counterfactual fixtures arrive with the
instruments that need them (§6.3), not before.

---

## 5. Deliberately not in v1

- **EV-Gates content** — B4, coordinator seat. This plan only promises the
  `RunRecord`/receipt shapes gates will read.
- **A Sim environment** (synthetic tape generator) — selftest fixtures only;
  a generative sim is speculative until something demands it.
- **Sampling rules** — R-ADMISS, coordinator (the 680-windows/coin re-sample
  decision remains open and this plan is indifferent to its outcome).
- **Live/venue anything** — out of scope for the programme.
- **Contract edits.** v1 runs as research code against the existing probes'
  conventions; the module-record/port formalisation lands with the DE §6.2
  batch when code starts, as additive records for `EV-Replay` mirroring §9's
  port table.

## 6. Build order within B3

1. This plan; coordinator sees it via D-4 (no freeze needed — it contains no
   decision rules; the only gated items it touches are already gated
   elsewhere: sampling, promotion bars).
2. Harness skeleton + golden-window parity gate against `edge_layer1` (the
   acceptance is measurable on day one — §4.1).
3. Port `warning_window`'s episode/envelope instrumentation as the first
   in-env instrument (it already carries the conformance discipline).
4. ~~The cancel grid (§2 of the frozen protocol) runs inside EV-Replay v1~~ —
   **DEAD-LETTERED (iteration 5): B7 returned DEAD-DEAD before this revision
   was a day old**, so the conditional failed and the grid is not built. The
   item stays struck rather than deleted so the harness's original demand
   provenance remains visible; the live consumers are §6.2–6.3 plus the
   surviving levers' replays (skew × terminal) and promotion parity. Revives
   only if the coordinator's 680-window re-sample ruling reopens the family.

## 7. What would falsify this design

1. **Parity is unachievable** — if v1 cannot reproduce the golden fills
   exactly, the five dialects disagree somewhere load-bearing, and THAT
   divergence must be characterized before any harness unification (it would
   mean published numbers depend on dialect).
2. **Determinism breaks under the plugin path** (hidden iteration order,
   float non-associativity across batching) — promotion parity would then be
   unauditable; fix before first use, not after.
3. **The boundary rule proves unenforceable at the seam** — if constructing
   `DecisionProblem` in-replay turns out to need any EV-derived input, the
   architecture's EV-reads-all rule and the promotion-parity rule collide,
   and the coordinator must arbitrate before code.
