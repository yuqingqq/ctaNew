# POLICY_OPTIMIZER_PROTOCOL — the pre-registered search, and the simulated actuator it runs on

REVISION: 1

**Status: DRAFT FOR COORDINATOR FREEZE — nothing here is frozen; nothing
runs before the freeze (R-110 item 4: the pre-registration is what made
R-109 citable). Drafted blind: no optimizer cell has been computed.**
Authored by DE under R-110 (user directive; the surface-freeze
authorization for exactly these two builds is the directive itself).

**What this is for:** R-109 compared two hand-picked policies and the
LEVELS said the loss is structural adverse selection — a difference
between two guesses was never going to move a level. This protocol
searches a SMALL, PRE-DECLARED policy space for any cell whose LEVEL
clears zero, on a train/holdout day split, with every cell reported.

---

## §0. Forbidden forms (R-44 §0, inherited verbatim in spirit)

The search is the danger here — an optimizer is a grid by construction.
Therefore: the grid below is FROZEN at ~20 cells; no cell may be added
after any receipt is read; **the headline is never the best cell** — it
is the full grid plus the pre-registered promotion test; promotion is
decided ONLY on the holdout days, at the day unit, by the §4 bar; a
promoted cell earns a CONFIRMATION protocol, not deployment. In-sample
(train-day) numbers rank nothing; they exist to show the search its own
overfitting (train-vs-holdout gap reported per cell).

## §1. Population and the day split, declared now

Day-grouped sampler (`select_by_day`, 30/coin/day, verdict coins
btc+eth), same as every receipt in this corpus; time-of-day skew
(earliest ~2.5 h per day) carried for comparability, stated per R-105
with n and as-of. **Split: TRAIN = 2026-08-20/21/22. HOLDOUT =
2026-08-23 + every COMPLETE era day that exists at run time and was
never trained on.** A partial day is a different population (R-109
item 4): partial days are reported beside, labeled, and NEVER decide
promotion. G on holdout is small ⇒ §4 uses signs and points, no
intervals (the R-109 ruled standard).

## §2. The axes — each with its existing receipt named, so the search
cannot re-run a closed marginal as if it were new

| axis | values | standing evidence, cited honestly |
|---|---|---|
| placement | JOIN, FRONT | R-109: FRONT ≥ JOIN (fills 5–6× day-robust; markout penalty nowhere). Depth-1 is EXCLUDED: measured `DEPTH_FAILS` both coins (POLICY_BOUNDS §8) — through-sweep selection, not room |
| terminal abstention `r_cut` | 0, 60, 120 s | **Premise correction carried into the design: this axis IS tested marginally** — POLICY_BOUNDS Lever T ran body-only (r_cut=60) for JOIN: `GATE_FAILS` both coins, body ≈ base, and R-50's inversion (the only positive bins sat IN the terminal minute). Its licence HERE is the INTERACTION space R-45 amendment 3 left open — abstention × FRONT is genuinely unmeasured, and FRONT's fill mass is formation-time, a different exposure to the terminal regime |
| size | 5, 10 | venue floor = pin = 5 (Class D); size marginally `DEAD_DEPLOYABLE` and the corpus size-invariant BELOW the pin (POLICY_BOUNDS §8) — the 10-share cell tests the interaction with FRONT's fill mass, not the closed marginal |
| skew (§1d rule) | off, on | §1d measured a 15× inventory-risk cut on ONE day as an UPPER BOUND with overlays; here it runs at book level, overlay-free, era days |
| cancellation | off, ww-envelope @ τ=500 ms | the REACTIVE family is CLOSED — DEAD across four channels at achievable rungs (R-11/R-49/R-54, 8/8 coin-days). Included as TWO interaction cells only because the directive orders the axis; expected NULL is stated in advance, and a non-null would challenge the closure, not quietly override it |

## §3. The grid — frozen at declaration

**Stage A (12 cells):** placement {JOIN, FRONT} × r_cut {0, 60, 120} ×
size {5, 10}, skew off, cancellation off.
**Stage B (+6 cells):** skew ON composed with ALL six FRONT cells of
Stage A (no selection — composition with the whole slice).
**Stage C (+2 cells):** cancellation @ τ=500 composed with
{JOIN, FRONT} at r_cut=0, size 5.
**Anchors (+2):** WAIT-only (identically zero — the abstention floor
every cell must beat to justify quoting at all) and the R-109 JOIN
baseline (continuity with the comparison receipt).
**Total: 22 cells. All run on train AND holdout; all reported.**

## §4. Objective and the promotion bar — declared before any number exists

**Objective: TOTAL M5 PnL per window** (Σ share×markout, ¢/window) —
NOT per-share markout. R-109's lesson is the reason: at negative
per-share levels, more fills is more loss; per-share ranking flattered
FRONT while total loss grew 5–6×. Per-share swm and fill shares are
reported beside as diagnostics.

**Promotion bar (holdout only, day unit, no intervals):** a cell
PROMOTES iff on the holdout days its total PnL/window is (a) > 0 on
EVERY complete holdout day, BOTH coins, and (b) > the WAIT-only anchor
(trivially 0) and the R-109 baselines on those same days. Anything
else: the cell is reported and nothing happens. If NO cell promotes,
**that is the expected outcome and it is a programme-level answer**:
the structural-adverse-selection reading survives a 22-cell search of
every axis this programme ever nominated.

## §5. The SIMULATED ACTUATOR — Actuator semantics, replay-side (R-110 item 3)

NOT the venue writer (out of scope, declined by the coordinator; module
plan line 46's "sole venue writer" path stays unbuilt). What the
optimizer needs, built against the tape:

- **Order lifecycle**: place → resting (queue via `RestingSide`) →
  filled / cancelled; every transition recorded RAW in the RunRecord.
- **Rate budget**: actions per window capped by a Class-A constant
  (declared in the receipt; the SP-Venue rate-limit row is `Unknown`,
  so v1 uses a generous cap and STAMPS it — a binding cap would be a
  finding, not a silent truncation).
- **Reconciliation**: position = fold of the fill stream, checked
  against the ledger every window end; divergence is fail-loud.
- **halt_in = HALTED ⇒ refuse-all except cancel_all** — the §6.2
  second-door semantics, executable in replay; exercised by a selftest
  that drives a mid-window halt and asserts the refuse-all path.
- **NULL-POINT CONFORMANCE, the load-bearing gate**: at the null
  parameters (r_cut=0, skew off, cancel off, size 5), the
  PolicySession's fills MUST reproduce `edge_layer1.replay_window`
  EXACTLY per window, both placements — this makes §4.1's parity gate
  REAL for the first non-reference engine that runs inside the
  harness rather than beside it. Any mismatch aborts the run.

## §6. Controls (before any cell is read)

Null-point parity (above, every window); determinism (repeat replay,
identical run_hash); §4.3 lag perturbation (+50 ms must move fills);
axis sensitivity must-fail (r_cut=300 must produce ZERO fills — an
abstention parameter that cannot reach the tape is not wired); the
promotion-bar comparator demonstrated on constructed cells (a doctored
holdout day flips the verdict).

## §7. Sequencing

1. This draft → coordinator freeze (the §0a ask names the decision-path
   item it gates). Build proceeds UNDER SEAL meanwhile: engine +
   actuator + selftests may run; no optimizer cell is READ before the
   freeze.
2. On freeze: run train + holdout, read against §4, report ALL cells —
   train, holdout, gap — cells first, promotion test second, per
   R-9/R-17 shape, every population with n and as-of (R-105).

---

## STAGE A — RUN AND ANSWERED, 2026-08-24 (appended per R-28; §§0–7 untouched)

**Receipt:** `derived/policy_optimizer_stageA.json`. **Controls first, all
PASS** (R-111 order): wiring must-fail r_cut=300 → **0 fills** (after the
lead-in semantic it caught on first launch was pinned — report #69); null
parity exact, sample AND every window in the main pass, both placements;
determinism; +50 ms moves fills. **Population: 300 windows — five COMPLETE
era days (08-20..24, 60/60 each; 08-24 completed between tick and run),
btc+eth, as-of 2026-08-24 run time.** TRAIN 08-20/21/22; HOLDOUT 08-23 +
08-24 (both complete; partial-beside list empty). Day-unit reporting:
points and signs, no intervals (G=2 holdout).

**PROMOTION: NO CELL PROMOTES — the §4 pre-declared expected outcome.**
All 12 cells × 2 coins × 5 days = **120 cell-days, every one NEGATIVE**
(total M5 PnL/window). The WAIT-only zero anchor dominates every quoting
cell on every day, both splits — signs are 120/120, no interval needed.
Best cell in the entire grid: JOIN:r120:s5 eth, −194..−252 ¢/window.
Worst: FRONT:r0:s10 btc, −6,176..−13,627 ¢/window.

**Structure in the cells (descriptive, no promotion implied):**
1. **Abstention scales the loss toward zero and cannot cross it** —
   monotone r120 < r60 < r0 in |loss| within every placement×size×coin.
   Terminal loss-density is MILD: JOIN's last 40 % of clock carries
   ~55–60 % of its loss, FRONT's ~46–54 % — consistent with Lever T's
   "the damage is everywhere" and R-50's inversion; nothing resembling a
   concentrated sink that abstention could excise.
2. **Size scales loss ≈ linearly** (s10 ≈ 1.6–2.0× s5 everywhere) — the
   corpus size-invariance again, now at the total-PnL level.
3. **FRONT loses 4–5× JOIN in TOTAL on every day** — R-109's volume
   arithmetic confirmed out-of-sample on two fresh holdout days.
4. **The abstention×FRONT interaction (this axis's licence) EXISTS and is
   small and unhelpful**: abstention removes proportionally slightly MORE
   of JOIN's loss than FRONT's; neither approaches zero.

**Programme-level answer (§4's own sentence, now earned):** structural
adverse selection survives a pre-registered search of every axis this
programme ever nominated — placement, terminal abstention, size, and
their interactions — with every cell losing on every day of a five-day
era population. Stages B (skew×FRONT) and C (cancellation interaction,
expected null declared) remain declared-but-unrun, awaiting order.
