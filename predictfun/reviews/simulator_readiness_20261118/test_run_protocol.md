# Test-run protocol — P-2026-005 keep/pull simulator

This protocol says what a test run of the keep/pull simulator must produce, and what its results may and may not claim. It applies to
every run through `predictfun/research/run_keep_pull_scenarios.sh`. The chain is:

- settlement times;
- the execution-window ledger;
- the known-order ledger with its clock census;
- the queue-aware scenario replay.

Check every item against FINDINGS.md section 1 (C1–C24) before quoting a number. The readiness loop's receipts are linked in
[plan.md](plan.md).

**State at commit.** Every module's selftest and mutation suite pass on the committed sources; the record is in plan.md's suite of record.
The reference run, Oct 23-27, is [scenario_checklist_v6_20261118](../scenario_checklist_v6_20261118/report.md). A new run is comparable to
it only through the same driver, sources and scenario bank.

## 1. What the simulator is

- **It re-simulates.** It replays one declared keep/pull rule through the validated queue / fill kernel (`pf_flow_control_kernel`,
  queue rule M), on consumed days, under declared execution scenarios. The rule is lane A+0, the fair-edge score: `pf_keep_pull.py`
  DECLARED; one 5-sh order per side; states every 50 ms.
- **Scenarios are assumption ranges, not measurements.** They cover:
  - match-time timelines S1 / S2 / S3 / ST / EV;
  - trade-clock shifts M0 / M50 / M250;
  - activation 15 / 58 / 100 ms;
  - cancel latency 0 / 37 / 100 ms, plus a slow mixture (13% of cancels at 4 s);
  - notice delay.

  Values are reported per scenario and pooled over the days, with by-day values.
- **Values are local marks** at END (+5 s), in cents. They are not settlement P&L and not a self-financing portfolio.
- **E is the declared score's improvement over its matched null.** KEEP is resting without pulls; PULL-ALL pulls at every state.

## 2. Gates: a run whose gates do not all pass is not quoted

| Gate | Where it is read | Must read |
|---|---|---|
| Synthetic selftests of every module | the driver's `selftest` phase | pass |
| Mutation suites (every mutant caught by a check, not a crash) | `*_mutants.py` logs in the receipt | n / n |
| Z0: the baseline reproduces the committed replay order by order (control a); KEEP = the sum of its fills (control i); every baseline fill's trade source = the committed one (control j) | scenario report | YES |
| Settlement times v2: controls C1-C4 per day; `hours_excluded` lists only sealed / forward hours, never opened | each day's settlement.json | pass |
| The ledger's per-day controls (V0: collapsed E / score = the committed per-day values) | ledger receipt | pass |
| `K6 GATE S3 (per clock domain)` on the known legs | known ledger v7 | PASS for any S3-based claim |
| N0 (the census reproduces the case audit) | census v4 | YES |
| C5: the cancel race study reproduces census v4 on its 86 filled orders | cancel race study | YES, for any claim about cancels |
| Memory: peak RSS per job under the 2 GiB self-guard (run_light's receipt line) | job logs | under 2 GiB |
| Reproduction of the previous version on the same day (`compare.py` in each step's receipt) | step receipts | ALL true |

## 3. What a run must report (FINDINGS checklist)

| Item | The run's output |
|---|---|
| C1 age | E by the order's age at the fill (< 100 / 100-500 / >= 500 ms after activation). Beside the headline: the headline without the < 100 ms fills, and B1a-weighted at FV3's M 0 / M 50 precision weights. Reading `C1 B1a-WEIGHTED E AT B` |
| C2 fill clocks | M0 / M50 / M250 scenarios; readings `C2 E AT M250 vs M50` and `C2 KEEP + SCORE AT M250 vs M50` (HOLDS-AT-M250 / SIGN-DIFFERS-AT-M250) |
| C3 cancel-pending | Fills landing while the score's pull is in flight: count, shares, value, share of the fills' gross |
| C12 anchor source | C1's age table split by each fill's trade-anchor source: WS / E2 / imputed / mixed. E2 anchors are uncertain by ~+-250 ms |
| C15 whole portfolio | Every fill, every clock, with and without young fills, against KEEP / PULL-ALL / the matched null; pooled and by day |
| C19 denominators | Every split names its population; shares on their own denominator |
| C20 / C23 / C24 windows | Match windows with bound types and clock domains (the ledger); ST bounds from above only |
| Exclusions | As statuses with counts (eligible, refused post-only, censored, missing path) |
| C8 days | Named; consumed days only |
| C9 | "Ours simulated, theirs real" wherever a replay is set beside live or public data |
| C11 strategy | Named: lane A+0, not the frozen pm_ladder v3 |

## 4. What a run MAY claim

- **Scenario-conditional values.** For example: "under the S3 windows, a 37 ms cancel and M50, E = … c; without the < 100 ms fills …;
  B1a-weighted …".
- **The strongest stability claim is "stable in tested scenarios"** (readings Z1-Z3, Z1-NY / Z3-NY). A NOT-STABLE reading is reported as such.
- **Instrument facts:** the gates above.
- **On the known legs (Nov 8, one consumed partial day):**
  - clock-domain bounds and the gate of record per scenario;
  - match-before-request evidence (K9);
  - the per-cancel race classes (the cancel race study);

  each with its stated premises (K7 for the status clock; the value link and the book-clock premise for K9).

## 5. What a run may NOT claim

| Claim | Why not |
|---|---|
| Profitability or P&L | Local marks of a replay whose fills are not validated at the order level (C10). The replay credits 1.5-2x live's shares (C4); young fills were 8-12% real, and ~47% of fills >= 500 ms (B1a weights) |
| That a pull avoids a fill | The match-versus-cancel order of public fills is not identified. Every certified avoidance rests on an assumed lower match bound (S3's WS delay, S1 / S2 ranges). A venue Canceled answer orders neither the match nor the cancel |
| A cancel-failure probability or a landing-time distribution | The cancel's effect time is never observed. No model may be trained from notice-after-answer, the cancel census or the race study's classes |
| Robustness to the queue rule (C5) | Only rule M is replayed. L is not shown, and M is unvalidated on live orders (a fifth of live's fills contradict the shown-size queue, B1a) |
| A keep / pull policy acceptance (C15-C17) | Needs an acceptance criterion declared before the run (C15), chronological fits and a frozen forward test on >= 5 untouched days (C16), and quote availability and guard survival (C17, not computed here) |
| Validation | Consumed days are development evidence. Validation needs a freeze (C8 / C16), and an order-level check needs a live run with capture (C10). Both are the USER's call |

## 6. Data and run rules

- Consumed days only: Oct 23-27, and Nov 8 read to 23:02Z. Never the sealed 10-28..30, the forward blocks 11-04..06 and 11-09..13, nor any
  untouched day.
- **Hour padding is where day rules break** (FINDINGS P4 / P11 / P35; the readiness loop broke it once:
  [sealed_read_fix_20261118](../sealed_read_fix_20261118/report.md)).
  - Every reader refuses sealed and forward hours before opening a file: the replay loader `part_allowed` / `SKIP_FOREIGN_PARTS`, and
    `pf_settlement_times.hour_allowed`.
  - Before a run whose days border a sealed or forward block (10-27, 10-31, 11-03), audit every module in the chain for `hours_of` /
    `extra_after` / `glob` and confirm its guard.
- Jobs only via `run_light.sh` with PF_SLOT=claude. No orders; live stays off.
- Readings are declared in the module before data is read. A step that moves the code reruns its predecessor's day and checks
  reproduction (`compare.py`).
- One commit per step (code + receipts + state).
