# OB dynamics optimization loop — charter

User directive (2026-08-25): "do more tests (reliable tests) on Orderbook
dynamics, check the results to improve the current model, double confirm the
results, proceed with a long loop to keep optimizing."

Binding: CLAUDE.md reliability rules. Everything below is development
evidence (consumed partial-era tape, train 08-24 → dev 08-25). Freeze
decisions are the user's. Commit after each gate, never before.

## Loop protocol (every iteration)

1. DECLARE the feature family + mechanism in code before any number exists.
2. Build behind the committed era-pure pipeline (recv_ns >= ledger boundary),
   cutoff T−1ms, no look-ahead; selftests must pass before any run.
3. Test as ONE combined arm vs the current best arm on IDENTICAL rows —
   paired increment, generation-native evaluator, matched randoms on NET.
   No per-family scoreboard on consumed data.
4. DOUBLE-CONFIRM any positive increment before it becomes "current best":
   a. reproduce the receipt at the named artifact from the committed code;
   b. time-shift null — features displaced +5s must kill the increment
      (alignment control, declared before running);
   c. per-hour breakdown — increment must not live in a single hour;
   d. multiplicity counter incremented in the receipt.
5. Update STATUS.yml + HANDOFF.md; commit.

## Iterations

- **I1 (in flight)**: OFI (bookTicker qty deltas, Cont-style, 0.1/0.5/1.0s)
  + big-print (max threat-side trade / trailing 60s median, 0.5/2.0s).
  Three arms: PM_ONLY / +reduced-fine / +extended. Multiplicity: 2 specs.
- **I2**: depth20 deep-book dynamics. Semantics probe done (100ms snapshots,
  20 absolute levels/side, L1 == bookTicker at same ns). Remaining
  verification before features: level-count constancy + hour-scale L1
  agreement with bookTicker. Candidates: depth-within-X-bps delta
  (pull/replenish), book-slope change.
- **I3**: PM-side book thinning — level-size drops near our quote from
  price_change events minus executed volume (cancel detection on the PM book
  itself).
- **I4**: confirmation pass on the cumulative winner (4a–4d above) + receipt.

## Standing constraints

- ~19h era tape only; day-unit intervals impossible below G=5 — report
  point estimates + matched-random bands, no fake CIs.
- btc is the economically live symbol; eth confirms direction.
- Any freeze proposal goes to the user with the multiplicity count attached.

## RESULTS (as of 2026-08-25, receipt harmful_fine_comparison_v2.json)

Five arms, identical rows (btc 605,243 / eth 520,033 — zero rows dropped by
the new families; run 1 arms reproduced to the cent = confirmation 4a).

| verdict | arm | evidence |
|---|---|---|
| CONFIRMED best | PM_PLUS_FINE (reduced) | 4b: T-5s shifted control collapses to PM_ONLY on BOTH coins. 4c: increment positive in 10/11 btc dev hours at every budget, top-hour 18-26%; eth top-hour <=48%. |
| unconfirmed small candidate | PM_FINE_EXTENDED (OFI+big-print) | +5%-budget-only bump, both coins (+529c btc, +435c eth); ~0 @10%, <=0 @15%. Held for forward tape, not adopted. |
| REJECTED | PM_FINE_PLUS_DEPTH (depth20) | hurts net at all btc budgets and eth 5/10%; deep book dilutes the L1 signal. Do not re-test this spec on consumed tape. |

Multiplicity: 3 candidate specs consumed on 08-24/25 development tape.
Score dumps: harmful_scores_{btc,eth}_v2.jsonl.gz (offline confirmations,
validated cent-exact vs receipt). Next: I3 PM-side thinning (multiplicity 4).

## I3 verdict (run 3, receipt harmful_fine_comparison_v3.json)

PM-thinning: NULL. Increments vs reduced are sign-mixed on BOTH coins
(btc +203/-476/+386; eth -214/+78/-425) — noise pattern, no coherent shape.
Not adopted; spec consumed (multiplicity 4). Anchors reproduced cent-exact
(4a). Next: I5 cross-symbol lead, btc->eth only (declared asymmetric —
btc leads price discovery; eth quotes may be protected by the btc book).
Multiplicity 5. After I5: saturation report + freeze proposal to the user.

## I5 + LOOP HOLD (2026-08-25 ~20:25 UTC)

I5 btc->eth lead: +229/+292/+161c over reduced at 5/10/15% — first family
positive at ALL budgets (receipt harmful_fine_comparison_v4lead.json). 4a
passed (anchors cent-exact); 4c borderline-pass (no single-hour
concentration: 40/50/46%, but hour-positivity weak 7/10, 6/10, 5/11).
4b (T-5s shifted-lead control, v5leadctl) was KILLED EXTERNALLY TWICE
(~19:40 and ~20:20 UTC) while live collectors stayed up. HELD per protocol —
no third relaunch without the user.

FLAG for the user: heavy backtest runs (99% CPU, ~8-9GB) on the collector
box during forward-tape accumulation may degrade recv_ns receive-latency
p99 — the exact sub-second precision the era depends on. If the kills were
deliberate protection, they were right. Options: (a) run 4b overnight/idle,
(b) nice/cpulimit the run, (c) move heavy runs off this box.

I5 status: HELD-UNCONFIRMED (4b pending). Loop otherwise saturated.

## SATURATION REPORT (5 candidate specs, consumed 08-24/25 era tape)

| spec | verdict | evidence |
|---|---|---|
| reduced fine (imb+midbps) | CONFIRMED best | 4a+4b+4c all passed, both coins; btc net +2492/+6575/+8289c, beats random max on NET at all budgets |
| + OFI+big-print | held, unconfirmed | consistent small @5%-only bump both coins; no confirmation run spent |
| + depth20 | REJECTED | hurts net both coins |
| + PM thinning | NULL | sign-mixed both coins |
| + btc->eth lead | held, 4b pending | all-budget positive on eth; 4c borderline; control run blocked |

Forward-race multiplicity if frozen now: reduced (primary) + extended +
btclead (held) = 3 specs. Freeze is the USER'S decision (rule 12: commit +
receipt + declared nulls + multiplicity at freeze time). Forward validation
needs >=5 complete untouched UTC days AFTER the freeze.

## HOLD LIFTED (2026-08-26, user: "Proceed the test, check if it is memory issue")

Probable cause found: the session shell carries `ulimit -v` = 12 GiB
(address space) with hard limit unlimited; heaviest completed run peaked
9.2 GB RSS (VSZ higher), the lead runs are heavier still. The machine had
23.9 GB free throughout — the cap, not the box, was the constraint. Same
cap explains the repack RLIMIT_DATA failures. 4b control relaunched with
`ulimit -v unlimited`, nice -19 (collector protection), and a 20s VSZ/RSS
sampler for a definitive trace either way.

## I5 FINAL (4b control, receipt harmful_fine_comparison_v5leadctl.json)

NOT ADOPTED. True-lead increments +229/+292/+161c; T-5s shifted control
+264/-871/-662c. The @10/15% gains collapse under the shift (timely), but
the @5% gain SURVIVES a 5s-stale btc book — slow-covariate, not timeliness;
per the pre-declared rule a surviving shifted gain flags the candidate.
BTCLEAD arm reproduced v4lead cent-exact (4a). Kill diagnosis: peak VSZ
10.1GB < the 12GiB cap — ulimit NOT confirmed as the killer (cap did break
git repack); completed run used nice -19 + unlimited (n=1).

## LOOP CLOSED — final state

CONFIRMED: reduced fine spec (PM + L1 imbalance + mid-move). Candidates
scored on consumed tape: 5. Held-unconfirmed: extended (@5%-only, both
coins). Rejected/null/flagged: depth20, PM-thinning, btc-lead.
Recommended freeze: reduced primary + extended held (race multiplicity 2;
5 specs consumed disclosed in the receipt). Freeze decision = USER'S.
After freeze: >=5 complete untouched UTC days before any forward verdict.
No further scoring on consumed tape — saturated.

## NEXT TODO — stateful predictor x cancel x skew

The ordered implementation and validation checklist is recorded in
`live/pm_research/plans/STATEFUL_HARMFUL_CANCEL_TODO.md`. It separates
candidate-independent market/queue state used by the harm predictor from
inventory, skew and cancel/repost lifecycle state used by the decision policy.
The sequence is: repair freeze/provenance -> one predeclared predictor-state
family -> one fixed nonlinear candidate -> stateful cancel/hold/repost replay
-> latency and queue-reset-cost grids -> immutable freeze -> five complete
untouched forward UTC days.
