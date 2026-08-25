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
