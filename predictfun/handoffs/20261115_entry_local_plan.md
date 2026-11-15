# Compact single order entry model

The candidate estimates one order's near-term execution value instead of the
terminal value of its recursive replacement chain. A separate cancellation
model remains responsible for resting orders. This is a local admission score,
not an estimate of optimal POST-minus-WAIT continuation value.

## Target and population

Use the existing neutral reference actions and independent hash-age probes.
Initial generations have inclusion probability 1/2; later generations 1/4,
with inverse-probability weights 2/4. The original age assignment stays fixed.
Every action has a status, including unfilled, refused, insufficient-horizon and
unreached actions. No filter selects favorable outcomes or completed fills.

For one posted order, count fills through five seconds after the probe or the
known research session's terminal cancellation tail, whichever comes first. Value
each executed tranche at its own fill time plus one second. Unfilled orders
have zero fill fraction and zero value. Missing/stale future marks remain
exclusion statuses. Late-window entries remain in training with their shorter
known exposure and the existing time-left feature. The actual order is not
cancelled when the measurement horizon ends.

F is total filled quantity divided by submitted quantity. D is the sum of
filled fractions times signed movement from the decision-known mid to each
tranche's future mid. The target reconciles exactly as entry edge times F plus
D. Both heads train on all eligible actions, so no independence assumption or
selection on fills is introduced. Subsequent replacement orders and other
slots' returns do not enter this label; competing liquidity remains in the
joint simulator. Full replay assesses the consequences excluded from the label.

## Data and models

Rebuild the relay basis counting each print once, propagate both basis and gap
before deriving fair edge, and load thirty seconds of admissible pre-window
books. Do not overwrite shared step-5 artifacts. Preserve boundary exclusions
and verify the neutral action identities against the previous dense build.

Use separate bid and ask models with a compact, predeclared set of market,
candidate queue, price, history support and inventory inputs. DIRECT predicts
local value. FACTOR predicts F and D and combines them with the known edge.
No test-day feature selection, threshold selection or horizon selection.
Positive scores post; exact zero ties also post to preserve queue opportunity.

This explicitly changes the old pinned learner to seven leaves, one hundred
rounds and L2=10 for both arms. It is a new compact-model experiment, not a
controlled attribution of all improvement to target changes. There are two
learned candidates and four arm/side local tests. The old full-chain study stays
as provenance, not an interchangeable target baseline.

Seven label dates: October 25/26/27/31 and November 1/2/3. Five chronological
evaluation dates start October 27; each uses strictly earlier label dates.
All are consumed development dates. No untouched data is read.

## Evaluation and cancellation interface

Compare local squared error with prior-mean and price/rung baselines. Local
selection uses five hundred draws matched on day, side, hour, rung,
initial/later generation, action count and weight mass, with Bonferroni four.
This is selection of local value, not evidence of an optimal waiting policy.

Full replay uses the frozen v6 stop-model predictions for each chronological
test fold, one-second risk and a predeclared probability threshold of 0.20.
Cancel only orders whose current depth is within that predicted adverse move.
No stop-model entry veto or cooldown is used. Empty slots reconsider the local
entry score every fifty milliseconds, with client notifications, placement and
cancellation latency and the existing hard capacity limit preserved.

Replay ALWAYS, price/rung BASELINE, DIRECT and FACTOR at M50 and M250, plus
M50 with blocked unknown queue. Keep thirty shares and the 1500-share cap fixed.
Report full terminal portfolio P&L, an accounting sensitivity excluding fills
younger than one hundred milliseconds, volume, posts, pulls and inventory.
Include quote availability, resting duration and integrated squared inventory.
Compare model arms to controls on every day and in aggregate; local prediction
alone is insufficient. These replay arms do not sum overlapping local labels.

The keeper is applied in replay, not in primitive training labels. A favorable
local result therefore does not establish keeper-conditioned Q accuracy. Public
queue estimates, fill clocks, gross marks without fees/rewards and retrospective
clock calibration remain limitations. No promotion or validation claim follows
from this consumed-data experiment. Larger sizes need their own label support.

## Verification and execution

The v3 correction was declared before any local model fit. Full-horizon v2
labels are unchanged and can be reused only with their frozen receipt and file
identities; one reused action per window is replayed and compared exactly.
Late-window labels are rebuilt. Previous attempts remain as provenance.

Synthetic controls cover per-tranche timestamp/sign accounting, zeros for
unfilled actions, horizon statuses, future input perturbation, deliberately
biased basis samples, pre-read date refusal, joint engine reconciliation and
matched-null power/refusal. Frozen hashes bind source, admissible raw inputs,
reference actions and keeper predictions. Saved models must reproduce their
predictions. Execute serially in the Codex core-8 slot with a two-GiB RSS guard.
