# Wait/post client-state and target repair — declared plan

User request: add missing inputs, review the model again, add necessary information
and improve the model. User design: preserve queue priority, decide entry from
signals, pull only affected orders, keep the rest. No fixed retry timer.

This session owns the entry model. The separate stop-model is not silently used
as a keeper: its learned policy is still pending. The experiment remains neutral
KEEP with engine safety, six rungs per side, 30 shares, hard exposure cap 1500,
50 ms decisions, 40 ms placement, cancellation 22 ms, fill notices 100 ms.

## Changes before any historical result

- New versioned lane observes the full client-live ladder, including pending
  placement and cancellation, submitted/activation times, prices, received fills,
  remaining quantities and per-slot generation/first-eligible time. Tuples are
  immutable. The venue's remaining quantity, queue, future refusals and unseen
  fills never enter the model.
- Features summarize every side/rung, preserving overlapping client-live orders:
  quantity, pending quantities, received fills, weighted/min/max price distance,
  oldest/newest age, activation age, public shown size and causal queue estimate,
  missing queue information, empty age and generation. Public liquidity is scaled
  by trailing known frames; best-price age is explicitly different from order age.
- Review found an unmasked start-price gap in the v2 entry inputs. Mask it in new
  model inputs AND the frozen bootstrap continuation. Rebuild labels; preserve all
  v2 results. Exclude Oct31's pre-00:30 windows with an explicit status because of
  the sealed previous-day volatility lookback.
- Split each economic target into the first order after the action and the
  remainder of the ladder. POST/WAIT share the same pre-decision information and
  frozen continuation. Local plus external contribution must equal the full
  portfolio label. WAIT's first subsequent order can be posted much later; no
  timer forces entry. Later replacement orders and displaced fills stay in the
  external component. All no-fill opportunities remain as zero local outcomes.

## Frozen experiment

Three arms, fixed pinned LightGBM parameters, separate bid/ask models:
1. BASE: causal v2 inputs, direct POST and POST-minus-WAIT heads.
2. LADDER: richer inputs, identical direct targets.
3. COMPONENTS: richer inputs, local/external POST and local/external timing heads;
   sum the predictions, then use POST > 0 and timing >= 0.

Same neutral action generations, original hash-selected probe ages/inclusion.
Inverse inclusion weights 2 for initial generations, 32 later, in training AND
evaluation. This does not create missing joint waiting configurations.

Rebuild seven consumed label days Oct25/26/27/31 and Nov1/2/3. Existing bootstrap
is trained on Oct23/24; its weights remain fixed and its gap input is repaired.
Five chronological tests Oct27/31 and Nov1/2/3, train on earlier label days only.
Source and input manifest committed before historical smoke/full runs. Each stage
writes a new directory and binds identities; no overwrite or silent retry.

Report raw n, effective n, statuses, per-day/side and pooled prediction errors vs
prior weighted means and fixed price/rung baselines. Report training errors and
component errors separately. Local economic null: 500 random selections matched
on day, side, hour, rung, initial/later generation, action count AND weighted mass.
Pass only with positive absolute POST, positive timing value above null p95, raw
selected/unselected n >=30, selected effective n >=30. Report all arms; no winner
picked on these consumed days. No row-level confidence intervals.

## Build review and required controls

- Unchanged execution at M0/M50/M250 with and without withheld entries.
- Hidden partial fills and future venue events cannot alter client snapshots or
  decision prefixes. Received fills MUST change the state. Corrupt exposures and
  incomplete snapshots must be refused.
- Future/unreceived public frames cannot alter features. Known-book perturbations
  must change them. Unknown deep queue sizes stay missing, never zero/front.
- Candidate and continuation mask the start-price gap. Reconstructed labels and
  artifact feature order match runtime inference.
- Bid/ask accounting controls, no-fill controls, and a displaced profitable fill
  where standalone value is positive but incremental portfolio value is negative.
- Every target component reconciles; a planted sign/value error must be refused.
- All historical runs use run_light codex/core8, 2 GiB RSS self-guard; previous
  size queue yields at worker boundaries and resumes on success/failure.

## Claim limits

This is a local model diagnostic, not a portfolio strategy acceptance run.
C1/C2/C3/C5/C12/C15/C17 remain portfolio gates; clock and young-fill performance
cannot be inferred from these labels. E2 timing uncertainty, assumed Binance
availability, unknown true queue position, missing jointly waiting support,
neutral KEEP versus the desired stop keeper and the long terminal valuation
horizon remain open. Local target decomposition narrows attribution, not time.
All used days are consumed; no untouched forward data will be read here.

Model quality is a measured outcome. New inputs, sound plumbing or a better fit
alone do not entitle a claim of profitable entry or promotion.
