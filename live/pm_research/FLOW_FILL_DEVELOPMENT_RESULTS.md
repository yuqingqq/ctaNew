# Flow and fills — Revision 4 development run

> **⚠ For current state read [`FLOW_MODEL_STATE.md`](FLOW_MODEL_STATE.md).** This
> document is **provenance** — correct about its own moment, not a statement of
> current belief. Where it conflicts with `FLOW_MODEL_STATE.md`, that page wins.


**Status: DEVELOPMENT · decision eligible: no.** This is an engineering and
within-design-data result, not forward evidence and not a profitability result.

## Receipt

- Protocol: `FLOW_AND_FILLS_V4`
- Probe: `flow_fill_development.py` (23 self-test checks)
- Source interval: 2026-08-20 17:45–19:45 UTC
- Sample: 24 consecutive five-minute windows per coin, seven coins
- State: unified Up book, 250 ms knowledge lag, gap-killed
- Arrivals admitted with state: 80,714
- Split: leave one five-minute window out within the two-hour design sample
- Shadow action: join the Up touch, 5 shares, both maker sides, horizons
  5/15/30 seconds
- Local receipt: `data/pm_5min/derived/flow_fill_development_v1.json`
  (gitignored; it carries source slugs, code/protocol hashes and its own
  `artifact_id`)

The run proves that the full data path is executable now. It does not answer
whether any fitted layer generalizes to another day.

## Nested baseline result

Numbers are held-window delta point-process NLL per admitted arrival; negative
is better than the immediately simpler layer.

| coin | events | B1−B0: price given time | B2−B1: tick-tail | B3−B2: book |
|---|---:|---:|---:|---:|
| BNB | 2,830 | −0.0207 | +0.0141 | −0.0020 |
| BTC | 50,578 | −0.0083 | +0.0000 | −0.0107 |
| DOGE | 3,038 | −0.0139 | −0.0029 | +0.0018 |
| ETH | 10,896 | −0.0155 | −0.0001 | −0.0017 |
| HYPE | 2,838 | −0.0336 | +0.0002 | −0.0388 |
| SOL | 4,471 | −0.0124 | +0.0000 | −0.0098 |
| XRP | 6,063 | −0.0165 | −0.0008 | −0.0033 |

Within these two hours, B1 improves on B0 for every coin. B2 has effectively no
support/effect on several folds and is actively worse on BNB. B3 improves on B2
for six coins but slightly worsens DOGE. These are useful implementation and
feature-triage signals, but five-minute held-out windows share the same regime;
they cannot promote a layer. A few optimizer folds reported numerical
precision-loss while ending with small gradients; this is retained in the JSON
instead of being hidden.

## Mark census

The event/type, unified side, execution-reach, size and native-notional fields
are all materialized. The extreme per-coin type mixture remains: the 0.02-share
class is 1.6% of BTC arrivals, versus 91.4% of HYPE arrivals in this slice.
The conditional M1–M4 law families are not yet frozen or validated; this run is
a plumbing census, not a claim that unconditional intensity determines order
size.

## Exploratory Hawkes residual

The scalar operational-time grid selects branching mass 0.40–0.55 for all
coins. Six coins select the shortest candidate half-life, 0.25 expected baseline
arrivals; BTC selects 0.5. All show positive within-design log-likelihood gains.

That says the residual contains very short-lived clustering worth testing. It
does **not** admit Hawkes: this development run resets history at every market,
has no continuous warm-up, uses correlated within-day folds, and has no forward
day. `HawkesDevelopmentDiagnostic` is consequently a different contract type
from the forward-gated `HawkesResidualFit`.

## Join-touch fill bounds

Fifteen-second any-fill rates for the frozen 5-share action:

| coin | available actions | front / optimistic | back-displayed / conservative |
|---|---:|---:|---:|
| BNB | 354 | 63.6% | 9.9% |
| BTC | 368 | 94.6% | 76.9% |
| DOGE | 332 | 71.1% | 9.6% |
| ETH | 356 | 84.8% | 55.6% |
| HYPE | 338 | 71.3% | 2.4% |
| SOL | 344 | 73.3% | 28.8% |
| XRP | 324 | 80.6% | 20.1% |

The wide bracket is the finding. Arrival rate alone plainly cannot determine a
fill, and displayed L2 cannot identify our exact queue position. The front bound
credits every reaching aggressive share; the back-displayed bound first consumes
the displayed queue and grants no cancellation credit. The midpoint is not a
result. Six BTC and two ETH 15-second actions crossing collector gaps remain
explicit unavailable rows; four additional ETH actions are unavailable because
the tick size changed inside their horizon.

No P&L is attached. The tape does not identify acknowledgement delay, our own
impact, cancellation success, hidden queue ordering, or the marginal entrant's
fill-conditional markout.

## A1 — micro/market independence: TESTED AND FAILED

The protocol previously declared this a tested null while nothing tested it. It
is now a test, and it fails.

| coin | verdict | ratio | direction | lead | follow |
|---|---|---:|---|---:|---:|
| BNB | DEPENDENT | 2.11 | bidirectional / common driver | 2.18 | 2.04 |
| BTC | DEPENDENT | 2.27 | bidirectional / common driver | 2.40 | 2.14 |
| DOGE | DEPENDENT | 2.43 | bidirectional / common driver | 2.65 | 2.20 |
| ETH | DEPENDENT | 2.37 | bidirectional / common driver | 2.40 | 2.34 |
| SOL | DEPENDENT | 2.79 | bidirectional / common driver | 3.05 | 2.52 |
| XRP | DEPENDENT | 1.75 | bidirectional / common driver | 1.75 | 1.75 |
| HYPE | NOT_REJECTED_NOT_EQUIVALENT | 1.18 | — | 1.40 | 0.93 |

Micro and market arrivals co-occur at roughly **twice** the circular-shift null.
`a1_holds` is false on **every** coin — six by rejection, HYPE by failing the
equivalence bound rather than by passing it.

**Direction decides the consequence, and it is the unfavourable one.** Had micro
merely *followed* market it would be reactive, and deleting it would leave the
market component's intensity and branching structure intact. Both channels are
elevated instead, so **ex-micro is a CONDITIONED SUB-PROCESS, not a component.**
Every ex-micro quantity inherits this: the Tier B grid, the ex-micro `f_r`
shapes, and any future ex-micro branching ratio.

**What is untouched, and it is the primary series.** R-DUAL has two mechanisms
and only one of them deletes. **Notional weighting reweights** — the micro class
carries 0.0145 % of notional and is negligible by construction rather than
removed — so no independence assumption is required. The notional-weighted
results stand. The specification's choice to make notional primary on five of
seven coins is vindicated by an argument it did not anticipate.

`tau` is load-bearing: at a saturating `tau` every micro event has a market
neighbour whatever the alignment, and a perfectly shadowing process returns
NOT_REJECTED. A selftest control pins that.

## B3 negative control — the book layer is real

Circularly shifting the covariate sequence preserves its marginal distribution
and autocorrelation while destroying alignment with arrivals. A gain surviving
the shift would be flexibility, not information.

| coin | real B3−B2 | placebo | placebo share |
|---|---:|---:|---:|
| BTC | −0.0107 | +0.0003 | −0.03 |
| HYPE | −0.0388 | −0.0009 | 0.02 |
| SOL | −0.0098 | +0.0016 | −0.17 |
| ETH | −0.0017 | −0.0007 | 0.42 |
| XRP | −0.0033 | −0.0008 | 0.25 |

The placebo does not reproduce the gain on any coin. **B3 carries genuine
covariate information.** Note the interaction: HYPE's gain is the largest, real,
and sits on a 91.4 % micro population — consistent with the micro class
responding to book state, which is also why A1's direction matters.

## Cumulative layer deltas — adjacent deltas hid a non-monotone stack

Reporting only B1−B0, B2−B1, B3−B2 lets "B3 improves on B2" read as "B3 is
best". Summed against B0, **BNB's best layer is B1** and the full stack is
materially worse. `cumulative_delta_nll_per_event_vs_b0`, `best_layer` and
`full_stack_is_best` are now emitted per coin.

## Hawkes grid — the previous selection was censored

The old floor of 0.25 expected baseline arrivals was selected by six of seven
coins, which is a boundary solution rather than a selection; on BTC it is ~36 ms,
order-splitting rather than market clustering. The grid now extends to 0.03, and
`half_life_boundary_hit` is reported separately from `branching_boundary_hit`.

## Verdict and next gate

Revision 4 fixes the plan-level category error: hours of data are enough to
build and falsify plumbing, while ten complete forward days are reserved for
candidate promotion. The next development item is to freeze and implement the
conditional M1–M4 mark families, then freeze a candidate implementation before
scoring any post-cutoff day. Until the baseline, marks and action-fill bounds
all pass the forward gates, `BE-FlowAndFills` remains `Unavailable`.
