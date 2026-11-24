# Why our opening fill loses while maker3's opening sequence earns

2026-11-24. User-selected Oct26 market3789484, saved CURRENT_ASYNC_FAST10
opening trace and the previously checked maker3 first-minute packet. This is
a descriptive diagnosis, not a candidate test, a latency measurement or a
generalization to other markets. No replay or new data was run.

## Executed path

|Time from opening|Our saved strategy|
|---|---|
|0ms|Posts100 shares each at bid49/48 and ask50/51, using the received best prices and one tick behind. The venue book also has49/50 around opening.|
|10ms|The four orders activate.|
|0–400ms decision grids|The cached adverse-signed250ms Binance return is valid and exactly zero on the ask side. No cancel request exists.|
|448ms, modeled fill time|Sells100 at50c and91.24 at51c, across12 simulated fill allocations. Every fill consumes a public trade priced52c: through-price fills, not a failure to get ahead of another order at our price.|
|450/500ms|The ask pressure value is0.26238, below its frozen0.285456 threshold.|
|550ms|The first threshold crossing is0.32485. Fill notices also make the client aware of−191.24 shares; inventory precedence requests cancellation of the remaining8.76 shares at51c.|
|560ms|That cancellation takes effect. It cannot undo the191.24 already filled shares.|
|Following seconds|The book moves up. We later buy back at58–60c. The opening asks'1s markout is−8.02c/share; completed turnover in the first15s loses$9.77.|

The initial asks were consistent with the opening book. The defect exposed by
this example is not simply a wrong displayed ask: the policy leaves cheap
sell exposure in place through an upward move, without a sufficiently early
warning from its chosen pressure signal. It is a default best-price ladder
with inventory sizing and a pressure guard, not maker3's recovered pricing
policy. The fill-notice delay affects the subsequent inventory response, but
these fills do not occur during a pending cancellation. Merely reducing the
10ms cancel latency cannot help without an earlier cancellation decision.

Queue priority alone also does not explain this opening loss: the public
trades pass through both of our ask prices. Initial external queue ahead was
78.96/140.46 shares; the through-price rule can fill these orders irrespective
of that queue. This is the saved simulator's mechanism, not a claim that all
possible real-world queue placements produce exactly the same fills.

## Maker3 is trading a different opening sequence

The previously reconciled first15s record has maker3 buying188.509 shares
at56c and subsequently selling at58/61/63c (264.917497 shares in total).
Its completed FIFO allocation earns$8.39 in this window. The first buy's
settlement upper bound is1.755s; the second56c hash's fills have upper bounds
3.999–4.201s. Ask58 fills have upper bounds7.428–7.518s, ask61 at8.537s,
and ask63 at8.943–8.954s. The hard windows separate the initial56c purchases
from those later sales. The quote-map bounds and their grades remain unchanged.

These observations establish different prices, sides and opportunities. They
do not establish that maker3 predicted the initial move, was faster, was
already at the queue front, or privately cancelled cheap asks. Never-filled
orders are invisible, its actual decision times are bounded, and its private
starting inventory is unknown. Its first observed fill is not the same event
as our first simulated fill. The minute later includes losing37–38c buys
followed by35c sells, so this is not evidence that all of its fills are good.

## Timing limitation

Our448ms uses the frozen M50 execution convention: the loader's anchor minus
50ms (`pf_queue_place_test.py:65–90`). Maker markers use recorded settlement
upper bounds. The native book moves from49/50 to higher levels after the
modeled fills, while the native frame updates and actual receipt times differ
from the strategy's modeled information clock. Consequently,550ms versus
448ms proves a late warning **in this saved replay**, not the true lead time
available to maker3 or to a real low-latency client. No execution-clock
sensitivity was rerun. Other market features might warn earlier; this packet
only checks the signal the executed guard actually uses.

The inventory-hysteresis defect already reported for this market begins at
138.5s, so it does not explain this opening loss.

## Evidence and next diagnostic

`opening_fill_evidence.json` binds the original ledger, observer arrays,
pressure cache and dashboard payload by SHA256. The raw fills reconcile to
the displayed fills; six deliberate changes to timing, quantity, signal,
pending-cancel status and notice time are refused. Native opening frames are
retained in opening_fill_probe.log. opening_signal_ticks.json preserves the
actual guard values. The first checker attempt used the maker-style `1s`
key for our numeric `1000` field and failed before output; v2 corrects that
adapter, v3 adds the public trade-price classification, with no source change.

The useful next question is whether the information available before these
cheap asks filled could justify a different price or side exposure, and
whether the same distinction recurs in the maker's losing sequences. Compare
decision-time context under the existing timing bounds; do not infer an
unconditional longer-rest or faster-cancel rule from this favorable maker
example. A strategy replay remains a separately authorized task.

All research reads used PF_SLOT=codex. Five seconds are conservatively charged
for this follow-up, including publication: packet upper82.11+5=87.11 CPU
seconds, below the user-approved180 seconds. No strategy files were edited.
