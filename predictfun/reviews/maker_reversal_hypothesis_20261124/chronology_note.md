# Post-run chronology readout, declared before this join

The frozen H1 run has four timing-only matching conditional-withdrawal orders at D0, but zero posting-compatible matches. Do not interpret zero strict matches as proof that makers ignore reversal: the posting upper bound is an uncertainty limit, not the true post time.

This readout joins each timing-only match, at both declared delays, to its original posting LOWER bound as well. It classifies every hit as before_earliest_post / inside_post_window / after_latest_post / post_unknown, retaining the original grade. This is deterministic chronology, not another threshold, horizon, sample, null or candidate. A hit inside the posting window permits both a before-reversal post and an after-reversal post; it cannot establish the chain's order. A hit before the earliest possible post cannot be a reaction by an already-resting order under those same premises. No estimate is narrowed using BTC; that would be circular.

Also report original signal census by day/market, exact ending statuses and both controls; do not equate a low strict-match rate with an identified cancellation-propensity estimate. The frozen primary result/checks remain unchanged. No extra null for the four timing-only hits is claimed.
