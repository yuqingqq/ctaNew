# Size changes without losing queue position (layered size): spec for the port session and the research engine

USER 2026-11-09: "We need to update the size adjustment logic without changing existing order position in the queue."
Research diagnosis; the port session implements live. The research engine needs the same change as an opt-in token, so that replay
and live stay identical. Evidence: `predictfun/research/pf_live_vs_replay.py churn`, 220 windows of 2026-11-08.

## Today (v3, live and replay)

- Each lane (touch and ladder, per side) holds one order.
- An order is kept only if the next decision has the same price and the same size (`pf_step4_ladder.py` l. 934-936; the live port
  matches). Any size change, 5 -> 25 or 25 -> 5 (the AM:5:25:95 rule, 25 when the guard value is at or above the size line), cancels
  the order and places a new one at the back of the queue. The venue has no amend.
- Measured on 11-08 (30,572 live cancels):
  - 4,389 resize cancels, 14% of all cancels;
  - the best prices were unchanged at 97% of them;
  - 42% returned to the old size within 100 ms, 63% within 1 s.
  - In the queue-miss analysis, resizes were 2,205 of the cancels that cost us an earlier place at the price.

## Change

Split each lane's quantity into a base order and a top-up order at the same price.

- **Base.** `min(small size, desired)` shares, 5 under AM:5:25:95. It stays as long as the lane's desired price is unchanged and the
  lane is quoted.
- **Top-up.** A separate order for `desired - base` shares (20 when the size rule says 25), at the same price.
  - Placed when the desired size rises above the base.
  - Cancelled alone when the desired size falls back.
  - The base keeps its queue position throughout.
- **Re-price or pull.** Cancel both orders, as today.
- **Desired size below the base's remaining quantity** (reduce-only sizing `min(rule, |known net|)` after 60 s). Cancel the top-up
  first. Replace the base only if the desired size is still below its remaining quantity: the venue has no amend, and the total must
  never exceed the desired size.
- **Partial fills.** Compare the desired size with base remaining + top-up remaining. Do not refill a partly filled base.
- **Order rules apply to each order:** minimum value (XMIN1, 1 USD), post-only and self-cross checks, and cancel-pending orders still
  counted as ours.
- **Caps use the lane total.** G6, CAP10 and reduce-only sizing count base + top-up. For the kernel this means NetCap (with the 1 s
  late-fill window), the loss budget, and `places_in_flight_per_side = 3`: a top-up is one more order per side and may meet that
  limit; report how often it does.
- **Requests drop.** A size change becomes one place or one cancel, instead of a cancel plus a place.

## Research engine (same change, opt-in)

- A rule token, for example `LZ`, in a generated engine (as for the inventory tokens). The frozen file is not edited.
- Parity checks:
  - without `LZ`, byte-identical to the frozen v3 outputs;
  - with `LZ` on a fixed-size arm (no size changes), identical to that arm without `LZ`;
  - with `LZ` on v3, the resize counter must be greater than 0 and the outputs must differ (falsifier).
- A replay result for `LZ` is only meaningful after the fill model is calibrated on live queue outcomes (STATUS 2026-11-09). The
  current replay overstates fills for short-lived orders, which is where the effect sits.

## Update 2026-11-09: incremental is the examples' setting (USER "Go")

After research's quote-mechanics study (handoffs/20261109_incremental_capture_results.md), the user approved switching the
default: pmmmrust `size_layering = "incremental_v1"` in both v3 examples; `base_topup_v1` (this spec) stays selectable.
- A new quote is ONE order of the whole size. A second order is added only when the want grows at the same tick past a resting
  order (5 -> 25: the 5 keeps its place, +20 beside it); a size-down back cancels only that second order.
- A size-down below a resting order's open size (25 quoted first, then 5) still cancels and re-places it (no amend). The study:
  1,104 resize cancels left vs 6,008 with one order per lane; requests -4.5% vs single (always split: +10.8%). A size-down that
  only follows the order's own fills (reduce-only) keeps it, where research's handler re-places at each.
- Rungs never grow (research's rung rule keeps a resting rung at its own size), so they stay one order.
- Everything else in this spec (logical order, target, front kept, member fills absorbed, pending members active) is unchanged.
- B4 (research token) must copy the incremental rule for a new quote.

## Verify live

- Base orders survive size switches: count front-order cancels caused by resizes. Under `incremental_v1` they remain only for
  size-downs below a resting order's open size (expected well under the 11-08 count of 4,389); under `base_topup_v1`, 0 outside
  reduce-only sizing.
- Requests per window, and NetCap / in-flight refusals before and after.
- Queue-miss share of first-minute touch trades (6% on 11-08) and live fill counts, with `pf_live_vs_replay` on the new capture.

## Not in this change

- Pulls (75% of cancels) and the guard's flicker around the pull line (48% of pulls are re-quoted within 100 ms, 77% at the same
  price). Hysteresis on the pull and size lines is a strategy change: a new version, tested first.
