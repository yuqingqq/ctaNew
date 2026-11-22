# Concurrent requests and venue latency: amendment for the development session

2026-11-22. Inspected source: **9f7fe7031252aa8003cb2f275eb38587acb89d0b**.
USER: "For posting, the real latency should be lower. For cancellation as
well, we don't wait for notifications and we cancel/post when other requests
are in flight."

This supersedes the FAST10 execution/notification specification in the
[previous amendment](../20261122_queue_latency_amendment/plan.md). No replay,
implementation or new performance result here. The development session owns
the implementation; bundle it into the proposed queue/timing experiment.

## What the current code does

- `pf_branch_engine.py:142-149`: a live cancellation takes effect at request
  +37 ms. The 100 ms cancellation acknowledgement is subsequent; it is not
  added to the order's executable lifetime.
- `pf_branch_ladder.py:191-220,223-228`: multiple commands, including both
  sides, can be submitted in one tick. There is no global single-request gate.
- `pf_branch_ladder.py:112-135,165-172,198-203`: each cancel-pending generation
  occupies a quote slot until closure is known. The two-per-side slot count
  includes retiring generations. This can delay a replacement independently
  of network submission speed. Pending prices also block duplicate-price
  proposals. The explicit fixture at `pf_ladder_rest_selftest.py:133-140`
  confirms this is part of the frozen contract, not an unexplained new bug.
- `pf_branch_ladder.py:191` exposes only live orders to the policy; pending
  posts cannot be cancelled by that callback. The structural rule at
  `:137-153` can request their cancellation. Supporting a cancel of the SAME
  unacknowledged order needs a declared addressing/race contract; concurrency
  on other orders does not establish that venue capability.
- `pf_branch_ladder.py:123-135,209-214,292-305`: reservations include every
  unclosed generation; hard-cap and desired-exposure checks can independently
  block a post. This accounting must survive concurrent submission.

These findings identify a restrictive client model. They do not establish
how much loss it caused or invalidate the earlier results under that model.
Actual request-to-venue-effect latency is not established by this inspection;
RPC response time and fill-notice delay are different quantities.

## Execution contract for the proposed faster arms

Rename the faster profile **ASYNC_FAST10**. Use post activation +10 ms and
live cancel effect +10 ms as sensitivity assumptions, not measured values.
Keep fill notices +100 ms, cancellation acknowledgement +100 ms after effect,
post notice +0 ms, the 50 ms decision grid, received market data and M50 tape
clocks unchanged. Shorter notifications are not a substitute for concurrency.
The earlier FAST10 proposal changing all four delays to 10 ms is superseded.

Separate two desired quote levels from the registry of outstanding requests
and retiring generations. An independent request may be sent while another
is pending; no wait solely for that response. A cancellation request can free
the *desired quote slot*, but cannot release its risk reservation. Avoid
unintended duplicate proposals; any replacement must identify the retiring
generation. Preserve own-cross protection until exposure is resolved.

Retain the existing 200-share hard cap and inventory sizing unless the USER
explicitly chooses an overlap allowance. At known inventory zero, two
outstanding 100-share bids reserve 200. Adding a new 100-share bid before an
old one is resolved would reserve 300, and remains refused. Even with a higher
hard cap, the desired-exposure sizing can still block it: both constraints
must be declared, not silently relaxed. Allowing concurrent requests is not
permission to assume the old order has already disappeared or to net unknown
opposite-side fills. Report risk/sizing blocks separately from slot blocks.

For cancelling a still-pending post, preserve the declared conservative
pending-race rule (FAST profile minimum submit+20 ms, also no earlier than
request+10 ms) unless the development session can bind a different supported
contract before the freeze. This is a per-order race convention, never a
global wait imposed on other commands. Retain trades-before-command ties.
Do not assume atomic cancel/replace or success of a cancel that arrives before
an order is addressable. No live API integration is part of this work.

## One experiment, not another sweep

Keep four cases: CURRENT_BASELINE, FRONT_BASELINE, CURRENT_ASYNC_FAST10 and
FRONT_ASYNC_FAST10. The baseline is immutable and must reproduce the accepted
FULL_11. The faster cases change the client execution contract as well as
action delay. Label their contrasts **combined execution-profile effects**;
they do not isolate transport latency. The queue contrasts at each profile,
economic measurements and consumed-market population remain as proposed.

One bounded preflight should demonstrate commands overlapping with available
headroom, refusal at the actual cap, pending fills accounted once, closure
without early reservation release, and post/cancel races and trade ties.
Preserve effective-setting/source identity, baseline parity and per-arm
feedback. Reuse existing fixtures where sufficient. No broad hardening or
minor-fix round. Existing traces should expose submit/effect/notice times and
why a new proposal was blocked, so we can distinguish idle-by-slot from
idle-by-risk rather than attributing both to latency.

The ONE next priority remains the bounded queue/execution sensitivity, with
this contract incorporated before its freeze. It does not guarantee that
full-size replacements are possible at a fully allocated 200-share cap, nor
that reduced delays make the strategy profitable. A larger transient cap is
a separate USER choice; no such increase is assumed here.

No extra arms or resource increase are added. The prior proposed development
cap of 0.40 additional core-hours and 2.45 cumulative including review still
require the resource amendment identified in the previous plan; the original
2.0-hour line limit remains in force. This note does not launch that run or
consume new label days. The stopped broader-pressure experiment stays stopped.
