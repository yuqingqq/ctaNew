# BTC feed mitigation — design for ruling (R-211(6))

2026-08-28T01:16Z · coordinator · evidence: `da_btc_gap_evidence_v1.json` (Q-DA-108,
as-of 4,162 ledger lines, sha 8a60e28a…) · prior record R-181/R-182 embedded there.
Nothing here is deployed. Deploy, if ruled, only at a UTC day boundary with a
`collector_runs` era stamp.

## Evidence summary (all figures as-of 01:13Z)

- Since the 08-25 step, btc: **1,734 gaps / 10,012s lost**; the six other coins on
  identical per-market connections in the same process: **36 gaps / 261s** (control).
  PING_TIMEOUT is btc-only (944 events; xrp 1, eth 1, zero elsewhere).
- **77.1% of all btc lost seconds is client-side detection lag**: PING_TIMEOUT
  median gap 11.305s of which 10.005s is time-to-notice — matching
  `ping_interval=10, ping_timeout=10` at `collect_pm.py:459` exactly. Reconnect
  after detection is ~1.3s across all causes.
- The slow-consumer story is refuted by our own telemetry: `ws_ever_paused` = 0
  in all 1,734 disconnects; local queue depth LOWER post-step (median 80 vs 123);
  consumer lag ~2ms. The server's 1013 closes are not corroborated locally.
- Trigger vs cost: the disconnect TRIGGER is consistent with peer/path behavior
  on the highest-rate market connection; the COST is overwhelmingly our detection
  timeout. (Q-DA-108's own refinement of Q-DA-99.)
- The day bar counts EVENTS (15/hr). Post-step gaps are frequent-but-tiny
  (median 1.3–11.3s; the failed 08-27 still had 96.2% coverage). Row-level
  GAP statuses already handle per-row admissibility in all scoring.

## Options

**O1 — shorten ping detection (client tunable, one line).**
`ping_interval=10, ping_timeout=10` → `3, 3`. Median PING_TIMEOUT gap
11.3s → ~4.3s; total btc loss falls ~65% (≈7,700s/3d → ≈2,300s). Does NOT
reduce the event COUNT, so days still fail the 15/hr bar. Risk: extra keepalive
traffic on the throttled connection (pings are ~2 bytes payload — judged
negligible, but stated). Verification is STRUCTURAL, not A/B (R-182: short
throughput A/Bs non-viable on this feed): post-change, the PING_TIMEOUT
duration distribution must collapse toward ~ping_interval + 1.3s. That check is
within-cause and distributional — viable.

**O2 — prospective re-derivation of the day bar (estimand fix, no collector change).**
The 15/hr count bar was set in the rare-long-gap regime. What scoring validity
actually needs is bounded DATA LOSS and bounded WINDOW CONTAMINATION, both
already measured. Proposal to be drafted as its own ruling if approved in
direction: bar on lost-seconds/hour (e.g. ≤120s/hr ≈ 3.3%) AND
windows-gap-affected share (e.g. ≤X%), derived from the scoring admissibility
logic, applied to days AFTER the ruling only. 08-27 stays excluded; nothing
retro-judged. Guard against chosen-to-pass: thresholds derived from first
principles and committed BEFORE the next day closes; the change named in every
receipt that relies on it.

**O3 — redundant second btc connection with merge (held).**
Only option that reduces the event count to ~zero (gap only when both legs are
down). But: doubles aggregate traffic to this IP (R-181: direction structural,
magnitude not estimable — may worsen the peer throttle for every connection),
large collector change, and effectiveness unverifiable quickly (R-182). Engage
only if O1+O2 leave btc days failing.

## Recommendation

O1 AND O2-in-direction together. O1 cuts real data loss regardless of any bar.
O2 makes the bar measure what scoring needs. O3 held. O1 deploys at
2026-08-29T00:00Z (next boundary) with a collector_runs stamp; its era effect is
benign (gap DURATIONS shrink; row semantics unchanged) and stamped anyway.
O2's exact thresholds come back as a separate pre-registered ruling before the
first day they would judge.

## Asks

1. Approve O1 (ping 10→3) for boundary deploy 08-29T00:00Z — yes/no.
2. Approve O2 in direction (I draft exact thresholds for pre-registration) — yes/no.
3. O3 stays held — objection?

## Addendum (01:25Z) — collector survey folds in (BE, read-only; Q-BE survey)

Four additional client-side facts, each with a small fix candidate. To keep ONE
era boundary instead of four, **O1 is amended to a single deployable PACKAGE**
(all client-side, one stamped change, structural verification per item):

- **O1a** ping_interval=10/ping_timeout=10 → **3/3** (worst-case dead-socket
  blindness ~20s → ~6s; median PING_TIMEOUT gap 11.3s → ~4.3s).
- **O1b** cause-aware backoff with jitter replacing the flat 1s retry
  (persistent faults are currently hammered at 1 Hz — consistent with the
  49%-within-60s burst clustering; network-type causes get exponential+jitter,
  SLOW_CONSUMER does not get faster retries).
- **O1c** subscribe CONFIRMATION probe: after re-subscribe, require a first
  message or explicit ack within a bound, else re-subscribe and record a
  distinct cause — today a silent no-subscribe is indistinguishable from a
  quiet market (an invisible-hole class).
- **O1d** gap-start stamping fix for never-connected sockets: stamp at
  last-coverage (window/connection start), not at the error — today such gaps
  are recorded SHORTER than they were, and this ledger is what the tape pins
  and the gate counts.

Unchanged and protected (per the survey): max_queue=2**16, the parse-free recv
loop, the finally-emitted gap_open_at_exit accounting, per-window scope.

Ask 1 becomes: approve the O1 PACKAGE (a–d) for one boundary deploy
08-29T00:00Z with a single collector_runs stamp — or name a subset.
