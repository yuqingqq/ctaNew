# DE 158 — the parallel null needs a slice change, and here is the number

**A proposal, not a change.** The coordinator rules; I have touched
nothing. `research.slice` is a shared surface.

## Why the code alone buys almost nothing

DE 154's parallel null is built and verified: the runner battery is green
at 403 with the parallel path reproducing the serial one **element by
element** across values, settlement values and cancel counts, plus a
can-fail control (the same parallel path at a different seed does not
reproduce). The sampler stays serial by design, so the RNG stream is
consumed identically.

S4 is **99.2 %** of a day run. By Amdahl on that split:

| effective cores | speedup | a ~2 h 47 m null becomes |
|---|---|---|
| 2 (today's cap) | **1.98×** | ~1 h 24 m |
| 4 | 3.91× | ~43 min |
| 8 | 7.58× | ~22 min |
| 14 | 12.68× | ~13 min |

**Today both caps bind and the slice is the tighter one:**

* the launch form pins the unit at `CPUQuota=100%` (1 core);
* **`research.slice` is itself capped at `CPUQuotaPerSecUSec=2s` — 200 %,
  2 cores** — so raising only the unit's quota leaves the slice as the
  wall and the 12.7× collapses to about **2×**.

## The proposal

For a day run using N null workers:

* unit: `CPUQuota=N×100%`
* **slice: `research.slice` `CPUQuota` from `200%` to `(N+2)×100%`**

At **N = 8**: unit 800 %, slice 1000 % — **7.6×**, a day's null ~22 min,
five days in under two hours instead of ~14.

## What it means for the other seats

* **The collectors are NOT in this slice.** `collect-hf`, `collect-hl`,
  `pm-collector-clob` and `pm-collector-prices` are in `collectors.slice`
  at `quota=infinity`; `resource-monitor` is in `app.slice`. Raising
  `research.slice` cannot cap them, and a quota is a CAP, not a
  reservation.
* **It does not weaken rule 20.** One heavy run at a time is enforced by
  the heavy lock, which is untouched. This changes how many cores that ONE
  run may use, not how many runs there are.
* **Headroom is deliberate.** `(N+2)` on a 16-core box leaves at least 6
  cores at N = 8 for light suites, a seat's battery and the collectors'
  scheduling. I would not propose N = 14: it leaves 2 and the collectors
  are latency-sensitive.
* **Memory is not the binding constraint here**, which is the other reason
  this is the good lever. The resident book measured **~450 MB, not the
  3.2 GB the day-run peak suggests** (parent RSS 446.6 MB after load,
  708.7 MB after the baseline replay, cgroup 0.42 GB against an 8 GB
  unit cap). Workers are forked AFTER the book and the flags exist, so
  they share it copy-on-write.

## What is NOT yet measured, and I would rather say so

The **per-worker** cost on a real book — how much of that 450 MB each
worker dirties through Python refcounting — was being measured when DE 154
was preempted; the probe got as far as the parent's numbers above and was
stopped to free the lock. The binding number is the CGROUP total (parent
and workers share the unit's `MemoryMax`), not the sum of RSS. **N = 8 is
proposed on the CPU arithmetic and the parent measurement; the worker
count should be confirmed against that probe before a five-day run**, and
it is ~10 minutes of lock to finish.
