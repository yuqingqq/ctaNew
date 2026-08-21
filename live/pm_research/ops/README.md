# Collector supervision (P-2026-003)

The two collectors ran as bare `nohup` processes orphaned to init: no restart, no
alert. Over a 10-day OOS window an unnoticed overnight death costs a whole UTC
day of the frozen `route_a_v1` gate, and nothing would have caught it once the
supervising session closed.

## Install collectors

```bash
cp live/pm_research/ops/pm-collector-*.service ~/.config/systemd/user/
loginctl enable-linger "$(id -un)"          # survive logout, not just this login
systemctl --user daemon-reload
systemctl --user enable --now pm-collector-prices.service pm-collector-clob.service
```

## Install the closed-day measurement batch

The timer retries hourly and processes one oldest uncommitted eligible day. It
does not run a model or trade; it only publishes normalized, validated research
artifacts. `--since 2026-08-20` makes the first deliberately collected UTC day
the catch-up boundary, so an earlier partial discovery day cannot block the
queue forever.

```bash
cp live/pm_research/ops/pm-measurement-pipeline.{service,timer} \
  ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now pm-measurement-pipeline.timer
systemctl --user start pm-measurement-pipeline.service  # immediate retry
```

Inspect the last attempts and the next scheduled retry with:

```bash
systemctl --user status pm-measurement-pipeline.timer
journalctl --user -u pm-measurement-pipeline.service -n 100 --no-pager
```

The service uses `--scheduled`: `BLOCKED` and `IDLE` are successful retry states,
while a build, validation, hash, or lock failure makes the unit fail visibly.
The coordinator holds `tier1/.locks/measurement_batch.lock`; do not overlap the
legacy per-coin CLI with the unattended batch writer.

## Install the Tier-2 evaluation batch

This timer is staggered to minute 40. It builds/reuses the complete `full` Tier-1
batch, then writes model-free terminal markouts and the normalized calibration
scaffold. It does not fit a model or emit an inferential result.

```bash
cp live/pm_research/ops/pm-evaluation-pipeline.{service,timer} \
  ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now pm-evaluation-pipeline.timer
systemctl --user start pm-evaluation-pipeline.service
```

```bash
systemctl --user status pm-evaluation-pipeline.timer
journalctl --user -u pm-evaluation-pipeline.service -n 100 --no-pager
```

The Tier-2 job is fail-closed on a missing/non-PASS G-FF1 artifact, incomplete
full batch, partition/hash mismatch, or writer-lock conflict. Its outputs and
claim boundary are documented in `../EVALUATION_PIPELINE.md`.

## Before starting, stop any nohup instance first

Duplicate collectors corrupt the tape and `recv_ns` dedup does **not** catch it,
because each process stamps its own receive time. Verify with `ps`/argv, never
`pgrep -f` — that matches the checking command itself:

```bash
ps -eo pid,args | awk '/python3 .*live\/pm_research\/collect/ && !/awk/'
```

## Two things that bit during the 2026-08-21 cutover

**The interpreter is the venv, not `/usr/bin/python3`.** `websockets` is
installed only in `/home/yuqing/pricer-sol/venv`. A unit pointing at the system
python restart-loops on `ModuleNotFoundError`, and collection is DOWN while it
does — which is how the 44 s gap at 01:42:15–01:42:58 happened.

**`StandardOutput=append:` matters.** `prices_collector.log` had acquired a
56 KB contiguous NUL run (18% of the file, genuinely sparse on disk) when a
restart truncated it while an old fd still held a high offset. GNU grep then
treats the file as binary and emits *nothing* — indistinguishable from "no
matches", which silently blinded several checks. `append:` is O_APPEND and
cannot reproduce it. The damaged file is preserved as
`prices_collector.log.nul-damaged-20260821`; read it with `grep -a`.

## Verify supervision actually works

`Restart=always` unverified is the same "reports success without checking"
pattern this programme keeps finding. Test it:

```bash
kill -9 $(ps -eo pid,args | awk '/[/]collect_pm.py/ {print $1}')
# expect a new pid within RestartSec=10, and exactly one of each afterwards
```

Measured 2026-08-21: restarted after 10 s, one of each, unit `active`.
