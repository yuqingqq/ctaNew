# Checked-in systemd drop-ins (BE 149)

The box can be rebuilt from the tree. These are COPIES of what is installed
under `~/.config/systemd/user/`; installing them is:

```
install -D live/pm_research/systemd/<unit>.service.d/heavy-lock.conf \
          ~/.config/systemd/user/<unit>.service.d/heavy-lock.conf
systemctl --user daemon-reload
```

## `heavy-lock.conf` — both pipelines WAIT for the heavy lock

`pm-measurement-pipeline` and `pm-evaluation-pipeline` run in
`research.slice` with **no lock**. Hourly they exit `IDLE` in ~1 s, but after
a day closes the catch-up takes **~12 GB for ~56 min**. The slice cap is
**15.0 GB** and a BE tape/book peaks **5.8–7.7 GB**, so 12 + 7.7 exceeds the
cap: one of them dies and a day is rebuilt.

The drop-in wraps each `ExecStart` in a **blocking** `flock` on
`/home/yuqing/ctaNew/data/.heavy_run.lock`, so the pipeline queues behind a
build and a build queues behind the catch-up.

**Two things that make this safe, both read from the units rather than
assumed:** `TimeoutStartUSec` is already `infinity` on both, so a blocking
wait cannot be killed mid-wait and cannot fire `OnFailure=pm-alert@`; and
`Slice=research.slice` is untouched, so the ≥8 GB out-of-slice guard still
sees them.

**The accepted trade:** while a long build holds the lock, the pipeline's
hourly IDLE check is **deferred, not skipped** — it sits in `activating` and
the timer will not double-start it. A catch-up can therefore be delayed by up
to a tape+book (~45 min). Nothing is dropped: `--scheduled` already makes
`IDLE` a successful exit.

**Driven, 2026-09-11** (BE 149): with the lock held, the unit sits
`activating/start` with its flock parked in `locks_lock_inode_wai` and the
payload never started; with the lock held by the same wrapper form, a heavy
offer refuses `ExecMainStatus=75` having run the payload **0** times; with the
lock free the same offer returns **0** and runs it **once**. Every field read
with `LoadState=loaded` — a unit that is `not-found` reports `0/success` as
DEFAULTS, not readings (R-648).
