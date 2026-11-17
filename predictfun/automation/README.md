# Scheduled commit review

User-authorized on 2026-11-17: this session reviews new fixes and identifies the
next priority; another session owns fixes and development.

A local user-systemd timer checks `refs/heads/mm-research` every five minutes.
The lightweight watcher queues a message with `codex queue --resume` in the
existing review thread only when new relevant Predict.fun commits appear.
The local host and Codex session/daemon must be available. This is a local
schedule, not a hosted ChatGPT automation.

The initial checkpoint is the completed review commit `38de5e66`. New commits
are batched through a pinned target SHA, including relevant fixes/results even
if their messages do not contain the word "fix". Shared state-only updates,
watcher files, automatic review reports, and commits carrying the exact trailer
`Predictfun-Review-Only: true` do not trigger another review.

One pending request is allowed. New commits arriving during a review wait until
that request is acknowledged. A crash or uncertain queue response leaves the
request pending rather than sending duplicates. Branch-history rewrites fail
explicitly instead of silently advancing the checkpoint.

The reviewer reads current repository instructions and prior findings, reviews
the exact committed source, runs necessary bounded checks on the Codex slot,
and writes findings plus one next priority under `predictfun/reviews/automatic/`.
It updates only its own state/documentation changes. It does not implement
fixes, launch full experiments, or alter development work. Review-only commits
must carry the trailer above. The generated request includes the exact `ack`
command; it accepts only a report in the review directory containing the full
target SHA. Acknowledgement advances to that target, never past newer commits.

Configuration/checkpoint and delivery state:
`~/.local/state/predictfun-commit-review/state.json`.

```bash
# Inspect schedule and delivery state
systemctl --user list-timers predictfun-commit-review.timer
python3 predictfun/automation/commit_review_watch.py status
journalctl --user -u predictfun-commit-review.service -n 20

# Check now (safe to repeat; one pending request prevents duplicates)
systemctl --user start predictfun-commit-review.service

# Pause / resume
systemctl --user disable --now predictfun-commit-review.timer
systemctl --user enable --now predictfun-commit-review.timer

# Run watcher tests: temporary repositories and fake transport only
python3 -m unittest discover -s predictfun/automation -p 'test_*.py' -v
```

Installation copies the `.service` and `.timer` here into
`~/.config/systemd/user/`. No Git hooks or other schedules are changed.
On `needs_attention`, `uncertain`, or history-rewrite errors, inspect the state
and queue result before retrying; do not clear a pending request blindly.
