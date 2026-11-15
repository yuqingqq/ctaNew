# Entry repair and larger size progress

The issue register was committed at 7a4aa584 and integrated into mm-research at
58159b5b. The repair pipeline and larger-size design are now implemented.
Source f7e76fba was corrected at ce6e645e; the authoritative pre-run freeze is
edca5c96 and `reviews/entry_repair_20261115/freeze_v2.json`.

As of 2026-11-15 06:32 UTC, all eight synthetic controls and the corrected
two-window historical smoke pass. The smoke contains 28 eligible actions,
65 unsampled generations and four unreached probes; all ten original eligible
actions retain their exact target and features. The separate validity pass
reconciled four original action rows. These are engineering checks on two
October 25 windows, not performance evidence.

The first smoke found an indexing error: setting action identity as an index
removed `rung`, which is also a model feature. No dense artifact or fitted model
was produced by that attempt. The corrected version keeps the identity fields,
passed the smoke, and supersedes the first freeze in-band. An earlier synthetic
fixture used the wrong fill-tuple unpacking; that fixture was corrected before
the first historical freeze. Failed attempts and logs are preserved.

The full serial study is running in `pf-entry-repair-20261115.service` from
`/tmp/cta-entry-repair-20261115`. Authoritative progress is
`data/predict_fun/research/entry_repair_20261115_v2/runner.json`, with stage logs
in its `logs/` directory. Source and input hashes are checked around each stage.
The runner will stop on any failed check and retain the attempt.

First come seven-day loader, target and size/queue checks. Each of 30, 50 and
100 shares has its own neutral path, with the same 1,500-share hard cap and both
unknown-queue assumptions. The comparison measures repeat fills, reposting,
resting time and inventory alongside gross terminal P&L. These are neutral
mechanism tests. Larger sizes need their own WAIT/POST labels before evaluating
size-specific learned entry models.

Then the builder increases later-generation sampling from 1/32 to 1/4 while
preserving every old probe age, target and evaluation action. The three model
arms compare sparse versus dense training and all compatible history versus
the latest two dates on identical evaluations. Models and representations are
fixed. Model changes require the subsequent review and a separate freeze.

The entry loader now rejects foreign book parts before reads and hashing. The
existing daily clock is retained consistently across features and labels and
compared with a median using only admissible parts. Its retrospective nature
and upstream reconstruction remain open qualifications. The unknown-queue
sensitivity is not real-fill calibration. Neutral KEEP labels still exclude
the separate selective keeper's benefit.

The older owned action-depth supervisor PID 2900451 is held at a stage boundary;
its current child finished normally. The new runner resumes that exact process
identity from `/tmp/entry-repair-resume-20261115.json` on completion or failure.
The other session's core 9 is unaffected. Read `launch.json` for artifact hashes
and the current receipt pointer; do not overwrite either frozen attempt.
