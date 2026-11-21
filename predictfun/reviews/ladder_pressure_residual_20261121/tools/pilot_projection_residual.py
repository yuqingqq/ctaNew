"""The declared pilot rule (declaration.md, Budget), WRITE-ONCE. From the saved-input pilot (the audit of the pilot part's 8 markets) and its run_light
record: the full audit's CPU = the pilot's CPU x (all fills / the pilot's fills) + 0.03 core-hours for the checks, the falsifier and the report
+ this round's CPU so far (every run_light record under logs/); the ladder line's recorded use = its six earlier receipts' logs + the six review
directories' logs + this round. run_full iff the pilot is done with rc 0, its peak below 1,536 MiB, this round's projection <= 0.12 core-hours
and the line's recorded use + the rest of this round <= 2.0 core-hours. Refuses when the output exists."""
import glob
import gzip
import json
import re
from pathlib import Path

REPO = Path('/home/yuqing/ctaNew'); R = REPO / 'predictfun/reviews/ladder_pressure_residual_20261121'; L = R / 'logs'; D = REPO / 'data/predict_fun/research/ladder_pressure_residual_v1_20261121'
AD = REPO / 'data/predict_fun/research/ladder_pressure_ablation_v1_20261121'
LINE = ['ladder_rest_20261121', 'ladder_audit_20261121', 'ladder_guard_20261121', 'ladder_pressure_20261121', 'ladder_pressure_lag_20261121', 'ladder_pressure_ablation_20261121']
REVIEWS = ['20261121_ladder_rest_review', '20261121_ladder_audit_review', '20261121_ladder_guard_review', '20261121_ladder_pressure_review', '20261121_ladder_pressure_lag_review',
           '20261121_ladder_pressure_ablation_review']
out = L / 'pilot_projection.json'
if out.exists():
    raise SystemExit(f'REFUSED: {out} exists (write-once)')
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')


def cpu_of(paths, jobs=None):
    tot = 0.
    for p in paths:
        for ln in Path(p).read_text().splitlines():
            m = RL.match(ln)
            if m:
                tot += float(m.group(6))
                if jobs is not None:
                    jobs[m.group(2)] = dict(rc=int(m.group(3)), peak=int(m.group(4)), cpu=float(m.group(6)))
    return tot


jobs = {}; so_far = cpu_of(glob.glob(str(L / '**' / '*.log'), recursive=True), jobs)
line = cpu_of([p for n in LINE for p in glob.glob(str(REPO / 'predictfun/reviews' / n / 'logs' / '**' / '*.log'), recursive=True)])
rev = cpu_of([p for n in REVIEWS for p in glob.glob(str(REPO / 'predictfun/reviews/automatic' / n / '*.log'))])
pj = jobs.get('gr_audit_pilot'); a = json.loads((D / 'pilot' / 'audit.json').read_text()) if (D / 'pilot' / 'audit.json').exists() else None
n_all = sum(len(json.loads(x)['FULL_11']['fills']) for p in ('pilot', 'rest') for f in sorted((AD / p).glob('*.jsonl.gz')) for x in gzip.decompress(f.read_bytes()).decode().splitlines() if x)
res = dict(pilot_done=bool(pj and pj['rc'] == 0 and a), pilot_job=pj)
if res['pilot_done']:
    full_h = pj['cpu'] * n_all / max(1, a['reconcile']['fills']) / 3600. + 0.03; proj = full_h + so_far / 3600.; line_h = (line + rev + so_far) / 3600.
    res.update(pilot_fills=a['reconcile']['fills'], all_fills=n_all, projected_core_hours=proj, cap=0.12, line_recorded_core_hours=line_h, line_projected_core_hours=line_h + full_h, line_cap=2.0,
               peak_ok=pj['peak'] < 1536, run_full=bool(pj['peak'] < 1536 and proj <= .12 and line_h + full_h <= 2.0))
else:
    res.update(run_full=False)
out.write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
