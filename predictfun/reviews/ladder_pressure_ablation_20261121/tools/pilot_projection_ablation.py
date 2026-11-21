"""The declared pilot rule (declaration.md, Budget), WRITE-ONCE. From the four pilot part records (load seconds, per-market seconds for the three run
arms) and the run_light records of the pilot jobs: the rest's wall = sum over the dates with rest members of (the pilots' mean load + the date's
rest markets x the pilots' mean seconds per market); CPU = wall x the pilots' CPU / wall + 0.08 core-hours for the rehearsal, measurement,
checks and report + this round's CPU so far (every run_light record under logs/). The ladder line's recorded use = its five earlier receipts'
logs + the five review directories' logs + this round. run_rest iff all four pilots are done with rc 0, peaks / high-waters below 1,536 MiB,
this round's projection <= 0.35 core-hours and the ladder line's recorded use + the rest of this round <= 2.0 core-hours. Refuses when the output
exists."""
import glob
import json
import re
from pathlib import Path

import numpy as np

REPO = Path('/home/yuqing/ctaNew'); R = REPO / 'predictfun/reviews/ladder_pressure_ablation_20261121'; L = R / 'logs'; D = REPO / 'data/predict_fun/research/ladder_pressure_ablation_v1_20261121'
LINE = ['ladder_rest_20261121', 'ladder_audit_20261121', 'ladder_guard_20261121', 'ladder_pressure_20261121', 'ladder_pressure_lag_20261121']
REVIEWS = ['20261121_ladder_rest_review', '20261121_ladder_audit_review', '20261121_ladder_guard_review', '20261121_ladder_pressure_review', '20261121_ladder_pressure_lag_review']
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
                    jobs[m.group(2)] = dict(rc=int(m.group(3)), peak=int(m.group(4)), wall=float(m.group(5)), cpu=float(m.group(6)))
    return tot


man = json.loads((R / 'manifest.json').read_text()); days = man['days']; jobs = {}
so_far = cpu_of(glob.glob(str(L / '**' / '*.log'), recursive=True), jobs)
line = cpu_of([p for n in LINE for p in glob.glob(str(REPO / 'predictfun/reviews' / n / 'logs' / '**' / '*.log'), recursive=True)])
rev = cpu_of([p for n in REVIEWS for p in glob.glob(str(REPO / 'predictfun/reviews/automatic' / n / '*.log'))])
recs = {d: (json.loads((D / 'pilot' / f'{d}.json').read_text()) if (D / 'pilot' / f'{d}.json').exists() else None) for d in days}; pj = {d: jobs.get(f'ga_pilot_{d}') for d in days}
done = all(recs[d] is not None and pj[d] is not None and pj[d]['rc'] == 0 for d in days); res = dict(pilots_done=done, pilot_jobs=pj)
if done:
    full = [d for d in days if recs[d]['members']]; load = float(np.mean([recs[d]['timing']['load'] for d in full])); per = [m['s'] for d in full for m in recs[d]['timing']['markets']]
    rest_days = [d for d in days if man['parts']['rest'][d]]; wall = sum(load + len(man['parts']['rest'][d]) * float(np.mean(per)) for d in rest_days)
    cpu_w = sum(pj[d]['cpu'] for d in full) / sum(pj[d]['wall'] for d in full); rest_h = wall * cpu_w / 3600. + 0.08; proj = rest_h + so_far / 3600.
    line_h = (line + rev + so_far) / 3600.; peaks_ok = all(pj[d]['peak'] < 1536 and float(recs[d].get('hwm_mib', 0.)) <= 1536. for d in days)
    res.update(load_s=load, market_s_mean=float(np.mean(per)), rest_days=rest_days, rest_markets=sum(len(man['parts']['rest'][d]) for d in days), cpu_per_wall=cpu_w,
               projected_rest_wall_s=wall, so_far_core_hours=so_far / 3600., projected_core_hours=proj, cap=0.35, line_recorded_core_hours=line_h,
               line_projected_core_hours=line_h + rest_h, line_cap=2.0, peaks_ok=peaks_ok, run_rest=bool(peaks_ok and proj <= .35 and line_h + rest_h <= 2.0))
else:
    res.update(run_rest=False)
out.write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
