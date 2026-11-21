"""The declared pilot rule (declaration.md, Budget), WRITE-ONCE. From the four pilot day records (load seconds, per-market seconds for both arms)
and the run_light records of the pilot jobs (CPU / wall): the full run's wall = sum over the four days of (the pilots' mean load + the day's
complete markets x the pilots' mean seconds per market); CPU = wall x the pilots' CPU / wall + 0.1 core-hours for selection, measurement and
checks + this round's CPU so far (every run_light record under logs/). run_rest iff all four pilots are DONE with rc 0, peaks / high-waters
below 1,536 MiB and the projection <= 0.75 core-hours. Refuses when the output exists."""
import glob
import json
import re
from pathlib import Path

import numpy as np

R = Path('/home/yuqing/ctaNew/predictfun/reviews/ladder_guard_20261121'); L = R / 'logs'; D = Path('/home/yuqing/ctaNew/data/predict_fun/research/ladder_guard_v1_20261121')
out = L / 'pilot_projection.json'
if out.exists():
    raise SystemExit(f'REFUSED: {out} exists (write-once)')
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
man = json.loads((R / 'manifest.json').read_text()); days = man['days']; so_far = 0.; jobs = {}
for p in glob.glob(str(L / '**' / '*.log'), recursive=True):
    for ln in Path(p).read_text().splitlines():
        m = RL.match(ln)
        if m:
            so_far += float(m.group(6)); jobs[m.group(2)] = dict(rc=int(m.group(3)), peak=int(m.group(4)), wall=float(m.group(5)), cpu=float(m.group(6)))
recs = {d: (json.loads((D / 'pilot' / f'{d}.json').read_text()) if (D / 'pilot' / f'{d}.json').exists() else None) for d in days}; pj = {d: jobs.get(f'gd_pilot_{d}') for d in days}
done = all(recs[d] is not None and pj[d] is not None and pj[d]['rc'] == 0 for d in days); res = dict(pilots_done=done, pilot_jobs=pj)
if done:
    load = float(np.mean([recs[d]['timing']['load'] for d in days])); per = [m['s'] for d in days for m in recs[d]['timing']['markets']]
    wall = sum(load + len(man['complete'][d]) * float(np.mean(per)) for d in days); cpu_w = sum(pj[d]['cpu'] for d in days) / sum(pj[d]['wall'] for d in days)
    proj = wall * cpu_w / 3600. + 0.1 + so_far / 3600.; peaks_ok = all(pj[d]['peak'] < 1536 and float(recs[d]['hwm_mib']) <= 1536. for d in days)
    res.update(load_s=load, market_s_mean=float(np.mean(per)), markets=sum(len(v) for v in man['complete'].values()), cpu_per_wall=cpu_w, projected_full_wall_s=wall,
               pilot_core_hours=sum(pj[d]['cpu'] for d in days) / 3600., so_far_core_hours=so_far / 3600., projected_core_hours=proj, cap=0.75, peaks_ok=peaks_ok,
               run_rest=bool(peaks_ok and proj <= .75))
else:
    res.update(run_rest=False)
out.write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
