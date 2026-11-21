"""The declared pilot rule (declaration.md, Budget), WRITE-ONCE. From the six pilot records (pilot_<day>.json: load, per-world census and label
seconds, queries) and the run_light records of the pilot jobs (CPU / wall): wall seconds projected for the build = sum over partitions of (the
pilots' mean fixed load - identity, worlds, day identity - + their per-member load - real inputs, tape - x members) + census seconds per world x the selected worlds + label seconds (a + b q fitted per pilot world,
a, b >= 0) at the query cap (4,608) over the selected worlds; CPU = wall x the pilot's CPU / wall ratio; + 0.3 core-hours for measure and
checks + the round's CPU so far (every run_light record under logs/ and source/logs/). run_rest iff all six pilots are DONE with job rc 0,
peaks and high-waters below 1,536 MiB, the pilot CPU <= 0.5 core-hours and the projection <= 8 core-hours. Refuses when the output exists."""
import glob
import json
import re
from pathlib import Path

import numpy as np

R = Path('/home/yuqing/ctaNew/predictfun/reviews/coverage_20261121'); L = R / 'logs'; D = Path('/home/yuqing/ctaNew/data/predict_fun/research/pi0_coverage_v1_20261121')
out = L / 'pilot_projection.json'
if out.exists():
    raise SystemExit(f'REFUSED: {out} exists (write-once)')
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
man = json.loads((R / 'manifest.json').read_text()); days = man['days']; so_far = 0.; jobs = {}
for p in glob.glob(str(L / '**' / '*.log'), recursive=True) + glob.glob(str(R / 'source' / 'logs' / '**' / '*.log'), recursive=True):
    for ln in Path(p).read_text().splitlines():
        m = RL.match(ln)
        if m:
            so_far += float(m.group(6)); jobs[m.group(2)] = dict(rc=int(m.group(3)), peak=int(m.group(4)), wall=float(m.group(5)), cpu=float(m.group(6)))
recs = {d: (json.loads((D / 'pilot' / f'pilot_{d}.json').read_text()) if (D / 'pilot' / f'pilot_{d}.json').exists() else None) for d in days}
pj = {d: jobs.get(f'cov_pilot_{d}') for d in days}
done = all(recs[d] is not None and pj[d] is not None and pj[d]['rc'] == 0 for d in days)
res = dict(pilots_done=done, pilot_jobs=pj)
if done:
    lp = [recs[d]['timing']['load_parts'] for d in days]; fixed = float(np.mean([x['identity'] + x['worlds'] + x['day_identity'] for x in lp]))
    load_pm = sum(x['real_inputs'] + x['tape'] for x in lp) / sum(recs[d]['members'] for d in days)
    cen = [x for d in days for x in recs[d]['timing']['census']]; lab = [x for d in days for x in recs[d]['timing']['labels']]
    nq = [n for d in days for n in recs[d]['timing']['label_queries']]
    if len(nq) == len(lab) and len(set(nq)) >= 2:
        A_ = np.c_[np.ones(len(nq)), np.asarray(nq, float)]; a, b = np.linalg.lstsq(A_, np.asarray(lab, float), rcond=None)[0]; a, b = max(0., float(a)), max(0., float(b))
    else:
        q_tot = sum(recs[d]['queries'] for d in days); a, b = 0., (sum(lab) / q_tot if q_tot else 0.)
    n_worlds = sum(len(man['sample'][d]['selected']) for d in days); members = [len(p['members']) for p in man['partitions']]
    wall = fixed * len(members) + load_pm * sum(members) + (np.mean(cen) if cen else 0.) * n_worlds + a * n_worlds + b * 4608.
    cpu_w = sum(pj[d]['cpu'] for d in days) / sum(pj[d]['wall'] for d in days); pilot_cpu = sum(pj[d]['cpu'] for d in days) / 3600.
    proj = wall * cpu_w / 3600. + 0.3 + so_far / 3600.
    peaks_ok = all(pj[d]['peak'] < 1536 and float(recs[d]['hwm_mib']) <= 1536. for d in days)
    res.update(load_fixed_s=fixed, load_s_per_member=load_pm, census_s_per_world=float(np.mean(cen)) if cen else None, label_a_s=a, label_b_s_per_query=b, worlds=n_worlds, query_cap=4608,
               cpu_per_wall=cpu_w, projected_build_wall_s=wall, pilot_core_hours=pilot_cpu, so_far_core_hours=so_far / 3600., projected_core_hours=proj, ceiling=8.0,
               peaks_ok=peaks_ok, run_rest=bool(peaks_ok and pilot_cpu <= .5 and proj <= 8.))
else:
    res.update(run_rest=False)
out.write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
