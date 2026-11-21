"""The declared pilot rule (declaration.md, Budget), WRITE-ONCE. From the Oct 24 pilot record (load seconds, per-market seconds for both run arms)
and its run_light record: the rest's wall = sum over Oct 25-27 of (the pilot's load + the date's complete markets x the pilot's mean seconds per
market); CPU = wall x the pilot's CPU / wall + 0.03 core-hours for the rehearsal, screen, measurement, checks and report + this round's CPU so far
(every run_light record under logs/). The ladder line's recorded use = its seven earlier receipts' logs + the seven review directories' logs + this
round. run_rest iff the pilot is done with rc 0, its peak and high-water below 1,536 MiB, this round's projection <= 0.14 core-hours and the line's
recorded use + the rest of this round <= 2.0 core-hours (otherwise stop and ask the USER). Refuses when the output exists."""
import glob
import json
import re
from pathlib import Path

REPO = Path('/home/yuqing/ctaNew'); R = REPO / 'predictfun/reviews/ladder_pressure_positive_20261121'; L = R / 'logs'; D = REPO / 'data/predict_fun/research/ladder_pressure_positive_v1_20261121'
LINE = ['ladder_rest_20261121', 'ladder_audit_20261121', 'ladder_guard_20261121', 'ladder_pressure_20261121', 'ladder_pressure_lag_20261121', 'ladder_pressure_ablation_20261121',
        'ladder_pressure_residual_20261121']
REVIEWS = ['20261121_ladder_rest_review', '20261121_ladder_audit_review', '20261121_ladder_guard_review', '20261121_ladder_pressure_review', '20261121_ladder_pressure_lag_review',
           '20261121_ladder_pressure_ablation_review', '20261121_ladder_pressure_residual_review']
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


man = json.loads((R / 'manifest.json').read_text()); jobs = {}; so_far = cpu_of(glob.glob(str(L / '**' / '*.log'), recursive=True), jobs)
line = cpu_of([p for n in LINE for p in glob.glob(str(REPO / 'predictfun/reviews' / n / 'logs' / '**' / '*.log'), recursive=True)])
rev = cpu_of([p for n in REVIEWS for p in glob.glob(str(REPO / 'predictfun/reviews/automatic' / n / '*.log'))])
pd_ = man['pilot_day']; rec = json.loads((D / 'pilot' / f'{pd_}.json').read_text()) if (D / 'pilot' / f'{pd_}.json').exists() else None; pj = jobs.get('gp2_pilot')
res = dict(pilot_done=bool(rec and pj and pj['rc'] == 0), pilot_job=pj)
if res['pilot_done']:
    load = float(rec['timing']['load']); per = sum(m['s'] for m in rec['timing']['markets']) / len(rec['timing']['markets'])
    rest = [d for d in man['days'] if man['parts']['rest'][d]]; wall = sum(load + len(man['parts']['rest'][d]) * per for d in rest); cpu_w = pj['cpu'] / pj['wall']
    rest_h = wall * cpu_w / 3600. + 0.03; proj = rest_h + so_far / 3600.; line_h = (line + rev + so_far) / 3600.; peak_ok = pj['peak'] < 1536 and float(rec['hwm_mib']) <= 1536.
    res.update(load_s=load, market_s_mean=per, rest_days=rest, rest_markets=sum(len(man['parts']['rest'][d]) for d in rest), cpu_per_wall=cpu_w, projected_rest_wall_s=wall,
               so_far_core_hours=so_far / 3600., projected_core_hours=proj, cap=0.14, line_recorded_core_hours=line_h, line_projected_core_hours=line_h + rest_h, line_cap=2.0,
               peak_ok=peak_ok, run_rest=bool(peak_ok and proj <= .14 and line_h + rest_h <= 2.0))
else:
    res.update(run_rest=False)
out.write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
