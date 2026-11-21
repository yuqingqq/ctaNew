"""The declared pilot rule (declaration.md, Budget): projection = the pilot's CPU x 12 + the round's CPU so far (recursive over logs/); the rest
runs only if the projection <= 2 core-hours and the pilot partition is done. Writes logs/pilot_projection.json and prints it."""
import glob, json, re
from pathlib import Path
R = Path('/home/yuqing/ctaNew/predictfun/reviews/entry_exposure_20261121'); L = R / 'logs'; D = Path('/home/yuqing/ctaNew/data/predict_fun/research/entry_exposure_v1_20261121')
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
so_far = 0.; pilot = None
for p in glob.glob(str(L / '**' / '*.log'), recursive=True):
    for ln in Path(p).read_text().splitlines():
        m = RL.match(ln)
        if m:
            so_far += float(m.group(6))
            if m.group(2) == 'exposure_part_000':
                pilot = dict(cpu=float(m.group(6)), peak=int(m.group(4)), rc=int(m.group(3)), wall=float(m.group(5)))
rec = D / 'parts' / 'part_000.json'; done = rec.exists()
proj = (pilot['cpu'] * 12 + so_far) / 3600. if pilot else None
out = dict(pilot_cpu_s=pilot and pilot['cpu'], pilot_wall_s=pilot and pilot['wall'], pilot_peak_mib=pilot and pilot['peak'], pilot_rc=pilot and pilot['rc'], pilot_record=done,
           pilot_hwm_mib=json.loads(rec.read_text())['hwm_mib'] if done else None, so_far_core_hours=so_far / 3600., projected_core_hours=proj, ceiling=2.0,
           run_rest=bool(done and pilot and pilot['rc'] == 0 and proj is not None and proj <= 2.0))
(L / 'pilot_projection.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out, indent=1))
