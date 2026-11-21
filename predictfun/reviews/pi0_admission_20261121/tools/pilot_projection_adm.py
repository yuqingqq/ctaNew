"""The declared pilot rule (declaration.md, Budget), WRITE-ONCE (review 99fb165d item 5): projection = the pilot's CPU x (selected generations /
the pilot's) + the round's CPU so far (recursive over logs/); the rest runs only if the projection <= 1 core-hour and the pilot is done. Refuses
when logs/pilot_projection.json exists."""
import glob, json, re
from pathlib import Path
R = Path('/home/yuqing/ctaNew/predictfun/reviews/pi0_admission_20261121'); L = R / 'logs'; D = Path('/home/yuqing/ctaNew/data/predict_fun/research/pi0_admission_v1_20261121')
out = L / 'pilot_projection.json'
if out.exists():
    raise SystemExit(f'REFUSED: {out} exists (write-once)')
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
so_far = 0.; pilot = None
for p in glob.glob(str(L / '**' / '*.log'), recursive=True):
    for ln in Path(p).read_text().splitlines():
        m = RL.match(ln)
        if m:
            so_far += float(m.group(6))
            if m.group(2) == 'admission_part_000':
                pilot = dict(cpu=float(m.group(6)), peak=int(m.group(4)), rc=int(m.group(3)), wall=float(m.group(5)))
man = json.loads((D / 'manifest.json').read_text()); n_all = len(man['selected']); n_p = sum(1 for g in man['selected'] if int(g['part']) == 0)
rec = D / 'parts' / 'part_000.json'; done = rec.exists()
proj = (pilot['cpu'] * n_all / max(1, n_p) + so_far) / 3600. if pilot else None
res = dict(pilot_cpu_s=pilot and pilot['cpu'], pilot_wall_s=pilot and pilot['wall'], pilot_peak_mib=pilot and pilot['peak'], pilot_rc=pilot and pilot['rc'], pilot_record=done,
           pilot_generations=n_p, selected_generations=n_all, so_far_core_hours=so_far / 3600., projected_core_hours=proj, ceiling=1.0,
           run_rest=bool(done and pilot and pilot['rc'] == 0 and proj is not None and proj <= 1.0))
out.write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
