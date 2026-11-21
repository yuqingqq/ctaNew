"""The declared pilot rule (declaration.md, Budget), WRITE-ONCE: projection = max(the pilot's CPU x the partitions with held fills, the pilot's CPU x
the worlds with held fills / the pilot's) + the round's CPU so far (every run_light record under logs/, recursive). The rest runs only if the pilot's
mids record exists, its job returned 0, its peak and the record's high-water are below 1,536 MiB and the projection is at most 1 core-hour.
Refuses when logs/pilot_projection.json exists."""
import glob
import gzip
import json
import re
from pathlib import Path

R = Path('/home/yuqing/ctaNew/predictfun/reviews/movement_audit_20261121'); L = R / 'logs'; D = Path('/home/yuqing/ctaNew/data/predict_fun/research/pi0_movement_audit_v1_20261121')
A = Path('/home/yuqing/ctaNew/data/predict_fun/research/pi0_admission_v1_20261121')
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
            if m.group(2) == 'mov_part_000':
                pilot = dict(cpu=float(m.group(6)), peak=int(m.group(4)), rc=int(m.group(3)), wall=float(m.group(5)))
worlds = {}
for k in json.loads((A / 'manifest.json').read_text())['parts']:
    rows = [json.loads(x) for x in gzip.decompress((A / 'parts' / f'part_{int(k):03d}.jsonl.gz').read_bytes()).decode().splitlines() if x]
    worlds[int(k)] = len({(r['m'], r['t_ms']) for r in rows if r['status'] == 'ok' and r['held']['fills']})
parts = sum(1 for v in worlds.values() if v); n_w = sum(worlds.values()); n_p = worlds.get(0, 0)
rec = D / 'parts' / 'mids_000.json'; done = rec.exists(); hw = json.loads(rec.read_text()).get('hwm_mib') if done else None
proj = (max(pilot['cpu'] * parts, pilot['cpu'] * n_w / max(1, n_p)) + so_far) / 3600. if pilot else None
res = dict(pilot_cpu_s=pilot and pilot['cpu'], pilot_wall_s=pilot and pilot['wall'], pilot_peak_mib=pilot and pilot['peak'], pilot_rc=pilot and pilot['rc'], pilot_record=done,
           pilot_hwm_mib=hw, partitions_with_fills=parts, worlds_with_fills=n_w, pilot_worlds=n_p, so_far_core_hours=so_far / 3600., projected_core_hours=proj, ceiling=1.0,
           run_rest=bool(done and pilot and pilot['rc'] == 0 and pilot['peak'] < 1536 and hw is not None and hw <= 1536. and proj is not None and proj <= 1.0))
out.write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))
