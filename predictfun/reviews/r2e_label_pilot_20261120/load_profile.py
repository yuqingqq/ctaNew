"""R2-E attempt-2 diagnosis: the memory of each loading step of one pilot partition, as pf_r2_label_sample.run performs them, on the consumed
day of attempt 2's latched partition 0 (Oct 24, its 8 worlds' markets). Inputs only: no census, pair or label is computed. After each step:
VmRSS and the never-reset VmHWM (MiB) -> load_profile_<part>.json. Usage (run_light): load_profile.py <plan dir> <part>"""
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
t0 = time.time(); prof = []


def mark(step):
    import pf_r2_runner as RN_
    r, h = RN_.rss(); prof.append(dict(step=step, rss_mib=round(r, 1), hwm_mib=round(h, 1), t_s=round(time.time() - t0, 1)))


mark('python start')
import pf_r2_label_sample as S                            # noqa: E402
mark('imports')
out, k = Path(sys.argv[1]), int(sys.argv[2]); plan, psha = S.load_plan(out); part = plan['partitions'][k]; day = part['day']; kp = Path(plan['kp_dir']); src = S.KP.Tape()
S.sources_now(); gc.collect(); mark('source closure (AST + hashes)')
M, sup, _ = S.L.teacher_assets(S.T.REAL, plan['freeze_sha256']); mark('teacher assets')
S.identity_now(day, kp, src, S.real_files, S.hours_tracked); mark('input identity')
allw = S.worlds_all(day, kp); mark(f'worlds_all ({len(allw)} worlds)')
allw = None; gc.collect(); mark('worlds_all released')
sel = {S.ckey(S.wkey(w)): w for w in plan['sample'][day]['selected']}; members = [sel[S.ckey(m)] for m in part['members']]; mk = sorted({w['m'] for w in members})
fair, depth, hrs, trk = S.X.real_inputs(day, mk); mark(f'real_inputs ({len(mk)} markets)')
data = src.load(day, mk, src.offset(day)); mark('tape load')
prs = {int(W_[0]['m']): W_ for W_ in src.pairs(data)}; mark('tape pairs')
data = None; gc.collect(); mark('tape data released')
print(json.dumps(prof, indent=1))
(out.parent / f'load_profile_{out.name}_part{k}.json').write_text(json.dumps(dict(day=day, part=k, markets=mk, steps=prof), indent=1))
