"""R2-A launch gate demonstration on the frozen 256 originals (review 4cd7b45f's next priority; consumed Oct 23), one mode per process:
A           the uninterrupted runner census (tag _<label>A; label gate, then gate2): the plan, 4 parts of 64, the audit sample r2_full_causal_v1, finalized from the records.
B1          the same plan (tag _<label>B) interrupted after 2 parts (the planted STOP_AFTER).
B2          B resumed in a new process: the 2 accepted parts kept, the 2 missing ones run, finalized.
refuse      on copies of the completed B directory: a corrupt part, a forged / overlapping record, another plan's record, a stale source, a changed
            input, a different plan - each must be refused with its declared message; B itself must be unchanged -> refuse.json.
refinalize  A finalized again on a copy with every all-part reader replaced by a sentinel -> refinalize.json (its readings)."""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_r2_runner as RN                                  # noqa: E402
import pf_r2_shadow as R2                                  # noqa: E402

OUT, DAY, HERE = Path(R2.OUT), '20261023', Path(__file__).resolve().parent
MODE, LABEL = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'gate')   # LABEL: the run's tag prefix (the rerun after the plan fix: gate2)
census = lambda tag: RN.census(day=DAY, per_side=128, out=OUT, part_size=64, tag=tag)
digest = lambda d: {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(d).rglob('*')) if p.is_file()}
if MODE == 'A':
    print(json.dumps(dict(mode='A', state=census(f'_{LABEL}A')[2].get('state'))), flush=True)
elif MODE == 'B1':
    RN.STOP_AFTER = 2; d, _, R = census(f'_{LABEL}B')
    print(json.dumps(dict(mode='B1', state=R.get('state'), records=len(list((d / 'parts').glob('part_*.json'))))), flush=True)
elif MODE == 'B2':
    print(json.dumps(dict(mode='B2', state=census(f'_{LABEL}B')[2].get('state'))), flush=True)
elif MODE == 'refuse':
    B = OUT / f'census_{DAY}_{LABEL}B'; before = digest(B); res = {}
    with tempfile.TemporaryDirectory(dir=str(R2.I.K.IO)) as tmp:
        def copy_of(name):
            dd = Path(tmp) / name / f'census_{DAY}_{LABEL}B'; shutil.copytree(B, dd); return dd
        def refused(fn, why):
            try:
                fn()
            except Exception as e:
                return dict(refused=why in str(e), message=str(e)[:200])
            return dict(refused=False, message='accepted')
        c = copy_of('corrupt'); pf = c / 'parts' / 'part_0001.parquet'; b_ = bytearray(pf.read_bytes()); b_[len(b_) // 2] ^= 0xFF; pf.write_bytes(bytes(b_))
        res['corrupt part'] = refused(lambda: RN.run(c), 'hash differs from the record')
        c = copy_of('overlap'); r_ = json.loads((c / 'parts' / 'part_0001.json').read_text()); r_['part'] = 2; (c / 'parts' / 'part_0002.json').write_text(json.dumps(r_))
        res['overlapping record'] = refused(lambda: RN.run(c), 'members differ')
        c = copy_of('other'); r_ = json.loads((c / 'parts' / 'part_0003.json').read_text()); r_['plan_sha256'] = '0' * 64; (c / 'parts' / 'part_0003.json').write_text(json.dumps(r_))
        res['another plan\'s record'] = refused(lambda: RN.run(c), 'another plan')
        c = copy_of('stale'); s0 = RN.sources
        try:
            RN.sources = lambda mod: dict(s0(mod), **{'pf_r2_shadow.py': '0' * 64}); res['stale source'] = refused(lambda: RN.run(c), 'stale source')
        finally:
            RN.sources = s0
        c = copy_of('input')
        res['changed input'] = refused(lambda: RN.run(c, inputs=lambda day, mk: R2.X.real_inputs(day, mk)[:3] + ({'a changed input': '0'},)), 'changed input')
        c = copy_of('plan')
        res['a different plan'] = refused(lambda: RN.make_plan(c, R2, DAY, 128, 'first_per_side', 32, None, None, None, lambda day, mk: ({}, None, [], {}), True, None,
                                                               False, None, f'_{LABEL}B'), 'a different plan exists')
    res['B unchanged'] = dict(refused=digest(B) == before, message=f'{len(before)} files hashed before and after')
    (HERE / f'{LABEL}_refuse.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1), flush=True)
elif MODE == 'refinalize':
    A = OUT / f'census_{DAY}_{LABEL}A'
    with tempfile.TemporaryDirectory(dir=str(R2.I.K.IO)) as tmp:
        c = Path(tmp) / f'census_{DAY}_{LABEL}A'; shutil.copytree(A, c)
        def sentinel(*a, **k):
            raise AssertionError('an all-part read during finalization')
        R2.materialize = R2.read_parts = sentinel
        _, S, R = RN.finalize(c, R2)
    (HERE / f'{LABEL}_refinalize.json').write_text(json.dumps(dict(readings=R, support_sha256=hashlib.sha256(json.dumps(S, sort_keys=True, default=str).encode()).hexdigest()),
                                                      indent=1, default=str))
    print(json.dumps(dict(mode='refinalize', state=R['state'])), flush=True)
else:
    sys.exit('usage: gate_run.py A | B1 | B2 | refuse | refinalize [label]')
