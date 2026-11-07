"""Parity of the generated pm_ladder_v3 engine (pf_step4_ladder_v3.py, pf_ladder_v3_build.py) against the in-sample runs that produced
the CAP10 + RO60-240 result (tags i2J / i2X, engine pf_step4_ladder_inv.py, report_inv2.json). Runs by run_v3_check.sh (run_light):
  v3pJ  11-03 J, PF_FORWARD unset, v3 arm alone, no PF_R60_FILE
  v3pX  11-03 X, same
  v3pF  11-02 J, PF_FORWARD=20261114 (the forward run's value)
PASS iff for each: fills / win / requests of the v3 arm byte-identical to i2J / i2X and the summary arm dict equal; the run's env is as
declared (PF_FORWARD, no PF_R60_FILE, the timing env); the engine file hash equals params.json's.
Falsifier (must fire): the same byte compare of the v3pJ output against i2J's RO60-240 and CAP10 arms (its two parents) must differ.
Writes step4/v3_engine_check.json.
Usage: python3 -m predictfun.research.pf_v3_engine_check
"""
import filecmp, hashlib, json, os, sys
from . import pf_io as io
from .pf_inv_report import name, stem, run_dir
from .pf_inv2 import ARMS, KINDS
from .pf_ladder_v3_build import ARM, OUT, DST

RUNS = [('v3pJ', 'i2J', '20261103', 'J', None), ('v3pX', 'i2X', '20261103', 'X', None), ('v3pF', 'i2J', '20261102', 'J', '20261114')]


def same_files(tag_a, spec_a, tag_b, spec_b, d):
    return all(filecmp.cmp(f'{run_dir(tag_a, d)}/{k}/{stem(spec_a)}.parquet', f'{run_dir(tag_b, d)}/{k}/{stem(spec_b)}.parquet', shallow=False)
               for k in KINDS)


def main():
    errs = []; rep = dict(runs={}, falsifier={})
    if ARMS['CAP10+RO60-240'] != ARM:
        errs.append('v3 arm differs from the tested CAP10+RO60-240 arm')
    eng = hashlib.sha256(open(DST, 'rb').read()).hexdigest()
    if eng != json.load(open(OUT))['engine']['sha256']:
        errs.append('engine hash differs from params.json')
    for tag, ref, d, tm, fwd in RUNS:
        s = json.load(open(f'{run_dir(tag, d)}/summary.json')); env = s.get('env', {})
        r = json.load(open(f'{run_dir(ref, d)}/summary.json'))
        ok_env = (env.get('PF_FORWARD') == fwd and 'PF_R60_FILE' not in env and env.get('PF_PREDS') == 'preds_t60k' and list(s['arms']) == [name(ARM)]
                  and ((tm == 'J') == (env.get('PF_TXJIT') == 'frame')) and ((tm == 'X') == (env.get('PF_TXSHIFT') == '-50' and env.get('PF_QACT') == 'max')))
        ok_files = same_files(tag, ARM, ref, ARM, d)
        ok_sum = s['arms'][name(ARM)] == r['arms'][name(ARM)]
        rep['runs'][f'{tag} {d}'] = dict(env=ok_env, files_identical=ok_files, summary_equal=ok_sum)
        for k, v in rep['runs'][f'{tag} {d}'].items():
            if not v:
                errs.append(f'{tag} {d} {k}')
    for k in ('RO60-240', 'CAP10'):
        diff = not same_files('v3pJ', ARM, 'i2J', ARMS[k], '20261103')
        rep['falsifier'][f'v3pJ vs i2J {k} differs'] = diff
        if not diff:
            errs.append(f'falsifier did not fire: v3 equals {k}')
    rep['engine_sha256'] = eng; rep['errors'] = errs; rep['PASS'] = not errs
    io.write_json(f'{io.OUT}/step4/v3_engine_check.json', rep)
    print(json.dumps(rep, indent=1))
    print('PASS' if not errs else f'FAIL {len(errs)}')
    return 0 if not errs else 1


if __name__ == '__main__':
    sys.exit(main())
