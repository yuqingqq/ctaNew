"""Checks for the inventory-mechanism runs (run_inv.sh; engine pf_step4_ladder_inv.py). Writes step4/inv_check.json; exit 1 on failure.
ctrl: on 10-24 and 11-01 (J), each never-trigger arm (SK1000, RX1000, CAP1000, RL1000000, RO300) gives fills / win / requests
      byte-identical to round i's v1 (lJi). Falsifier: the engaged SK5+RX5 arm must differ from v1 (else the comparison cannot fire).
main: (1) v1 run by the generated engine is byte-identical to round i (lJi / lXi) on every day and timing, summary arm entry equal;
      (2) every run has the declared arms, PF_R60_FILE = step4/inv_r60.json and its timing env;
      (3) every mechanism engaged: each counter group of pf_inv_report.arms() sums > 0 over the 9 days, per timing.
Usage: python3 -m predictfun.research.pf_inv_check ctrl|main
"""
import filecmp, json, sys
from . import pf_io as io
from .pf_inv_report import arms, name, stem, run_dir, DAYS, CTRL_DAYS, S4

KINDS = ('fills', 'win', 'requests')


def same(tag_a, spec_a, tag_b, spec_b, d):
    return all(filecmp.cmp(f'{run_dir(tag_a, d)}/{k}/{stem(spec_a)}.parquet', f'{run_dir(tag_b, d)}/{k}/{stem(spec_b)}.parquet', shallow=False) for k in KINDS)


def main(mode):
    MAIN, CTRL, ENG, L = arms(); V1 = MAIN['v1']; errs = []; rep = dict(mode=mode)
    path = f'{S4}/inv_check.json'
    prev = json.load(open(path)) if mode == 'main' else {}
    if mode == 'ctrl':
        rep['ctrl'] = {}
        for d in CTRL_DAYS:
            for lab, spec in CTRL.items():
                eq = same('ivc', spec, 'lJi', V1, d); rep['ctrl'][f'{d} {lab}'] = eq
                if lab == 'SK5+RX5' and eq:
                    errs.append(f'falsifier: SK5+RX5 equals v1 on {d}')
                if lab != 'SK5+RX5' and not eq:
                    errs.append(f'never-trigger {lab} differs from v1 on {d}')
    else:
        if not prev.get('ctrl_PASS'):
            errs.append('ctrl checks not PASS')
        rep['parity'] = {}; tot = {}
        for tm in ('J', 'X'):
            ref = 'l' + tm + 'i'; tag = 'iv' + tm
            for d in DAYS:
                s = json.load(open(f'{run_dir(tag, d)}/summary.json')); env = s.get('env', {})
                if sorted(s['arms']) != sorted(name(x) for x in MAIN.values()):
                    errs.append(f'{tm} {d} arms')
                if env.get('PF_R60_FILE') != f'{S4}/inv_r60.json' or env.get('PF_PREDS') != 'preds_t60k':
                    errs.append(f'{tm} {d} env {env.get("PF_R60_FILE")} {env.get("PF_PREDS")}')
                if (tm == 'J') != (env.get('PF_TXJIT') == 'frame') or (tm == 'X') != (env.get('PF_TXSHIFT') == '-50' and env.get('PF_QACT') == 'max'):
                    errs.append(f'{tm} {d} timing env')
                eq = same(tag, V1, ref, V1, d) and json.load(open(f'{run_dir(ref, d)}/summary.json'))['arms'][name(V1)] == s['arms'][name(V1)]
                rep['parity'][f'{tm} {d}'] = eq
                if not eq:
                    errs.append(f'v1 parity {tm} {d}')
                for lab, spec in MAIN.items():
                    for k, c in s['arms'][name(spec)]['withheld_decisions'].items():
                        tot.setdefault((tm, lab), {}).setdefault(k, 0); tot[(tm, lab)][k] += c
        rep['engaged'] = {}
        for (tm, lab), c in tot.items():
            for grp in ENG.get(lab, []):
                n = sum(c.get(k, 0) for k in grp); rep['engaged'][f'{tm} {lab} {"+".join(grp)}'] = n
                if n == 0:
                    errs.append(f'{tm} {lab} never engaged {grp}')
        if any(c.get(k, 0) for k in ('SK_shift', 'RX_exempt', 'RL', 'LD_RL', 'RO', 'LD_RO', 'CAPN', 'LD_CAPN') for (tm, lab), c in tot.items() if lab == 'v1'):
            errs.append('v1 shows a new-mechanism counter')
    rep['errors'] = errs
    out = dict(prev); out.update(rep)
    out[f'{mode}_PASS'] = not errs
    out['PASS'] = bool(out.get('ctrl_PASS')) and bool(out.get('main_PASS'))
    io.write_json(path, out)
    print(json.dumps({k: v for k, v in rep.items() if k != 'parity'}, indent=1)); print(mode, 'PASS' if not errs else f'FAIL {len(errs)}')
    return 0 if not errs else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
