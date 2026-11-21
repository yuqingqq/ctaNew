"""ENTRY-RETENTION-DISCRIMINATION-v1: THE DECLARED FALSIFIER, v2 (2026-11-21, session aea5bfb9). Supersedes the frozen falsifier's argument
plumbing only. The frozen falsify() (retention_checks.py) unpacked `args_c, args_x = (d, clean_e) if on == 'out' else (clean, d)` into two
paths instead of one argument tuple per case, so its first case raised TypeError before any case was recorded (logs/retention_checks.log).
The frozen checks A-H ran and are all true (checks.json, kept). This file is the frozen falsify() with that one line corrected and the logs
switch made on the frozen module's global; the cases, corruptions and checks are the frozen ones. Writes checks_falsifier_v2.json.
Usage (run_light): retention_checks_v2.py"""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import retention_checks as X                             # noqa: E402  the frozen checks

M, C, pd, LS = X.M, X.C, X.pd, X.LS


def falsify_v2(out=M.OUT, expo=M.EXPO):
    out, expo = Path(out), Path(expo); res = {}

    def safe(f, *a):
        try:
            return f(*a)
        except Exception:                                                                 # noqa: BLE001
            return False
    with tempfile.TemporaryDirectory(dir=str(out)) as td:
        td = Path(td)

        def copy_(name, base):
            d = td / name; (d / 'parts').mkdir(parents=True)
            for q in base.iterdir():
                if q.is_file():
                    shutil.copy(q, d / q.name)
            for q in (base / 'parts').iterdir():
                if q.is_file():
                    (d / 'parts' / q.name).symlink_to(q.resolve())
            if (base / 'v2').exists():
                (d / 'v2').symlink_to((base / 'v2').resolve(), target_is_directory=True)
            return d

        def corrupt(d, fn):
            """Replace the first partition of copy d holding a row fn applies to; its record's hash updated."""
            for k in range(12):
                gp, rp = d / 'parts' / f'part_{k:03d}.jsonl.gz', d / 'parts' / f'part_{k:03d}.json'; rows = LS.gz_rows(gp.read_bytes())
                if any(fn(r) for r in rows if r.get('status') == 'ok'):
                    nb = LS.gz_bytes(rows); rec = json.loads(rp.read_text()); gp.unlink(); gp.write_bytes(nb); rp.unlink(); rp.write_text(json.dumps(dict(rec, sha256=hashlib.sha256(nb).hexdigest())))
                    return True
            return False
        once = lambda fn: (lambda st: (lambda r: (not st) and fn(r) and (st.append(1) or True)))([])

        def drop(r):
            return False
        def t_review(r):
            x = r['stages']['first_review']
            if x.get('status') != 'ok':
                return False
            x['t'] = float(x['t']) + 50.; return True
        def missing_cb(r):
            r['checks']['callbacks'] -= 1; r['checks']['expected_callbacks'] -= 1; return True
        def comp(r):
            x = r['stages']['first_review']
            if x.get('status') != 'ok':
                return False
            x['target']['value'] += 1.; x['unavoidable'] -= 1.; return True
        def wrong_gid(r):
            r['held']['gid'] = int(r['held']['gid']) + 1; return True
        def an_input(r):
            x = r['stages']['entry']
            if not x.get('inputs'):
                return False
            x['inputs'][M.FEATS.index('pf_mid')] += .5; return True
        def wrong_end(r):
            r['arms']['POST_PI0']['end'] = 6000.; return True
        clean, clean_e = copy_('clean', out), copy_('clean_expo', expo)
        for name, chk, fn, on in (('B_review_time', X.check_B, t_review, 'out'), ('B_missing_callback', X.check_B, missing_cb, 'out'), ('C_compensating', X.check_C, comp, 'out'),
                                  ('C_wrong_gid', X.check_C, wrong_gid, 'out'), ('D_input', X.check_D, an_input, 'out'), ('B_wrong_END', X.check_B, wrong_end, 'expo')):
            d = copy_(name, out if on == 'out' else expo); ap = corrupt(d, once(fn))
            args = (d, clean_e) if on == 'out' else (clean, d)                                     # v2: one argument tuple per case
            res[name] = dict(clean=safe(chk, clean, clean_e, {}), corrupted=safe(chk, *args, {}), applied=bool(ap))
        d = copy_('A_drop', out); gp = d / 'parts' / 'part_000.jsonl.gz'; rows = LS.gz_rows(gp.read_bytes()); rp = d / 'parts' / 'part_000.json'; rec = json.loads(rp.read_text())
        nb = LS.gz_bytes(rows[:-1]); gp.unlink(); gp.write_bytes(nb); rp.unlink(); rp.write_text(json.dumps(dict(rec, sha256=hashlib.sha256(nb).hexdigest(), rows=len(rows) - 1)))
        res['A'] = dict(clean=safe(X.check_A, clean, clean_e, {}), corrupted=safe(X.check_A, d, clean_e, {}), applied=True)
        d = copy_('E', out); c_ = pd.read_csv(d / 'calibration.csv', float_precision='round_trip'); c_.loc[0, 'mae_V'] += 1.; c_.to_csv(d / 'calibration.csv', index=False)
        res['E'] = dict(clean=safe(X.check_E, clean, clean_e, {}), corrupted=safe(X.check_E, d, clean_e, {}), applied=True)
        d = copy_('F', out); j = json.loads((d / 'retention.json').read_text()); j['predicates']['readings'] = {k_: (not v) for k_, v in j['predicates']['readings'].items()}
        (d / 'retention.json').write_text(json.dumps(j)); res['F'] = dict(clean=safe(X.check_F, clean, clean_e, {}), corrupted=safe(X.check_F, d, clean_e, {}), applied=True)
        saved = X.LOGS
        try:
            h = td / 'G_logs'; shutil.copytree(saved, h); X.LOGS = h; clG = safe(X.check_G, out, expo, {})
            (h / 'over_budget.log').write_text('2026-11-21T00:00:00Z tag=over_budget rc=0 peak_rss_mib=100 wall_s=3700 cpu_user_s=3700.0\n')
            res['G'] = dict(clean=clG, corrupted=safe(X.check_G, out, expo, {}), applied=True)
            h2 = td / 'H_logs'; shutil.copytree(saved, h2); X.LOGS = h2; clH = safe(X.check_H, {})
            shutil.copy(h2 / 'mutants_ret' / 'mut_ret_0.log', h2 / 'mutants_ret' / 'mut_ret_1.log'); res['H'] = dict(clean=clH, corrupted=safe(X.check_H, {}), applied=True)
        finally:
            X.LOGS = saved
    ok = all(v['clean'] is True and v['corrupted'] is False and v['applied'] for v in res.values())
    (HERE / 'checks_falsifier_v2.json').write_text(json.dumps(dict(cases=res, falsifier_ok=ok, asof=C.stamp()), indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(cases=res, falsifier_ok=ok)))


if __name__ == '__main__':
    falsify_v2()
