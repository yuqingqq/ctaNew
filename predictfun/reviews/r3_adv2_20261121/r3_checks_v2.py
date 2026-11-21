"""R3 DECLARED CHECKS v2: SUPERSEDES r3_checks.py (frozen in freeze 2, c8655f75) FOR CHECK G ONLY (P-2026-005; 2026-11-21, session
aea5bfb9). checks.json (v1) stays as written. v1's G required at least one binding check on every ok reached pair. Two ok KEEP pairs reached
at the final tick (4,950 ms, side 0) made none, and that is correct: pf_r2_labels.pair_bound forces the decision at tick k, decides the
other side at k only when it moves first ((episode + k) % 2 == side), then steps k+1..NT-1. At k = NT-1 with the other side already
decided, the frozen pi0 continuation is never called. This is the zero-callback case review 38febc0b closed for the label validator.
G2 = v1's G with that one predicate: binding_bad == 0 always, and binding_checked > 0 unless no callback is possible (k = NT-1 and the
other side moved first). A-F, H and I are v1's functions, run unchanged.
FALSIFIER (rule 15), on copies of the real parts (G2 against its own corruptions):
 - the clean copy is true;
 - a zero-callback ok pair moved to a non-final tick is false (a missing required callback);
 - a binding failure on one ok pair is false;
 - one episode's cap_ok false is false.
 E and H are re-run on v1's corruptions.
Writes checks_v2.json and checks_falsifier_v2.json. Usage (run_light): r3_checks_v2.py"""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r3_checks as V1                                   # noqa: E402  frozen (freeze 2): A-F, H, I and the readers

A, AE, AR, C, pd = V1.A, V1.AE, V1.AR, V1.C, V1.pd
NT = A.L.NT


def callback_possible(key):
    """A reached pair's arms can make a callback unless its tick is the last and the other side already moved first (episode 0)."""
    k = int(round(float(key['decision_ms']) / A.STEP))
    return k < NT - 1 or (k % 2) == int(key['side'])


def check_G2(out, R):
    plan = json.loads((AE.EVAL_DIR / 'plan.json').read_text()); parts = AR.population(plan); n_pop = sum(len(ws) for _, ws in parts); worlds, bad = 0, []
    reached = dict(ok=0, missing_mark=0, no_eligible_decision=0, other=0); zero_cb = 0
    for k in range(len(parts)):
        rec = json.loads((AR.OUT / 'parts' / f'part_{k:03d}.json').read_text()); b = (AR.OUT / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes()
        if hashlib.sha256(b).hexdigest() != rec['sha256']:
            bad.append(f'part {k} bytes'); continue
        for r in A.LS.gz_rows(b):
            worlds += 1
            if r['status'] != 'ok':
                continue
            if set(r['episodes']) != set(A.POLICIES):
                bad.append(f"{r['m']} {r['t_ms']} policies")
            for name, e in r['episodes'].items():
                cz = e.get('causal') or {}
                if not (e.get('ledger_ok') is True and e.get('binding_bad') == 0 and cz.get('mismatch', 0) == 0 and cz.get('parity_mismatch', 0) == 0
                        and e.get('cap_ok') is True and len(e.get('churn', [None])) == e.get('reviewed_generations') and (name != 'RICH' or isinstance(e.get('exposure'), dict))):
                    bad.append(f"{r['m']} {r['t_ms']} {name}")
                if name == 'N0' and (e.get('value') != 0. or e.get('generations') != 0):
                    bad.append(f"{r['m']} {r['t_ms']} N0")
            if sorted((x['key']['side'], x['key']['head']) for x in r['reached']) != sorted((s, h) for s, h in A.HEADS):
                bad.append(f"{r['m']} {r['t_ms']} reached entries")
            for x in r['reached']:
                st = x.get('status'); reached[st if st in reached else 'other'] += 1
                if st == 'ok':
                    cb_ok = x.get('binding_checked', 0) > 0 or not callback_possible(x['key'])
                    zero_cb += int(x.get('binding_checked', 0) == 0)
                    if not (x.get('reconciled') is True and x.get('binding_bad') == 0 and cb_ok):
                        bad.append(f"{r['m']} {r['t_ms']} reached pair")
                if (x.get('eligible', 0) > 0) != (st != 'no_eligible_decision'):
                    bad.append(f"{r['m']} {r['t_ms']} reached inclusion count")
    R['G'] = dict(population=n_pop, worlds=worlds, bad=bad[:10], reached=reached, zero_callback_ok_pairs=zero_cb)
    return bool(worlds == n_pop and not bad and reached['other'] == 0)


def main(out=A.OUT):
    out = Path(out); R = {}; ck = {}
    for k, f in (('A_extraction', V1.check_A), ('B_fit', V1.check_B), ('C_order', V1.check_C), ('D_sample', V1.check_D), ('E_readings', V1.check_E),
                 ('F_control', V1.check_F), ('G_replay', check_G2)):
        try:
            ck[k] = f(out, R)
        except Exception as e:                                                            # a check that cannot run is false, with its reason
            ck[k] = False; R[k[0]] = dict(error=repr(e)[:300])
    for k, f in (('H_suites', V1.check_H), ('I_resources', V1.check_I)):
        try:
            ck[k] = f(R)
        except Exception as e:
            ck[k] = False; R[k[0]] = dict(error=repr(e)[:300])
    v1 = json.loads((HERE / 'checks.json').read_text())
    J = dict(checks=ck, all_true=all(ck.values()), readings=R, supersedes=dict(file='checks.json', version='v1 (r3_checks.py, freeze 2)', v1_checks=v1['checks'],
             changed='G: zero binding checks allowed only where no callback is possible'), source_sha256=C.sha(Path(__file__)), asof=C.stamp())
    (HERE / 'checks_v2.json').write_text(json.dumps(J, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(checks=ck, all_true=J['all_true']), indent=1))


def falsify(out=A.OUT):
    out = Path(out); res = {}

    def part_edit(g, fn):
        """Rewrite part 000 of the copy with fn applied to its ok rows' list, its record's hash updated; returns whether fn found a target."""
        rec = json.loads((g / 'part_000.json').read_text()); rows = A.LS.gz_rows((g / 'part_000.jsonl.gz').read_bytes()); hit = fn(rows)
        b = A.LS.gz_bytes(rows); (g / 'part_000.jsonl.gz').write_bytes(b); rec['sha256'] = hashlib.sha256(b).hexdigest(); (g / 'part_000.json').write_text(json.dumps(rec))
        return hit

    def zero_cb_nonfinal(rows):
        for r in rows:
            for x in r.get('reached', []):
                if x.get('status') == 'ok':
                    x['binding_checked'] = 0; x['key']['decision_ms'] = 1000; return True
        return False

    def bind_bad(rows):
        for r in rows:
            for x in r.get('reached', []):
                if x.get('status') == 'ok':
                    x['binding_bad'] = 1; return True
        return False

    def cap_bad(rows):
        for r in rows:
            if r['status'] == 'ok':
                r['episodes']['RICH']['cap_ok'] = False; return True
        return False
    saved_out = AR.OUT
    with tempfile.TemporaryDirectory(dir=str(out)) as td:
        td = Path(td); src = AR.OUT / 'parts'
        try:
            for name, fn in (('clean', None), ('zero_callback_at_a_non_final_tick', zero_cb_nonfinal), ('binding_failure', bind_bad), ('cap_ok_false', cap_bad)):
                g = td / name / 'parts'; g.mkdir(parents=True)
                for q in src.glob('part_*'):
                    shutil.copy(q, g / q.name)
                hit = part_edit(g, fn) if fn else True; AR.OUT = td / name
                res[f'G2 {name}'] = dict(applied=hit, value=check_G2(out, {}))
        finally:
            AR.OUT = saved_out
    v1f = json.loads((HERE / 'checks_falsifier.json').read_text())['cases']
    ok = (res['G2 clean']['value'] is True and all(v['applied'] and v['value'] is False for k, v in res.items() if k != 'G2 clean')
          and all(v1f[c]['clean'] is True and v1f[c]['corrupted'] is False for c in ('E', 'H')))
    J = dict(cases=res, v1_cases_E_H={c: v1f[c] for c in ('E', 'H')}, falsifier_ok=ok, asof=C.stamp())
    (HERE / 'checks_falsifier_v2.json').write_text(json.dumps(J, indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(cases=res, falsifier_ok=ok)))


if __name__ == '__main__':
    main(); falsify()
