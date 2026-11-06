"""Confirmation C2 report (declared in STATUS.yml before any C2 day was read): computed predicates, never printed verdicts.

Arms (MICRO policy, tokyo20, E2 timing, fill model v2, model = pf_step5_c2 fit on the six read days):
  P   D1 stack + M025, adaptive A1 5..25 (>= 0)        primary candidate (O2's B2 + A1)
  S   combo G1+G5+M025, adaptive A1 5..25 (>= 0)       secondary candidate (O2's B1 + A1)
  R2  D1 stack + M025, fixed 25        R3  combo, fixed 25        R1  D1 stack + M025, fixed 5
  R0  D1 stack alone, fixed 5          R4  D1 stack alone, fixed 25
  C1c D1 stack + M05, fixed 5 (C1's candidate, now with the C2 model)      R5  combo, fixed 5 (descriptive)
Each predicate on mid timing over the admitted days: sum > 0 AND > 0 on >= 4 of 5 days (needs exactly 5 admitted days;
otherwise status INSUFFICIENT_DAYS and every predicate is None). Differences are per day, paired by day.
Decisions: P_CONFIRMED = P_POSITIVE and P_ADAPTIVE_BEATS_FIXED; S_REPLACES_P = S_POSITIVE and S_BEATS_P.
Reported, not deciding: early/late timing, +1 s mark c/sh, inventory p95, rebate, 90% t-interval of the per-day mean
(n = 5 days, descriptive), model R2 per day, day admission statuses.
No permutation null: 5 day-level deltas allow only 2^5 = 32 sign-flip draws (< 200), so the rule is a count rule.
Usage: python3 -m predictfun.research.pf_c2_report            (reads step4/<C2 dir>/<day> for the admitted days)
       python3 -m predictfun.research.pf_c2_report --selftest (synthetic positive / negative controls; must pass)
"""
import sys, os, json, math, datetime as dt
from . import pf_io as io

ST = 'MICRO+G4+G2+G6+G5+G3+G1'; CB = 'MICRO+G5+G1+M025'; A1 = '@A1:5:25:0'
ARMS = {'P': f'{ST}+M025/mid{A1}', 'P_early': f'{ST}+M025/early{A1}', 'P_late': f'{ST}+M025/late{A1}',
        'S': f'{CB}/mid{A1}', 'S_early': f'{CB}/early{A1}', 'S_late': f'{CB}/late{A1}',
        'R2': f'{ST}+M025/mid@25', 'R3': f'{CB}/mid@25', 'R1': f'{ST}+M025/mid', 'R0': f'{ST}/mid', 'R4': f'{ST}/mid@25',
        'C1c': f'{ST}+M05/mid', 'R5': f'{CB}/mid'}
PRED = {'P_POSITIVE': ('P', None), 'P_ADAPTIVE_BEATS_FIXED': ('P', 'R2'), 'MODEL_ADDS_AT_SIZE': ('P', 'R4'),
        'MODEL_GUARD_ADDS_5': ('R1', 'R0'), 'C1_REPLICATES': ('C1c', 'R0'),
        'S_POSITIVE': ('S', None), 'S_ADAPTIVE_BEATS_FIXED': ('S', 'R3'), 'S_BEATS_P': ('S', 'P')}
N_DAYS, MIN_POS = 5, 4
DIR = f'{io.OUT}/step4/tokyo20_custom_e2_preds_c2_c2'; ADMIT = f'{io.OUT}/c2/admission.jsonl'
T90 = 2.1318  # t_{0.95, 4}


def predicates(G, days):
    """G[label][day] = gross USD. Returns {name: dict(by_day, sum, pos, value)}; value None unless exactly N_DAYS days."""
    out = {}
    for name, (a, b) in PRED.items():
        d = [G[a][x] - (G[b][x] if b else 0.0) for x in days]
        s = sum(d); pos = sum(v > 0 for v in d)
        val = None if len(days) != N_DAYS else bool(s > 0 and pos >= MIN_POS)
        m = s / len(d) if d else float('nan')
        sd = math.sqrt(sum((v - m) ** 2 for v in d) / (len(d) - 1)) if len(d) > 1 else float('nan')
        ci = [round(m - T90 * sd / math.sqrt(len(d)), 2), round(m + T90 * sd / math.sqrt(len(d)), 2)] if len(d) == N_DAYS else None
        out[name] = dict(a=a, b=b, by_day=[round(v, 2) for v in d], sum=round(s, 2), pos=pos, value=val, t90_mean_per_day=ci)
    dec = {'P_CONFIRMED': ('P_POSITIVE', 'P_ADAPTIVE_BEATS_FIXED'), 'S_REPLACES_P': ('S_POSITIVE', 'S_BEATS_P')}
    for k, parts in dec.items():
        vs = [out[p]['value'] for p in parts]
        out[k] = dict(parts=parts, value=None if any(v is None for v in vs) else all(vs))
    return out


def selftest():
    days = [f'd{i}' for i in range(5)]
    def mk(f):
        return {lab: {d: f(lab, i) for i, d in enumerate(days)} for lab in ARMS}
    good = predicates(mk(lambda lab, i: {'P': 10, 'R2': 5, 'R4': 1, 'R1': 3, 'R0': 1, 'C1c': 2, 'S': 20, 'R3': 4}.get(lab, 0) + 0.0), days)
    assert all(good[k]['value'] is True for k in PRED) and good['P_CONFIRMED']['value'] and good['S_REPLACES_P']['value'], good
    # 3 of 5 days positive with a positive sum must FAIL (count rule), and a negative sum with 4/5 positive must FAIL
    three = predicates(mk(lambda lab, i: (100.0 if i < 3 else -1.0) if lab == 'P' else 0.0), days)
    assert three['P_POSITIVE']['value'] is False and three['P_POSITIVE']['sum'] > 0, three['P_POSITIVE']
    negsum = predicates(mk(lambda lab, i: (1.0 if i < 4 else -100.0) if lab == 'P' else 0.0), days)
    assert negsum['P_POSITIVE']['value'] is False and negsum['P_POSITIVE']['pos'] == 4, negsum['P_POSITIVE']
    assert negsum['P_CONFIRMED']['value'] is False
    four = predicates(mk(lambda lab, i: (5.0 if i < 4 else -1.0) if lab == 'P' else 0.0), days)
    assert four['P_POSITIVE']['value'] is True
    short = predicates(mk(lambda lab, i: 10.0), days[:4])
    assert all(short[k]['value'] is None for k in list(PRED) + ['P_CONFIRMED', 'S_REPLACES_P']), 'INSUFFICIENT_DAYS must give None'
    print('selftest PASS: positive control fires True, 3/5 and negative-sum controls fire False, 4-day input gives None')


def admitted_days():
    if not os.path.exists(ADMIT):
        return [], []
    recs = [json.loads(l) for l in open(ADMIT) if l.strip()]
    last = {}
    for r in recs:
        last[r['day']] = r
    adm = sorted(d for d, r in last.items() if r['status'] == 'ADMITTED')
    return adm[:N_DAYS], [last[d] for d in sorted(last)]


def main():
    days, statuses = admitted_days()
    S = {d: json.load(open(f'{DIR}/{d}/summary.json')) for d in days}
    for d in days:
        missing = [lab for lab, arm in ARMS.items() if arm not in S[d]['arms']]
        if missing:
            sys.exit(f'REFUSED: day {d} lacks arms {missing}')
        if S[d]['preds'] != 'preds_c2' or S[d]['fill_model'] != 'v2' or S[d]['timing'] != 'e2' or S[d]['scenario'] != 'tokyo20':
            sys.exit(f'REFUSED: day {d} was not run with the declared settings: {S[d]}')
    G = {lab: {d: S[d]['arms'][arm]['gross_usd'] for d in days} for lab, arm in ARMS.items()}
    out = dict(generated_at=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), days=days, n_days=len(days),
               status='OK' if len(days) == N_DAYS else 'INSUFFICIENT_DAYS', admission=statuses, predicates=predicates(G, days), arms={})
    for lab, arm in ARMS.items():
        v = [S[d]['arms'][arm] for d in days]; sh = sum(x['shares'] for x in v)
        out['arms'][lab] = dict(arm=arm, by_day=[round(x['gross_usd'], 2) for x in v], sum=round(sum(x['gross_usd'] for x in v), 2),
                                pos=sum(x['gross_usd'] > 0 for x in v), shares=round(sh), windows=[x['windows'] for x in v],
                                c_sh=round(100 * sum(x['gross_usd'] for x in v) / max(sh, 1e-9), 3),
                                mo1_c_sh=round(100 * sum(x['mo1_usd'] for x in v) / max(sh, 1e-9), 3),
                                inv_p95_max=max((x['abs_net_end_p95'] for x in v), default=None),
                                rebate_unverified=round(sum(x['rebate_usd_unverified'] for x in v), 2))
    ap = {d: f'{io.OUT}/step5/{d}/apply_c2.json' for d in days}
    out['model'] = {d: json.load(open(p)) if os.path.exists(p) else 'MISSING' for d, p in ap.items()}
    os.makedirs(f'{io.OUT}/c2', exist_ok=True)
    io.write_json(f'{io.OUT}/c2/report_c2.json', out)
    print(f"C2 report {out['generated_at']}: status {out['status']}, days {days}")
    for lab, a in out['arms'].items():
        print(f"  {lab:8s} {a['sum']:10.2f} {a['by_day']} pos {a['pos']} c/sh {a['c_sh']:6.3f} mo1 {a['mo1_c_sh']:6.3f} inv95 {a['inv_p95_max']}")
    for k, p in out['predicates'].items():
        print(f"  {k:24s} value {p['value']}  " + (f"sum {p['sum']:9.2f} {p['by_day']} pos {p['pos']} t90 {p['t90_mean_per_day']}" if 'sum' in p else str(p['parts'])))


if __name__ == '__main__':
    selftest() if '--selftest' in sys.argv else main()
