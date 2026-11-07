"""CAP10 + RO60-240 on pm_ladder_v1 (user 2026-11-07: "Go test", on: limit 10 shares in the first minute plus the 60 -> 240 s sell-down
that keeps the pull line). Engine pf_step4_ladder_inv.py (generated, pf_ladder_inv_build.py; unchanged since 0a8b69bf), runner
run_inv2.sh (tags i2J / i2X), the 9 consumed round-i days, J and X. IN-SAMPLE, exploratory: CAP10 and RO60-240 were picked from the
10 arms of report_inv_v1 / report_spread_capture_v1 on these days; nothing here validates.
Not run: the sell-down to 290 s with the line. preds_t60k has no guard value at t >= 240 s (0 rows on 10-24 and 11-02; max 239.95 s),
so GS withholds every side after 240 s and RO60+S290 cannot quote after 240 s with the line.
Arms: v1, CAP10, RO60-240 (controls: byte-identical to round i / ivJ / ivX), CAP10+RO60-240 (new).
check: v1 fills / win / requests byte-identical to lJi / lXi and its summary arm equal (18 runs); CAP10 and RO60-240 byte-identical to
       ivJ / ivX (18 runs each); the new arm differs from both CAP10 and RO60-240 on >= 1 day per timing (falsifier: it must) and its
       CAPN+LD_CAPN and RO+LD_RO counters are > 0; arms and env as declared.
report: per arm and timing: settle, +1 s $ (engine mo1, like-for-like), mark@stop (RO arms at 240 s: not like-for-like), day sd, worst
       day, max drawdown, $150 cap refusal / halt within 1 day, carry sd, |net| at stop, fills, shares; the new arm vs v1, vs CAP10 and
       vs RO60-240: settle and +1 s $ / day, day t (8 df) and exact 512 sign flips (raw p). -> step4/report_inv2.json
Usage: python3 -m predictfun.research.pf_inv2 arms | check | report
"""
import filecmp, json, sys
import numpy as np, pandas as pd
from . import pf_io as io
from .pf_inv_report import arm, name, stem, run_dir, DAYS, S4, T8
from .pf_cap_check import first_refusal, first_halt, mdd, PER_DAY

ARMS = {'v1': arm(''), 'CAP10': arm('CAP10'), 'RO60-240': arm('', 'RO60+S240'), 'CAP10+RO60-240': arm('CAP10', 'RO60+S240')}
NEW = 'CAP10+RO60-240'
KINDS = ('fills', 'win', 'requests')


def same(tag_a, tag_b, spec, d):
    return all(filecmp.cmp(f'{run_dir(tag_a, d)}/{k}/{stem(spec)}.parquet', f'{run_dir(tag_b, d)}/{k}/{stem(spec)}.parquet', shallow=False) for k in KINDS)


def check():
    errs = []; rep = dict(parity={}, differs={}, engaged={})
    for tm in ('J', 'X'):
        tag = 'i2' + tm; cnt = {}
        for d in DAYS:
            s = json.load(open(f'{run_dir(tag, d)}/summary.json')); env = s.get('env', {})
            if sorted(s['arms']) != sorted(name(x) for x in ARMS.values()):
                errs.append(f'{tm} {d} arms')
            if env.get('PF_PREDS') != 'preds_t60k' or env.get('PF_R60_FILE') != f'{S4}/inv_r60.json':
                errs.append(f'{tm} {d} env')
            if (tm == 'J') != (env.get('PF_TXJIT') == 'frame') or (tm == 'X') != (env.get('PF_TXSHIFT') == '-50' and env.get('PF_QACT') == 'max'):
                errs.append(f'{tm} {d} timing env')
            ref = json.load(open(f'{run_dir("l" + tm + "i", d)}/summary.json'))['arms'][name(ARMS['v1'])]
            ok = {'v1': same(tag, 'l' + tm + 'i', ARMS['v1'], d) and ref == s['arms'][name(ARMS['v1'])],
                  'CAP10': same(tag, 'iv' + tm, ARMS['CAP10'], d), 'RO60-240': same(tag, 'iv' + tm, ARMS['RO60-240'], d)}
            for k, v in ok.items():
                rep['parity'][f'{tm} {d} {k}'] = v
                if not v:
                    errs.append(f'parity {tm} {d} {k}')
            for k in ('CAP10', 'RO60-240'):
                rep['differs'].setdefault(f'{tm} vs {k}', 0)
                ne = not filecmp.cmp(f'{run_dir(tag, d)}/fills/{stem(ARMS[NEW])}.parquet', f'{run_dir(tag, d)}/fills/{stem(ARMS[k])}.parquet', shallow=False)
                rep['differs'][f'{tm} vs {k}'] += int(ne)
            for k, c in s['arms'][name(ARMS[NEW])]['withheld_decisions'].items():
                cnt[k] = cnt.get(k, 0) + c
        for grp in (('CAPN', 'LD_CAPN'), ('RO', 'LD_RO')):
            n = sum(cnt.get(k, 0) for k in grp); rep['engaged'][f'{tm} {"+".join(grp)}'] = n
            if n == 0:
                errs.append(f'{tm} {NEW} never engaged {grp}')
        for k in ('CAP10', 'RO60-240'):
            if rep['differs'][f'{tm} vs {k}'] == 0:
                errs.append(f'falsifier: {NEW} equals {k} on every day ({tm})')
    rep['errors'] = errs; rep['PASS'] = not errs
    io.write_json(f'{S4}/inv2_check.json', rep)
    print(json.dumps({k: v for k, v in rep.items() if k != 'parity'}, indent=1)); print('parity', sum(rep['parity'].values()), '/', len(rep['parity']))
    print('PASS' if not errs else f'FAIL {len(errs)}')
    return 0 if not errs else 1


def ci(x):
    x = np.asarray(x, float); m = x.mean(); h = T8 * x.std(ddof=1) / np.sqrt(len(x))
    return dict(mean=round(float(m), 2), lo=round(float(m - h), 2), hi=round(float(m + h), 2), days_pos=int((x > 0).sum()),
                excludes_0=bool(m - h > 0 or m + h < 0))


def report():
    from .pf_cap10_attribution import signflip, controls
    if not json.load(open(f'{S4}/inv2_check.json')).get('PASS'):
        sys.exit('REFUSED: step4/inv2_check.json is not PASS')
    controls()
    out = dict(as_of=pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%MZ'), days=DAYS, arms=ARMS, timings={})
    for tm in ('J', 'X'):
        R = {}
        for lab, spec in ARMS.items():
            per = {d: json.load(open(f'{run_dir("i2" + tm, d)}/summary.json'))['arms'][name(spec)] for d in DAYS}
            W = pd.concat([pd.read_parquet(f'{run_dir("i2" + tm, d)}/win/{stem(spec)}.parquet') for d in DAYS]).sort_values('ws').reset_index(drop=True)
            p, mw = W.pnl.values, W.min_worst.values; n = len(W)
            st = np.array([per[d]['gross_usd'] for d in DAYS]); mk = np.array([per[d]['mtm_stop_usd'] for d in DAYS]); m1 = np.array([per[d]['mo1_usd'] for d in DAYS])
            ref = [first_refusal(p[i:i + PER_DAY], mw[i:i + PER_DAY]) for i in range(n - PER_DAY + 1)]
            hlt = [first_halt(p[i:i + PER_DAY]) for i in range(n - PER_DAY + 1)]
            R[lab] = dict(settle=round(float(st.sum()), 1), settle_days_pos=int((st > 0).sum()), plus1=round(float(m1.sum()), 1), mark=round(float(mk.sum()), 1),
                          settle_day_sd=round(float(st.std(ddof=1)), 1), settle_worst_day=round(float(st.min()), 1), max_drawdown=round(mdd(p), 1),
                          cap150_refusal_1d_pct=round(100.0 * np.mean([x is not None for x in ref]), 1), cap150_halt_1d_pct=round(100.0 * np.mean([x is not None for x in hlt]), 1),
                          carry_sd_window=round(float(W.carry_stop.std()), 2), abs_net_stop_mean=round(float(W.net_end.abs().mean()), 2),
                          abs_net_stop_p90=round(float(W.net_end.abs().quantile(0.9)), 1), fills=int(sum(per[d]['fills'] for d in DAYS)),
                          shares=round(float(sum(per[d]['shares'] for d in DAYS)), 0),
                          per_day=dict(settle=dict(zip(DAYS, st.round(2).tolist())), plus1=dict(zip(DAYS, m1.round(2).tolist()))), _st=st, _m1=m1)
        fam = [(b, met) for b in ('v1', 'CAP10', 'RO60-240') for met in ('_st', '_m1')]
        M = np.column_stack([R[NEW][met] - R[b][met] for b, met in fam]); tt, praw, _ = signflip(M)
        R[NEW]['vs'] = {f'{b} {"settle" if met == "_st" else "plus1"}': dict(ci(R[NEW][met] - R[b][met]), t=round(float(tt[i]), 2), p_signflip=round(float(praw[i]), 4))
                        for i, (b, met) in enumerate(fam)}
        for r in R.values():
            del r['_st'], r['_m1']
        out['timings'][tm] = R
    io.write_json(f'{S4}/report_inv2.json', out)
    for tm in ('J', 'X'):
        print(f'== {tm}  ($ over 9 days; day sd; worst day; max dd; cap150 refuse / halt % within 1 day; |net| at stop)')
        for lab, r in out['timings'][tm].items():
            print(f"{lab:15s} settle {r['settle']:7.1f} ({r['settle_days_pos']}/9) +1s {r['plus1']:6.1f} mark {r['mark']:6.1f} daysd {r['settle_day_sd']:6.1f} "
                  f"worst {r['settle_worst_day']:7.1f} dd {r['max_drawdown']:7.1f} cap {r['cap150_refusal_1d_pct']:5.1f}/{r['cap150_halt_1d_pct']:5.1f} net {r['abs_net_stop_mean']:5.2f} "
                  f"fills {r['fills']:5d} sh {r['shares']:6.0f}")
        for k, v in out['timings'][tm][NEW]['vs'].items():
            print(f"   {NEW} - {k:16s} {v['mean']:+7.2f} $/day ({v['lo']:+.2f}..{v['hi']:+.2f}) days>0 {v['days_pos']}/9 p {v['p_signflip']:.4f}")
    return 0


if __name__ == '__main__':
    if sys.argv[1] == 'arms':
        print(';'.join(ARMS.values())); sys.exit(0)
    sys.exit(check() if sys.argv[1] == 'check' else report())
