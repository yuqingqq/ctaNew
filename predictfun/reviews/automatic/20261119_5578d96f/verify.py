"""Bounded result review: pinned sources, saved artifacts, independent economics.
No tape, fit, new policy, or untouched-day access. Run in the Codex light slot.
"""
from pathlib import Path
from datetime import datetime, timezone
import contextlib
import hashlib
import io
import json
import subprocess
import sys

ROOT = Path('/home/yuqing/ctaNew')
HERE = Path(__file__).resolve().parent
TARGET = '5578d96f684d472b04ed25325295a9a2a6989b2e'
PUB = ROOT/'predictfun/reviews/keep_pull_variants_20261119'
DAY = '20261023'


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def equal(a, b):
    require(abs(a-b) <= 1e-6, 'independent economic reconciliation')


def direct(E, Fs, policy):
    """Direct fill mask and explicit per-step risk sets; no production score/null call."""
    import numpy as np
    import pandas as pd
    E = E.reset_index(drop=True)
    sign = np.where(E.side.to_numpy() == 0, 1., -1.)
    fair = E[[f'f{j:02d}' for j in range(99)]].to_numpy()
    theta = float(policy[1:])
    signal = sign[:, None]*(fair[:, 2:]-E.p.to_numpy()[:, None]) < theta if policy[0] == 'A' else (
        sign[:, None]*(fair[:, 2:]-fair[:, :1]) <= -theta)
    steps = np.where(signal.any(axis=1), signal.argmax(axis=1)+2, 99)
    limit = np.fmin(E.g_struct.to_numpy(), E.t_full_current.to_numpy()+100.)
    limit = np.where(np.isnan(limit), np.inf, limit)
    active = (steps <= 98) & (50.*steps < limit)
    cell = E.groupby(['day', 'm', 'side', 'phase'], sort=True).ngroup().to_numpy()
    ix = pd.Series(np.arange(len(E)), index=E.oid)
    ri = ix.loc[Fs.oid].to_numpy()
    ft = Fs.t.to_numpy(); value = Fs.q.to_numpy()*sign[ri]*(E.mark.to_numpy()[ri]-E.p.to_numpy()[ri])
    removed = active[ri] & (ft >= 50.*steps[ri]+37.)
    imp = float(-value[removed].sum()); keep = float(value.sum())
    pa = float(-value[(100. < limit[ri]) & (ft >= 137.)].sum())
    # Each column j is the local suffix value affected by a pull at j.
    suffix = np.zeros((len(E), 99))
    last = np.clip(np.searchsorted(50.*np.arange(99)+37., ft, side='right')-1, 0, 98)
    np.add.at(suffix, (ri, last), value)
    suffix = np.cumsum(suffix[:, ::-1], axis=1)[:, ::-1]
    equal(imp, float(-suffix[np.flatnonzero(active), steps[active]].sum()))
    null = 0.; nc = int(cell.max())+1
    for j in np.unique(steps[active]):
        live = 50.*j < limit
        count = np.bincount(cell[active & (steps == j)], minlength=nc)
        nlive = np.bincount(cell[live], minlength=nc)
        total = np.bincount(cell[live], weights=-suffix[live, j], minlength=nc)
        require(bool(np.all(count <= nlive)), 'executed pulls outside live risk set')
        null += float(np.sum(count*total/np.maximum(nlive, 1)))
    return dict(KEEP=keep, score=imp, null=null, E=imp-null, PULLALL=pa,
                V=keep+imp, V_pullall=keep+pa, pulls=int(active.sum()), eligible=len(E))


def controls():
    import numpy as np
    import pandas as pd
    # Two opportunities, the sole selected pull avoids the bad fill; the matched
    # random pull has expectation zero. Unavoidable early fill stays in both.
    E = pd.DataFrame(dict(day=['x']*2, m=[1]*2, side=[0]*2, phase=[1]*2,
                         oid=[0, 1], p=[50., 50.], mark=[49., 51.],
                         g_struct=[np.nan]*2, t_full_current=[np.nan]*2))
    E = pd.concat([E, pd.DataFrame({f'f{j:02d}': [49., 51.] for j in range(99)})], axis=1)
    f = pd.DataFrame(dict(oid=[0, 0, 1], t=[100., 137., 500.], q=[2., 10., 10.]))
    r = direct(E, f, 'A+0')
    for k, v in dict(KEEP=-2., score=10., E=10., null=0., PULLALL=0., V=8., V_pullall=-2.).items():
        equal(r[k], v)
    try:
        equal(r['V_pullall'], r['PULLALL'])
    except ValueError:
        pass
    else:
        raise ValueError('delta-as-total falsifier did not fire')
    return dict(independent_positive_control=True, delta_as_total_refused=True)


def main():
    import pandas as pd
    out = dict(target_commit=TARGET, asof=datetime.now(timezone.utc).isoformat(), controls=controls())
    hashes = {}
    for line in (PUB/'sources.sha256').read_text().splitlines():
        h, name = line.split()
        p = ROOT/'predictfun/research'/name
        committed = subprocess.check_output(['git', 'show', TARGET+':predictfun/research/'+name], cwd=ROOT)
        require(hashlib.sha256(committed).hexdigest() == h == sha(p), 'source pin: '+name)
        hashes[name] = h
    out['source_hashes'] = hashes
    sys.path.insert(0, str(ROOT/'predictfun/research'))
    import pf_keep_pull_variants as V
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        V.selftest()
    (HERE/'selftest.txt').write_text(capture.getvalue())
    out['variants_selftest'] = 'passed'
    saved = json.loads((PUB/'variants_20261023.json').read_text())
    require(saved['days'] == [DAY], 'result days')
    for name, h in saved['outputs'].items():
        require(sha(PUB/name) == h, 'published output hash: '+name)
    for stem in ('scenario_report', 'keep_pull_report'):
        require(sha(saved['inputs'][stem]) == saved['inputs'][stem+'_sha256'], 'input report hash')
    require(sha(V.OUT/'variants_20261023.json') == sha(PUB/'variants_20261023.json'), 'published/source result identity')
    O, Fl, S, SF, Op, rec = V.load([DAY], V.SC.OUT, V.KP.OUT)
    require(rec == saved['inputs']['days'], 'input receipt identity')
    bad = V.controls(saved['values'], [DAY], O, Fl, saved['inputs']['scenario_report'], saved['inputs']['keep_pull_report'])
    require(not bad, str(bad))
    recalculated = V.readings(saved['values']); recalculated['CONTROLS'] = 'YES'
    require(recalculated == saved['readings'], 'reading reproduction')
    text, table, comparison = V.tables(saved['values'], recalculated, [DAY], Op, rec)
    pd.testing.assert_frame_equal(table, pd.read_csv(PUB/'variants_20261023_policies.csv'), atol=1e-7, rtol=1e-12)
    pd.testing.assert_frame_equal(comparison, pd.read_csv(PUB/'variants_20261023_compare.csv'), atol=1e-7, rtol=1e-12)
    require('\n'.join(text)+'\n' == (PUB/'variants_20261023.txt').read_text(), 'text reproduction')
    independent = {}
    for clock in (0, 50, 250):
        sc = dict(V.SC.B0, tl=f'M{clock}'); nm = V.SC.name(sc)
        Os, Fs = V.scen_inputs(sc, Op, Fl, S, SF)
        E = O.merge(Os[['oid', 'refused', 't_full']].rename(columns={'t_full': 't_full_current'}), on='oid', validate='one_to_one')
        E = E[~E.refused]
        independent[nm] = {}
        for policy in V.KP.POLICIES:
            r = direct(E, Fs, policy); original = saved['values'][policy][nm]
            for k in ('KEEP', 'score', 'null', 'E', 'PULLALL', 'pulls', 'eligible'):
                equal(r[k], original[k])
            independent[nm][policy] = r
    out.update(result_asof=saved['asof'], input_asof=rec[DAY]['scenario_asof'], consumed_day=DAY,
               original_orders=len(O), path_orders=len(Op), original_statuses=O.status.value_counts().to_dict(),
               saved_readings_and_all_tables_reproduced=True, policy_scenario_rows=len(table),
               independent=independent, summary=saved['readings']['SUMMARY'],
               candidate_counts={p: dict(positive_E=int((g.E > 0).sum()), positive_V=int((g.V > 0).sum()),
                                        above_pullall=int((g.score > g.PULLALL).sum()), scenarios=len(g)) for p, g in table.groupby('policy')},
               never_post_value=0., never_post_role='mechanical no-exposure control only, no quote provision',
               raw_tape_read=False, training=False, untouched_validation=False,
               checks_finished_asof=datetime.now(timezone.utc).isoformat())
    with (HERE/'receipt.json').open('x') as f:
        json.dump(out, f, indent=2); f.write('\n')
    print(json.dumps(dict(passed=True, result_rows=len(table), independent_cells=18, receipt=str(HERE/'receipt.json'))))


if __name__ == '__main__':
    main()
