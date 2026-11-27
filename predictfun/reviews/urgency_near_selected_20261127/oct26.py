"""One previously consumed control date. Use the selected policy implementation directly."""
from collections import Counter
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'urgency_early_strong_20261127'
DAY = '20261026'
strong = runpy.run_path(str(SOURCE / 'run.py'))
R, U, ref, pilot, common = (strong[k] for k in ('R', 'U', 'ref', 'pilot', 'common'))


def main():
    strong['fixture']()
    fit_path = R.HERE / 'fit_v2.json'
    bind = U.S.read(ref['S']['L']['E'].REPLICATION / 'result.json')
    assert R.sha(fit_path) == bind['fit_sha'] and R.sha(R.CENSUS) == bind['census_sha']
    fit = U.S.read(fit_path)['cells']
    btc_dir = R.X.DATA / 'maker_trace_v2_20261123/btc'
    for line in (strong['PRIOR'] / 'btc.sha256').read_text().splitlines():
        digest, name = line.split()
        assert R.sha(btc_dir / Path(name).name) == digest
    btc = R.BTC.load_btc(btc_dir, DAY, Counter())
    controls = HERE.parent / 'urgency_cap150_eight_dates_20261126/markets_20261026.csv'
    assert R.sha(controls) == '7aca5b26a8e414a061a2fbaffdf98af4e3d370d71e9ed4681801906a97886c06'
    original = pd.read_csv(controls, float_precision='round_trip').set_index('mid')
    census = [r for r in U.S.read(R.CENSUS)['rows'] if r['day'] == DAY]
    reg = R.MM.registry(DAY).set_index('mid')
    eligible = {r['mid']: r['tape_id'] for r in census if r['status'] == 'complete'}
    assert reg.index.is_unique and len(reg) == len(census) == 288
    assert set(reg.index) == {r['mid'] for r in census}
    assert eligible == original.tape_id.to_dict() and len(original) == len(eligible)
    pay = pd.read_parquet(R.MM.S1 / DAY / 'registry.parquet').set_index('mid')
    tape = R.MM.KP.Tape()
    edges = R.MM.TR.T.P.frozen_edges()
    offset = tape.offset(DAY)
    records = []
    for _, group in reg.groupby('hour'):
        batch = sorted(set(group.index) & eligible.keys())
        if not batch:
            continue
        for mid, status, path, info in R.MM.worlds(DAY, batch, tape, offset, reg):
            assert status == 'complete'
            ws = 1000. * reg.at[mid, 'ws']
            ident = R.MM.ident_of(DAY, mid, ws, info, path)
            assert ident['tape_id'] == eligible[mid]
            flags, z = U.S.signal_data(btc, info, ws, edges)
            times = info['t_start'] + R.BE.STEP * np.arange(len(z))
            _, limit = strong['curves']((ws + 300000 - times) / 1000, z)
            args = path, info, ident, flags, fit, R.VM.V.payout_of(pay.loc[mid])
            for arm in ref['ARMS']:
                base = ref['simulate'](arm, *args)
                if arm == 'nominal':
                    for key, value in base.items():
                        assert abs(value - original.at[mid, key]) < 1e-6, (mid, key)
                result = pilot['simulate'](arm, args, limit)
                for policy, row in [('original', base), ('near', result)]:
                    records.append(dict(day=DAY, mid=mid, tape_id=eligible[mid], arm=arm, policy=policy, **row))
    frame = pd.DataFrame(records)
    assert len(frame) == 4 * len(eligible)
    common['validate'](frame.query("policy=='near'"), frame.query("policy=='original'"))
    assert frame.query("arm=='short'").post_request_shares.eq(0).all()
    frame.to_csv(HERE / 'markets.csv', index=False)
    rows = pd.read_csv(HERE / 'markets.csv', float_precision='round_trip')
    results = [dict(arm=arm, policy=pol, **common['metrics'](g)) for (arm, pol), g in rows.groupby(['arm','policy'])]
    pd.DataFrame(results).to_csv(HERE / 'comparison.csv', index=False)
    print(dict(Counter(r['status'] for r in census)), flush=True)
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
