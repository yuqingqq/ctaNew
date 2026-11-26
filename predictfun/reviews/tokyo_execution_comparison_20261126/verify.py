"""Independent saved-artifact checks, including the original booked native-token fills."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
DATA=HERE.parents[2]/'data/predict_fun/research'
ARCH=Path('/home/yuqing/pmmm-data/review-exports/live-samewindow-20261108')
TOKYO=Path('/home/yuqing/pmmm-data/tokyo-live-data-20261109/tokyo-live-20261109')


def reconcile(actual,expected):
    if abs(actual-expected)>1e-5:
        raise ValueError(f'accounting disagreement {actual} versus {expected}')


def run():
    try: reconcile(1,0)
    except ValueError: pass
    else: raise AssertionError('corruption accepted')
    markets=pd.read_csv(HERE/'fixed_markets.csv')
    cap=pd.read_csv(HERE/'cap_markets.csv')
    order_path=DATA/HERE.name/'orders.parquet'
    orders=pd.read_parquet(order_path)
    wallet=pd.read_parquet(DATA/'fill_clock_20261109/clock_cases.parquet')
    scope=pd.DataFrame(json.loads((ARCH/'scope.json').read_text())['windows']).set_index('market').run
    actual=markets[markets.world=='wallet_notice'].set_index('market')
    native=[]
    for k in ('v6','basis'):
        source=json.loads((TOKYO/k/'stageA/ledger.json').read_text())
        for f in source['fills']:
            if f['booked'] and not f['reversed'] and scope.get(f['market'])==k:
                native.append(f)
    legs=pd.DataFrame(native)
    join=legs.merge(wallet,left_on=['trade','order','cid'],right_on=['trade_id','order_hash','cid'],
                    suffixes=('_native','_clock'),validate='one_to_one')
    assert len(join)==len(legs)==len(wallet)==99
    assert (join.market==join.mid).all() and np.allclose(join.shares_native,join.shares_clock)
    assert np.allclose(np.where(join.outcome=='up',join.price,1-join.price)*100,join.ptick)
    assert (join.payout.notna()).all()
    cash=((join.payout-join.price)*join.shares_native).groupby(join.market).sum().reindex(actual.index,fill_value=0)
    assert np.allclose(cash,actual.settle,atol=1e-5)
    for world,g in orders.groupby('world'):
        assert g.cid.is_unique and len(g)==30221
        m=markets[markets.world==world].set_index('market')
        for k1,k2 in [('pred','shares'),('pred_close','closed')]:
            assert np.allclose(g.groupby('market')[k1].sum().reindex(m.index,fill_value=0),m[k2])
    assert np.allclose(markets.turnover+markets.residual,markets.settle)
    assert np.allclose(cap.turnover+cap.residual,cap.settle)
    assert np.allclose(cap.shares,2*cap.closed+cap.abs_end)
    assert cap[cap.cap==150].peak_position.max()<=150.00001
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [order_path,HERE/'fixed_markets.csv',HERE/'fixed.json',HERE/'cap_markets.csv']}
    (HERE/'verification.json').write_text(json.dumps(dict(markets=len(actual),wallet_legs=len(join),by_run=cash.groupby(scope).sum().to_dict(),sha256=hashes),indent=2)+'\n')


if __name__=='__main__':
    run()
