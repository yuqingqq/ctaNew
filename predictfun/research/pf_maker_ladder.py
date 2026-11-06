"""Who is in the queue: maker concentration and laddering from matched trade legs (8 read days, live phase, real taker orders).
ladder = one maker signer filled at >= 2 different prices inside one taker order that swept >= 2 prices.
control: the same count with signers shuffled within the day (keeps each maker's volume share, breaks the price link)."""
import numpy as np, pandas as pd
from . import pf_io as io
R = f"{io.OUT}/step1"
D8 = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
rng = np.random.default_rng(5)


def ladder(L, sig):
    X = pd.DataFrame(dict(tk=L.tk.values, p=L.ptick.values, s=sig, q=L.shares.values))
    lv = X.groupby('tk').p.nunique(); multi = lv.index[lv >= 2]
    M = X[X.tk.isin(multi)]
    per = M.groupby(['tk', 's']).p.nunique()
    lad_tk = per[per >= 2].index.get_level_values(0).unique()
    lad_legs = M.set_index(['tk', 's']).index.isin(per[per >= 2].index)
    return len(multi), len(lad_tk) / max(len(multi), 1), M.q.values[lad_legs].sum() / max(M.q.sum(), 1e-9)


rows = []
for d in D8:
    L = pd.read_parquet(f'{R}/{d}/legs_timed2.parquet', columns=['mid', 'side', 'ptick', 'shares', 'signer', 'phase'])
    T = pd.read_parquet(f'{R}/{d}/legs_tk.parquet', columns=['tk', 'tk_status'])
    assert len(T) == len(L)
    L['tk'] = T.tk.values
    L = L[(L.phase == 'live') & (L.tk >= 0)]
    vol = L.groupby('signer').shares.sum().sort_values(ascending=False); tot = vol.sum()
    n_multi, lad, lad_sh = ladder(L, L.signer.values)
    _, lad_c, lad_sh_c = ladder(L, rng.permutation(L.signer.values))
    # makers per (taker order, price): how many different makers one level's fill was split across
    mk = L.groupby(['tk', 'ptick']).signer.nunique()
    rows.append(dict(day=d, legs=len(L), makers=len(vol), top1=vol.iloc[0] / tot, top5=vol.iloc[:5].sum() / tot, top10=vol.iloc[:10].sum() / tot,
                     multi_level_taker_orders=n_multi, ladder_share=lad, ladder_share_shuffled=lad_c, ladder_sh=lad_sh, ladder_sh_shuffled=lad_sh_c,
                     makers_per_level_mean=mk.mean(), levels_with_ge3_makers=(mk >= 3).mean()))
D = pd.DataFrame(rows).set_index('day')
pd.set_option('display.width', 250); print(D.round(3).to_string()); print('\nmean'); print(D.mean().round(3).to_string())
