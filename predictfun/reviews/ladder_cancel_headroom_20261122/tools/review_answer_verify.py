"""Review 32d102bd finding 3: the chosen-cut pure class versus the first-callback class (descriptive, on the frozen outputs)."""
import json, pandas as pd
D = '/home/yuqing/ctaNew/data/predict_fun/research/ladder_cancel_headroom_v1_20261122'
G = pd.read_csv(f'{D}/generations_core.csv', dtype={'day': str}); out = {}
for arm, g in G.groupby('arm'):
    p = g[g.b_cls == 'pure_observed_loss']
    out[arm] = dict(b_pure=int(len(p)), a_class_of_b_pure=p.a_cls.value_counts().to_dict(), b_pure_cut_at_first=int((p.b_cut == p.a_cut).sum()),
                    btg_flagged=int(g.seq_bad_then_good.sum()), btg_feasible=int(g.seq_btg_feasible.sum()), btg_L_feasible=round(float(g[g.seq_btg_feasible].seq_btg_L.sum()), 2))
print(json.dumps(out, indent=1))
