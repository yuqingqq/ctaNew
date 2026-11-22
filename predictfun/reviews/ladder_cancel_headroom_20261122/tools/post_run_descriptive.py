"""POST-RUN DESCRIPTIVE ARITHMETIC (undeclared; labeled as such in the receipt): where the oracle cuts relative to the first callback."""
import json, pandas as pd, numpy as np
D = '/home/yuqing/ctaNew/data/predict_fun/research/ladder_cancel_headroom_v1_20261122'
G = pd.read_csv(f'{D}/generations_core.csv', dtype={'day': str}); out = {}
for arm, g in G.groupby('arm'):
    c = g[g.b_cut.notna()]
    out[arm] = dict(oracle_cuts=int(len(c)), at_first_callback=int((c.b_cut == c.a_cut).sum()), b_age_p25=float(c.b_age.quantile(.25)), b_age_p50=float(c.b_age.median()),
                    b_age_p75=float(c.b_age.quantile(.75)), b_age_p90=float(c.b_age.quantile(.9)), L_removed_at_first_callback=round(float(c[c.b_cut == c.a_cut].b_suf_L.sum()), 2),
                    L_removed_total=round(float(c.b_suf_L.sum()), 2), cut_end_causes=c.end_cause.value_counts().to_dict(), cut_why=c.why.fillna('').value_counts().to_dict())
print(json.dumps(out, indent=1))
