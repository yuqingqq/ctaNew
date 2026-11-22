"""Review 838f4be2 finding 1 (descriptive, on the frozen outputs): Oct 26 REVIEW components; per-matched-share profit as the mean of arm ratios
and as the ratio of sums; new-lot-only ratios."""
import json, pandas as pd
D = '/home/yuqing/ctaNew/data/predict_fun/research/ladder_turnover_attribution_v1_20261122'
P = pd.read_csv(f'{D}/pairs.csv', dtype={'day': str}); A = pd.read_csv(f'{D}/arms.csv', dtype={'day': str}); L = pd.read_csv(f'{D}/lots.csv')
r26 = P[(P.kind == 'REVIEW') & (P.day == '20261026')][['Y', 'dMatched', 'dResidual']].sum().round(2).to_dict()
out = dict(review_oct26=r26)
for (k, a), g in A.groupby(['kind', 'arm']):
    nl = L[(L.key.isin(g.key)) & (L.arm == a) & (~L.carried)]
    out[f'{k}/{a}'] = dict(mean_arm_ratio=round(float(g.profit_per_share.mean()), 3), ratio_of_sums=round(float(g.matched.sum() / g.matched_q.sum()), 3),
                           new_lots_ratio_of_sums=round(float(nl.profit.sum() / nl.q.sum()), 3))
print(json.dumps(out, indent=1))
