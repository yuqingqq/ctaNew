"""Which v3 columns would be all-null inside one 64-original partition (a parquet null-typed part -> an object column after the concat)?"""
import pandas as pd
D = pd.read_parquet('/home/yuqing/ctaNew/data/predict_fun/research/r2_shadow_20261119/census_20261023_v3.parquet')
o = D[['instrument_id', 'original_episode_id', 'side']].drop_duplicates().sort_values(['instrument_id', 'original_episode_id'], kind='stable').reset_index(drop=True)
o['part'] = o.index // 64; D = D.merge(o, on=['instrument_id', 'original_episode_id', 'side'], how='left')
print('originals', len(o), 'parts', o.part.nunique(), 'rows per part', D.groupby('part').size().to_dict())
risk = []
for c in D.columns:
    if c == 'part':
        continue
    nn = D[c].notna().groupby(D.part).sum()
    if (nn == 0).any() and (nn > 0).any():
        risk.append((c, str(D[c].dtype), nn.to_dict()))
print('dtypes:', D.dtypes.astype(str).value_counts().to_dict())
print('columns all-null in some part but not all parts:', risk)
print('object columns:', {c: D[c].map(type).value_counts().to_dict() for c in D.columns if D[c].dtype == object})
