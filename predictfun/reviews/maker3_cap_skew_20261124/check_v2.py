import json
import pandas as pd
import check as V
import observe_v2 as A
import amendment
amendment.bound();V.A=A;V.run()
# Reconcile the new inventory map against previously frozen independent context output.
s=pd.read_csv(A.E.OUT/'sizes.csv',dtype={'day':str},float_precision='round_trip')
old=pd.read_csv(A.H.parent/'maker_action_population_20261124/outputs/contexts.csv',float_precision='round_trip')
old=old[(old['rank']==3)&old.stage.eq('post')]
j=s.merge(old[['order_hash','lag_ms','inventory_side','sampled_states','coverage']],left_on=['hash','lag_ms'],right_on=['order_hash','lag_ms'],validate='one_to_one')
assert len(j)==len(s)
for r in j.itertuples():
 assert r.inventory==r.inventory_side.replace('offsetting','reducing'),(r.hash,r.inventory,r.inventory_side)
 assert r.n==r.sampled_states
z=json.loads((A.H/'checks.json').read_text());z['independent_prior_context_rows']=len(j);A.T.dump(A.H/'checks_v2.json',z)
print('PASS independent prior inventory/window context',len(j),'rows')
