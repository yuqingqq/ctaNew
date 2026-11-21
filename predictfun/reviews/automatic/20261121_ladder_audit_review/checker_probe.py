"""Review-only corruptions of in-memory copies; no development artifact changed."""
from pathlib import Path
import copy, importlib.util, json, sys
PIN=Path('/tmp/pf-la-review-6b14fbf9')
sys.path.insert(0,str(PIN/'predictfun/research'))
spec=importlib.util.spec_from_file_location('ck',PIN/'predictfun/reviews/ladder_audit_20261121/audit_checks_v3.py')
CK=importlib.util.module_from_spec(spec);spec.loader.exec_module(CK)
CK.REPO=PIN;CK.R=PIN/'predictfun/reviews/ladder_audit_20261121';CK.REV=PIN/'predictfun/reviews/automatic/20261121_ladder_rest_review'
E=CK.load(); base=CK.run_checks(E); assert base['all_ok']
X=copy.deepcopy(E)
b=X['bins']; mask=(b.kind=='live')&(b.feature=='ctx_bid_reserved_shares')&(b.bin=='q5')
for i in b.index[mask]:
    z=b.loc[i]; q1=b[(b.kind==z.kind)&(b.feature==z.feature)&(b.day==z.day)&(b.side==z.side)&(b.bin=='q1')].iloc[0]
    b.loc[i,'mo1s_mean']=float(q1.mo1s_mean)+1.
bad=CK.run_checks(X)
Y=copy.deepcopy(E);Y['audit']['cross']['generation_tails']['mo1s_censored']=-999999
bad2=CK.run_checks(Y)
out=dict(clean=base['all_ok'], reversed_selected_bin_means_accepted=bad['all_ok'], changed_censor_count_accepted=bad2['all_ok'],
    altered_mean_rows=int(mask.sum()), note='Current saved means independently reconcile in evidence.json; these expose checker coverage, not corrupt real data.')
assert out['reversed_selected_bin_means_accepted'] and out['changed_censor_count_accepted']
Path(__file__).with_name('checker_probe.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
