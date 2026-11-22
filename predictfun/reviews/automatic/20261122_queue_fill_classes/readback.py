"""Post-run artifact readback and descriptive arithmetic; changes no class/choice."""
from pathlib import Path
import json,math
import pandas as pd
HERE=Path(__file__).resolve().parent;O=HERE/'outputs'
R=pd.read_csv(O/'generations.csv.gz',float_precision='round_trip',dtype={'day':str})
S=pd.read_csv(O/'classes.csv',float_precision='round_trip',dtype={'day':str})
X=pd.read_csv(O/'sequences.csv',float_precision='round_trip')
C=pd.read_csv(O/'reconciliation.csv',float_precision='round_trip',dtype={'day':str})
assert not R.duplicated(['day','m','arm','gid']).any()
assert len(R)==int(C.generations.sum()) and R.whole_fills.sum()==C.fills.sum()
for (a,d,m),rr in R.groupby(['arm','day','m']):
    cc=C[(C.arm==a)&(C.day==d)&(C.m==m)];assert len(cc)==1
    assert math.isclose(rr.whole_1s.sum(),cc.iloc[0].value_1s,abs_tol=1e-6)
    assert math.isclose(rr.whole_end.sum(),cc.iloc[0].end,abs_tol=1e-6)
for keys in (['arm'],['arm','day'],['arm','side']):
    sx=S[S.grouping=='|'.join(keys)]
    for kk,rr in R.groupby(keys+['category']):
        ss=sx
        for k,v in zip(keys+['category'],kk):
            ss=ss[ss[k].astype(str)==str(v)] if k!='side' else ss[pd.to_numeric(ss[k])==v]
        assert len(ss)==1,(keys,kk,len(ss));z=ss.iloc[0]
        for val,want in ((z.generations,len(rr)),(z.suffix_good,rr.good.sum()),(z.suffix_loss,rr.loss.sum()),
                         (z.suffix_net,rr.net.sum()),(z.suffix_shares,rr.suffix_shares.sum()),(z.prefix_loss,rr.prefix_loss.sum())):
            assert math.isclose(val,want,rel_tol=1e-9,abs_tol=1e-6)
for (a,c,s),rr in R.groupby(['arm','category','sequence']):
    xx=X[(X.arm==a)&(X.category==c)&(X.sequence==s)];assert len(xx)==1
    assert xx.iloc[0].generations==len(rr) and xx.iloc[0].separable==rr.separable_good_then_bad.sum()
result={}
for a,rr in R.groupby('arm'):
    first=rr[rr.anchor.notna()];sc=rr[rr.category=='PURE_BAD']
    mixed=rr[rr.category.str.startswith('MIXED_')]
    bg=mixed[mixed.sequence=='BAD_THEN_GOOD'];gb=mixed[mixed.sequence=='GOOD_THEN_BAD']
    result[a]=dict(generations=len(rr),markets=rr[['day','m']].drop_duplicates().shape[0],
        class_counts={k:int(v) for k,v in rr.category.value_counts().items()},
        complete_suffix_with_fill=int(((first.missing_suffix==0)&(first.suffix_fills>0)).sum()),
        pure_bad_loss=float(sc.loss.sum()),observed_suffix_gross_loss=float(first.loss.sum()),
        pure_bad_share_of_observed_suffix_gross_loss=float(sc.loss.sum()/first.loss.sum()),
        pure_bad_end_causes={k:int(v) for k,v in sc.known_end_cause.value_counts().items()},
        bad_then_good=int(len(bg)),bad_then_good_compensated=int((bg.category=='MIXED_COMPENSATED').sum()),
        good_then_bad=int(len(gb)),good_then_bad_separable=int(gb.separable_good_then_bad.sum()),
        whole_1s=float(rr.whole_1s.sum()),suffix_observed_1s=float(rr.net.sum()),
        prefix_observed_1s=float((rr.prefix_good-rr.prefix_loss).sum()))
(O/'interpretation.json').write_text(json.dumps(dict(kind='post_run_descriptive_readback_no_reclassification',checks_pass=True,arms=result),indent=2)+'\n')
print(json.dumps(result,indent=2));print('Exported rows, all class summaries, sequences and accepted totals: READBACK PASS.')
