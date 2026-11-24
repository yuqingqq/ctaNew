"""Hand cases, including positive conditional deductions and false-action refusals."""
import json,numpy as np,pandas as pd
import build as B
checks={}
def t(name,actual,expected):assert actual==expected,(name,actual,expected);checks[name]=True
own=[dict(lo=100,hi=120,q=20),dict(lo=300,hi=320,q=30)]
t('before',B.phase(40,80,own),'pre_first');t('between',B.phase(140,260,own),'between_fills');t('tie',B.phase(120,260,own),'overlap_or_tie');t('after',B.phase(330,350,own),'after_last');t('unknown',B.phase(-1,350,own),'unbounded')
a=np.array([0,100,200,300]);p=np.array([0.,20.,10.,0.]);c=np.array([False,True,True,True])
t('insufficient pending',B.pending(a,p,c,105,180,50),('visible_insufficient',20.));t('pending may cover',B.pending(a,p,c,105,180,15),('could_cover',20.))
t('no future bracketing point',B.pending(a,p,c,250,301,50),('unknown',None));t('gap cannot prove absence',B.pending(a,p,np.array([False,True,False,True]),105,180,50),('unknown',None));t('nan not zero',B.pending(a,np.array([0.,np.nan,10.,0.]),c,105,180,50),('unknown',None))
b=dict(post_A_lo=0,old_end_lo=320,end_hi_conditional=900,post_hi_unchanged=120)
ops=[dict(witness_status='conditional_available_priority_violation',phase='pre_first',relation='through',lo_ms=30,hi_ms=60,sid='a'),dict(witness_status='conditional_available_priority_violation',phase='after_last',relation='same_price',lo_ms=400,hi_ms=420,sid='b'),dict(witness_status='conditional_available_priority_violation',phase='between_fills',relation='through',lo_ms=140,hi_ms=260,sid='c')]
r=B.refine(b,ops,'conditional_withdrawal',True,1000);t('positive post lower',(r['post_lo_ms'],r['post_witness']),(30,'a'));t('positive end upper',(r['end_hi_ms'],r['end_witness']),(420,'b'));t('interruption not terminal',r['interruption_records'],1)
t('exhaustion suppresses terminal pull',B.refine(b,ops,'exhausted',True,1000)['end_witness'],'');t('expiry remains alternative',B.refine(b,ops,'conditional_withdrawal',True,400)['end_witness'],'');t('refused grade cannot improve',B.refine(b,ops,'unresolved',False,1000)['post_lo_ms'],0)
t('ordered successor',B.successor_order(100,200,250,300,True),'sequential_after_under_bounds');t('early successor',B.successor_order(100,200,0,90,True),'posted_before_prior_end');t('overlap not sequential',B.successor_order(100,200,150,250,True),'ordering_ambiguous');t('touching endpoints ambiguous',B.successor_order(100,200,200,250,True),'ordering_ambiguous');t('refused successor',B.successor_order(100,200,0,90,False),'refused_or_missing_window')
t('BTC mixed cannot become trigger',B.categories(np.array([-2.,2.]),lambda x:'against' if x<0 else 'with'),'against|with');t('unknown category preserved',B.categories(np.array([np.nan,2.]),str),'unknown')
print(json.dumps(dict(passed=len(checks),checks=checks),indent=2))
