"""Check a lockstep trace run of pf_step4_ladder_trace.py (pm_ladder_v1): the candidate's fills / win / requests parquet must be
byte-identical to the round-i run (falsifier: the round-i pm arm must differ), and every traced window's requests / fills / end record must
reconcile with the run's outputs. Run from data/predict_fun/research/step4.
Usage: python3 <this file> <round-i tag lJi|lXi> <trace run tag> <trace dir> <day>
"""
import json, filecmp, sys, pandas as pd, numpy as np
ref, run, tdir, day = sys.argv[1:5]
A=f'ladder_tokyoreal_custom_e2_preds_t60k_{ref}/{day}'; B=f'ladder_tokyoreal_custom_e2_preds_t60k_{run}/{day}'
stem='MICRO_G4_G2_G6_G5_G3_G1_PM90_GS_LD2_XNA_XMIN1_XV_S60__mid_szAM-5-25-95.parquet'; other='MICRO_G4_G2_G6_G5_G3_G1_PM90_GS_XNA_XMIN1_XV_S60__mid_szAM-5-25-95.parquet'
name='MICRO+G4+G2+G6+G5+G3+G1+PM90+GS+LD2+XNA+XMIN1+XV+S60/mid@AM:5:25:95'
bad=0
for d in ('fills','win','requests'):
    s=filecmp.cmp(f'{A}/{d}/{stem}',f'{B}/{d}/{stem}',shallow=False); fz=filecmp.cmp(f'{A}/{d}/{other}',f'{B}/{d}/{stem}',shallow=False)
    print(d,'identical',s,'falsifier identical',fz); bad+=int(not s)+int(fz)
eq=json.load(open(f'{A}/summary.json'))['arms'][name]==json.load(open(f'{B}/summary.json'))['arms'][name]; print('summary arm equal',eq); bad+=int(not eq)
fl=pd.read_parquet(f'{B}/fills/{stem}'); W=pd.read_parquet(f'{B}/win/{stem}'); Q=pd.read_parquet(f'{B}/requests/{stem}')
reg=pd.read_parquet(f'../step1/{day}/registry.parquet',columns=['mid','ws']).set_index('mid')
FL=json.load(open(f'{tdir}/flags.json'))
for m in json.load(open(f'{tdir}/mids.json')):
    recs=[json.loads(l) for l in open(f'{tdir}/{m}.jsonl')]
    ev=[r for r in recs if r['rec']=='event']; rq=[r for r in recs if r['rec']=='requests'][0]['rows']; end=[r for r in recs if r['rec']=='end'][0]
    q=Q[Q.mid==m]; f=fl[fl.mid==m]; w=W[W.mid==m].iloc[0]; ws=reg.loc[m,'ws']*1000
    evreq=[x for r in ev for x in r.get('new_requests',[])]; evf=[x for r in ev for x in r.get('new_fills',[])]
    ok_q = len(rq)==len(q)==len(evreq) and all(abs(a['t_ms']-b)<1e-6 and a['side']==s and a['kind']==k and a['lane']==l for a,(b,s,k,l) in zip(rq, q[['t_ms','side','kind','lane']].itertuples(index=False))) and rq==evreq
    ok_f = len(evf)==len(f) and all(abs((a['t']-ws)/1000-b)<1e-9 and a['side']==s and a['pt']==p and abs(a['q']-qq)<1e-9 and a['lane']==l for a,(b,s,p,qq,l) in zip(evf, f[['t','side','pt','q','lane']].itertuples(index=False)))
    ok_e = abs(end['pnl']-w.pnl)<1e-9 and end['n_fills']==w.n_fills and len(ev)==FL[str(m)]['events']
    nfr=sum(r['rec']=='frame' for r in recs); kinds=pd.Series([r['kind'] for r in ev]).value_counts().to_dict()
    print(m,'requests',ok_q,len(rq),'fills',ok_f,len(evf),'end/events',ok_e,'frames',nfr,'events',len(ev),kinds)
    bad+=int(not(ok_q and ok_f and ok_e))
print('CHECKS', 'PASS' if bad==0 else f'FAIL {bad}')
sys.exit(0 if bad == 0 else 1)
