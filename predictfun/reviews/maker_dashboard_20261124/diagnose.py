import extract as E,collections
v,ranks=E.top_fills();r=E.load_records();a=E.read(E.MR/'v2/matched_gap_r1/per_market.csv')
for day,m,part,_ in E.CASES:
 rec=r[m][E.ARMS[0]];lo=rec['ident']['start_abs_ms'];hi=rec['ident']['end_abs_ms'];mark=rec['mark']/100
 for rank in [1,2,3]:
  fs=[f for f in v if f['mid']==m and ranks[f['addr'].lower()]==rank and lo<=f['settle_ms']<=hi and f['status']=='ok']
  allv=sum(f['s']*f['q']*(mark-f['p'])-f['fee_fill'] for f in fs);non=sum(f['s']*f['q']*(mark-f['p'])-f['fee_fill'] for f in fs if not f['self_trade']);prior=next(x for x in a if int(x['m'])==m and x['party']==f'rank{rank}' and x['view']=='zero')
  print(m,rank,'mark',mark,'legs',len(fs),'all',allv,'nonself',non,'expected',prior['total'],'status',collections.Counter(f['status'] for f in v if f['mid']==m),'example',[(f['role'],f['s'],f['y_side'],f['q'],f['p'],f['self_trade']) for f in fs[:2]])
