"""Independent checker performance-only adapter, declared before full results.
Caches immutable frame tuples and skips unselected level cells. The same raw
cell values, predicates, intervals and falsifiers are evaluated.
"""
import inspect
import check as C
src=inspect.getsource(C.main)
changes={"pend={};bad={}":"pend={};bad={};frame_rows={int(r.fid):r for r in ff.itertuples()}","for g in lv.itertuples():":"wanted=set(zip(nr.side,nr.tick));selected=lv.side.astype(str)+'|'+lv.ptick.astype(str);lv=lv[selected.isin({s+'|'+str(k) for s,k in wanted})]\n  for g in lv.itertuples():","fr=fm.loc[fid]":"fr=frame_rows[fid]","fm.loc[int(prev['fid'])].conn":"frame_rows[int(prev['fid'])].conn"}
for old,new in changes.items():assert src.count(old)==1,(old,src.count(old));src=src.replace(old,new)
ns=dict(C.__dict__);exec(compile(src,'<bound independent fast check>','exec'),ns)
if __name__=='__main__':ns['main']()
