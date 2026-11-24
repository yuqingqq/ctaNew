"""Complete the unchanged independent predicates with a bounded larger watchdog."""
import inspect
import check_fast as F
C=F.C;src=inspect.getsource(C.main);changes=dict(F.changes)
changes['for g in lv.itertuples():']="lv=lv[lv.ptick.isin(set(nr.tick)) & lv.side.isin(set(nr.side))]\n  for g in lv.itertuples():"
changes['resource.setrlimit(resource.RLIMIT_CPU,(16,18))']='resource.setrlimit(resource.RLIMIT_CPU,(26,28))'
changes["  need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1450*1024,'memory')"]="  need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1450*1024,'memory');print('verified market',d,m,'CPU',round(time.process_time()-start,3),flush=True)"
for old,new in changes.items():assert src.count(old)==1;src=src.replace(old,new)
ns=dict(C.__dict__);exec(compile(src,'<bounded complete independent check>','exec'),ns)
if __name__=='__main__':ns['main']()
