import build as B
ps=[B.HERE/x for x in ('recover.py','recovery_declaration.md','freeze_recovery.py','check_fast.py','resource_launch.py')]+sorted((B.DATA/'v3').glob('*_normalized.csv.gz'))+[B.DATA/'v3/pilot.json']
p=B.HERE/'recovery_sources.json';B.need(not p.exists(),'write once');B.dump(p,{str(x.relative_to(B.ROOT)):B.sha(x) for x in ps})
