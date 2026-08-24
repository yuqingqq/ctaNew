import sys, json, hashlib, pathlib
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,'/home/yuqing/ctaNew')
from live.pm_research.tier1_pipeline import _iter_wire_file, _raw_message_key, _canonical_json, ParseStats

def scan(path):
    p=pathlib.Path(path); st=ParseStats()
    try: recs=list(_iter_wire_file(p, st, source_file_id='p'))
    except Exception as e: return (path,-1,[])
    recs.sort(key=lambda r:(r[0],r[1],r[2]))
    seen={}; out=[]
    for r in recs:
        msg=r[3]; k=_raw_message_key(msg)
        d=hashlib.sha256(_canonical_json(msg)).hexdigest()
        if k in seen:
            if seen[k][0]!=d:
                a,b=seen[k][1],msg
                diff=sorted({kk for kk in set(a)|set(b) if a.get(kk)!=b.get(kk)})
                out.append({"event_type":msg.get("event_type"),"diff":diff,
                            "A":{kk:a.get(kk) for kk in diff},"B":{kk:b.get(kk) for kk in diff},
                            "key":str(k)[:180]})
            continue
        seen[k]=(d,msg)
    return (path,len(recs),out)

if __name__=='__main__':
    files=sys.argv[1:]
    with ProcessPoolExecutor(max_workers=12) as ex:
        for path,n,out in ex.map(scan, files):
            if out:
                print("SHARD", pathlib.Path(path).name, "records", n, "conflicts", len(out))
                for c in out[:3]:
                    print(json.dumps(c)[:1800]); print("---")
