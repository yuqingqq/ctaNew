import base64,gzip,json
from pathlib import Path
import core as C
data=json.loads(gzip.decompress((C.OUT/'data.json.gz').read_bytes()));data['risk']=json.loads((C.H/'risk.json').read_text())
dashboard = C.H.parent / 'maker_dashboard_20261124'
trend_markets = json.loads((dashboard / 'markets/selection.json').read_text())
receipt = C.H.parent / 'inventory_exit_exception_20261125/summary.json'
records = {(row['mid'], row['arm']): row for row in json.loads(receipt.read_text())['rows']}
for market in data['cases']:
    trend = trend_markets.get(str(market['mid']))
    if trend is None:
        continue
    market['trend_window'] = trend
    for arm in ('BASE', 'WIDE'):
        ledger = C.D / 'inventory_exit_exception_20261125' / f"{market['mid']}_{arm}.json.gz"
        assert C.sha(ledger) == records[market['mid'], arm]['sha256']
        replay = json.loads(gzip.decompress(ledger.read_bytes()))
        assert replay['mid'] == market['mid'] and replay['arm'] == arm
        offset = (replay['ident']['start_abs_ms'] - market['ws']) / 1000
        market['own'][arm] = dict(metrics=replay['metrics'], fills=[
            dict(t=offset+t/1000, price=price, s=1-2*side, q=q, gid=gid)
            for t,gid,side,q,price,_ in replay['fills']], curve=[
            [offset+row[0]/1000, row[4]] for row in replay['grid'][::5]])
assert sum(bool(market['own']) for market in data['cases']) == 10
payload=gzip.compress(json.dumps(data,separators=(',',':'),allow_nan=False).encode(),mtime=0)
template=(C.H/'view.html').read_text();script=(C.H/'view.js').read_text();html=template.replace('__PAYLOAD__',base64.b64encode(payload).decode()).replace('__SCRIPT__',script)
assert '__PAYLOAD__' not in html and '__SCRIPT__' not in html
assert json.loads(gzip.decompress(base64.b64decode(html.split('<script id="payload" type="application/octet-stream">')[1].split('</script>')[0])))==data
p=C.H.parent/'maker_dashboard_20261124/maker3-research.html';p.write_text(html)
C.dump(C.H/'chart.json',dict(path=str(p.relative_to(C.R)),sha256=C.sha(p),bytes=p.stat().st_size,markets=len(data['cases']),trend=data['trend'],source_sha256=C.sha(C.OUT/'data.json.gz'),risk_sha256=C.sha(C.H/'risk.json')))
print(p,p.stat().st_size,'bytes; payload exact round trip')
