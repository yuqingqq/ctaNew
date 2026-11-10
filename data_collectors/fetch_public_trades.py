"""Fetch BTC trades from public exchange APIs for given UTC days (P-2026-005, 2026-11-04). Read-only public endpoints, no keys.

  coinbase  Exchange API GET /products/BTC-USD/trades (1000 per page, cursor 'after' = older); 'side' is the MAKER side, so the
            taker side is its opposite (Coinbase docs). Day bounds found by bisecting trade ids on their times.
  kraken    GET /0/public/Trades?pair=XBTUSD&since=... (1000 per call, cursor 'last'); side 'b'/'s' is the taker side.
  deribit   history API get_last_trades_by_instrument_and_time, BTC-PERPETUAL, ascending (1000 per call); 'direction' is the
            taker side; amount is USD, size_btc = amount / price.
Output: <out>/<venue>/<YYYYMMDD>.csv.gz with t_ms, price, size_btc, taker (+1 buy / -1 sell), id. Existing day files are skipped.
Usage: python3 -I data_collectors/fetch_public_trades.py <out_dir> <venue,venue> <YYYYMMDD> [...]
"""
import gzip, json, os, sys, threading, time, urllib.request, urllib.error
from datetime import datetime, timezone

UA = {'User-Agent': 'ctaNew-research/1.0'}


def get(url, tries=8):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode()), dict(r.headers)
        except urllib.error.HTTPError as e:
            wait = 2 ** k if (e.code in (429, 500, 502, 503, 504) or (e.code == 400 and k < 3)) else None   # a 400 seen once was transient
            if wait is None:
                raise
            time.sleep(min(wait, 60))
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(min(2 ** k, 60))
    raise RuntimeError(f'gave up: {url}')


def day_ms(d):
    t0 = datetime.strptime(d, '%Y%m%d').replace(tzinfo=timezone.utc).timestamp() * 1000.0
    return t0, t0 + 86400e3


def write(path, rows):
    tmp = path + '.tmp'
    with gzip.open(tmp, 'wt') as f:
        f.write('t_ms,price,size_btc,taker,id\n')
        for r in sorted(rows):
            f.write(f'{r[0]:.3f},{r[1]},{r[2]:.8f},{r[3]},{r[4]}\n')
    os.replace(tmp, path)


def cb_time(iso):
    return datetime.strptime(iso.replace('Z', '+00:00').split('+')[0], '%Y-%m-%dT%H:%M:%S.%f' if '.' in iso else '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).timestamp() * 1000.0


CB = 'https://api.exchange.coinbase.com/products/BTC-USD/trades'


def cb_id_at(t_ms, hi):
    """Smallest trade id whose time is >= t_ms (bisection on 'after' = id + 1, limit 1)."""
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        js, _ = get(f'{CB}?limit=1&after={mid + 1}'); time.sleep(0.15)
        if not js or cb_time(js[0]['time']) < t_ms:
            lo = mid + 1
        else:
            hi = mid
    return lo


def coinbase(out, days, log):
    js, _ = get(f'{CB}?limit=1'); latest = int(js[0]['trade_id'])
    for d in days:
        path = f'{out}/coinbase/{d}.csv.gz'
        if os.path.exists(path):
            continue
        t0, t1 = day_ms(d); a = cb_id_at(t0, latest); b = cb_id_at(t1, latest)
        rows, cur = [], b
        while cur > a:
            js, _ = get(f'{CB}?limit=1000&after={cur}'); time.sleep(0.2)
            if not js:
                break
            for x in js:
                i = int(x['trade_id'])
                if a <= i < b:
                    rows.append((cb_time(x['time']), x['price'], float(x['size']), -1 if x['side'] == 'buy' else 1, i))
            cur = min(int(x['trade_id']) for x in js)
        write(path, rows); log(f'coinbase {d}: {len(rows)} trades (ids {a}..{b - 1})')


def kraken(out, days, log):
    for d in days:
        path = f'{out}/kraken/{d}.csv.gz'
        if os.path.exists(path):
            continue
        t0, t1 = day_ms(d); since = str(int(t0 / 1000)); rows = []
        while True:
            js, _ = get(f'https://api.kraken.com/0/public/Trades?pair=XBTUSD&count=1000&since={since}'); time.sleep(1.1)
            if js.get('error'):
                time.sleep(5); continue
            res = js['result']; key = [k for k in res if k != 'last'][0]; tr = res[key]
            for x in tr:
                tm = float(x[2]) * 1000.0
                if t0 <= tm < t1:
                    rows.append((tm, x[0], float(x[1]), 1 if x[3] == 'b' else -1, int(x[6]) if len(x) > 6 else 0))
            if not tr or float(tr[-1][2]) * 1000.0 >= t1 or res['last'] == since:
                break
            since = res['last']
        write(path, rows); log(f'kraken {d}: {len(rows)} trades')


def deribit(out, days, log):
    for d in days:
        path = f'{out}/deribit/{d}.csv.gz'
        if os.path.exists(path):
            continue
        t0, t1 = day_ms(d); start = int(t0); seen = set(); rows = []
        while start < t1:
            js, _ = get('https://history.deribit.com/api/v2/public/get_last_trades_by_instrument_and_time?instrument_name=BTC-PERPETUAL'
                        f'&start_timestamp={start}&end_timestamp={int(t1) - 1}&count=1000&sorting=asc'); time.sleep(0.12)
            tr = js['result']['trades']
            new = [x for x in tr if x['trade_id'] not in seen]
            for x in new:
                seen.add(x['trade_id']); rows.append((float(x['timestamp']), x['price'], float(x['amount']) / float(x['price']), 1 if x['direction'] == 'buy' else -1, x['trade_seq']))
            if not js['result'].get('has_more') or not tr:
                break
            last = int(tr[-1]['timestamp']); start = last if new else last + 1
        write(path, rows); log(f'deribit {d}: {len(rows)} trades')


if __name__ == '__main__':
    out, venues, days = sys.argv[1], sys.argv[2].split(','), sys.argv[3:]
    bad = [d for d in days if '20261028' <= d <= '20261030' or d >= '20261103']
    if bad:
        sys.exit(f'REFUSED: days {bad} are sealed or not yet allowed')
    lock = threading.Lock()

    def log(msg):
        with lock:
            print(datetime.now(timezone.utc).strftime('%H:%M:%SZ'), msg, flush=True)
    for v in venues:
        os.makedirs(f'{out}/{v}', exist_ok=True)
    def runv(v):
        try:
            globals()[v](out, days, log); log(f'{v} done')
        except Exception as e:                                   # one venue failing must not stop the others; reported, not hidden
            log(f'{v} FAILED: {type(e).__name__}: {e}')
    th = [threading.Thread(target=runv, args=(v,)) for v in venues]
    [t.start() for t in th]; [t.join() for t in th]
    log('ALL_DONE')
