"""P-2026-005 shared readers: Predict WS books, REST matches/categories, the
Predict reference relay and the Binance USD-M futures bookTicker tape.

All times are kept as integer ns (receipt) or integer ms (venue source) until
the caller converts. Readers stream one hour file at a time; nothing here holds
a whole day of raw JSON.
"""
import gzip, json, os, resource, sys
import datetime as dt

ROOT = '/home/yuqing/ctaNew'
PF_RUN = f'{ROOT}/data/predict_fun/runs/20261021T162823Z_0dd45f2d5a'
REF_RUN = f'{ROOT}/data/predict_reference/runs/20261022T132901Z_8c043d17b6'
BN_DIR = f'{ROOT}/data/mm_hf/raw/bookTicker/BTCUSDT'
OUT = f'{ROOT}/data/predict_fun/research'
E18 = 10 ** 18
MEM_GUARD_MIB = 2048  # self-abort at ~50% of the 4 GiB launcher cap


def rss_mib():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) // 1024
    return 0


def mem_check(stage=''):
    r = rss_mib()
    if r > MEM_GUARD_MIB:
        sys.exit(f'[mem] ABORT {stage}: VmRSS {r} MiB > guard {MEM_GUARD_MIB} MiB')
    return r


def hours_of(day, extra_after=0, extra_before=0):
    """Hour-file stems for a UTC day (YYYYMMDD), optionally padded."""
    d0 = dt.datetime.strptime(day, '%Y%m%d').replace(tzinfo=dt.timezone.utc)
    return [(d0 + dt.timedelta(hours=h)).strftime('%Y%m%d_%H')
            for h in range(-extra_before, 24 + extra_after)]


def _lines(path):
    if not os.path.exists(path):
        return
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield line


def ws_books(hour):
    """Yield (recv_ns, data) for every predictOrderbook frame in one hour file."""
    for line in _lines(f'{PF_RUN}/predict_ws/{hour}.jsonl.gz'):
        if 'predictOrderbook' not in line:
            continue
        r = json.loads(line)
        raw = r.get('raw')
        if not raw:
            continue
        m = json.loads(raw)
        if m.get('type') != 'M' or not str(m.get('topic', '')).startswith('predictOrderbook/'):
            continue
        d = m.get('data')
        if isinstance(d, dict) and 'updateTimestampMs' in d:
            yield r['recv_ns'], d


def rest_rows(hour, path_prefix):
    """Yield (recv_ns, params, payload) for HTTP-200 REST rows on one endpoint family."""
    prefixes = (path_prefix,) if isinstance(path_prefix, str) else tuple(path_prefix)
    for line in _lines(f'{PF_RUN}/rest/{hour}.jsonl.gz'):
        j = line.find('"path":"', 0, 400)
        if j < 0 or not line.startswith(prefixes, j + 8):
            continue
        r = json.loads(line)
        p = r.get('path') or ''
        if not p.startswith(prefixes) or r.get('status') != 200:
            continue
        try:
            payload = json.loads(r['raw'])
        except (TypeError, ValueError):
            continue
        yield r['recv_ns'], r.get('params'), p, payload


def iso_s(s):
    return dt.datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


def yes_side(leg):
    """Map a maker/taker leg to (YES-space side, YES-space price).
    YES bid at p -> bid p; YES ask at p -> ask p; NO bid at q -> ask 1-q; NO ask at q -> bid 1-q."""
    p = int(leg['price']) / E18
    yes = leg['outcome']['indexSet'] == 1
    bid = leg['quoteType'] == 'Bid'
    if yes:
        return ('bid' if bid else 'ask'), round(p, 2)
    return ('ask' if bid else 'bid'), round(1 - p, 2)


def bn_rows(hour):
    """Yield (recv_ns, E_ms, update_id, bid, bid_qty, ask, ask_qty) from the Binance futures tape."""
    for line in _lines(f'{BN_DIR}/{hour}.csv.gz'):
        if not line or not line[0].isdigit():
            continue
        a = line.rstrip('\n').split(',')
        if len(a) < 8:
            continue
        yield int(a[0]), int(a[1]), int(a[3]), float(a[4]), float(a[5]), float(a[6]), float(a[7])


def ref_rows(hour):
    """Yield (recv_ns, publish_time_s, price_str, duplicate) from the Predict reference relay."""
    for line in _lines(f'{REF_RUN}/reference_prices/{hour}.jsonl.gz'):
        r = json.loads(line)
        if r.get('kind') != 'PRICE':
            continue
        yield r['recv_ns'], r['publish_time_s'], r['price'], r.get('duplicate_previous_payload', False)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)
