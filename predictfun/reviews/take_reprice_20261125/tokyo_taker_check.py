"""Research check of Tokyo's taker-timing receipt (pmmmrust f3b3db4, raw records on data/tokyo-taker-timing-20261128 1185f65),
answering docs/receipts/20261128_tokyo_taker_timing_request (pmmmrust cd97f9a).

Read straight from git (nothing extracted): the raw files are checked against MANIFEST.sha256 and the hashes the receipt names, then
  1. every order is recomputed from the create records and the RAW wallet frames (not the probe's own pair summaries) and compared
     field by field with Tokyo's report (run1 report.json on the receipt commit, checks1.report.json on the data commit);
  2. pooled timing per group (BUY after a move, BUY quiet, SELL): median / p90 / max, p90 = nearest rank as Tokyo's;
  3. settlement: match (settlementId UUIDv7 ms) -> the orderTransactionSuccess notice, per fill; the refused SELL of check 4 against it;
  4. was Tokyo's process slow when the after-move BUYs were? Binance E -> receipt in the 400 ms after each send, against the run;
     and Predict's book feed (updateTimestampMs -> receipt) against the replay's 53.1 ms (pf_clock_rl.PF_P50);
  5. the replay's entry fill rule (take_markout.minbook + vwalk on the frame current at the match on the venue clock and the next one)
     on Tokyo's own book frames for every order, at the actual match (venue acceptance + 151 ms) and at the replay's nominal match
     (send + NET + HOLD), against the live outcome, at the probe's sizes;
  6. signed orders and cash against the request's caps.
Point estimates from one 11-minute run on one day (n per row); no interval. No forward data of this research host is read.
`python3 tokyo_taker_check.py selftest` | `python3 tokyo_taker_check.py run` (write-once tokyo_taker_check.json)."""
import gzip
import hashlib
import io
import json
import re
import statistics as st
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
REPO = H.parents[2]
sys.path.insert(0, str(H))
sys.path.insert(0, str(REPO / 'predictfun' / 'research'))
import exit_follow as E  # noqa: E402
import pf_clock_rl as RL  # noqa: E402
import take_markout as TK  # noqa: E402

PM = REPO / 'pmmmrust'
RECEIPT_C, DATA_C, REQUEST_C = 'f3b3db4', '1185f65', 'cd97f9a'
RDIR = 'docs/receipts/20261128_tokyo_taker_timing_request'
DD = 'tokyo-taker-timing-20261128'
OUT = H / 'tokyo_taker_check.json'
HOLD_LIVE = 151.                                    # measured acceptance -> match (all 24 fills 150..152, median 151)
WIN_MS, TOL_MS = 400., 1e-4
FIELDS = ('create_rtt_ms', 'accept_wallet_after_send_ms', 'accept_wallet_to_match_ms', 'send_to_match_ms', 'match_to_first_notice_ms',
          'accept_to_not_accepted_ms', 'match_to_wallet_success_ms')


class Refused(SystemExit):
    pass


def require(c, why):
    if not c:
        raise Refused(f'REFUSED: {why}')


def git_bytes(commit, path):
    return subprocess.run(['git', '-C', str(PM), 'show', f'{commit}:{path}'], capture_output=True, check=True).stdout


def full(c):
    return subprocess.run(['git', '-C', str(PM), 'rev-parse', c], capture_output=True, text=True, check=True).stdout.strip()


def verified(b, sha, what):
    got = hashlib.sha256(b).hexdigest()
    require(got == sha, f'{what}: sha256 mismatch ({got[:12]} != {sha[:12]})')
    return b


def uuid7_ms(u):
    return int(u.replace('-', '')[:12], 16)


def iso_ms(s):
    return round(datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp() * 1000)


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def summ(xs):
    xs = [float(x) for x in xs if x is not None]
    return dict(n=len(xs), median=st.median(xs), p90=q(xs, .9), max=max(xs), min=min(xs)) if xs else dict(n=0)


# ------------------------------------------------------------------------------------------------------------------ inputs
def load_files():
    """The receipt's named hashes, then every raw file through MANIFEST.sha256 (whose own hash the receipt names)."""
    readme = git_bytes(RECEIPT_C, f'{RDIR}/tokyo/README.md').decode()
    m_sha = re.search(r'`MANIFEST\.sha256` sha256 `([0-9a-f]{64})`', readme)
    raw = dict(re.findall(r'`(probe\.jsonl|binance\.jsonl\.gz|books\.jsonl\.gz)` `([0-9a-f]{64})`', readme))
    require(m_sha and len(raw) == 3, 'the receipt does not name the manifest and the three raw hashes')
    man = verified(git_bytes(DATA_C, 'MANIFEST.sha256'), m_sha.group(1), 'MANIFEST.sha256').decode()
    want = {p: s for s, p in (ln.split(None, 1) for ln in man.strip().splitlines())}
    for k, s in raw.items():
        require(want.get(f'{DD}/run1/{k}') == s, f'manifest and receipt differ on run1/{k}')
    files = {}
    for p, s in want.items():
        if p.startswith(f'{DD}/') and p.split('/')[1] in ('run1', 'checks1') and not p.endswith('.err'):
            files[p[len(DD) + 1:]] = verified(git_bytes(DATA_C, p), s, p)
    files['report_run1'] = git_bytes(RECEIPT_C, f'{RDIR}/tokyo/report.json')
    request = git_bytes(REQUEST_C, f'{RDIR}/README.md').decode()
    return files, request, readme


def jl(b):
    return [json.loads(x) for x in b.decode().splitlines() if x.strip()]


def gzl(b):
    return [json.loads(x) for x in gzip.open(io.BytesIO(b), 'rt')]


# ------------------------------------------------------------------------------------------------------------------ 1. orders
def orders_of(recs, tag):
    creates, plans, wallet, kind = {}, {}, {}, {}
    for r in recs:
        k, v = r['kind'], r['v']
        if k == 'taker_create':
            a = v['answer']
            b = a.get('body') or {}
            d = b.get('data') or {}
            creates[v['hash']] = dict(side=v['side'], sent_ms=v['sent_ns'] / 1e6, rtt_ms=(v['answered_ns'] - v['sent_ns']) / 1e6, status=a.get('status'),
                                      code=d.get('code') or b.get('error') or b.get('message'), lock=d.get('removalLockedUntil'),
                                      trigger_to_send_us=v.get('trigger_to_send_us'))
        elif k in ('taker_buy_plan', 'taker_sell_plan'):
            plans[v['hash']] = v
        elif k == 'wallet' and isinstance(v.get('input'), dict) and 'data' in v['input']:
            d = v['input']['data']
            wallet.setdefault(d.get('orderHash'), []).append(dict(type=d['type'], ts=d.get('timestamp'), at_ms=v['at_ns'] / 1e6, d=d))
        elif k == 'taker_pair':
            for s in ('buy', 'sell'):
                if v.get(s):
                    kind[v[s]['hash']] = (v['n'], v['kind'], v[s])
        elif k == 'taker_check':
            for lg in v['legs']:
                kind[lg['hash']] = (None, f"check:{v['check']}", lg)
    out = []
    for h, c in creates.items():
        ev = wallet.get(h, [])
        g = lambda t: [e for e in ev if e['type'] == t]  # noqa: E731
        acc, na, sub, suc = g('orderAccepted'), g('orderNotAccepted'), g('orderTransactionSubmitted'), g('orderTransactionSuccess')
        n, kd, lg = kind.get(h, (None, None, None))
        o = dict(tag=tag, hash=h, n=n, kind=kd, side=c['side'], status=c['status'], code=c['code'], sent_ms=c['sent_ms'],
                 create_rtt_ms=c['rtt_ms'], trigger_to_send_us=c['trigger_to_send_us'],
                 market=lg and lg['market'], outcome=lg and lg['outcome'], limit=lg and lg['limit'], shares=lg and lg['shares'],
                 usdt_signed=(plans.get(h) or {}).get('usdt'), n_fail=len(g('orderTransactionFailed')))
        if acc:
            o['accept_ms'] = acc[0]['ts']
            o['accept_wallet_after_send_ms'] = acc[0]['ts'] - c['sent_ms']
            o['accepted_notice_lag_ms'] = acc[0]['at_ms'] - acc[0]['ts']
            if c['lock']:
                o['lock_minus_accept_ms'] = iso_ms(c['lock']) - acc[0]['ts']
        if na:
            o['not_accepted_reason'] = na[0]['d'].get('reason')
            o['accept_to_not_accepted_ms'] = na[0]['ts'] - acc[0]['ts'] if acc else None
        if sub:
            ms = sorted({uuid7_ms(e['d']['settlementId']) for e in sub})
            o['match_ms'] = ms[0]
            o['settlements'] = len(sub)
            o['one_match_ms'] = len(ms) == 1
            o['sid_eq_stamp'] = all(uuid7_ms(e['d']['settlementId']) == e['ts'] for e in sub)
            o['accept_wallet_to_match_ms'] = ms[0] - acc[0]['ts']
            o['send_to_match_ms'] = ms[0] - c['sent_ms']
            o['match_to_first_notice_ms'] = min(e['at_ms'] for e in ev if e['type'] != 'orderAccepted') - ms[0]
            o['filled'] = sum(int(e['d']['fill']['executedSizeWei']) for e in sub) / 1e18
            o['value'] = sum(int(e['d']['fill']['executedValueWei']) for e in sub) / 1e18
            o['prices'] = sorted({int(e['d']['fill']['executedPriceWei']) / 1e16 for e in sub})
        if suc:
            o['match_to_wallet_success_ms'] = min(e['at_ms'] for e in suc) - o.get('match_ms', np.nan)
            o['success_notice_ms'] = min(e['at_ms'] for e in suc)
            fees = [(e['d'].get('fee') or {}) for e in suc]
            o['fee'] = sum(int(f.get('amountWei', 0)) for f in fees) / 1e18
            o['fee_type'] = sorted({f.get('type') for f in fees})
        out.append(o)
    return out


def parity(mine, theirs):
    """Field-by-field against Tokyo's report rows (by hash) -> (compared values, mismatches)."""
    by = {r['hash']: r for r in theirs}
    require(by, 'no report rows to compare')
    n, bad = 0, []
    for o in mine:
        r = by.get(o['hash'])
        if r is None:
            bad.append((o['hash'][:10], 'missing in the report'))
            continue
        for k in FIELDS:
            a, b = o.get(k), r.get(k)
            if a is None and b is None:
                continue
            n += 1
            if a is None or b is None or abs(float(a) - float(b)) > TOL_MS:
                bad.append((o['hash'][:10], k, a, b))
    require(n > 0, 'no values compared')
    return n, bad


# ------------------------------------------------------------------------------------------------------------------ 4. feeds
def feeds(bn_rows, book_rows):
    bn = sorted((r[0] / 1e6, r[0] / 1e6 - json.loads(r[2])['data']['E']) for r in bn_rows if '"E"' in r[2])
    pf = []
    for r in book_rows:
        d = json.loads(r[2])['data']
        if d.get('updateTimestampMs'):
            pf.append(r[0] / 1e6 - d['updateTimestampMs'])
    return np.array(bn), np.array(pf)


def window_lag(bn, t0):
    i0, i1 = np.searchsorted(bn[:, 0], [t0, t0 + WIN_MS])
    x = bn[i0:i1, 1]
    return dict(n=int(len(x)), median=float(np.median(x)), max=float(x.max())) if len(x) else dict(n=0)


# ------------------------------------------------------------------------------------------------------------------ 5. fill rule
def frames_of(book_rows):
    fr = {}
    for recv, _mono, t in book_rows:
        x = json.loads(t)
        d, m = x['data'], int(x['topic'].split('/')[1])
        row = np.full(22, -1.)
        row[0], row[1] = d['updateTimestampMs'], recv / 1e6
        row[7:12] = 0.
        row[17:22] = 0.
        bids = sorted(((round(p * 100, 6), s) for p, s in d.get('bids', []) if s > 0), reverse=True)[:5]
        asks = sorted((round(p * 100, 6), s) for p, s in d.get('asks', []) if s > 0)[:5]
        for k, (p, s) in enumerate(bids):
            row[2 + k], row[7 + k] = p, s
        for k, (p, s) in enumerate(asks):
            row[12 + k], row[17 + k] = p, s
        fr.setdefault(m, []).append(row)
    return {m: np.array(sorted(v, key=lambda r: (r[0], r[1]))) for m, v in fr.items()}


def up_book_order(side, outcome, limit):
    """An order on outcome `outcome` as a walk of the Up book: (d, limit in Up cents); d = +1 takes Up asks, -1 takes Up bids."""
    buys_up = (side == 'BUY') == (outcome == 'Up')
    lim = float(limit) if outcome == 'Up' else 100. - float(limit)
    return (1., lim) if buys_up else (-1., lim)


def fill_at(f, act, d, lim, q_):
    """The replay's entry rule at match time act (venue ms): snapshot A alone and the AB minbook -> (filled A, filled AB)."""
    f0 = f[:, 0]
    jv = int(np.searchsorted(f0, act, 'right') - 1)
    require(jv >= 0, 'no frame before the match')
    s_ = 1 if d > 0 else 0
    pa, sa = TK.side_levels(f[jv], s_)
    pb, sb = TK.side_levels(f[jv + 1], s_) if jv + 1 < len(f) else (np.full(5, -1.), np.zeros(5))
    a = TK.vwalk(pa[None], sa[None], np.array([d]), np.array([lim]), q=q_)[0][0]
    ab = TK.vwalk(*TK.minbook(pa[None], sa[None], pb[None], sb[None]), np.array([d]), np.array([lim]), q=q_)[0][0]
    return float(a), float(ab)


def fill_rule(orders, frames):
    rows = []
    for o in orders:
        if 'accept_ms' not in o or o['market'] is None:
            continue
        d, lim = up_book_order(o['side'], o['outcome'], o['limit'])
        f = frames[o['tag']][o['market']]
        a = fill_at(f, o['accept_ms'] + HOLD_LIVE, d, lim, float(o['shares']))
        b = fill_at(f, o['sent_ms'] + E.NET + E.HOLD, d, lim, float(o['shares']))
        rows.append(dict(tag=o['tag'], n=o['n'], kind=o['kind'], side=o['side'], live=o.get('filled', 0.) > 0,
                         actual_A=a[0] > 0, actual_AB=a[1] > 0, nominal_A=b[0] > 0, nominal_AB=b[1] > 0))
    return rows


def agree(rows, sel):
    xs = [r for r in rows if sel(r)]
    return dict(n=len(xs), live_fills=sum(r['live'] for r in xs),
                **{k: dict(agree=sum(r[k] == r['live'] for r in xs), fills=sum(r[k] for r in xs)) for k in ('actual_A', 'actual_AB', 'nominal_A', 'nominal_AB')})


# ------------------------------------------------------------------------------------------------------------------ run
GROUPS = (('BUY_after_move', lambda o: o['tag'] == 'run1' and o['side'] == 'BUY' and o['kind'] == 'Move'),
          ('BUY_quiet', lambda o: o['tag'] == 'run1' and o['side'] == 'BUY' and o['kind'] == 'Quiet'),
          ('SELL', lambda o: o['tag'] == 'run1' and o['side'] == 'SELL'),
          ('run_all', lambda o: o['tag'] == 'run1'))
POOLED = ('create_rtt_ms', 'accept_wallet_after_send_ms', 'accept_wallet_to_match_ms', 'accept_to_not_accepted_ms', 'send_to_match_ms',
          'match_to_first_notice_ms', 'match_to_wallet_success_ms', 'accepted_notice_lag_ms')


def compute(files, request):
    recs = {t: jl(files[f'{t}/probe.jsonl']) for t in ('run1', 'checks1')}
    orders = orders_of(recs['run1'], 'run1') + orders_of(recs['checks1'], 'checks1')
    rep_run = json.loads(files['report_run1'])['orders']
    rep_chk = json.loads(files['checks1/checks1.report.json'])['orders']
    n_cmp, bad = parity(orders, rep_run + rep_chk)
    require(not bad, f'recomputed orders differ from Tokyo\'s report: {bad[:5]}')
    pooled = {g: {k: summ([o.get(k) for o in orders if sel(o)]) for k in POOLED} | dict(orders=sum(map(sel, orders)),
                                                                                       matched=sum(1 for o in orders if sel(o) and 'match_ms' in o))
              for g, sel in GROUPS}
    fills = [o for o in orders if 'match_ms' in o]
    chk = [o for o in orders if o['tag'] == 'checks1']
    refused = [o for o in chk if o['status'] != 201]
    require(len(refused) == 1 and refused[0]['side'] == 'SELL', 'expected exactly one refused SELL among the checks')
    buy3 = next(o for o in chk if o['side'] == 'BUY' and 'match_ms' in o)
    sell5 = next(o for o in chk if o['side'] == 'SELL' and 'match_ms' in o)
    settle = dict(
        match_to_success_notice_all_fills=summ([o['match_to_wallet_success_ms'] for o in fills]),
        match_to_success_notice_buy_fills=summ([o['match_to_wallet_success_ms'] for o in fills if o['side'] == 'BUY']),
        success_stamp_whole_seconds=True,
        check4_refused=dict(code=refused[0]['code'], sent_after_buy_match_ms=refused[0]['sent_ms'] - buy3['match_ms'],
                            sent_before_buy_success_notice_ms=buy3['success_notice_ms'] - refused[0]['sent_ms']),
        check5_filled=dict(sent_after_buy_match_ms=sell5['sent_ms'] - buy3['match_ms'],
                           sent_after_buy_success_notice_ms=sell5['sent_ms'] - buy3['success_notice_ms']),
        failed_settlements=sum(o['n_fail'] for o in orders))
    bn, pf = feeds(gzl(files['run1/binance.jsonl.gz']), gzl(files['run1/books.jsonl.gz']))
    per = [dict(n=o['n'], kind=o['kind'], side=o['side'], accept_after_send_ms=o['accept_wallet_after_send_ms'], **{f'bn_lag_{k}': v for k, v in window_lag(bn, o['sent_ms']).items()})
           for o in orders if o['tag'] == 'run1']
    mv = [p for p in per if p['kind'] == 'Move' and p['side'] == 'BUY']
    feed = dict(binance_E_to_recv_run=dict(median=float(np.median(bn[:, 1])), p99=float(np.quantile(bn[:, 1], .99)), n=int(len(bn))),
                binance_after_move_sends=dict(max_of_window_max=max(p['bn_lag_max'] for p in mv), max_of_window_median=max(p['bn_lag_median'] for p in mv),
                                              min_window_n=min(p['bn_lag_n'] for p in mv), max_window_n=max(p['bn_lag_n'] for p in mv)),
                predict_update_to_recv_run=dict(median=float(np.median(pf)), p90=float(np.quantile(pf, .9)), n=int(len(pf))),
                replay_predict_p50_ms=RL.PF_P50, per_order=per)
    feed['host_not_slow'] = bool(feed['binance_after_move_sends']['max_of_window_max'] < 10.)
    frames = {t: frames_of(gzl(files[f'{t}/books.jsonl.gz'])) for t in ('run1', 'checks1')}
    fr = fill_rule(orders, frames)
    rule = {g: agree(fr, sel) for g, sel in (('BUY_after_move', lambda r: r['tag'] == 'run1' and r['side'] == 'BUY' and r['kind'] == 'Move'),
                                             ('BUY_quiet', lambda r: r['tag'] == 'run1' and r['side'] == 'BUY' and r['kind'] == 'Quiet'),
                                             ('SELL', lambda r: r['tag'] == 'run1' and r['side'] == 'SELL'), ('checks', lambda r: r['tag'] == 'checks1'),
                                             ('all', lambda r: True))}
    caps = re.search(r'At most (\d+) BUYs and (\d+) SELLs, at most (\d+) USDT bought', request)
    require(caps, 'the request\'s caps not found')
    cb, cs, cu = (int(x) for x in caps.groups())
    nb, ns = sum(o['side'] == 'BUY' for o in orders), sum(o['side'] == 'SELL' for o in orders)
    usdt_signed = sum(o['usdt_signed'] or 0. for o in orders if o['side'] == 'BUY')
    cash = {t: sum((-1 if o['side'] == 'BUY' else 1) * o.get('value', 0.) - (o.get('fee', 0.) if o['side'] == 'SELL' else 0.) for o in orders if o['tag'] == t)
            for t in ('run1', 'checks1')}
    rep_cash = dict(run1=json.loads(files['report_run1'])['cash_usdt'], checks1=json.loads(files['checks1/checks1.report.json'])['cash_usdt'])
    signed = dict(buys=nb, sells=ns, run=dict(buys=sum(o['side'] == 'BUY' for o in orders if o['tag'] == 'run1'), sells=sum(o['side'] == 'SELL' for o in orders if o['tag'] == 'run1')),
                  request_caps=dict(buys=cb, sells=cs, usdt=cu), buy_cap_exceeded_by=max(0, nb - cb), sell_cap_exceeded_by=max(0, ns - cs),
                  usdt_signed_buys=usdt_signed, usdt_cap_held=bool(usdt_signed <= cu))
    fees = dict(buy_shares=sum(o.get('fee', 0.) for o in orders if o['side'] == 'BUY' and o['tag'] == 'run1'),
                sell_usdt=sum(o.get('fee', 0.) for o in orders if o['side'] == 'SELL' and o['tag'] == 'run1'),
                buy_fee_types=sorted({t for o in orders if o['side'] == 'BUY' for t in o.get('fee_type', [])}),
                sell_fee_types=sorted({t for o in orders if o['side'] == 'SELL' for t in o.get('fee_type', [])}))
    hold = dict(lock_minus_accept_ms=sorted({o['lock_minus_accept_ms'] for o in orders if 'lock_minus_accept_ms' in o}),
                sid_eq_submitted_stamp=sum(o.get('sid_eq_stamp', False) for o in fills), fills=len(fills),
                one_match_ms_per_order=all(o['one_match_ms'] for o in fills))
    return dict(asof=datetime.now().astimezone().isoformat(timespec='seconds'), receipt_commit=full(RECEIPT_C), data_commit=full(DATA_C), request_commit=full(REQUEST_C),
                parity=dict(values_compared=n_cmp, mismatched=len(bad), orders=len(orders)), pooled=pooled, hold=hold, settlement=settle, feeds=feed,
                fill_rule=dict(groups=rule, hold_live_ms=HOLD_LIVE, nominal_ms=E.NET + E.HOLD, rows=fr),
                signed=signed, cash_usdt=dict(recomputed=cash, report=rep_cash, cash_matches_report=all(abs(cash[t] - rep_cash[t]) < 1e-9 for t in cash)),
                fees=fees, orders=[{k: v for k, v in o.items() if k != 'sent_ms'} | dict(sent_ms=o['sent_ms']) for o in orders])


def run():
    require(not OUT.exists(), f'{OUT.name} exists (write-once)')
    files, request, _ = load_files()
    res = compute(files, request)
    OUT.write_text(json.dumps(res, indent=1, default=float) + '\n')
    print('wrote', OUT.name, 'parity', res['parity'], 'host_not_slow', res['feeds']['host_not_slow'])


# ------------------------------------------------------------------------------------------------------------------ selftest
def selftest():
    def refuses(fn, why):
        try:
            fn()
        except Refused as e:
            require(why in str(e), f'refused for another reason: {e}')
            return
        raise AssertionError(f'did not refuse: {why}')
    refuses(lambda: verified(b'abc', hashlib.sha256(b'abd').hexdigest(), 'x'), 'sha256 mismatch')
    a = dict(hash='0xaa', create_rtt_ms=40., accept_wallet_after_send_ms=25., send_to_match_ms=176.)
    n, bad = parity([a], [dict(a)])
    require(n == 3 and not bad, 'parity flags an identical row')
    _, bad = parity([a], [dict(a, accept_wallet_after_send_ms=26.)])
    require(len(bad) == 1 and bad[0][1] == 'accept_wallet_after_send_ms', 'parity misses a 1 ms change')
    _, bad = parity([a, dict(a, hash='0xbb')], [dict(a)])
    require(bad == [('0xbb', 'missing in the report')], 'parity misses a row absent from the report')
    refuses(lambda: parity([a], [dict(hash='0xbb')]), 'no values compared')
    refuses(lambda: parity([a], []), 'no report rows')
    # fill rule: Up ask 50 x 10 then 55 x 10; a BUY Up at limit 53 fills on A; the next frame empties 50 -> AB does not fill
    f = np.full((2, 22), -1.)
    f[:, 7:12] = 0.
    f[:, 17:22] = 0.
    f[0, [0, 1, 2, 7, 12, 13, 17, 18]] = [1000., 1050., 48., 10., 50., 55., 10., 10.]
    f[1, [0, 1, 2, 7, 12, 17]] = [1100., 1150., 48., 10., 55., 10.]
    require(fill_at(f, 1050., *up_book_order('BUY', 'Up', 53), 2.) == (2., 0.), 'fill rule: A fills, AB must not (level emptied next)')
    require(fill_at(f, 1150., *up_book_order('BUY', 'Up', 53), 2.) == (0., 0.), 'fill rule: no level at or under the limit')
    # BUY Down at 54 = take Up bids at >= 46: the 48 bid fills; at Down 51 (Up >= 49) nothing
    require(fill_at(f, 1050., *up_book_order('BUY', 'Down', 54), 2.)[0] == 2., 'fill rule: a Down BUY takes the Up bid')
    require(fill_at(f, 1050., *up_book_order('BUY', 'Down', 51), 2.)[0] == 0., 'fill rule: a Down BUY under the bid fills')
    require(up_book_order('SELL', 'Down', 40) == (1., 60.), 'a Down SELL is an Up ask take at 100 - limit')
    refuses(lambda: fill_at(f, 900., 1., 53., 2.), 'no frame before the match')
    bn = np.array([[0., 1.], [100., 2.], [500., 50.]])
    require(window_lag(bn, 0.) == dict(n=2, median=1.5, max=2.), 'window lag')
    print('selftest ok')


if __name__ == '__main__':
    {'selftest': selftest, 'run': run}[sys.argv[1]]()
