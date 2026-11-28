"""Research check of Tokyo's big-move acceptance receipt (pmmmrust 2d53b0b, raw records on data/tokyo-bigmove-acceptance-20261128
bd65d08), answering docs/receipts/20261128_tokyo_bigmove_acceptance_request (pmmmrust 21ae192). tokyo_taker_check.py's method and
functions, read straight from git (nothing extracted):
  1. hashes: MANIFEST.sha256 against the receipt, every raw file against the manifest, the receipt's report.json against the data's;
  2. every order recomputed from the create records and the RAW wallet frames, field by field against Tokyo's report;
  3. pooled timing (BUY, SELL): median / p90 (nearest rank, as Tokyo's) / max; the hold (lock - acceptance, acceptance -> match or kill);
  4. the trigger, from the raw Binance stream: each BUY's trigger frame recomputed with Tokyo's rule (probe/taker.rs at 0b1f4b5: every
     uncrossed bookTicker in receipt order, the mid now less the mid of the last change at or before E - 100), its size against the
     request's threshold and its sign against the outcome bought; trigger -> decision -> send; spacing, window margin, run length, count;
     the late triggers (taker_trigger_late), recomputed the same way;
  5. the replay's entry fill rule (take_markout AB) on Tokyo's frames at the actual match (acceptance + 151) and the nominal (send + 22
     + 150), against the live outcome; the book's best at the hold's end against Tokyo's;
  6. the move statistics of the line's own definition (chase_move_size.move_stats: the mid collapsed per E ms) at each BUY's and each
     late trigger's E; Tokyo's two counts of the changes after the send reproduced, and a bridge between the counts; Spearman of the
     acceptance delay; the first probe's after-move BUYs (chase_move_size.json, 1ac2e0529) pooled with these;
  7. signed orders and USDT against the request's caps; cash recomputed in wei against the on-chain balances and the report.
Point estimates from one 9-minute run (n per row); no interval. No forward data of this research host is read.
`python3 tokyo_bigmove_check.py selftest` | `python3 tokyo_bigmove_check.py run` (write-once tokyo_bigmove_check.json)."""
import bisect
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import tokyo_taker_check as T  # noqa: E402
import chase_move_size as CMS  # noqa: E402

RECEIPT_C, DATA_C, REQUEST_C = '2d53b0b', 'bd65d08', '21ae192'
RDIR = 'docs/receipts/20261128_tokyo_bigmove_acceptance_request'
DD = 'tokyo-bigmove-acceptance-20261128'
OUT = H / 'tokyo_bigmove_check.json'
FIRST_MOVES = H / 'chase_move_size.json'
SPAN_MS, POST_MS, MOVE_GAP_MS = 100, 200, 1000.
MOVE_KEYS = ('up100', 'up1000', 'n100', 'n_post200', 'up_post200')
require, summ = T.require, T.summ


# ------------------------------------------------------------------------------------------------------------------ inputs
def load_files():
    readme = T.git_bytes(RECEIPT_C, f'{RDIR}/tokyo/README.md').decode()
    m_sha = re.search(r'`MANIFEST\.sha256` sha256 `([0-9a-f]{64})`', readme)
    raw = dict(re.findall(r'`(probe\.jsonl|binance\.jsonl\.gz|books\.jsonl\.gz)` `([0-9a-f]{64})`', readme))
    require(m_sha and len(raw) == 3, 'the receipt does not name the manifest and the three raw hashes')
    man = T.verified(T.git_bytes(DATA_C, 'MANIFEST.sha256'), m_sha.group(1), 'MANIFEST.sha256').decode()
    want = {p: s for s, p in (ln.split(None, 1) for ln in man.strip().splitlines())}
    for k, s in raw.items():
        require(want.get(f'{DD}/run1/{k}') == s, f'manifest and receipt differ on run1/{k}')
    files = {p[len(DD) + 1:]: T.verified(T.git_bytes(DATA_C, p), s, p) for p, s in want.items() if p.startswith(f'{DD}/') and not p.endswith('.err')}
    rep = T.git_bytes(RECEIPT_C, f'{RDIR}/tokyo/report.json')
    same_report(rep, want[f'{DD}/run1/run1.report.json'])
    files['report_run1'] = rep
    toml = T.git_bytes(RECEIPT_C, f'{RDIR}/tokyo/taker.toml')
    first_toml = T.git_bytes(T.RECEIPT_C, f'{T.RDIR}/tokyo/taker.toml')
    request = T.git_bytes(REQUEST_C, f'{RDIR}/README.md').decode()
    return files, request, dict(toml_same_as_data=hashlib.sha256(toml).hexdigest() == want[f'{DD}/taker.toml'],
                                toml_same_as_first_probe=toml == first_toml)


def same_report(rep, sha):
    require(hashlib.sha256(rep).hexdigest() == sha, "the receipt's report.json is not the data branch's run1.report.json")


def request_params(request):
    """The request's declared trigger, count, spacing, margin, run limit and caps (read from its text, not retyped)."""
    pats = dict(min_move=r'at least \*\*\$(\d+(?:\.\d+)?)\*\* over the last 100 ms', buys=r'(\d+) BUYs on BTC', spacing_s=r'at least (\d+) s apart',
                margin_s=r'at least (\d+) s before its window ends', run_min=r'Stop after (\d+) minutes',
                caps=r'At most (\d+) BUYs and (\d+) SELLs \*\*in total, any checks included\*\*, and at most (\d+) USDT committed to BUYs')
    got = {k: re.search(p, request) for k, p in pats.items()}
    require(all(got.values()), f"the request's parameters not found: {[k for k, v in got.items() if not v]}")
    cb, cs, cu = (int(x) for x in got['caps'].groups())
    return dict(min_move=float(got['min_move'].group(1)), buys=int(got['buys'].group(1)), spacing_ms=1000. * int(got['spacing_s'].group(1)),
                margin_ms=1000. * int(got['margin_s'].group(1)), run_ms=60000. * int(got['run_min'].group(1)), cap_buys=cb, cap_sells=cs, cap_usdt=cu)


# ------------------------------------------------------------------------------------------------------------------ 4. trigger
def bn_parse(rows):
    """Every Binance record in receipt order -> (recv_ns, E, bid, ask, is_book_ticker)."""
    out = []
    for recv, _mono, txt in rows:
        x = json.loads(txt).get('data') or {}
        bt = x.get('e') == 'bookTicker'
        out.append((int(recv), x.get('E'), float(x['b']) if bt else None, float(x['a']) if bt else None, bt))
    return out


def tokyo_mid_moves(parsed, at_recv):
    """Tokyo's Mid (probe/taker.rs at 0b1f4b5): every bookTicker with 0 < bid < ask, in receipt order; its history keeps (E, mid) at each
    mid change; move_over(100) = the mid now less the mid of the last history entry at or before E - 100; changes_over(100) = history
    entries with E + 100 > E now. -> {recv_ns: dict(E, move_100ms, changes_100ms)} at the frames received at `at_recv`."""
    want, hist, mid, at = set(at_recv), [], None, []
    for recv, e, b, a, bt in parsed:
        if not bt or not (b > 0. and a > b):
            continue
        m = (b + a) / 2.
        if m != mid:
            hist.append((int(e), m))
        mid = m
        if recv in want:
            at.append((recv, int(e), m, len(hist)))
    he = np.array([h[0] for h in hist], float)
    hm = np.array([h[1] for h in hist], float)
    out = {}
    for recv, e, m, k in at:
        prior = np.flatnonzero(he[:k] <= e - SPAN_MS)
        out[recv] = dict(E=e, move_100ms=None if not len(prior) else float(m - hm[prior[-1]]), changes_100ms=int(np.sum(he[:k] + SPAN_MS > e)))
    return out


def tokyo_changes(parsed):
    """pf_taker_report.binance_changes: every record with a bid and an ask (no crossed-book filter), kept where the mid differs from the
    previous record's -> [(recv_ns, E, mid)] in receipt order."""
    out, last = [], None
    for recv, e, b, a, bt in parsed:
        if not bt:
            continue
        m = (b + a) / 2.
        if m != last:
            out.append((recv, e, m))
            last = m
    return out


def tokyo_counts(changes, sent_ns):
    """pf_taker_report's two counts after a send: changes received in [send, send + 200 ms), and changes whose E is in [send ms, + 200)."""
    i = bisect.bisect_left(changes, (sent_ns,))
    by_recv = sum(1 for c in changes[i:] if c[0] < sent_ns + POST_MS * 1_000_000)
    sm = sent_ns / 1e6
    by_e = sum(1 for c in changes if c[1] is not None and sm <= c[1] < sm + POST_MS)
    return by_recv, by_e


def count_bridge(changes, E_dec):
    """Between the counts: every change (not collapsed per ms) with E in (E_dec, E_dec + 200], the line's window on Tokyo's changes."""
    return sum(1 for c in changes if c[1] is not None and E_dec < c[1] <= E_dec + POST_MS)


def moves_of(late, gap=MOVE_GAP_MS):
    """Late triggers -> separate moves: a new move where the trigger E is more than `gap` ms after the previous late trigger's."""
    E = sorted(x['E'] for x in late)
    return int(len(E) > 0) + int(np.sum(np.diff(E) > gap)) if E else 0


# ------------------------------------------------------------------------------------------------------------------ 5. book
def best_at(f, act, side, outcome):
    """The outcome's best price (cents) on the frame current at venue ms `act`: a BUY's ask, a SELL's bid; Down = 100 - the Up book's."""
    jv = int(np.searchsorted(f[:, 0], act, 'right') - 1)
    require(jv >= 0, 'no frame before the instant')
    up_ask, up_bid = f[jv, 12], f[jv, 2]
    if outcome == 'Up':
        return float(up_ask) if side == 'BUY' else float(up_bid)
    return float(100. - up_bid) if side == 'BUY' else float(100. - up_ask)


# ------------------------------------------------------------------------------------------------------------------ run
def compute(files, request, idn):
    P = request_params(request)
    recs = T.jl(files['run1/probe.jsonl'])
    orders = T.orders_of(recs, 'run1')
    rep = json.loads(files['report_run1'])
    n_cmp, bad = T.parity(orders, rep['orders'])
    require(not bad, f"recomputed orders differ from Tokyo's report: {bad[:5]}")
    by_h = {o['hash']: o for o in orders}
    rep_h = {r['hash']: r for r in rep['orders']}
    plans = {r['v']['hash']: r for r in recs if r['kind'] == 'taker_buy_plan'}
    creates = {r['v']['hash']: r['v'] for r in recs if r['kind'] == 'taker_create'}
    late = [r['v'] for r in recs if r['kind'] == 'taker_trigger_late']
    pair_end = sorted(r['recv_ns'] / 1e6 for r in recs if r['kind'] == 'taker_pair')
    start_ms = next(r['recv_ns'] for r in recs if r['kind'] == 'taker_start') / 1e6
    buys = [o for o in orders if o['side'] == 'BUY']
    sells = [o for o in orders if o['side'] == 'SELL']
    require(len(buys) == len(plans) and all(o['kind'] == 'Move' for o in buys), 'every BUY is a planned move BUY')
    # 3. timing
    groups = dict(BUY=buys, SELL=sells, BUY_filled=[o for o in buys if o.get('filled', 0.) > 0], BUY_killed=[o for o in buys if not o.get('filled', 0.) > 0])
    pooled = {g: {k: summ([o.get(k) for o in xs]) for k in T.POOLED} | dict(orders=len(xs), matched=sum('match_ms' in o for o in xs)) for g, xs in groups.items()}
    hold = dict(lock_minus_accept_ms=sorted({o['lock_minus_accept_ms'] for o in orders if 'lock_minus_accept_ms' in o}),
                accept_to_match_ms=sorted({round(o['accept_wallet_to_match_ms'], 3) for o in orders if 'accept_wallet_to_match_ms' in o}),
                accept_to_kill_ms=sorted({round(o['accept_to_not_accepted_ms'], 3) for o in orders if o.get('accept_to_not_accepted_ms') is not None}),
                kill_reasons=sorted({o.get('not_accepted_reason') for o in buys if o.get('not_accepted_reason')}),
                sid_eq_submitted_stamp=sum(o.get('sid_eq_stamp', False) for o in orders if 'match_ms' in o), fills=sum('match_ms' in o for o in orders))
    # 4. trigger, from the raw stream
    parsed = bn_parse(T.gzl(files['run1/binance.jsonl.gz']))
    trig_recv = [plans[o['hash']]['v']['trigger']['recv_ns'] for o in buys] + [x['trigger']['recv_ns'] for x in late]
    mv = tokyo_mid_moves(parsed, trig_recv)
    changes = tokyo_changes(parsed)
    E_t, M_t = CMS.tokyo_series(files)
    rows = []
    for o in buys:
        v = plans[o['hash']]['v']
        tf, c = v['trigger'], creates[o['hash']]
        re_ = mv.get(tf['recv_ns'])
        require(re_ is not None, f"pair {v['n']}: its trigger frame is not in the raw stream")
        d = 1. if v['outcome'] == 'Up' else -1.
        by_recv, by_e = tokyo_counts(changes, c['sent_ns'])
        st = CMS.move_stats(E_t, M_t, [float(v['binance']['E'])], [d])
        r_ = rep_h[o['hash']]
        rows.append(dict(n=v['n'], market=v['market'], outcome=v['outcome'], accept_ms=o['accept_wallet_after_send_ms'], filled=o.get('filled', 0.) > 0,
                         trigger_E=tf['E'], decision_E=v['binance']['E'], trigger_move_recorded=tf['move_100ms_usd'], trigger_move_recomputed=re_['move_100ms'],
                         trigger_changes_recorded=tf['changes_100ms'], trigger_changes_recomputed=re_['changes_100ms'],
                         frame_recv_after_E_ms=tf['recv_ns'] / 1e6 - tf['E'], trigger_to_decision_ms=(v['decided_mono'] - v['trigger_mono']) / 1e6,
                         trigger_to_send_ms=(c['sent_mono'] - v['trigger_mono']) / 1e6, window_left_ms=v['end_s'] * 1000. - c['sent_ns'] / 1e6,
                         sent_ms=c['sent_ns'] / 1e6, tokyo_changes_200_recv=by_recv, tokyo_changes_200_E=by_e,
                         report_changes_200_recv=r_.get('binance_changes_200ms_after_send'), report_changes_200_E=r_.get('binance_changes_E_200ms_after_send'),
                         bridge_all_changes_line_window=count_bridge(changes, float(v['binance']['E'])),
                         **{k: float(st[k][0]) for k in MOVE_KEYS}))
    rows.sort(key=lambda r: r['n'])
    sends = np.array([r['sent_ms'] for r in rows])
    gaps_pair = [s - max([p for p in pair_end if p < s], default=-np.inf) for s in sends[1:]]
    comp = dict(
        trigger_recomputed_equal=all(r['trigger_move_recomputed'] is not None and abs(r['trigger_move_recomputed'] - r['trigger_move_recorded']) < 1e-9
                                     and r['trigger_changes_recomputed'] == r['trigger_changes_recorded'] for r in rows),
        move_at_least_threshold=all(abs(r['trigger_move_recomputed']) + 1e-9 >= P['min_move'] for r in rows),
        bought_the_moves_way=all((r['trigger_move_recomputed'] > 0) == (r['outcome'] == 'Up') for r in rows),
        trigger_to_send_ms_max=max(r['trigger_to_send_ms'] for r in rows), send_within_5ms=all(r['trigger_to_send_ms'] <= 5. for r in rows),
        trigger_to_decision_ms_max=max(r['trigger_to_decision_ms'] for r in rows),
        frame_recv_after_E_ms=summ([r['frame_recv_after_E_ms'] for r in rows]),
        buy_to_buy_ms_min=float(np.min(np.diff(sends))), pair_end_to_next_buy_ms_min=float(min(gaps_pair)),
        spacing_held=bool(np.min(np.diff(sends)) >= P['spacing_ms'] and min(gaps_pair) >= P['spacing_ms']),
        window_left_ms_min=min(r['window_left_ms'] for r in rows), margin_held=all(r['window_left_ms'] >= P['margin_ms'] for r in rows),
        run_ms=float(sends.max() - start_ms), run_limit_held=bool(sends.max() - start_ms <= P['run_ms']), buys=len(rows), buys_requested=P['buys'],
        tokyo_counts_reproduced=all(r['tokyo_changes_200_recv'] == r['report_changes_200_recv'] and r['tokyo_changes_200_E'] == r['report_changes_200_E'] for r in rows))
    require(comp['trigger_recomputed_equal'], 'the trigger frames do not recompute from the raw stream')
    require(comp['tokyo_counts_reproduced'], "Tokyo's counts after the send do not reproduce")
    # late triggers
    lr = []
    for x in late:
        tf = x['trigger']
        re_ = mv[tf['recv_ns']]
        dd = 1. if (re_['move_100ms'] or 0.) > 0 else -1.
        st = CMS.move_stats(E_t, M_t, [float(tf['E'])], [dd])
        lr.append(dict(n=x['n'], lag_ms=x['lag_us'] / 1000., E=tf['E'], move_recorded=tf['move_100ms_usd'], move_recomputed=re_['move_100ms'],
                       **{k: float(st[k][0]) for k in MOVE_KEYS}))
    srt = sorted(lr, key=lambda r: r['E'])
    first_of_move = [r for i, r in enumerate(srt) if i == 0 or r['E'] - srt[i - 1]['E'] > MOVE_GAP_MS]
    late_s = dict(n=len(lr), moves=moves_of(lr), lag_ms=summ([r['lag_ms'] for r in lr]), by_pair={int(k): int(v) for k, v in zip(*np.unique([r['n'] for r in lr], return_counts=True))},
                  recomputed_equal=all(r['move_recomputed'] is not None and abs(r['move_recomputed'] - r['move_recorded']) < 1e-9 for r in lr),
                  at_least_threshold=all(abs(r['move_recomputed']) + 1e-9 >= P['min_move'] for r in lr),
                  moves_line_def={k: CMS.quant([r[k] for r in lr]) for k in MOVE_KEYS}, first_of_each_move={k: CMS.quant([r[k] for r in first_of_move]) for k in MOVE_KEYS},
                  rows=lr)
    sent_m = {k: CMS.quant([r[k] for r in rows]) for k in MOVE_KEYS}
    late_s['busier_after_than_sent'] = dict(all=bool(late_s['moves_line_def']['n_post200']['p50'] > sent_m['n_post200']['p50']),
                                            first_of_each_move=bool(late_s['first_of_each_move']['n_post200']['p50'] > sent_m['n_post200']['p50']))
    # 5. fill rule and the book at the hold's end
    frames = T.frames_of(T.gzl(files['run1/books.jsonl.gz']))
    fr = T.fill_rule(orders, {'run1': frames})
    rule = {g: T.agree(fr, sel) for g, sel in (('BUY', lambda r: r['side'] == 'BUY'), ('SELL', lambda r: r['side'] == 'SELL'), ('all', lambda r: True))}
    book = []
    for o in orders:
        if 'accept_ms' not in o:
            continue
        mine = best_at(frames[o['market']], o['accept_ms'] + T.HOLD_LIVE, o['side'], o['outcome'])
        r_ = rep_h[o['hash']]
        book.append(dict(n=o['n'], side=o['side'], filled=o.get('filled', 0.) > 0, limit=o['limit'], best_at_hold_end=mine, tokyo_best_at_bump_end=r_.get('best_at_bump_end'),
                         tokyo_best_at_send=r_.get('best_at_send'), limit_reaches=(o['limit'] >= mine) if o['side'] == 'BUY' else (o['limit'] <= mine)))
    book_s = dict(agree_with_tokyo=sum(b['tokyo_best_at_bump_end'] is not None and abs(b['best_at_hold_end'] - b['tokyo_best_at_bump_end']) < 1e-9 for b in book), n=len(book),
                  killed_buys_limit_short=sum(1 for b in book if b['side'] == 'BUY' and not b['filled'] and not b['limit_reaches']),
                  killed_buys=sum(1 for b in book if b['side'] == 'BUY' and not b['filled']),
                  filled_buys_limit_reaches=sum(1 for b in book if b['side'] == 'BUY' and b['filled'] and b['limit_reaches']),
                  filled_buys=sum(1 for b in book if b['side'] == 'BUY' and b['filled']), rows=book)
    # 6. Spearman and the first probe pooled
    acc = [r['accept_ms'] for r in rows]
    sp = {k: CMS.spearman(acc, [r[k] for r in rows]) for k in MOVE_KEYS} | dict(
        tokyo_changes_200_recv=CMS.spearman(acc, [r['tokyo_changes_200_recv'] for r in rows]), tokyo_changes_200_E=CMS.spearman(acc, [r['tokyo_changes_200_E'] for r in rows]),
        abs_trigger_move=CMS.spearman(acc, [abs(r['trigger_move_recomputed']) for r in rows]))
    fm = json.loads(FIRST_MOVES.read_text())
    first = [b for b in fm['probe']['buys'] if b['kind'] == 'Move']
    require(len(first) == 10, 'expected the first probe\'s 10 after-move BUYs')
    both = [dict(probe='first', accept_ms=b['accept_ms'], filled=b['filled'], **{k: b[k] for k in MOVE_KEYS}) for b in first] + \
           [dict(probe='bigmove', accept_ms=r['accept_ms'], filled=r['filled'], **{k: r[k] for k in MOVE_KEYS}) for r in rows]
    pooled_both = dict(n=len(both), accept_ms=summ([b['accept_ms'] for b in both]), filled=sum(b['filled'] for b in both),
                       spearman={k: CMS.spearman([b['accept_ms'] for b in both], [b[k] for b in both]) for k in MOVE_KEYS},
                       line=fm['line']['MOVE_top10'], rows=both)
    # 7. signed orders, USDT, cash in wei
    usdt_signed = sum(o['usdt_signed'] or 0. for o in buys)
    signed = dict(creates=len(creates), buys=len(buys), sells=len(sells), cap_buys=P['cap_buys'], cap_sells=P['cap_sells'], usdt_signed_buys=usdt_signed, cap_usdt=P['cap_usdt'],
                  caps_held=bool(len(buys) <= P['cap_buys'] and len(sells) <= P['cap_sells'] and usdt_signed <= P['cap_usdt']))
    wei = {'BUY': 0, 'SELL': 0, 'sell_fee': 0, 'buy_fee_shares': 0}
    for r in recs:
        if r['kind'] == 'wallet' and isinstance(r['v'].get('input'), dict) and 'data' in r['v']['input']:
            dd = r['v']['input']['data']
            o = by_h.get(dd.get('orderHash'))
            if o is None:
                continue
            if dd['type'] == 'orderTransactionSubmitted':
                wei[o['side']] += int(dd['fill']['executedValueWei'])
            elif dd['type'] == 'orderTransactionSuccess':
                wei['sell_fee' if o['side'] == 'SELL' else 'buy_fee_shares'] += int((dd.get('fee') or {}).get('amountWei', 0))
    bal = {r['v']['when']: int(r['v']['balance']['wei']) for r in recs if r['kind'] == 'taker_balance'}
    cash_wei = wei['SELL'] - wei['sell_fee'] - wei['BUY']
    held = [r['v'] for r in recs if r['kind'] == 'taker_held']
    cash = dict(recomputed_usdt=cash_wei / 1e18, report_usdt=rep['cash_usdt'], onchain_delta_usdt=(bal['end'] - bal['start']) / 1e18,
                equals_onchain_to_the_wei=bool(bal['end'] - bal['start'] == cash_wei), equals_report=bool(abs(cash_wei / 1e18 - rep['cash_usdt']) < 1e-9),
                buy_fee_shares=wei['buy_fee_shares'] / 1e18, sell_fee_usdt=wei['sell_fee'] / 1e18, held=held)
    return dict(asof=datetime.now().astimezone().isoformat(timespec='seconds'), receipt_commit=T.full(RECEIPT_C), data_commit=T.full(DATA_C), request_commit=T.full(REQUEST_C),
                identities=idn, request=P, parity=dict(values_compared=n_cmp, mismatched=len(bad), orders=len(orders)), pooled=pooled, hold=hold,
                trigger=comp, late_triggers=late_s, fill_rule=dict(groups=rule, hold_live_ms=T.HOLD_LIVE, rows=fr), book_at_hold_end=book_s,
                moves_sent=sent_m, spearman_accept=sp, pooled_with_first_probe=pooled_both, signed=signed, cash=cash, buys=rows,
                orders=[{k: v for k, v in o.items() if k != 'sent_ms'} | dict(sent_ms=o['sent_ms']) for o in orders])


def run():
    require(not OUT.exists(), f'{OUT.name} exists (write-once)')
    files, request, idn = load_files()
    res = compute(files, request, idn)
    OUT.write_text(json.dumps(res, indent=1, default=float) + '\n')
    print('wrote', OUT.name, 'parity', res['parity'], 'trigger', {k: res['trigger'][k] for k in ('move_at_least_threshold', 'bought_the_moves_way', 'send_within_5ms')},
          'fill rule', {g: (v['n'], v['actual_AB']['agree']) for g, v in res['fill_rule']['groups'].items()})


# ------------------------------------------------------------------------------------------------------------------ selftest
def selftest():
    def refuses(fn, why):
        try:
            fn()
        except T.Refused as e:
            require(why in str(e), f'refused for another reason: {e}')
            return
        raise AssertionError(f'did not refuse: {why}')
    refuses(lambda: same_report(b'{}', hashlib.sha256(b'{ }').hexdigest()), 'is not the data branch')
    same_report(b'{}', hashlib.sha256(b'{}').hexdigest())
    # Tokyo's Mid rule: a crossed frame is dropped, a size-only frame adds no history, the move reads back to the last change <= E - 100
    bt = lambda recv, e, b, a: (recv, e, b, a, True)  # noqa: E731
    p = [bt(1, 1000, 100., 101.), bt(2, 1150, 100., 101.), bt(3, 1200, 101., 102.), bt(4, 1230, 102., 101.5), bt(5, 1240, 103.5, 104.5),
         (6, 1241, None, None, False), bt(7, 1300, 103., 104.)]
    m = tokyo_mid_moves(p, [5, 7])
    require(m[5]['move_100ms'] == 104. - 100.5 and m[5]['changes_100ms'] == 2, f'move at 1240: {m[5]}')
    require(m[7]['move_100ms'] == 103.5 - 101.5 and m[7]['changes_100ms'] == 2, f'move at 1300 (E 1200 exactly 100 back): {m[7]}')
    require(tokyo_mid_moves(p[:2], [2])[2]['move_100ms'] == 0., 'a size-only update: the mid then is the mid now')
    require(tokyo_mid_moves(p[:1], [1])[1]['move_100ms'] is None, 'no history 100 ms back: None, not 0')
    # the report's counts keep a crossed record and every change; the line's collapse would not
    ch = tokyo_changes(p)
    require([c[0] for c in ch] == [1, 3, 4, 5, 7], f'changes {ch}')
    require(tokyo_counts(ch, 3) == (4, 0), 'by receipt counts [send, +200 ms); by E nothing near a 3 ns send')
    ch2 = [(10_000_000, 1000, 1.), (11_000_000, 1000, 2.), (12_000_000, 1001, 3.), (300_000_000, 1300, 4.)]
    require(tokyo_counts(ch2, 10_000_000) == (3, 0) and tokyo_counts(ch2, 1_000_000_000)[1] == 3, f'counts {tokyo_counts(ch2, 10_000_000)}')
    require(count_bridge(ch2, 999.) == 3 and count_bridge(ch2, 1000.) == 1, 'the bridge: every change in (E, E + 200]')
    require(moves_of([dict(E=0.), dict(E=500.), dict(E=2000.), dict(E=2100.), dict(E=9000.)]) == 3 and moves_of([]) == 0, 'moves_of')
    # the book's best: an Up BUY reads the Up ask, a Down BUY 100 - the Up bid
    f = np.full((1, 22), -1.)
    f[0, [0, 2, 12]] = [1000., 40., 43.]
    require(best_at(f, 1000., 'BUY', 'Up') == 43. and best_at(f, 1000., 'BUY', 'Down') == 60. and best_at(f, 1000., 'SELL', 'Down') == 57., 'best_at')
    refuses(lambda: best_at(f, 999., 'BUY', 'Up'), 'no frame before')
    # the request's parameters come from its text; a reworded cap refuses
    req = ('at least **$3.5** over the last 100 ms ... 20 BUYs on BTC ... at least 10 s apart. Each BUY goes out at least 45 s before its window ends. '
           'Stop after 120 minutes ... At most 20 BUYs and 20 SELLs **in total, any checks included**, and at most 40 USDT committed to BUYs.')
    P = request_params(req)
    require(P == dict(min_move=3.5, buys=20, spacing_ms=10000., margin_ms=45000., run_ms=7200000., cap_buys=20, cap_sells=20, cap_usdt=40), f'params {P}')
    refuses(lambda: request_params(req.replace('any checks included', 'checks excluded')), "the request's parameters not found")
    keep = globals()['OUT']
    globals()['OUT'] = Path(__file__)
    try:
        refuses(run, 'write-once')
    finally:
        globals()['OUT'] = keep
    print('tokyo_bigmove_check selftest PASS')


if __name__ == '__main__':
    {'selftest': selftest, 'run': run}[sys.argv[1]]()
