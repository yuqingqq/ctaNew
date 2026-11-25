"""TAKE-EXIT FLOOR v1 (amendment.md): the frozen take + maker exit under STRICT58 with a break-even price floor on the exit, on the five
v3 days, valued at the 298 s mark and at settlement. BASE = the saved v3 STRICT58 ledgers (not re-run); PARITY = the floor code with
the floor out of reach on every 40th complete market, which must reproduce BASE.
Usage: PF_SLOT=claude predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_exit_floor_20261125/floor_replay.py \
       selftest|run|check|report
"""
import collections
import gzip
import hashlib
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
R = H.parents[2]
sys.path.insert(0, str(H.parent / 'take_exit_replay_20261125'))
import other_days as OD  # noqa: E402  (v3, frozen: imported, never edited)

MM, SR = OD.MM, OD.SR
ER, TR, X, LB = MM.ER, MM.TR, MM.X, SR.LB
BE, C = LB.BE, LB.C
OUT = X.DATA / 'take_exit_floor_20261125'
DAYS = OD.DAYS3
CPU_CAP = 900.
RSS_GUARD_MIB = 2500.
PARITY_EVERY = 40
PARITY_SLACK = 1000.
FLAGS = LB.STRUCT_FLAGS + ('floor',)
FLOOR_KEYS = ('floor', 'limit', 'ref')
SAME = ['fills', 'orders', 'commands', 'value', 'cash', 'position', 'signal_log', 'struct_log']
FILES = ['amendment.md', 'floor_replay.py']
EPS = 1e-9
LAST = []


# ---------------------------------------------------------------------------------------------------------------- the floor (pure parts)
def basis(fills, fee_of):
    """Average-cost basis (I, C) of the open position over fills in order. Long: C = cost paid incl. taker fees; short: C = proceeds net
    of fees. Adds average in, reductions keep the average, a flip restarts at the flipping fill (its fee pro rata)."""
    I = Cb = 0.
    for t, gid, s, q, p, it in fills:
        sg = 1. if s == 0 else -1.
        f = fee_of(int(gid), it)
        if abs(I) < EPS or (I > 0) == (sg > 0):
            Cb += q * p + f if sg > 0 else q * p - f
            I += sg * q
            continue
        close = min(q, abs(I))
        Cb *= (abs(I) - close) / abs(I)
        I += sg * close
        if abs(I) < EPS:
            I, Cb = 0., 0.
        rem = q - close
        if rem > EPS:
            fr = f * rem / q
            Cb = rem * p + fr if sg > 0 else rem * p - fr
            I = sg * rem
    return I, Cb


def limit_of(I, Cb, s, slack=0.):
    """The exit price limit on side s: the ask floor of a long (ceil of average cost), the bid ceiling of a short (floor of average
    net proceeds); None when side s does not close the position. slack moves it out of reach (PARITY)."""
    if s == 1 and I > EPS:
        return float(math.ceil(Cb / I - 1e-9)) - slack
    if s == 0 and I < -EPS:
        return float(math.floor(Cb / -I + 1e-9)) + slack
    return None


def fee_map(rec):
    """IOC gid -> fee (cents) from a ledger's IOC results; one fill per IOC."""
    fee = {}
    for x in rec['ioc_results']:
        if x['filled'] > EPS:
            C.require(int(x['gid']) not in fee, f"two fills on IOC {x['gid']}")
            fee[int(x['gid'])] = float(x['fee_c'])
    return fee


def known_limit(rec, fee, g, s, slack=0.):
    """The limit on side s recomputed from the ledger's client-known fills at g (fill time + notice <= g) -> (limit, I)."""
    I, Cb = basis([f for f in rec['fills'] if f[0] + BE.FILL_NOTICE <= g], lambda gid, it: fee[gid] if it < 0 else 0.)
    return limit_of(I, Cb, s, slack), I


# ---------------------------------------------------------------------------------------------------------------- the floor branch
class FloorBranch(SR.StrictBranch):
    """STRICT58's branch with the closing side's exit price limited by the break-even floor (amendment.md)."""

    def __init__(self, path, end, ident, bps, host, take, slack=0.):
        super().__init__(path, end, ident, bps, host, take, strict=True)
        self.slack = float(slack)
        self.floor_log = []
        self._lim = (None, None)
        self.settings = dict(self.settings, exit_floor=dict(rule='break_even_average_cost_v1', slack=self.slack))
        LAST.append(self)

    def limits(self, g):
        """{side: limit or None} from the client-known fills at g, cached per tick; the basis must reproduce the known position."""
        if self._lim[0] != g:
            fee = {gid: f for _, gid, f in self.E.fees}
            I, Cb = basis([x for x in self.E.fills if x[0] + BE.FILL_NOTICE <= g], lambda gid, it: fee[gid] if it < 0 else 0.)
            Ik = self.reserve(g)[0]
            C.require(abs(I - Ik) < 1e-6, f'basis position {I} differs from the known position {Ik} at {g}')
            self._lim = (g, {s: limit_of(I, Cb, s, self.slack) for s in (0, 1)})
        return self._lim[1]

    def structural(self, g):
        """pf_branch_ladder.LadderBranch.structural, except: distance is measured from the floored target, and an order beyond the
        current floor is cancelled (flag `floor`)."""
        fk = self.E.f[self.E.known]
        lim = self.limits(g)
        for s in (0, 1):
            for gid in self.unclosed[s]:
                o = self.E.o[gid]
                if o['req'] is not None or g <= o['submit']:
                    continue
                best, opp = fk[2 + s * 10], fk[2 + (1 - s) * 10]
                L = lim[s]
                ref = best if (L is None or best < 0) else (min(best, L) if s == 0 else max(best, L))
                behind = ref - o['price'] if s == 0 else o['price'] - ref
                cross = o['price'] >= opp if s == 0 else o['price'] <= opp
                flags = dict(no_book=bool(best < 0), distance=bool(behind > BE.HOLD + o['rung']), cross=bool(cross),
                             floor=bool(L is not None and (o['price'] > L + EPS if s == 0 else o['price'] < L - EPS)))
                if any(flags.values()):
                    self.E.request(gid, g, 'structural')
                    self.struct_log.append(dict(t=float(g), gid=int(gid), side=int(s), rung=float(o['rung']), price=float(o['price']), best=float(best), opp=float(opp),
                                                behind=float(behind), primary=next(f for f in FLAGS if flags[f]), mode=self.mode, effect=float(o['remove']),
                                                ack=float(o['remove'] + BE.CANCEL_ACK), received=float(fk[1]), frame=int(self.E.known), **flags, limit=L,
                                                ref=float(ref)))

    def side_commands(self, g, s, pol):
        """pf_ladder_async_fast.ProfileBranch.side_commands, except the rung prices start from the floored target on the closing side."""
        C.require(s not in self.acted, f'side {s} already processed at {g}'); self.acted.add(s); v = self.side_view(g, s)
        if not v['book_ok']:
            for o in v['orders']:
                if o['state'] == 'live':
                    self.fallbacks.append((float(g), s, 'live', 'feature_unavailable')); self.decisions.append((float(g), s, 'live', 'KEEP', o['gid']))
                    self._log(g, s, 'live', o['gid'], o['rung'], o['price'], None, 'KEEP', 'fallback')
            if v['vacancy'] > 0:
                self.fallbacks.append((float(g), s, 'absent', 'feature_unavailable')); self.decisions.append((float(g), s, 'absent', 'WAIT', None))
                self._log(g, s, 'slot', None, None, None, None, 'WAIT', 'fallback')
            self.check_position(g); return
        for o in [x for x in v['orders'] if x['state'] == 'live']:
            a, why = pol.live(self.side_view(g, s), o); C.require(a in ('KEEP', 'PULL'), f'a live action {a}')
            if a == 'PULL':
                self.E.request(o['gid'], g, 'policy')
            self.decisions.append((float(g), s, 'live', a, o['gid'])); self._log(g, s, 'live', o['gid'], o['rung'], o['price'], None, a, why); self.check_position(g)
        raw = v['bid'] if s == 0 else v['ask']
        L = self.limits(g)[s]
        best = raw if L is None else (min(raw, L) if s == 0 else max(raw, L))
        for r in range(self.levels):
            v = self.side_view(g, s)
            if v['vacancy'] <= 0:
                break
            price = best - LB.TICK * r if s == 0 else best + LB.TICK * r
            if any(o['price'] == price for o in v['orders']):
                continue
            if not 1 <= price <= 99:
                self.constraints.append((float(g), s, 'unavailable_price')); self._log(g, s, 'slot', None, r, price, None, 'WAIT', 'unavailable_price'); continue
            if self.own_cross and any((price >= p_ if s == 0 else price <= p_) for p_ in v['opp_prices']):
                self.constraints.append((float(g), s, 'own_cross_blocked')); self._log(g, s, 'slot', None, r, price, None, 'WAIT', 'own_cross_blocked'); continue
            a, q, why = pol.slot(v, r, price); gid = None
            if a == 'POST':
                C.require(LB.QMIN <= q <= self.q0 and q == math.floor(q), f'a POST quantity {q} outside the whole shares [{LB.QMIN}, {self.q0}]')
                room = (self.cap - v['position'] - v['B']) if s == 0 else (self.cap + v['position'] - v['A'])
                if q > room + LB.TOL:
                    self.constraints.append((float(g), s, 'cap')); a, why = 'WAIT', 'cap'
                elif g + self.prof['place'] >= self.end:
                    self.constraints.append((float(g), s, 'near_end')); a, why = 'WAIT', 'near_end'
                else:
                    gid = self.E.add_order(s, price, g, q, r); self.unclosed[s].append(gid)
                    self.floor_log.append(dict(t=float(g), gid=int(gid), side=int(s), rung=int(r), raw_best=float(raw), limit=L, price=float(price)))
            else:
                C.require(a == 'WAIT', f'a slot action {a}')
            self.decisions.append((float(g), s, 'absent', a, gid)); self._log(g, s, 'slot', gid, r, price, q, a, why); self.check_position(g)


def one(p, cache, old, edges, bps, slack):
    """One market through exit_replay.run_one (PASSIVE_EXIT) with the floor branch and the STRICT58 take delay."""
    orig, d0 = TR.SignalBranch, TR.TAKE_DELAY
    TR.SignalBranch = lambda *a, **k: FloorBranch(*a, slack=slack, **k)
    TR.TAKE_DELAY = 58.
    del LAST[:]
    try:
        rec = ER.run_one(p, cache, old, edges, bps, 'PASSIVE_EXIT')
    finally:
        TR.SignalBranch, TR.TAKE_DELAY = orig, d0
    C.require(len(LAST) == 1, 'one floor branch per run')
    C.require(all(abs(v['act'] - v['submit'] - 58.) < 1e-9 for v in rec['iocs'].values()), 'an IOC delay other than 58 ms')
    rec['floor_log'] = TR.clean(LAST[0].floor_log)
    rec['floor_slack'] = float(slack)
    del LAST[:]
    return rec


# ---------------------------------------------------------------------------------------------------------------- accounting
def account(rec, yup):
    """Round trips and held lots of one ledger with fees, at the mark and at settlement (amendment.md); identities enforced."""
    C.require(yup in (0., 1.), f'settlement outcome {yup}')
    fee = fee_map(rec)
    mark, S = float(rec['mark']), 100. * yup
    lots = collections.deque()
    rt = collections.Counter()
    cash = pos = fees = taker = maker = 0.
    for t, gid, s, q, p, it in rec['fills']:
        sg = 1. if s == 0 else -1.
        f = fee[int(gid)] if it < 0 else 0.
        fps = f / q
        cash -= sg * q * p; pos += sg * q; fees += f
        if it < 0:
            taker += q
        else:
            maker += q
        kind = 'take' if it < 0 else 'exit'
        n = q
        while n > EPS and lots and lots[0][0] != sg:
            ls, lq, lp, lf = lots[0]
            x = min(n, lq)
            gr, fe = ls * x * (p - lp), x * (lf + fps)
            for k in ('', kind + '_'):
                rt[k + 'gross_c'] += gr; rt[k + 'fees_c'] += fe; rt[k + 'shares'] += x
            n -= x; lq -= x
            if lq < EPS:
                lots.popleft()
            else:
                lots[0] = (ls, lq, lp, lf)
        if n > EPS:
            lots.append((sg, n, p, fps))
    held_mark = sum(s * q * (mark - p) for s, q, p, _ in lots)
    held_settle = sum(s * q * (S - p) for s, q, p, _ in lots)
    held_fees = sum(q * lf for _, q, _, lf in lots)
    m = dict(yup=yup, mark=mark, end_position=pos, taker_shares=taker, maker_shares=maker, fees_c=fees,
             rt_shares=rt['shares'], rt_gross_c=rt['gross_c'], rt_fees_c=rt['fees_c'], rt_net_c=rt['gross_c'] - rt['fees_c'],
             rt_exit_shares=rt['exit_shares'], rt_exit_net_c=rt['exit_gross_c'] - rt['exit_fees_c'],
             rt_take_shares=rt['take_shares'], rt_take_net_c=rt['take_gross_c'] - rt['take_fees_c'],
             held_shares=sum(q for _, q, _, _ in lots), held_fees_c=held_fees, held_mark_net_c=held_mark - held_fees,
             held_settle_net_c=held_settle - held_fees,
             held_max_loss_c=sum(q * (p + lf) if s > 0 else q * (100. - p + lf) for s, q, p, lf in lots),
             held_won_shares=sum(q for s, q, _, _ in lots if (s > 0) == (yup == 1.)),
             carry_c=pos * (S - mark))
    m['mark_net_c'] = m['rt_net_c'] + m['held_mark_net_c']
    m['settle_net_c'] = m['rt_net_c'] + m['held_settle_net_c']
    tol = 1e-6 * max(1., abs(cash) + abs(pos) * 100.)
    C.require(abs(cash - rec['cash']) < tol and abs(pos - rec['position']) < 1e-6, 'cash / position differ from the ledger')
    C.require(abs(fees - rec['fees_c']) < tol and abs(rt['fees_c'] + held_fees - fees) < tol, 'fees do not add up')
    C.require(abs(m['mark_net_c'] - rec['net_value_c']) < tol, 'round trips + held at the mark != ledger net')
    C.require(abs(m['settle_net_c'] - (cash + pos * S - fees)) < tol, 'round trips + held at settlement != cash + position x settlement - fees')
    C.require(abs(m['settle_net_c'] - m['mark_net_c'] - m['carry_c']) < tol, 'settlement - mark != carry')
    ff = X.fifo([tuple(f) for f in rec['fills']], mark)
    C.require(abs(ff['completed_c'] - rt['gross_c']) < tol and abs(ff['residual_c'] - held_mark) < tol and abs(ff['matched_shares'] - rt['shares']) < 1e-6,
              'fee-carrying FIFO differs from the frozen fifo')
    return m


def floor_stats(rec):
    fl = rec['floor_log']
    st = collections.Counter(x['primary'] for x in rec['struct_log'])
    return dict(exit_posts=len(fl), floor_bound_posts=sum(abs(x['price'] - x['raw_best']) > EPS for x in fl),
                floor_cancels=st.get('floor', 0), struct_by_primary=dict(st))


# ---------------------------------------------------------------------------------------------------------------- run
def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bound():
    OD.bound()
    for f in FILES:
        C.require(subprocess.check_output(['git', 'show', 'HEAD:' + str((H / f).relative_to(R))], cwd=R) == (H / f).read_bytes(), f'{f} is not committed')


def worlds_yup(day, mids, src, off, reg):
    """more_markets.worlds with the window's settlement outcome (same statuses, same path builder)."""
    data = src.load(day, list(mids), off)
    seen = set()
    for W in src.pairs(data):
        m = int(W[0]['m'])
        seen.add(m)
        ys = {float(W[s]['yup']) for s in W}
        yup = ys.pop() if len(ys) == 1 else None
        if int(reg.loc[m, 'decimal_precision']) != 2:
            yield m, 'precision_not_2', None, None, yup
            continue
        p, st, info = X.M.market_path(W, float(reg.loc[m, 'ws']) * 1000.)
        yield m, st, p, info, yup
    for m in mids:
        if m not in seen:
            yield int(m), 'loader_excluded', None, None, None


def write_once_gz(fn, rec):
    b = gzip.compress(json.dumps(rec, allow_nan=False).encode(), mtime=0)
    if fn.exists():
        C.require(fn.read_bytes() == b, f'{fn.name} exists with different bytes (determinism)')
    else:
        fn.write_bytes(b)
    return hashlib.sha256(b).hexdigest()


def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.


def market(day, h, m, p, info, yup, reg, edges, v3_row, with_parity):
    """One complete market: the FLOOR run, the saved BASE ledger, both accounted; PARITY when asked -> (row, FLOOR ledger, BASE ledger, parity)."""
    ws = float(reg.loc[m, 'ws']) * 1000.
    bps = float(reg.loc[m, 'fee_bps'])
    C.require(bps == 200., 'fee_bps')
    cache = OD.make_cache3(float(info['t_start']), float(info['end_rel']), OD.SEG_OF[day])
    old = dict(end=float(info['end_rel']), ident=MM.ident_of(day, m, ws, info, p))
    rf = one(p, cache, old, edges, bps, 0.)
    rf.update(day=day, mid=m, hour=int(h), arm='FLOOR', yup=yup)
    rf = json.loads(json.dumps(TR.clean(rf), allow_nan=False))
    bfn = OD.OUT3 / f'{m}_STRICT58.json.gz'
    C.require(sha_file(bfn) == v3_row['sha256'], f'BASE ledger {m} changed since v3')
    rb = json.loads(gzip.decompress(bfn.read_bytes()))
    C.require(rb['mark'] == rf['mark'] and rb['ident'] == rf['ident'], f'{m}: BASE and FLOOR paths differ')
    row = dict(day=day, mid=m, hour=int(h), yup=yup, base=account(rb, yup), floor=account(rf, yup), floor_stats=floor_stats(rf),
               v3_net_c=float(v3_row['metrics']['net_c']))
    par = None
    if with_parity:
        rp = json.loads(json.dumps(TR.clean(one(p, cache, old, edges, bps, PARITY_SLACK)), allow_nan=False))
        sl = [{k: v for k, v in x.items() if k not in FLOOR_KEYS} for x in rp['struct_log']]
        par = dict(day=day, mid=m, same={k: (sl if k == 'struct_log' else rp[k]) == rb[k] for k in SAME}, posts=len(rp['floor_log']))
    return row, rf, rb, par


def rehearse():
    """The production per-market path on two real markets (the flow page's typical market and the first complete market), nothing
    written: PARITY equals BASE, FLOOR respects its floor, BASE is flagged by the checker, accounting identities hold."""
    v3 = json.loads((OD.OUT3 / 'summary.json').read_text())
    v3_rows = {(r['day'], int(r['mid'])): r for r in v3['rows'] if r['setting'] == 'STRICT58'}
    first = next(c for c in v3['census'] if c['status'] == 'complete')
    edges, src = TR.T.P.frozen_edges(), MM.KP.Tape()
    for day, mid in (('20261102', 4125703), (first['day'], int(first['mid']))):
        reg = MM.registry(day).set_index('mid')
        for m, st, p, info, yup in worlds_yup(day, [mid], src, src.offset(day), reg):
            C.require(st == 'complete' and (day, m) in v3_rows, f'{m} {st}')
            row, rf, rb, par = market(day, int(reg.loc[m, 'hour']), m, p, info, yup, reg, edges, v3_rows[(day, m)], True)
            vf, vb = violations(rf, fee_map(rf)), violations(rb, fee_map(rb))
            print(json.dumps(dict(market=m, yup=yup, parity=par['same'], parity_posts=par['posts'], floor_orders=vf[0], floor_violations=len(vf[1]),
                                  base_orders=vb[0], base_violations=vb[1][:3], stats=row['floor_stats'],
                                  base={k: round(row['base'][k], 2) for k in ('mark_net_c', 'settle_net_c', 'rt_net_c', 'held_settle_net_c', 'held_shares')},
                                  floor={k: round(row['floor'][k], 2) for k in ('mark_net_c', 'settle_net_c', 'rt_net_c', 'held_settle_net_c', 'held_shares')},
                                  first_posts=rf['floor_log'][:4])))
            C.require(all(par['same'].values()) and not vf[1] and vf[2] == 0, f'{m}: rehearsal failed')


def run():
    resource.setrlimit(resource.RLIMIT_CPU, (int(CPU_CAP) + 60, int(CPU_CAP) + 61))
    bound()
    C.require(not (OUT / 'summary.json').exists(), 'summary.json exists (write-once)')
    OUT.mkdir(parents=True, exist_ok=True)
    v3 = json.loads((OD.OUT3 / 'summary.json').read_text())
    v3_census = {(c['day'], int(c['mid'])): c['status'] for c in v3['census']}
    v3_rows = {(r['day'], int(r['mid'])): r for r in v3['rows'] if r['setting'] == 'STRICT58'}
    edges = TR.T.P.frozen_edges()
    src, rows, census, parity = MM.KP.Tape(), [], [], []
    t0 = time.process_time()
    n_complete = 0
    for di, day in enumerate(DAYS):
        reg = MM.registry(day).set_index('mid')
        off = src.offset(day)
        for h, g in reg.groupby('hour'):
            for m, st, p, info, yup in worlds_yup(day, [int(x) for x in g.index], src, off, reg):
                ws = float(reg.loc[m, 'ws']) * 1000.
                if ws + 300000. >= OD.FWD_MS:
                    st = 'boundary_censored_forward_tail'
                census.append(dict(day=day, mid=m, hour=int(h), status=st, yup=yup))
                if st != 'complete':
                    continue
                C.require((day, m) in v3_rows, f'{day} {m} complete here but not in v3')
                row, rf, rb, par = market(day, h, m, p, info, yup, reg, edges, v3_rows[(day, m)], n_complete % PARITY_EVERY == 0)
                row['sha256'] = write_once_gz(OUT / f'{m}_FLOOR.json.gz', rf)
                if par is not None:
                    parity.append(par)
                n_complete += 1
                rows.append(row)
                del rf, rb
                C.require(rss_mib() < RSS_GUARD_MIB, f'RSS {rss_mib():.0f} MiB over the self-guard')
        spent = time.process_time() - t0
        print(f'{day}: {sum(r["day"] == day for r in rows)} markets, cpu {spent:.0f} s, rss {rss_mib():.0f} MiB', flush=True)
        if di == 0:
            proj = spent * len(DAYS) * 1.1
            rec = dict(asof='2026-11-25', first_day=day, spent_cpu_s=spent, projected_cpu_s=proj, cap_cpu_s=CPU_CAP, go=proj <= CPU_CAP)
            fn = H / 'projection.json'
            C.require(not fn.exists(), 'projection.json exists (write-once)')
            TR.dump(fn, rec)
            if not rec['go']:
                sys.exit(f'REFUSED: projected {proj:.0f} CPU s > cap {CPU_CAP:.0f}')
    mine = {(c['day'], int(c['mid'])): c['status'] for c in census}
    TR.dump(OUT / 'summary.json', dict(asof='2026-11-25', census=census, rows=rows, parity=parity, census_equal_v3=mine == v3_census,
                                       cpu_s=time.process_time() - t0, peak_rss_mib=rss_mib(), parity_every=PARITY_EVERY, parity_slack=PARITY_SLACK))
    print('census equal v3:', mine == v3_census, '| parity all same:', all(all(x['same'].values()) for x in parity), len(parity))


# ---------------------------------------------------------------------------------------------------------------- checks
def violations(rec, fee):
    """Maker exit orders priced beyond the floor recomputed at their submit time; plus orders with no closing basis."""
    bad, nob, n = [], 0, 0
    for o in rec['orders']:
        if o['why'] == 'IOC':
            continue
        n += 1
        L, I = known_limit(rec, fee, o['submit'], o['side'])
        if L is None:
            nob += 1
        elif (o['price'] < L - EPS) if o['side'] == 1 else (o['price'] > L + EPS):
            bad.append((int(o['gid']), float(o['price']), L))
    return n, bad, nob


def check():
    sm = json.loads((OUT / 'summary.json').read_text())
    v3r = json.loads((H.parent / 'take_exit_replay_20261125/v3_readings.json').read_text())
    res = {}

    def ok(name, cond, detail=''):
        res[name] = dict(ok=bool(cond), detail=detail)
        print(('PASS ' if cond else 'FAIL ') + name, detail)

    rows = sm['rows']
    ok('K1_census_equals_v3', sm['census_equal_v3'] and len(rows) == 408, f'{len(rows)} complete markets')
    d = max(abs(r['base']['mark_net_c'] - r['v3_net_c']) for r in rows)
    pooled = sum(r['base']['mark_net_c'] for r in rows) / 100.
    ok('K1_base_equals_v3', d < 1e-6 and abs(pooled - v3r['STRICT58']['net_usd']) < 1e-6, f'max |diff| {d:.2e} c, pooled {pooled:.2f} USD')
    ok('K2_parity', len(sm['parity']) >= 10 and all(all(x['same'].values()) for x in sm['parity']) and sum(x['posts'] for x in sm['parity']) > 0,
       f"{len(sm['parity'])} markets, {sum(x['posts'] for x in sm['parity'])} exit posts")
    b = collections.Counter()
    first_bad = None
    for r in rows:
        rf = json.loads(gzip.decompress((OUT / f"{r['mid']}_FLOOR.json.gz").read_bytes()))
        C.require(hashlib.sha256(gzip.compress(json.dumps(rf, allow_nan=False).encode(), mtime=0)).hexdigest() == r['sha256'], 'FLOOR ledger changed')
        rb = json.loads(gzip.decompress((OD.OUT3 / f"{r['mid']}_STRICT58.json.gz").read_bytes()))
        for arm, rec in (('floor', rf), ('base', rb)):
            fee = fee_map(rec)
            n, bad, nob = violations(rec, fee)
            b[f'{arm}_orders'] += n; b[f'{arm}_violations'] += len(bad); b[f'{arm}_no_basis'] += nob
            if arm == 'base' and bad and first_bad is None:
                first_bad = (r['mid'], bad[0])
            m2 = account(rec, r['yup'])
            b[f'{arm}_account_mismatch'] += any(abs(m2[k] - r[arm][k]) > 1e-6 for k in m2)
        fee_f = fee_map(rf)
        for x in rf['floor_log']:
            L, I = known_limit(rf, fee_f, x['t'], x['side'])
            b['logged'] += 1
            b['logged_mismatch'] += not (L == x['limit'] and x['price'] == (max(x['raw_best'], L) if x['side'] == 1 else min(x['raw_best'], L)))
            b['closing_rung0_violation'] += not (x['rung'] == 0 and ((x['side'] == 1 and I > EPS) or (x['side'] == 0 and I < -EPS)))
        b['delay'] += any(abs(v['act'] - v['submit'] - 58.) > 1e-9 for v in rf['iocs'].values())
        b['bound'] += r['floor_stats']['floor_bound_posts']
        b['yup_bad'] += r['yup'] not in (0., 1.)
    ok('K3_floor_respected', b['floor_orders'] > 0 and b['floor_violations'] == 0 and b['floor_no_basis'] == 0,
       f"{b['floor_orders']} FLOOR exit orders, {b['floor_violations']} below the floor, {b['floor_no_basis']} without a basis")
    ok('K3_falsifier_base_flagged', b['base_violations'] > 0, f"{b['base_violations']} of {b['base_orders']} BASE exit orders beyond break-even; first {first_bad}")
    ok('K4_logged_floor_rederived', b['logged'] > 0 and b['logged_mismatch'] == 0, f"{b['logged']} posts, {b['logged_mismatch']} mismatches")
    ok('K5_accounting_rederived', b['floor_account_mismatch'] == 0 and b['base_account_mismatch'] == 0, 'identities enforced inside account()')
    ok('K6_yup_delays_rung0_bound', b['yup_bad'] == 0 and b['delay'] == 0 and b['closing_rung0_violation'] == 0 and b['bound'] > 0,
       f"yup bad {b['yup_bad']}, delay {b['delay']}, closing/rung0 {b['closing_rung0_violation']}, floor-bound posts {b['bound']}")
    res['all_pass'] = all(v['ok'] for v in res.values() if isinstance(v, dict))
    fn = H / 'checks.json'
    C.require(not fn.exists(), 'checks.json exists (write-once)')
    TR.dump(fn, res)
    print('ALL PASS' if res['all_pass'] else 'SOME CHECK FAILED')
    sys.exit(0 if res['all_pass'] else 1)


# ---------------------------------------------------------------------------------------------------------------- readings
def quant(a, p):
    return float(np.quantile(np.asarray(a, float), p)) if len(a) else None


def arm_readings(rows, arm):
    M = [r[arm] for r in rows]
    tot = lambda k: float(sum(m[k] for m in M))
    by_day = collections.defaultdict(float)
    for r in rows:
        by_day[r['day']] += r[arm]['settle_net_c']
    held = [m for m in M if m['held_shares'] > EPS]
    hs = [m['held_settle_net_c'] for m in held]
    return dict(
        markets=len(M),
        money_usd=dict(round_trips=tot('rt_net_c') / 100, round_trips_closed_by_exit=tot('rt_exit_net_c') / 100, round_trips_closed_by_take=tot('rt_take_net_c') / 100,
                       held_to_settlement=tot('held_settle_net_c') / 100, total_at_settlement=tot('settle_net_c') / 100,
                       held_at_mark=tot('held_mark_net_c') / 100, total_at_mark=tot('mark_net_c') / 100, fees=tot('fees_c') / 100, carry_settlement_minus_mark=tot('carry_c') / 100),
        shares=dict(taken=tot('taker_shares'), round_trips=tot('rt_shares'), closed_by_exit=tot('rt_exit_shares'), closed_by_take=tot('rt_take_shares'),
                    held_to_settlement=tot('held_shares')),
        per_market_usd=dict(total_at_settlement=tot('settle_net_c') / 100 / len(M), total_at_mark=tot('mark_net_c') / 100 / len(M)),
        day_total_at_settlement_usd={k: v / 100 for k, v in sorted(by_day.items())},
        days_positive_at_settlement=int(sum(v > 0 for v in by_day.values())),
        risk=dict(markets_with_position=len(held), markets_with_1_share_or_more=sum(m['held_shares'] >= 1 for m in M),
                  held_share_of_taken=tot('held_shares') / tot('taker_shares'),
                  held_shares_per_market_p50=quant([m['held_shares'] for m in held], .5), held_shares_per_market_p90=quant([m['held_shares'] for m in held], .9),
                  held_shares_per_market_max=max((m['held_shares'] for m in held), default=0.),
                  max_loss_if_all_settle_against_usd=tot('held_max_loss_c') / 100, max_loss_one_market_usd=max((m['held_max_loss_c'] for m in M), default=0.) / 100,
                  held_settlement_usd_per_market_with_position=dict(mean=float(np.mean(hs)) / 100 if hs else None, std=float(np.std(hs)) / 100 if hs else None,
                                                                     p5=quant(hs, .05) / 100 if hs else None, min=min(hs) / 100 if hs else None, max=max(hs) / 100 if hs else None),
                  held_shares_won_share=tot('held_won_shares') / tot('held_shares') if tot('held_shares') > EPS else None))


def readings():
    sm = json.loads((OUT / 'summary.json').read_text())
    ck = json.loads((H / 'checks.json').read_text())
    C.require(ck['all_pass'], 'checks did not pass')
    rows = sm['rows']
    out = dict(asof='2026-11-25', markets=len(rows), days=list(DAYS), base=arm_readings(rows, 'base'), floor=arm_readings(rows, 'floor'))
    diff = {}
    for val in ('settle_net_c', 'mark_net_c', 'rt_net_c', 'held_settle_net_c'):
        d = np.array([r['floor'][val] - r['base'][val] for r in rows])
        dd = collections.defaultdict(float)
        for r, x in zip(rows, d):
            dd[r['day']] += x
        a = np.sort(np.abs(d))[::-1]
        diff[val] = dict(total_usd=float(d.sum()) / 100, by_day_usd={k: v / 100 for k, v in sorted(dd.items())}, days_positive=int(sum(v > 0 for v in dd.values())),
                         markets_better=int((d > 0.5).sum()), markets_worse=int((d < -0.5).sum()), markets_within_half_cent=int((np.abs(d) <= 0.5).sum()),
                         top10_abs_share=float(a[:10].sum() / a.sum()) if a.sum() > 0 else None)
    out['floor_minus_base'] = diff
    fs = collections.Counter()
    for r in rows:
        for k in ('exit_posts', 'floor_bound_posts', 'floor_cancels'):
            fs[k] += r['floor_stats'][k]
        for k, v in r['floor_stats']['struct_by_primary'].items():
            fs['cancel_' + k] += v
    out['floor_mechanics'] = dict(fs)
    fb, bb = out['floor'], out['base']
    out['readings'] = dict(
        floor_better_at_settlement=bool(fb['money_usd']['total_at_settlement'] > bb['money_usd']['total_at_settlement'] and diff['settle_net_c']['days_positive'] >= 4),
        floor_better_at_mark=bool(fb['money_usd']['total_at_mark'] > bb['money_usd']['total_at_mark'] and diff['mark_net_c']['days_positive'] >= 4),
        floor_round_trips_higher=bool(fb['money_usd']['round_trips'] > bb['money_usd']['round_trips']),
        floor_carries_more=bool(fb['shares']['held_to_settlement'] > bb['shares']['held_to_settlement']),
        base_settlement_total_positive=bool(bb['money_usd']['total_at_settlement'] > 0),
        floor_settlement_total_positive=bool(fb['money_usd']['total_at_settlement'] > 0))
    fn = H / 'readings.json'
    C.require(not fn.exists(), 'readings.json exists (write-once)')
    TR.dump(fn, out)
    print(json.dumps(out['readings']))


# ---------------------------------------------------------------------------------------------------------------- selftest
def selftest():
    fee = {0: 48., 2: 8., 3: 60.}
    fo = lambda gid, it: fee[gid] if it < 0 else 0.
    # buy 50 @ 52 (fee 48): average 52.96 -> floor 53
    f = [(158., 0, 0, 50., 52., -1)]
    I, Cb = basis(f, fo)
    assert abs(I - 50) < 1e-9 and abs(Cb - 2648.) < 1e-9 and limit_of(I, Cb, 1) == 53. and limit_of(I, Cb, 0) is None
    # a maker sale keeps the average; an add averages in
    f += [(847., 1, 1, 3.67, 52., 0), (900., 2, 0, 10., 60., -1)]
    I, Cb = basis(f, fo)
    avg = (46.33 * 52.96 + 608.) / 56.33
    assert abs(I - 56.33) < 1e-9 and abs(Cb / I - avg) < 1e-9 and limit_of(I, Cb, 1) == float(math.ceil(avg))
    # a flip: sell 60 @ 50 (fee 60) -> short 3.67 at proceeds 50 - 1 = 49 per share -> ceiling 49
    f += [(950., 3, 1, 60., 50., -1)]
    I, Cb = basis(f, fo)
    assert abs(I + 3.67) < 1e-9 and abs(Cb / -I - 49.) < 1e-9 and limit_of(I, Cb, 0) == 49. and limit_of(I, Cb, 1) is None
    assert limit_of(I, Cb, 0, PARITY_SLACK) == 49. + PARITY_SLACK
    # a non-integer average rounds up for a long and down for a short
    assert limit_of(1., 52.01, 1) == 53. and limit_of(1., 52., 1) == 52. and limit_of(-1., 48.99, 0) == 48.
    # fee-carrying FIFO on a synthetic ledger vs the frozen fifo; identities; settlement both ways
    rec = dict(fills=[[0., 0, 0, 50., 52., -1], [500., 1, 1, 46., 46., 0], [900., 2, 1, 10., 60., 0]], ioc_results=[dict(gid=0, filled=50., fee_c=48.)],
               mark=98., fees_c=48.)
    cash = -50 * 52. + 46 * 46. + 10 * 60.
    pos = 50. - 56.
    rec.update(cash=cash, position=pos, net_value_c=cash + pos * 98. - 48.)
    up, dn = account(rec, 1.), account(rec, 0.)
    assert abs(up['rt_shares'] - 50) < 1e-9 and abs(up['rt_exit_shares'] - 50) < 1e-9 and abs(up['held_shares'] - 6) < 1e-9
    assert abs(up['rt_net_c'] - (46 * (46 - 52.) + 4 * (60 - 52.) - 48.)) < 1e-9
    assert abs(up['held_settle_net_c'] - (-6 * (100 - 60.))) < 1e-9 and abs(dn['held_settle_net_c'] - (-6 * (0 - 60.))) < 1e-9
    assert abs(up['held_max_loss_c'] - 6 * 40.) < 1e-9 and up['held_won_shares'] == 0. and abs(dn['held_won_shares'] - 6.) < 1e-9
    # falsifiers: a wrong fee total and an unknown outcome must be refused
    for bad, why in ((dict(rec, fees_c=47.), 'fees'), (rec, 'outcome')):
        try:
            account(bad, 0.5 if why == 'outcome' else 1.)
        except Exception:
            continue
        raise AssertionError(f'account() accepted a bad {why}')
    # floor checker: an exit below the recomputed floor is flagged, one at the floor is not
    led = dict(fills=[[0., 0, 0, 50., 52., -1]], ioc_results=[dict(gid=0, filled=50., fee_c=48.)],
               orders=[dict(gid=0, why='IOC', side=0, price=52., submit=0.), dict(gid=1, why=None, side=1, price=46., submit=200.),
                       dict(gid=2, why=None, side=1, price=53., submit=200.), dict(gid=3, why=None, side=1, price=52., submit=50.)])
    n, bad, nob = violations(led, fee_map(led))
    assert n == 3 and [b_[0] for b_ in bad] == [1] and nob == 1, (n, bad, nob)
    print('selftest PASS')


if __name__ == '__main__':
    dict(selftest=selftest, rehearse=rehearse, run=run, check=check, report=readings)[sys.argv[1]]()
