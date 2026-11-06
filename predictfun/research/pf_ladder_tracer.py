"""Lockstep decision trace for the ladder engine (exec-port session request, 2026-11-06; candidate pm_ladder_v1).
Used only by pf_step4_ladder_trace.py, which pf_ladder_trace_build.py generates from the frozen pf_step4_ladder.py by anchored
insertions. The hooks only read engine state: they never call pv() / policy_open() / cur_order() (those count or expire), and
never write to engine structures.
Env: PF_TRACE_DIR (output dir, required), PF_TRACE_MIDS (comma list: full JSONL for these windows; every window gets coverage flags).
Per traced window <dir>/<mid>.jsonl, one JSON object per line:
  rec=window   mid, ws, rules, size
  rec=frame    i, t_ms, recv_ns, kf (knowledge time after the max-accumulate), b1..b5 / a1..a5 price and size
  rec=binance  KB / MB / BC kept rows from the last row before ws - 2 s to ws + 300 s (k = knowledge ms, m = mid, c = G2 class)
  rec=v        side bid / ask: tau and value arrays exactly as the engine reads them (PREDVB / PREDVA, incl. the -1 rows)
  rec=event    n, t, prio, seq, kind (D / A / P / K / R / T / U), payload, handler outcome (D: want + decision internals; A: keep / noop /
               pull / cancel / throttled / place; P: venue_refused / own_refused / live / zombie; R: resend), and the state after the
               handler: knet, kcash, intent, known, orders, inflight, zombies, pb, cancel reservations, pend, counter deltas, new
               requests and new fills
  rec=requests the window's request list as the engine returns it (t_ms, side, kind, lane)
  rec=end      pnl, net_end, mtm_stop, carry_stop, n_fills, counters
NaN -> null, +-inf -> "inf" / "-inf".
"""
import collections, json, math, os
import numpy as np

FLAG_W = ('G6_soft', 'G6_hard', 'LD_G6_soft', 'LD_G6_hard', 'GS_nocover', 'BTC', 'LD_dup_touch', 'MIN_USD', 'LD_MIN_USD')
FLAG_L = ('postonly_refused', 'postonly_refused_own', 'throttled', 'place_delayed', 'zombie_fills', 'zombies', 'acted_on_unknown_fill',
          'rej_reset', 'view_filled_unknown', 'qact_raised', 'prio_reordered')


def c(x):
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else (None if math.isnan(x) else ('inf' if x > 0 else '-inf'))
    if isinstance(x, dict):
        return {str(k): c(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [c(v) for v in x]
    if isinstance(x, np.ndarray):
        return [c(v) for v in x.tolist()]
    return repr(x)


OKEYS = ('lane', 'side', 'pt', 'size', 'rem', 'ahead', 'ahead0', 't0', 'ci', 't_send', 't_act', 'cancelled', 't_rm', 't_end', 'kq')


def o(x):
    """an order / create record without object references."""
    if x is None:
        return None
    return {k: c(x[k]) for k in OKEYS if k in x}


class Tracer:
    def __init__(self):
        self.dir = os.environ['PF_TRACE_DIR']; os.makedirs(self.dir, exist_ok=True)
        self.mids = {int(m) for m in os.environ.get('PF_TRACE_MIDS', '').split(',') if m}
        self.flags = {}; self.fh = None; self.e = None; self.d = {}; self.mid = None; self.inp = None

    def set_inputs(self, KB, MB, BC, PVB, PVA):
        self.inp = (KB, MB, BC, PVB, PVA)

    def _w(self, obj):
        self.fh.write(json.dumps(c(obj), separators=(',', ':')) + '\n')

    def win_begin(self, mid, ws, rules, size, F, kf):
        if self.mid is not None and self.fh is not None:          # a window that returned before win_end (not expected here)
            self.fh.close(); self.fh = None
        self.mid = int(mid); self.ws = float(ws); self.fl = collections.Counter(); self.e = None; self.n = 0
        self.nreq_ev = 0; self.nfill_ev = 0
        self.full = self.mid in self.mids
        if not self.full:
            return
        self.fh = open(f'{self.dir}/{self.mid}.jsonl', 'w')
        self._w(dict(rec='window', mid=self.mid, ws=self.ws, rules=sorted(rules), size=size))
        for i, (r, k) in enumerate(zip(F, kf)):
            self._w(dict(rec='frame', i=i, t_ms=r['t_ms'], recv_ns=int(r['recv_ns']), kf=k,
                         **{f'{s}{j}_{q}': r[f'{s}{j}_{q}'] for s in 'ba' for j in range(1, 6) for q in 'ps'}))
        KB, MB, BC, PVB, PVA = self.inp
        a = max(int(np.searchsorted(KB, ws - 2000.0, side='left')) - 1, 0); b = int(np.searchsorted(KB, ws + 300_000.0, side='right'))
        self._w(dict(rec='binance', k=KB[a:b], m=MB[a:b], c=BC[a:b]))
        for side, P in (('bid', PVB), ('ask', PVA)):
            v = P.get(mid)
            self._w(dict(rec='v', side=side, tau=(v[0] if v is not None else []), value=(v[1] if v is not None else [])))

    def dclear(self):
        self.d = {}

    def dset(self, **kw):
        self.d.update(kw)

    def e_set(self, **kw):
        self.e.update(kw)

    def ev(self, t, pr, seq, kind, pl, withheld, live_ctr, intent, requests, fills, knet):
        self.n += 1
        self.e = dict(rec='event', n=self.n, t=t, prio=pr, seq=seq, kind=kind)
        self._w0 = {k: withheld.get(k, 0) for k in withheld}; self._l0 = {k: live_ctr.get(k, 0) for k in live_ctr}
        self._nreq = len(requests); self._nfill = len(fills); self._k0 = knet
        if not self.full:
            if kind == 'K':
                self.e['done'] = bool(pl[2]); self.e['lane'] = int(pl[0])
            return
        if kind == 'D':
            self.e['frame_i'] = pl
        elif kind == 'A':
            self.e['lane'], self.e['p'], self.e['zsz'] = pl
        elif kind == 'P':
            self.e['rec_'] = o(pl)
        elif kind == 'K':
            self.e.update(lane=pl[0], qd=pl[1], done=pl[2], order=o(pl[3]))
        elif kind == 'R':
            self.e.update(lane=pl[0], key=pl[1])
        elif kind == 'U':
            self.e['side'] = pl
        else:
            self.e['trade'] = dict(t=pl[0], side=pl[1], pt=pl[2], shares=pl[3], timed=pl[4])

    def post(self, knet, kcash, intent, known, orders, inflight, zombies, TB, cancel_wait, requests, fills, withheld, live_ctr, pend):
        e = self.e
        if e is None:
            return
        self.e = None
        dw = {k: withheld.get(k, 0) - self._w0.get(k, 0) for k in withheld if withheld.get(k, 0) != self._w0.get(k, 0)}
        dl = {k: live_ctr.get(k, 0) - self._l0.get(k, 0) for k in live_ctr if live_ctr.get(k, 0) != self._l0.get(k, 0)}
        new_req = requests[self._nreq:]; new_fill = fills[self._nfill:]
        self.nreq_ev += len(new_req); self.nfill_ev += len(new_fill)
        fl = self.fl
        for k in FLAG_W:
            fl[k] += dw.get(k, 0)
        for k in FLAG_L:
            fl[k] += dl.get(k, 0)
        if e['kind'] == 'K':
            fl['full_fill' if e['done'] else 'partial_fill'] += 1
            fl[f"k_lane{e['lane']}"] += 1
        if e['kind'] == 'D' and 'want' in e and 'b1' in self.d:
            w = e['want']
            for s_ in (0, 1):
                L_ = s_ + 2
                if L_ < len(w) and known.get(L_) is not None:
                    best = self.d['b1'] if s_ == 0 else self.d['a1']
                    if w[L_][0] == known[L_]['pt'] == best:
                        fl['rung_kept_at_best'] += 1
            vv = self.d.get('vv', (None, None))
            if any(v is not None and v == -1.0 for v in vv):
                fl['startknown_v_minus1'] += 1
            if e['kind'] == 'D' and self.d.get('rel') is not None and 59.0 <= self.d['rel'] < 60.0:
                fl['d_last_second'] += 1
        for r in new_fill:
            fl['fill_lane%d' % r[8]] += 1
        fl['events'] += 1
        if not self.full:
            return
        if e['kind'] == 'D':
            e['d'] = dict(self.d)
        e['post'] = dict(knet=knet, kcash=kcash, intent={L_: intent[L_] for L_ in intent}, known={L_: o(known[L_]) for L_ in known},
                         orders={L_: o(orders[L_]) for L_ in orders}, inflight={L_: [o(r) for r in inflight[L_]] for L_ in inflight},
                         zombies={s_: [o(z) for z in zombies[s_]] for s_ in zombies}, pb=TB['pb'], tb_last=TB['last'],
                         cancel_wait=[dict(order=o(v[0]), ack=v[1]) for v in cancel_wait.values()],
                         pend={L_: (None if pend[L_] is None else dict(pend[L_])) for L_ in pend})
        if dw:
            e['withheld_delta'] = dw
        if dl:
            e['counters_delta'] = dl
        if new_req:
            e['new_requests'] = [dict(t_ms=r[0], side=r[1], kind=r[2], lane=r[3]) for r in new_req]
        if new_fill:
            e['new_fills'] = [dict(t=r[0], side=r[1], pt=r[2], q=r[3], kind=r[4], timed=r[5], osz=r[6], zombie=r[7], lane=r[8], ahead0=r[9], age_ms=r[10])
                              for r in new_fill]
        for k in ('cancel', 'place', 'p_out'):
            if k in e:
                v = dict(e[k])
                for kk in ('cur', 'rec', 'o'):
                    if kk in v:
                        v[kk] = o(v[kk])
                e[k] = v
        self._w(e)

    def win_end(self, res, requests, fills):
        if self.nreq_ev != len(requests) or self.nfill_ev != len(fills):
            raise SystemExit(f'TRACE RECONCILIATION FAILED {self.mid}: requests {self.nreq_ev} vs {len(requests)}, fills {self.nfill_ev} vs {len(fills)}')
        self.flags[self.mid] = dict(self.fl)
        if self.full:
            self._w(dict(rec='requests', rows=[dict(t_ms=r[0], side=r[1], kind=r[2], lane=r[3]) for r in requests]))
            self._w(dict(rec='end', **res))
            self.fh.close(); self.fh = None
        self.mid = None

    def close(self):
        with open(f'{self.dir}/flags.json', 'w') as f:
            json.dump(c(self.flags), f, indent=0)
