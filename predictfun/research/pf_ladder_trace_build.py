"""Generate pf_step4_ladder_trace.py = the FROZEN ladder engine (pm_ladder_v1: pf_step4_ladder.py sha256 c682229e...) plus read-only
trace hooks (pf_ladder_tracer.py). The frozen file is never edited. Refuses unless the source hash matches and every anchor occurs
exactly once; the generated file records both hashes in its header. Request: exec-port session, 2026-11-06 (lockstep decision trace).
Usage: python3 -m predictfun.research.pf_ladder_trace_build
"""
import hashlib, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = f'{HERE}/pf_step4_ladder.py'; DST = f'{HERE}/pf_step4_ladder_trace.py'
FROZEN = 'c682229e4fbe606b217cfa633816bcba67bce701c7e690b4cb8be9aa7ff6e37f'
POST = ('_TR.post(knet, kcash, intent, known, orders, inflight, zombies, TB, cancel_wait, requests, fills, withheld, live_ctr, pend)')
PATCHES = [   # (anchor, replacement); each anchor must occur exactly once
    ('from . import pf_io as io\n', 'from . import pf_io as io\nfrom . import pf_ladder_tracer as _TRM\n_TR = _TRM.Tracer()\n'),
    ('BCV = {v_: np.concatenate(x)[o] for v_, x in cbv.items()}\n',
     'BCV = {v_: np.concatenate(x)[o] for v_, x in cbv.items()}\n_TR.set_inputs(KB, MB, BC, PREDVB, PREDVA)\n'),
    ('    ws = float(R.ws) * 1000.0\n', '    ws = float(R.ws) * 1000.0\n    _TR.win_begin(mid, ws, rules, size, F, kf)\n'),
    ('    def desired(t):\n        rel = (t - ws) / 1000.0\n', '    def desired(t):\n        rel = (t - ws) / 1000.0\n        _TR.dclear()\n'),
    ('        btc_ = (ba, sa)                                     # Binance-only flags (LH: a resting rung is pulled only by these)\n',
     '        btc_ = (ba, sa)                                     # Binance-only flags (LH: a resting rung is pulled only by these)\n'
     '        _TR.dset(btc=(ba, sa))\n'),
    ('        live = {s: sum(pv_rem(o_) for o_ in (pv(L_) for L_ in LANES_OF[s]) if o_) for s in (0, 1)}\n',
     '        live = {s: sum(pv_rem(o_) for o_ in (pv(L_) for L_ in LANES_OF[s]) if o_) for s in (0, 1)}\n'
     '        _TR.dset(rel=rel, b1=b1, b1s=bs, a1=a1, a1s=as_, q_micro=tuple(q), ba=ba, sa=sa, vv=tuple(vv_), live=(live[0], live[1]))\n'),
    ('        return tuple(out_)\n', '        _TR.dset(rung=tuple(rung), rz=tuple(rz), zs=tuple(zs), q_final=tuple(q))\n        return tuple(out_)\n'),
    ('    while ev:\n        t, pr, _, kind, pl = heapq.heappop(ev)\n',
     f'    while ev:\n        {POST}\n        t, pr, _, kind, pl = heapq.heappop(ev)\n'
     '        _TR.ev(t, pr, _, kind, pl, withheld, live_ctr, intent, requests, fills, knet)\n'),
    ("        if kind == 'D':\n            want = desired(t)\n", "        if kind == 'D':\n            want = desired(t)\n            _TR.e_set(want=want)\n"),
    ('            if cur is None and p is None:\n                continue\n',
     "            if cur is None and p is None:\n                _TR.e_set(out='noop'); continue\n"),
    ('                continue                                      # same price and size: keep\n',
     "                _TR.e_set(out='keep'); continue               # same price and size: keep\n"),
    ('                reserve_cancel(cur, t_rm)\n',
     '                reserve_cancel(cur, t_rm)\n                _TR.e_set(cancel=dict(t_rm=t_rm, pending=pending_cur, cur=cur))\n'),
    ('            if p is None:\n                continue\n            wc = worst_case(',
     "            if p is None:\n                _TR.e_set(out='pull'); continue\n            wc = worst_case("),
    ("                    live_ctr['throttled'] += 1\n                    heapq.heappush(ev, (t + pd_['bo'], 0, seq, 'R', (lane, key))); seq += 1\n",
     "                    live_ctr['throttled'] += 1\n                    heapq.heappush(ev, (t + pd_['bo'], 0, seq, 'R', (lane, key))); seq += 1\n"
     "                    _TR.e_set(throttled=dict(pb=TB['pb'], open_n=open_n, sent_n=len(sent[side]), bo=pd_['bo'], key=key))\n"),
    ("            inflight[lane].append(rec); heapq.heappush(ev, (t + L_PL, 0, seq, 'P', rec)); seq += 1\n",
     "            inflight[lane].append(rec); heapq.heappush(ev, (t + L_PL, 0, seq, 'P', rec)); seq += 1\n"
     "            _TR.e_set(place=dict(rec=rec, pb=TB['pb'], open_n=(open_n if TH else None), sent_n=(len(sent[side]) if TH else None)))\n"),
    ("                live_ctr['postonly_refused'] += 1\n",
     "                live_ctr['postonly_refused'] += 1\n"
     "                _TR.e_set(p_out=dict(r='venue_refused', b1=f['b1_p'], a1=f['a1_p'], frame_t=f['t_ms']))\n"),
    ("                    live_ctr['postonly_refused_own'] += 1\n                    cancel_wait.pop(id(rec), None)\n",
     "                    live_ctr['postonly_refused_own'] += 1\n                    cancel_wait.pop(id(rec), None)\n"
     "                    _TR.e_set(p_out=dict(r='own_refused', opp=[x['pt'] for x in op_]))\n"),
    ("            o['ahead0'] = o['ahead']\n",
     "            o['ahead0'] = o['ahead']\n"
     "            _TR.e_set(p_out=dict(r=('zombie' if (rec['cancelled'] and not SEQ) else 'live'), o=o, frame_t=(f['t_ms'] if f is not None else None)))\n"),
    ("            if pend[lane] is not None and pend[lane]['key'] == key and intent[lane] == key:\n",
     "            if pend[lane] is not None and pend[lane]['key'] == key and intent[lane] == key:\n                _TR.e_set(resend=True)\n"),
    ("    mids = np.array([((r['a1_p'] + r['b1_p']) / 2.0) if (r['b1_p'] >= 0 and r['a1_p'] > r['b1_p']) else np.nan for r in F])\n",
     f"    {POST}\n"
     "    mids = np.array([((r['a1_p'] + r['b1_p']) / 2.0) if (r['b1_p'] >= 0 and r['a1_p'] > r['b1_p']) else np.nan for r in F])\n"),
    ('    return dict(pnl=cash + net * y, net_end=net, n_fills=len(fills),',
     '    _TR.win_end(dict(pnl=cash + net * y, net_end=net, mtm_stop=mtm_stop, carry_stop=carry_stop, n_fills=len(fills), '
     'counters=dict(live_ctr), withheld=dict(withheld)), requests, fills)\n'
     '    return dict(pnl=cash + net * y, net_end=net, n_fills=len(fills),'),
]


def main():
    src = open(SRC, 'rb').read()
    h = hashlib.sha256(src).hexdigest()
    if h != FROZEN:
        sys.exit(f'REFUSED: {SRC} sha256 {h} is not the frozen engine {FROZEN}')
    s = src.decode()
    for a, b in PATCHES:
        n = s.count(a)
        if n != 1:
            sys.exit(f'REFUSED: anchor occurs {n} times: {a[:80]!r}')
        s = s.replace(a, b)
    s += '_TR.close()\n'
    hdr = (f'# GENERATED by pf_ladder_trace_build.py from pf_step4_ladder.py sha256 {FROZEN} (frozen engine of pm_ladder_v1)\n'
           f'# plus read-only trace hooks (pf_ladder_tracer.py), {len(PATCHES)} anchored insertions. Do not edit; regenerate.\n')
    out = hdr + s
    open(DST, 'w').write(out)
    print(f'wrote {DST} sha256 {hashlib.sha256(out.encode()).hexdigest()} ({len(PATCHES)} insertions)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
