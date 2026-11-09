"""Generate opt-in LZ for frozen v3; no frozen engine or input bounds change.

LZ copies native incremental_v1 pairs, not the older INCSIZE multi-tranche
prototype. Native parity applies to paired sizing and the existing XV handler;
the research fill/venue model and its calibration limits remain explicit.
"""
import hashlib
from pathlib import Path
from textwrap import dedent

HERE=Path(__file__).resolve().parent
SRC=HERE/'pf_step4_ladder_v3.py'
DST=HERE/'pf_step4_ladder_pairs.py'
FROZEN='d678e7c3c4e47c6d60ebc031edf8d7fc7317210d5e58ff301d92d7ae0ece5f17'

HELPERS=dedent('''\
    def lz_snapshot():
        # No pv() call here: observation must not inflate legacy view counters.
        out = {}
        for m in LANES:
            fl = [r for r in inflight[m] if not r['cancelled']]
            o = fl[-1] if fl else known[m]
            if o is not None:
                out[m] = dict(pt=o['pt'], size=o['size'], remaining=pv_rem(o), t_send=o['t_send'])
        return out

    def lz_trace(kind, now, **fields):
        callback = globals().get('_LZ_TRACE')
        if callback is not None:
            callback(dict(kind=kind, t=now, **fields))

    def lz_rejected(rec, now):
        # A terminal create rejection answers that create immediately. The
        # frozen handler keeps its timestamp until L_ANS and can wrongly block
        # a new order in that interval. Native rejected() releases the slot.
        side = rec['side']; stamp = rec['t_send']
        if now < stamp + L_ANS and stamp in sent[side]:
            sent[side].remove(stamp)
            live_ctr['lz_rejection_answer_release'] += 1
        lz_trace('rejected', now, id=rec['lz_id'])

''')

PATCHES=[
    ('from . import pf_io as io\n','from . import pf_io as io\nfrom .pf_ladder_pairs import PairMapper\n'),
    ('    LANES = (0, 1, 2, 3) if LDK is not None else (0, 1)\n',
     "    LZ = 'LZ' in rules\n"
     "    if LZ and (not SPLIT or not V2 or POLICYVIEW != 'known' or not LV['noamend'] or LEGACY_INTENT or LV['seq'] is not None or LV['hy'] or LV['bk'] is not None or LV['plock'] is not None or LV['qlock'] is not None or FILLPRIO != 'price' or LEGORDER != 'price' or REJRESET != 'on' or OWNCROSS != 'on'):\n"
     "        raise SystemExit('REFUSED: LZ requires known-view v2, concurrent split/no-amend, standard price priority and rejection reset')\n"
     '    BASE_LANES = (0, 1, 2, 3) if LDK is not None else (0, 1)\n'
     '    LANES = BASE_LANES + tuple(m + 4 for m in BASE_LANES) if LZ else BASE_LANES\n'
     "    pair = PairMapper(LV['min_usd']) if LZ else None\n"
     '    lz_ids = itertools.count() if LZ else None\n'),
    ('    NOQ = tuple((None, 0.0) for L_ in LANES)\n','    NOQ = tuple((None, 0.0) for L_ in BASE_LANES)\n'),
    ('    def desired(t):\n', ''.join('    '+line+'\n' if line else '\n' for line in HELPERS.rstrip().split('\n'))+'    def desired(t):\n'),
    ("                L_ = s_ + 2; o_ = cur_order(L_); flag = ba if s_ == 0 else sa\n",
     "                L_ = s_ + 2; o_ = cur_order(L_); flag = ba if s_ == 0 else sa\n"
     "                if LZ:\n"
     "                    logical = pair.rung(L_, lz_snapshot(), pend, intent)\n"
     "                    o_ = dict(pt=logical[0], size=logical[1]) if logical is not None else None\n"),
    ("        if kind == 'D':\n            want = desired(t)\n",
     "        if kind == 'D':\n            want = desired(t)\n"
     "            if LZ:\n"
     "                logical = tuple(want) + ((None, 0.0),) * (4 - len(want))\n"
     "                want = pair.wants(logical, lz_snapshot(), pend, intent)\n"
     "                if pair.last_resizes: live_ctr['lz_resize'] += pair.last_resizes\n"
     "                if pair.last_withheld: live_ctr['lz_topup_withheld'] += pair.last_withheld\n"
     "                lz_trace('D', t, logical=logical, physical=want, total=list(pair.total), filled=list(pair.filled))\n"),
    ('                reserve_cancel(cur, t_rm)\n',
     "                reserve_cancel(cur, t_rm)\n"
     "                if LZ:\n"
     "                    cur['lz_cancelled'] = True\n"
     "                    lz_trace('cancel', t, id=cur['lz_id'], lane=lane, end=t_rm)\n"),
    ("            inflight[lane].append(rec); heapq.heappush(ev, (t + L_PL, 0, seq, 'P', rec)); seq += 1\n",
     "            if LZ:\n"
     "                rec['lz_id'] = next(lz_ids)\n"
     "                lz_trace('place', t, id=rec['lz_id'], lane=lane, tick=p, size=zsz, answer=t+L_ANS)\n"
     "            inflight[lane].append(rec); heapq.heappush(ev, (t + L_PL, 0, seq, 'P', rec)); seq += 1\n"),
    ("                live_ctr['postonly_refused'] += 1\n                cancel_wait.pop(id(rec), None)",
     "                live_ctr['postonly_refused'] += 1\n"
     "                if LZ: lz_rejected(rec, t)\n"
     "                cancel_wait.pop(id(rec), None)"),
    ("                    live_ctr['postonly_refused_own'] += 1\n                    cancel_wait.pop(id(rec), None)\n",
     "                    live_ctr['postonly_refused_own'] += 1\n"
     "                    if LZ: lz_rejected(rec, t)\n"
     "                    cancel_wait.pop(id(rec), None)\n"),
    ("            if id(rec) in cancel_wait:\n",
     "            if LZ:\n"
     "                o.update(t_send=rec['t_send'], lz_id=rec['lz_id'], lz_cancelled=rec['cancelled'])\n"
     "                lz_trace('accepted', t, id=rec['lz_id'])\n"
     "            if id(rec) in cancel_wait:\n"),
    ("        elif kind == 'R':                                   # throttle: retry a refused place while its intent stands\n            lane, key = pl\n",
     "        elif kind == 'R':                                   # throttle: retry a refused place while its intent stands\n            lane, key = pl\n"
     "            if LZ: lz_trace('retry', t, lane=lane, key=key)\n"),
    ("            o_k['kq'] = o_k.get('kq', 0.0) + abs(qd)\n",
     "            o_k['kq'] = o_k.get('kq', 0.0) + abs(qd)\n"
     "            if LZ:\n"
     "                pair.notify(lane, o_k['pt'], abs(qd), o_k.get('lz_cancelled', False))\n"
     "                lz_trace('fill', t, id=o_k['lz_id'], qty=abs(qd))\n"),
    ("                known[lane] = None; intent[lane] = None     # slot vacated as notified: re-place now (D1's repost)\n",
     "                known[lane] = None\n"
     "                if LZ:\n"
     "                    partner = pair.partner(lane)\n"
     "                    if pair.active(partner, o_k['pt'], lz_snapshot(), pend, intent):\n"
     "                        intent[lane] = (None, 0.0)\n"
     "                        continue  # member depleted; the logical pair remains active\n"
     "                    pair.reset(lane); intent[partner] = None\n"
     "                intent[lane] = None     # whole logical order vacated: re-place now\n"),
    ("            sh_left = sh\n",
     "            if LZ and any(o_.get('lane', 0) >= 4 for o_ in book_):\n"
     "                book_.sort(key=lambda o_: (-o_['pt'] if side == 0 else o_['pt'], o_['t0'], o_['t_send'], o_['lz_id']))\n"
     "            sh_left = sh\n"),
]

def generate(source=None):
    source=SRC.read_text() if source is None else source
    if hashlib.sha256(source.encode()).hexdigest()!=FROZEN:raise ValueError('frozen v3 source hash mismatch')
    for old,new in PATCHES:
        if source.count(old)!=1:raise ValueError('LZ patch anchor mismatch: '+old[:90])
        source=source.replace(old,new,1)
    return '# GENERATED by pf_ladder_pairs_build.py; frozen v3 '+FROZEN+'; opt-in LZ incremental pairs.\n'+source

def main():
    result=generate();DST.write_text(result)
    print('generated',DST,hashlib.sha256(result.encode()).hexdigest())

if __name__=='__main__':main()
