"""Generate opt-in LC50 paired quote retention, without altering B4 or frozen v3."""
import hashlib
from pathlib import Path
from textwrap import dedent

from . import pf_ladder_pairs_build as pairs

HERE = Path(__file__).resolve().parent
DST = HERE/'pf_step4_ladder_continuity.py'
PARENT = '1097826bf128191b4544dd3cc6f92486cc6bc6ff513c99913473b0a28187b41f'

HELPERS = dedent('''\
def lc_record(kind, t, lane, **fields):
    callback = globals().get('_LC_TRACE')
    if callback is not None:
        callback(dict(kind=kind, t=t, lane=lane, **fields))

def lc_context(t):
    # LC50's supported rules have no stateful desired() cooldown mutations.
    # Read-only PM relaxation must not change legacy diagnostic counters.
    saved_withheld, saved_live = dict(withheld), dict(live_ctr)
    relaxed = desired(t, relax_pm=True)
    withheld.clear(); withheld.update(saved_withheld)
    live_ctr.clear(); live_ctr.update(saved_live)
    return relaxed

def lc_order(lane, ident):
    for member in (lane, lane + 4):
        o = known[member]
        if o is not None and o['lz_id'] == ident and not o.get('lz_cancelled', False):
            if any(not r['cancelled'] for r in inflight[member]):
                return None
            return o
    return None

def lc_reason(t, lane, o, relaxed, opening=False):
    rel = (t-ws)/1000.
    # Retention stops at reduce-only; no new policy beyond the studied phase.
    if rel >= 300 or rel >= stop or (LV['ro'] is not None and rel >= LV['ro']):
        return 'phase_end'
    if o is None or pv_rem(o) <= 1e-9:
        return 'order_unavailable'
    value = model_value(mid, t, PREDVB if lane == 0 else PREDVA)
    line = minute_line(PMK, rel)
    if value is None or not math.isfinite(value):
        return 'missing_score'
    if value <= line - .01:
        return 'lower_line'
    if opening and value > line:
        return 'not_soft_exit'
    if relaxed[lane][0] != o['pt'] or relaxed[lane][1] + 1e-9 < pv_rem(o):
        return 'price_size_or_hard_guard'
    return None

def lc_close(lane, t, reason):
    hold = lc_holds[lane]
    if hold is not None:
        lc_record('close', t, lane, id=hold['id'], start=hold['start'], deadline=hold['deadline'], reason=reason)
        lc_holds[lane] = None

def lc_pull(lane, t, reason):
    lc_close(lane, t, reason)
    pair.reset(lane)
    lz_trace('pull', t, lane=lane)
    for member in (lane, lane+4):
        intent[member] = (None, 0.)
        heapq.heappush(ev, (t, 0, next(SEQX), 'A', (member, None, 0.)))

def lc_observe(t):
    if not any(lc_holds):
        return
    relaxed = lc_context(t)
    for lane in (0, 1):
        hold = lc_holds[lane]
        if hold is None:
            continue
        reason = lc_reason(t, lane, lc_order(lane, hold['id']), relaxed)
        if reason is None and t >= hold['deadline']:
            reason = 'deadline'
        if reason is not None:
            lc_pull(lane, t, reason)

def lc_adjust(t, raw):
    relaxed = lc_context(t)
    out = list(raw)
    for lane in (0, 1):
        hold = lc_holds[lane]
        if hold is not None:
            o = lc_order(lane, hold['id'])
            if raw[lane][0] is not None:
                lc_close(lane, t, 'same_price_reopen' if o is not None and raw[lane][0] == o['pt'] else 'other_price_reopen')
            else:
                reason = lc_reason(t, lane, o, relaxed)
                if reason is not None:
                    lc_close(lane, t, reason)
        if lc_holds[lane] is None and lc_previous[lane][0] is not None and raw[lane][0] is None:
            candidates = [o for member in (lane, lane+4) if (o := known[member]) is not None
                and lc_order(lane, o['lz_id']) is o and o['pt'] == lc_previous[lane][0]
                and o['size'] <= 5. and 0. < pv_rem(o) <= 5.]
            reason = 'no_accepted_small_front'
            if len(candidates) == 1:
                o = candidates[0]
                reason = lc_reason(t, lane, o, relaxed, opening=True)
                if reason is None and raw[lane+2][0] == o['pt']:
                    reason = 'duplicate_rung'
                if reason is None:
                    old = pair.total[lane]
                    # Preserve the current pair's fill budget when it exactly
                    # describes the retained member. Otherwise shrink to its
                    # known open quantity; never replenish while the guard is off.
                    total = old[1] if old is not None and old[0] == o['pt'] and abs(old[1]-pair.filled[lane]-pv_rem(o)) < 1e-9 else pv_rem(o)
                    hold = dict(id=o['lz_id'], tick=o['pt'], total=total, start=t, deadline=t+50.)
                    lc_holds[lane] = hold
                    heapq.heappush(ev, (t+50., 3, next(SEQX), 'C', None))
                    lc_record('open', t, lane, **hold, remaining=pv_rem(o))
            if reason is not None:
                lc_record('refused', t, lane, reason=reason)
        hold = lc_holds[lane]
        if hold is not None:
            out[lane] = (hold['tick'], hold['total'])
        lc_previous[lane] = raw[lane]
    return tuple(out)
''')


def generate():
    source = pairs.DST.read_text()
    if hashlib.sha256(source.encode()).hexdigest() != PARENT or source != pairs.generate():
        raise ValueError('paired parent identity changed')
    patches = [
        ("    LZ = 'LZ' in rules\n", "    LZ = 'LZ' in rules\n"
         "    LC = 'LC50' in rules\n"
         "    if LC:\n"
         "        required = set('LZ+LC50+G1+G2+G3+G4+G5+G6+PM90+GS+LD2+XNA+XMIN1+XV'.split('+'))\n"
         "        extra = rules - required\n"
         "        if not required <= rules or any(not any(r.startswith(p) and r[len(p):].isdigit() for p in ('CAP', 'RO', 'S')) for r in extra) or policy != 'MICRO' or size != 'AM:5:25:95':\n"
         "            raise SystemExit('REFUSED: LC50 requires the declared v3 paired MICRO/AM rules')\n"
         "    lc_holds = [None, None]; lc_previous = [(None, 0.), (None, 0.)]\n"),
        ('    def desired(t):\n', ''.join('    '+line+'\n' if line else '\n' for line in HELPERS.rstrip().split('\n')) +
         '    def desired(t, relax_pm=False):\n'),
        ('            ba = ba or pmf_[0]; sa = sa or pmf_[1]\n',
         '            if not relax_pm:\n                ba = ba or pmf_[0]; sa = sa or pmf_[1]\n'),
        ("    for t in sorted({ws + stop * 1000.0, ws + 240_000, ws + 300_000}):\n",
         "    if LC:\n"
         "        for P_ in (PREDVB, PREDVA):\n"
         "            if mid in P_:\n"
         "                for stamp in P_[mid][0]:\n"
         "                    for at in (stamp, stamp + 100.001):\n"
         "                        if ws <= at <= ws + min(stop, LV['ro'] if LV['ro'] is not None else 300.)*1000.:\n"
         "                            ev.append((float(at), 3, seq, 'C', None)); seq += 1\n"
         "    for t in sorted({ws + stop * 1000.0, ws + 240_000, ws + 300_000}):\n"),
        ("            want = desired(t)\n            if LZ:\n", "            want = desired(t)\n            if LC:\n                want = lc_adjust(t, want)\n            if LZ:\n"),
        ("        elif kind == 'R':                                   # throttle: retry a refused place while its intent stands\n",
         "        elif kind == 'C':\n            lc_observe(t)\n"
         "        elif kind == 'R':                                   # throttle: retry a refused place while its intent stands\n"),
        ("                lz_trace('fill', t, id=o_k['lz_id'], qty=abs(qd))\n",
         "                lz_trace('fill', t, id=o_k['lz_id'], qty=abs(qd))\n"
         "                if LC and any(lc_holds):\n"
         "                    heapq.heappush(ev, (t, 3, next(SEQX), 'C', None))\n"),
    ]
    for old, new in patches:
        if source.count(old) != 1:
            raise ValueError('LC50 patch anchor changed: '+old[:80])
        source = source.replace(old, new, 1)
    return '# GENERATED by pf_ladder_continuity_build.py; opt-in LC50, parent '+PARENT+'\n'+source


def main():
    DST.write_text(generate())
    print('generated', DST, hashlib.sha256(DST.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
