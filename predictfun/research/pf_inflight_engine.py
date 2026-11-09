"""Reserve reducing liquidity across lanes and uncertain order lifecycle states."""
import math
from . import pf_entry_engine as entry
from . import pf_sizing_engine as parent
from .pf_inflight_study import HERE, context, sha

NEW = ('reduce_reserved', 'both_reserved')
ENTRY_SHA = '7d4f898cda8db105a4356614c93508d3b663bc388eb3d8caba66614c2e61ff5d'


def room(net, side, opens, mode, exclude=None):
    if mode not in NEW:
        raise ValueError('unknown reservation policy')
    adding = entry.adding(net, side)
    if adding and mode == 'reduce_reserved':
        return float('inf')
    unique = {id(o): o for o in opens}
    pending = sum(entry.remainder(o) for o in unique.values() if o is not exclude and o['side'] == side)
    allowance = 10 - (net if side == 0 else -net) if adding else abs(net)
    return max(0., allowance - pending)


def plan(net, wants, currents, opens, mode):
    if mode not in NEW:
        return entry.plan(net, wants, currents, opens, mode)
    out = []; new = [0., 0.]
    for lane, (price, size) in enumerate(wants):
        side = lane % 2
        if price is None or (mode == 'reduce_reserved' and entry.adding(net, side)):
            out.append((price, size)); continue
        if not math.isfinite(size) or size < 0:
            raise ValueError('invalid desired quantity')
        cur = currents[lane]
        same = cur is not None and cur['pt'] == price
        cap = min(size, max(0., room(net, side, opens, mode, cur if same else None) - new[side]))
        if same and 1e-9 < entry.remainder(cur) <= cap + 1e-9:
            out.append((price, cur['size'])); continue
        cap = min(size, max(0., room(net, side, opens, mode) - new[side]))
        q = parent.lots(cap)
        if not entry.legal(price, side, q):
            out.append((None, 0.)); continue
        out.append((price, q)); new[side] += q
    return tuple(out)


def admission(net, side, size, opens, mode):
    if mode not in NEW:
        return entry.admission(net, side, size, opens, mode)
    return min(size, room(net, side, opens, mode))


def assert_admission(net, side, size, opens, mode):
    if mode not in NEW:
        return entry.assert_admission(net, side, size, opens, mode)
    if not math.isfinite(size) or size < 0 or size > room(net, side, opens, mode) + 1e-7:
        raise ValueError('inflight reservation submission violation')


def release_time(ack, now, fill_delay, mode):
    if mode not in NEW:
        return max(ack, now)
    if not all(math.isfinite(x) for x in (ack, now, fill_delay)) or fill_delay < 0:
        raise ValueError('invalid release clock')
    # No cumulative execution quantity accompanies this simulator's cancel acknowledgement.
    # Keep the reservation until every fill before cancellation must have been reported.
    return max(ack, now) + fill_delay + .001


def assert_path(net, mode):
    if mode == 'both_reserved' and (not math.isfinite(net) or abs(net) > 10 + 1e-7):
        raise ValueError('actual inventory exceeded ten-share envelope')


def source():
    if sha(HERE/'pf_entry_engine.py') != ENTRY_SHA:
        raise ValueError('wrong entry adapter')
    s = entry.source()
    s = parent.replace(s, "    ENTRY_KIND = 'reserved10' if 'ERES10' in rules else 'uniform10' if 'EUN10' in rules else None",
        "    ENTRY_KIND = 'both_reserved' if 'IBOTH' in rules else 'reduce_reserved' if 'IRED' in rules else 'reserved10' if 'ERES10' in rules else 'uniform10' if 'EUN10' in rules else None")
    s = parent.replace(s, '            cancel_wait.setdefault(id(o_), (o_, max(ack, t + STUDY_ACK_MS)))',
        '            cancel_wait.setdefault(id(o_), (o_, inflight_release(ack, t + STUDY_ACK_MS, L_FILL, ENTRY_KIND)))')
    s = parent.replace(s, "                    fills.append((t, side, o['pt'], q,",
        "                    inflight_assert_path(net, ENTRY_KIND)\n                    fills.append((t, side, o['pt'], q,")
    return s


def main():
    context()
    exec(compile(source(), str(HERE/'pf_step4_ladder_v3.py'), 'exec'), dict(
        __name__='__main__', __package__=__package__, __file__=str(HERE/'pf_step4_ladder_v3.py'),
        lots=parent.lots, headroom=parent.headroom, size_state=parent.size_state, MARGIN=parent.MARGIN,
        cap_cannot_bind=parent.cap_cannot_bind, capacity_cannot_clip=parent.capacity_cannot_clip,
        CAP_CACHE=False, STUDY_EXEC=True, STUDY_ACK_MS=0., STUDY_SHA=sha(__file__),
        entry_plan=plan, entry_admission=admission, entry_assert_admission=assert_admission,
        entry_adding=entry.adding, ENTRY_AUDIT=[], ENTRY_PARENT_SHA=ENTRY_SHA,
        inflight_release=release_time, inflight_assert_path=assert_path))


if __name__ == '__main__':
    main()
