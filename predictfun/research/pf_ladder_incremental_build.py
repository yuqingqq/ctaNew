"""Generate an opt-in incremental-size research variant; frozen v1 is untouched.

Usage: python3 -m predictfun.research.pf_ladder_incremental_build
Replay with pf_step4_ladder_incremental and add INCSIZE to PF_ARMS.
The variant also contains the existing opt-in D1 inventory tokens.
For v1, the arm is:
MICRO|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+INCSIZE+XNA+XMIN1+XV+S60|mid|AM:5:25:95
Keep the established clock/prediction environment, use a new PF_TAG, and run
through run_light.sh. Sealed/forward input bounds remain those of frozen v1.

INCSIZE retains same-price orders, adds the missing remaining quantity, and
cancels surplus tranches newest first. Cancel-pending exposure is reserved
until acknowledgement. Sub-minimum additions are skipped. Partial fills alone
do not replenish; a target change or complete depletion of a logical level
does. All tranches retain ordinary guards, post-only checks, throttling and
loss-cap accounting. Unsupported lifecycle modes fail closed.
New placements enforce aggregate projected G6/CAP/RO quantity headroom,
including cancel-pending exposure. Indivisible reductions/reprices may wait
for cancellation acknowledgement before replacing the reserved quantity.
The inc_add/inc_cancel/inc_keep counters describe plans; the requests log
records actual sends. No profitability claim or live deployment is implied.

Recorded fills/requests retain logical lane (touch/rung) plus order_slot for
tranche identity. The old single-order pf_view_check diagnostic is not an
INCSIZE validator; use run_incremental_checks.sh's paired causal tests.
"""
from textwrap import dedent

from .pf_ladder_fwd_build import HERE, generate
from .pf_ladder_inv_build import PATCHES as INVENTORY_PATCHES


HELPERS = dedent('''\
    def inc_active(parent):
        out = []
        for slot in LANES:
            if slot % NBASE != parent:
                continue
            fl = [r for r in inflight[slot] if not r['cancelled']]
            o = fl[-1] if fl else pv(slot)
            if o is not None and pv_rem(o) > 1e-9:
                out.append(SizeOrder(slot, int(o['pt']), pv_rem(o), o.get('t_send', o.get('t0', 0.))))
            elif pend[slot] is not None and intent[slot] == pend[slot]['key']:
                p, q = intent[slot]
                out.append(SizeOrder(slot, int(p), q, pend[slot]['t_first'], False))
        return out

    def inc_reserved(parent, now):
        return [SizeOrder(o['lane'], int(o['pt']), pv_rem(o), o.get('t_send', o.get('t0', 0.)))
                for o in policy_open(now) if o['lane'] % NBASE == parent and pv_rem(o) > 1e-9]

    def inc_slot(parent):
        slot = inc_next[parent]; inc_next[parent] += NBASE
        if slot not in intent:
            LANES.append(slot); LANES_OF[slot % 2].append(slot)
            orders[slot] = known[slot] = intent[slot] = busy[slot] = pend[slot] = hy_pend[slot] = None
            inflight[slot] = []
        return slot

    def inc_decide(want, now):
        nonlocal seq
        # Keep matching/planning proportional to outstanding orders. Old slot
        # dictionaries remain for already-queued fill notifications/retries.
        held = {o['lane'] for o in policy_open(now)}
        LANES[:] = [slot for slot in LANES if slot in BASE_LANES or slot in held
                    or orders[slot] is not None or inflight[slot] or pend[slot] is not None]
        for side in (0, 1):
            LANES_OF[side][:] = [slot for slot in LANES if slot % 2 == side]
        for parent in BASE_LANES:
            goal = want[parent]
            active, reserved = inc_active(parent), inc_reserved(parent, now)
            if goal != inc_goals[parent] or (goal[0] is not None and not active and not reserved):
                inc_work[parent] = True
            inc_goals[parent] = goal
            if not inc_work[parent]:
                continue
            p, qty = goal
            unit = (p if parent % 2 == 0 else 100 - p) / 100.0 if p is not None else 1.0
            minimum = LV['min_usd'] / unit if LV['min_usd'] is not None else 0.0
            plan = plan_size(int(p) if p is not None else None, qty, active, reserved, minimum)
            for slot in plan.cancel:
                pend[slot] = None
                intent[slot] = (None, 0.0)
                heapq.heappush(ev, (now, 0, seq, 'A', (slot, None, 0.0))); seq += 1
            if plan.add > 1e-9:
                slot = inc_slot(parent)
                intent[slot] = (p, plan.add)
                heapq.heappush(ev, (now, 0, seq, 'A', (slot, p, plan.add))); seq += 1
                live_ctr['inc_add'] += 1
            live_ctr['inc_keep'] += len(plan.keep)
            live_ctr['inc_cancel'] += len(plan.cancel)
            live_ctr['inc_below_minimum'] += int(plan.below_minimum)
            inc_work[parent] = plan.waiting

''')


PATCHES = [
    ("    LANES = (0, 1, 2, 3) if LDK is not None else (0, 1)\n",
     "    INC = 'INCSIZE' in rules\n"
     "    if INC and (not SPLIT or not V2 or POLICYVIEW != 'known' or FILLPRIO != 'price'\n"
     "                or LV['seq'] is not None or LV['hy'] or LEGACY_INTENT or LV['lock']\n"
     "                or LV['qlock'] is not None or LV['plock'] is not None):\n"
     "        raise SystemExit('REFUSED: INCSIZE needs known-view split V2, price/time priority, no legacy/locking modes')\n"
     "    if INC:\n"
     "        from predictfun.research.pf_incremental_size import SizeOrder, plan_size\n"
     "    BASE_LANES = (0, 1, 2, 3) if LDK is not None else (0, 1)\n"
     "    NBASE = len(BASE_LANES)\n"
     "    LANES = list(BASE_LANES) if INC else BASE_LANES\n"
     "    inc_next = {L: L for L in BASE_LANES}\n"
     "    inc_goals = {L: None for L in BASE_LANES}; inc_work = {L: False for L in BASE_LANES}\n"),
    ("    LANES_OF = {s_: tuple(L_ for L_ in LANES if L_ % 2 == s_) for s_ in (0, 1)}\n",
     "    LANES_OF = {s_: (list if INC else tuple)(L_ for L_ in LANES if L_ % 2 == s_) for s_ in (0, 1)}\n"),
    ("    def reserve_cancel(o_, ack):\n        if POLICYVIEW == 'known':\n            cancel_wait.setdefault(id(o_), (o_, ack))\n",
     "    def reserve_cancel(o_, ack):\n"
     "        nonlocal seq\n"
     "        if POLICYVIEW == 'known':\n"
     "            cancel_wait.setdefault(id(o_), (o_, ack))\n"
     "        if INC:\n"
     "            heapq.heappush(ev, (ack, 2, seq, 'D', None)); seq += 1\n"),
    ("            refs = [o_ for o_ in known.values() if o_ is not None]\n"
     "            refs += [r_ for fl_ in inflight.values() for r_ in fl_ if not r_['cancelled']]\n",
     "            refs = [o_ for o_ in (known[L_] for L_ in LANES) if o_ is not None]\n"
     "            refs += [r_ for L_ in LANES for r_ in inflight[L_] if not r_['cancelled']]\n"),
    ("        fl = [r_ for r_ in inflight[L_] if not r_['cancelled']]\n        return fl[-1] if fl else pv(L_)\n",
     "        if INC:\n"
     "            active = sorted(inc_active(L_), key=lambda o: (o.sent, o.slot))\n"
     "            if not active:\n"
     "                return None\n"
     "            goal = inc_goals[L_]\n"
     "            return dict(pt=active[0].price, size=goal[1] if goal and goal[0] == active[0].price else sum(o.remaining for o in active))\n"
     "        fl = [r_ for r_ in inflight[L_] if not r_['cancelled']]\n"
     "        return fl[-1] if fl else pv(L_)\n"),
    ("                    if increasing and abs(knet) + live[s_] + z_ > 5 * cap:\n",
     "                    if not INC and increasing and abs(knet) + live[s_] + z_ > 5 * cap:\n"),
    ("                if increasing and abs(knet) + live[side] + sz_ > 5 * cap:\n",
     "                if not INC and increasing and abs(knet) + live[side] + sz_ > 5 * cap:\n"),
    ("    while ev:\n", ''.join('    ' + line if line.strip() else line for line in HELPERS.splitlines(keepends=True)) + "    while ev:\n"),
    ("            want = desired(t)\n            for side in LANES:\n",
     "            want = desired(t)\n"
     "            if INC:\n"
     "                inc_decide(want, t)\n"
     "                continue\n"
     "            for side in LANES:\n"),
    ("            lane, p, zsz = pl; side = lane % 2\n",
     "            lane, p, zsz = pl; side = lane % 2\n"
     "            if INC and intent[lane] != (p, zsz):\n"
     "                continue  # obsolete retry/decision\n"),
    ("            wc = worst_case(dict(side=side, pt=p, rem=zsz))\n"
     "            min_worst[0] = min(min_worst[0], wc)\n"
     "            if LV['losscap'] is not None and realised0 + wc < -LV['losscap']:\n"
     "                live_ctr['refused'] += 1; intent[lane] = None; continue\n",
     "            if INC:\n"
     "                parent = lane % NBASE\n"
     "                opens = policy_open(t)\n"
     "                reserved = sum(pv_rem(o) for o in opens if o['lane'] % NBASE == parent)\n"
     "                # Recheck at send time, including any earlier sends at this timestamp.\n"
     "                goal = inc_goals[parent]\n"
     "                room = max(0.0, goal[1] - reserved) if goal and goal[0] == p else 0.0\n"
     "                same_side = sum(pv_rem(o) for o in opens if o['side'] == side)\n"
     "                signed_net = knet if side == 0 else -knet\n"
     "                if 'G6' in rules:\n"
     "                    room = min(room, max(0.0, 5 * cap - signed_net - same_side))\n"
     "                if LV['capn'] is not None:\n"
     "                    room = min(room, max(0.0, LV['capn'] - signed_net - same_side))\n"
     "                if LV['ro'] is not None and (t - ws) / 1000.0 >= LV['ro']:\n"
     "                    room = min(room, max(0.0, -signed_net - same_side))\n"
     "                zsz = min(zsz, room)\n"
     "                if zsz <= 1e-9 or (LV['min_usd'] is not None and\n"
     "                        (p if side == 0 else 100 - p) * zsz / 100.0 < LV['min_usd'] - 1e-9):\n"
     "                    intent[lane] = None; pend[lane] = None\n"
     "                    live_ctr['inc_quantity_refused'] += 1\n"
     "                    continue\n"
     "                intent[lane] = (p, zsz)\n"
     "            wc = worst_case(dict(side=side, pt=p, rem=zsz))\n"
     "            min_worst[0] = min(min_worst[0], wc)\n"
     "            if LV['losscap'] is not None and realised0 + wc < -LV['losscap']:\n"
     "                live_ctr['refused'] += 1; intent[lane] = None; continue\n"),
    ("                live_ctr['refused'] += 1; intent[lane] = None; continue\n",
     "                live_ctr['refused'] += 1; intent[lane] = None\n"
     "                if INC:\n"
     "                    pend[lane] = None\n"
     "                continue\n"),
    ("                    intent[lane] = None; live_ctr['rej_reset'] += 1   # rejected: the next decision re-sends it if still wanted\n",
     "                    intent[lane] = None; live_ctr['rej_reset'] += 1   # rejected: the next decision re-sends it if still wanted\n"
     "                    if INC:\n"
     "                        inc_work[lane % NBASE] = True\n"),
    ("                        intent[rec['lane']] = None                     # refused: the next decision re-places it if still wanted\n",
     "                        intent[rec['lane']] = None                     # refused: the next decision re-places it if still wanted\n"
     "                        if INC:\n"
     "                            inc_work[lane % NBASE] = True\n"),
    ("            o['ahead0'] = o['ahead']\n            if rec['cancelled'] and not SEQ:\n",
     "            o['ahead0'] = o['ahead']\n"
     "            if INC:\n"
     "                o['t_send'] = rec['t_send']\n"
     "            if rec['cancelled'] and not SEQ:\n"),
    ("                live_ctr['own_dup_price'] += int(len(lv_) != len(set(lv_)))\n",
     "                live_ctr['inc_same_price_tranches' if INC else 'own_dup_price'] += int(len(lv_) != len(set(lv_)))\n"),
    ("            o_k['kq'] = o_k.get('kq', 0.0) + abs(qd)\n",
     "            o_k['kq'] = o_k.get('kq', 0.0) + abs(qd)\n"
     "            if INC:\n"
     "                heapq.heappush(ev, (t, 2, seq, 'D', None)); seq += 1\n"),
    ("                book_.sort(key=lambda o_: -o_['pt'] if side == 0 else o_['pt'])\n",
     "                if INC:\n"
     "                    book_.sort(key=lambda o_: ((-o_['pt'] if side == 0 else o_['pt']), o_['t0'], o_['lane']))\n"
     "                else:\n"
     "                    book_.sort(key=lambda o_: -o_['pt'] if side == 0 else o_['pt'])\n"),
    ("for f in ('pf_step4_ladder.py', 'pf_clock_rl.py', 'pf_io.py')},\n",
     "for f in ('pf_step4_ladder.py', 'pf_clock_rl.py', 'pf_io.py', 'pf_step4_ladder_incremental.py',\n"
     "                         'pf_incremental_size.py', 'pf_ladder_incremental_build.py', 'pf_ladder_inv_build.py', 'pf_inv_r60.py')},\n"),
    ("    Q = pd.DataFrame(reqs.pop(name), columns=['t_ms', 'side', 'kind', 'mid', 'lane']); Q.to_parquet(f'{OUT}/requests/{stem}.parquet', index=False)\n"
     "    f = pd.DataFrame(allf.pop(name), columns=FCOLS)\n",
     "    Q = pd.DataFrame(reqs.pop(name), columns=['t_ms', 'side', 'kind', 'mid', 'lane'])\n"
     "    f = pd.DataFrame(allf.pop(name), columns=FCOLS)\n"
     "    rs_ = name.split('/')[0].split('+')[1:]\n"
     "    if 'INCSIZE' in rs_:\n"
     "        nb_ = 4 if ladder_k(set(rs_)) is not None else 2\n"
     "        for panel_ in (Q, f):\n"
     "            panel_['order_slot'] = panel_['lane']\n"
     "            panel_['lane'] = panel_['lane'] % nb_\n"
     "    Q.to_parquet(f'{OUT}/requests/{stem}.parquet', index=False)\n"),
]


def main():
    generate(INVENTORY_PATCHES + PATCHES, f'{HERE}/pf_step4_ladder_incremental.py',
             'opt-in INCSIZE and D1 inventory tokens (pf_ladder_incremental_build)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
