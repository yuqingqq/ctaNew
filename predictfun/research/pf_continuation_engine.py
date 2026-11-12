"""Fork the existing execution loop; never approximate re-entry in a second matcher."""
import ast
import textwrap
from .pf_default_pull_engine import source as base_source
from .pf_order_policy_engine import replace

STATE = ('requests zombies live_ctr min_worst busy TB sent pend delays inflight '
         'hy_pend CDS SEQX ev seq orders intent known cancel_wait net knet cash '
         'kcash fills withheld own t').split()


def source(queue='M'):
    s = base_source(queue)
    anchor = '''            for at,target in own.probes:
                heapq.heappush(ev,(at,0,seq,'Y',target));seq+=1
            own.probes.clear()
'''
    replacement = '''            probes = list(own.probes)
            own.probes.clear()
            if not own.in_fork:
                for _, target in probes:
                    row = own.records[target['_gid']]
                    if own.selected(row):
                        for alternative in ('keep', 'pull'):
                            result = fork_path(row, alternative)
                            own.forks.setdefault(row['id'], {})[alternative] = result
'''
    s = replace(s, anchor, replacement)
    tree = ast.parse(s)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_window')
    loop = next(n for n in fn.body if isinstance(n, ast.While))
    lines = s.splitlines(keepends=True)
    original = ''.join(lines[loop.lineno-1:loop.end_lineno])
    body = textwrap.dedent(original).replace('while ev:', 'while ev and ev[0][0] < until:', 1)
    # Only these outer scalars are assigned by the original event loop. Containers
    # retain their outer identities; helper closures see the same restored objects.
    drain = '    def drain(until=float("inf")):\n'
    drain += '        nonlocal seq, net, knet, cash, kcash, t\n'
    drain += textwrap.indent(body, '        ')
    names = ', '.join(STATE)
    pack = '{' + ', '.join(repr(k)+': '+k for k in STATE) + '}'
    restore = '\n'.join('            '+k+' = saved['+repr(k)+']' for k in STATE)
    install = '\n'.join('        '+k+' = cloned['+repr(k)+']' for k in STATE)
    fork = f'''
    def fork_path(row, alternative):
        nonlocal {names}
        import copy
        saved = {pack}
        # Keep only events inside this fork's finite horizon. Copy the complete
        # bundle together to preserve aliases across notices/orders/client view.
        bundle = dict(saved)
        bundle['ev'] = [e for e in ev if e[0] < row['decision'] + 30000.]
        cloned = copy.deepcopy(bundle)
{install}
        cancel_wait = {{id(o): (o, ack) for o, ack in cancel_wait.values()}}
        own.in_fork = True
        own.probes.clear()
        start = row['decision']
        before = len(fills)
        initial_net = net
        if alternative == 'pull':
            own.cool[int(row['side'])] = start + 1000.
        elif alternative != 'keep':
            raise ValueError('unknown fork alternative')
        heapq.heappush(ev, (start, 2, -1, 'D', None))
        result = {{'initial_net': initial_net, 'initial_cash': cash, 'horizons': {{}}}}
        try:
            for horizon in (2000., 10000., 30000.):
                drain(start + horizon)
                result['horizons'][int(horizon)] = dict(
                    fills=list(fills[before:]), net=net, cash=cash,
                    requests=[r for r in requests if r[0] >= start],
                    placements=[(r['t_send'], r['side'], r['order_price'], r['initial_size'])
                                for r in own.records.values() if r['t_send'] >= start])
            return result
        finally:
{restore}

'''
    # t is needed by worst_case's closure, even before the first event.
    new = '    t = ws\n' + drain + fork + '    drain()\n'
    s = ''.join(lines[:loop.lineno-1]) + new + ''.join(lines[loop.end_lineno:])
    compile(s, 'continuation_engine', 'exec')
    return s


def verify_transform():
    for rule in ('M', 'L'):
        s = source(rule)
        tree = ast.parse(s)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_window')
        drain = next(n for n in fn.body if isinstance(n, ast.FunctionDef) and n.name == 'drain')
        assignments = {n.id for n in ast.walk(drain) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        # If a future engine edit starts replacing a state container in the loop,
        # it must explicitly become nonlocal; otherwise refuse the transform.
        required = assignments.intersection(STATE)
        declared = {x for n in drain.body if isinstance(n, ast.Nonlocal) for x in n.names}
        if required != declared:raise ValueError(('unbound fork state', required, declared))
    try:replace('xx', 'x', 'z')
    except ValueError:pass
    else:raise ValueError('ambiguous source anchor accepted')
    return dict(status='PASS', state_fields=STATE, queues=['M','L'])
