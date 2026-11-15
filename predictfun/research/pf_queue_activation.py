"""Initial queue from the last book snapshot at or before venue activation.

Callers select that snapshot using their existing activation clock. This
function deliberately receives no later snapshots: post-activation additions
cannot acquire priority ahead of an already resting order. Unknown depth is
still an explicit replay assumption, not evidence of an empty queue.
"""
import math

CONTRACT = 'activation-snapshot-queue-v1'


def initial_queue(shown, rule, unknown='zero'):
    """Return (assumed displayed queue, queue ahead, initially unknown).

    M/L join behind the activation snapshot; U is the explicit front scenario.
    'block' and 'inf' name the same conservative unknown-depth assumption in
    the entry and stop lanes. Later queue depletion is owned by those lanes.
    """
    if rule not in ('M', 'L', 'U'):
        raise ValueError('unknown queue rule')
    if unknown not in ('zero', 'block', 'inf'):
        raise ValueError('unknown queue-depth assumption')
    shown = float(shown)
    if shown < 0:
        raise ValueError('negative displayed queue')
    missing = not math.isfinite(shown)
    assumed = (0.0 if unknown == 'zero' else math.inf) if missing else shown
    return assumed, 0.0 if rule == 'U' else assumed, missing
