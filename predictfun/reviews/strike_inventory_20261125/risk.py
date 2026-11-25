"""Additional order sizing for the frozen strike_inventory contract, in dollars."""
import math

BUDGET = 50.0

def risks(book, probability):
    if not (0 <= book <= 1 and 0 <= probability <= 1):
        raise ValueError("invalid binary market probability")
    return book * math.sqrt(1 - probability), (1-book) * math.sqrt(probability)

def allowed(size, signed_position, reserved, unit_risk):
    """Position is signed in the proposal's direction. Reservations include retiring orders."""
    if size < 0 or reserved < 0 or (unit_risk is not None and unit_risk < 0):
        raise ValueError("negative size or risk")
    if unit_risk == 0:
        return size
    limit = BUDGET/unit_risk if unit_risk is not None else 0.0
    room = max(0., limit - signed_position - reserved)
    return min(size, math.floor(room+1e-9))

def selftest():
    assert risks(.5,.5)[0] == risks(.5,.5)[1]
    long, short = risks(.8,.4)
    assert math.isclose(long**2,.8**2*.6)
    assert math.isclose(short**2,.2**2*.4)
    assert allowed(100,80,15,.5) == 5
    assert allowed(100,-80,15,None) == 65
    assert allowed(100,150,15,.5) == 0
    assert allowed(100,-150,15,.9) == 100
    try:
        risks(50,.5)
    except ValueError:
        return
    raise AssertionError('cent price accepted as probability')
