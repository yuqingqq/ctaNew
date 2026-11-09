"""Stop superseded J/X drivers before they consume additional data or write results."""
from .pf_trade_tape import refuse_legacy_clock

if __name__ == '__main__':refuse_legacy_clock()
