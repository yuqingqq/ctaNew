import hashlib,json,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2]
paths=[H/'quote_contract.md',H/'quote_relative.py',H/'quote_check.py',H/'quote_freeze.py',H/'outputs/chains.csv',R/'predictfun/reviews/maker_trace_20261122/v2/chains/take_logic_v2/panel.parquet',R/'predictfun/reviews/maker_timing_estimates_20261124/outputs/post_estimates.csv.gz']
assert not (H/'quote_bindings.json').exists() and not (H/'outputs/quote_relative.csv').exists()
(H/'quote_bindings.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
print('quote inputs and sources frozen; no quote-relative output exists')
