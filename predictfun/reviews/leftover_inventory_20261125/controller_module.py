"""Load this receipt's script under its own name; the upstream stack imports another replay.py."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

spec = spec_from_file_location('leftover_controller_replay', Path(__file__).with_name('replay.py'))
controller = module_from_spec(spec)
spec.loader.exec_module(controller)
