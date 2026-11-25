"""Distinct module name prevents collision with the frozen cap experiment."""
import importlib.util,sys
from pathlib import Path
p=Path(__file__).with_name('experiment.py');spec=importlib.util.spec_from_file_location('exit_experiment',p);E=importlib.util.module_from_spec(spec);sys.modules[spec.name]=E;spec.loader.exec_module(E)
if __name__=='__main__':
 if sys.argv[1] in ['prepare','run']:getattr(E,sys.argv[1])()
 else:
  name=sys.argv[1];assert name in ['selftest','check','report']
  spec=importlib.util.spec_from_file_location('exit_'+name,p.with_name(name+'.py'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.run(E)
