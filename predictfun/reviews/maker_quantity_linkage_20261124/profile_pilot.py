"""Timing-only rerun of the frozen first market, separate output path; no new sample."""
import cProfile,pstats
import build as B
p=B.DATA/'profile';p.mkdir(exist_ok=False)
i=B.input_tables();pr=cProfile.Profile();pr.enable();B.market('20261024',3697902,i,p);pr.disable();pstats.Stats(pr).strip_dirs().sort_stats('cumtime').print_stats(22)
