"""Run the single-factor module from the dedicated runtime copy.

The shared venv sitecustomize prepends /gs/history-repo; remove it so
bdlm_serving resolves to the experiment copy, then run the module normally.
"""
import runpy
import sys

REPO = "/gs/work/llada-hfull-sf"
sys.path[:] = [p for p in sys.path if p not in (REPO, "/gs/history-repo", "")]
sys.path.insert(0, REPO)
sys.argv[0] = "experiments.t2t.serving_slots.llada_hfull_single_factor"
runpy.run_module("experiments.t2t.serving_slots.llada_hfull_single_factor", run_name="__main__")
