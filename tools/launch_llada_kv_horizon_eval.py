"""Run the research LLaDA KV-horizon evaluation from the dedicated runtime copy."""
import runpy
import sys

REPO = "/gs/work/llada-hfull-sf"
sys.path[:] = [p for p in sys.path if p not in (REPO, "/gs/history-repo", "")]
sys.path.insert(0, REPO)
sys.argv[0] = "experiments.t2t.serving_slots.llada_kv_horizon_eval"
runpy.run_module("experiments.t2t.serving_slots.llada_kv_horizon_eval", run_name="__main__")
