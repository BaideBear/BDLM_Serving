"""Fixed native input-embedding context vs position, CPU only."""
import argparse
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np

from bdlm_serving.methods.t2t.embedding_context import KEY, Projector, context, input_table
from tools.edit_opportunity import load
from tools.lexical_hazard import BUDGETS, TOP, run_ablation, write_rows


NAME = 'position_embedding_context'


def compare(rows):
    by = {(r['index'], r['metric'], r['model']): r for r in rows}
    indices = sorted({r['index'] for r in rows})
    budgets = [f'Q{q}/{cap}' for q, cap in BUDGETS]
    result = []
    for m in budgets + [f'top-{k}' for k in TOP]:
        pairs = [(by[i, m, 'position'], by[i, m, NAME]) for i in indices]
        deltas = [b['hits'] - a['hits'] for a, b in pairs]
        result.append(dict(metric=m, seats=sum(a['seats'] for a, _ in pairs),
                           targets=sum(a['targets'] for a, _ in pairs),
                           position_hits=sum(a['hits'] for a, _ in pairs),
                           context_hits=sum(b['hits'] for _, b in pairs), delta=sum(deltas),
                           requests_up=sum(d > 0 for d in deltas), requests_down=sum(d < 0 for d in deltas)))
    physical = [r['delta'] for r in result if r['metric'] in budgets]
    return result, min(physical) >= 0 and sum(d > 0 for d in physical) >= 2


def benchmark(cal, table):
    states = []
    for t in json.loads((cal / 'weights.json').read_text())['samples']:
        calls = []
        def capture(*inputs):
            calls.append(inputs)
            return np.zeros((len(inputs[0]), 3))
        path = load(cal / f"{t['index']:02d}.trace.jsonl.gz", t, context_features=capture)
        states.extend(c for c, step in zip(calls, (s for s in path if s[3]), strict=True)
                      if step[0] == 'selected')
    p = Projector(table)
    def sweep():
        start = perf_counter_ns()
        for inputs in states:
            context(p.unit, *inputs)
        return (perf_counter_ns() - start) / 1e6
    cold = sweep()
    warm = median(sweep() for _ in range(21))
    return dict(dataset='calibration24', selected_states_with_old=len(states),
                old_candidates=sum(len(s[0]) for s in states), cached_rows=len(p.cache),
                miss_read_project_normalize_ms=p.miss_ns / 1e6, cold_sweep_ms=cold,
                warm_sweep_median_ms=warm, warm_us_per_state=1000 * warm / len(states))


def main(cal, native, checkpoint, out):
    out.mkdir(parents=True, exist_ok=True)
    with input_table(checkpoint) as table:
        p = Projector(table)
        # The shared runner also writes coefficients. Keep only aggregate counts.
        with TemporaryDirectory(dir=out) as scratch:
            rows, _ = run_ablation(cal, native, Path(scratch), NAME, {},
                                   context_features=lambda *args: context(p.unit, *args))
        result, passed = compare(rows)
        write_rows(out / 'comparison.csv', result)
        timing = benchmark(cal, table)
        (out / 'timing.json').write_text(json.dumps(timing, indent=2) + '\n')
    print('Frozen practical rule:', 'PASS' if passed else 'FAIL')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cal', 'native', 'checkpoint', 'out'):
        parser.add_argument('--' + key, required=True, type=Path)
    main(**vars(parser.parse_args()))
