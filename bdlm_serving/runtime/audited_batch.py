"""Large-batch execution with independent transformer-row observation."""
import concurrent.futures
import gc
import queue
import time
import types
import torch
from bdlm_serving.runtime.batch import ForwardBatcher
from bdlm_serving.methods.flowblock.control import TailCache


class AuditedBatcher(ForwardBatcher):
    def __init__(self, model, kind, batch_size):
        super().__init__(model, kind, batch_size)
        self.observed = []
        self.handles = []
        self.checkpoint = None
        for index, layer in enumerate(model.model.layers):
            def observe(module, args, kwargs, index=index):
                hidden = args[0] if args else kwargs["hidden_states"]
                self.observed.append([index, hidden.shape[0], hidden.shape[1]])
            self.handles.append(layer.register_forward_pre_hook(observe, with_kwargs=True))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def run(self, functions):
        # Gather a full wave of active requests; the old 3ms coalescing timer
        # often launches tiny batches while GPU-synchronized controllers prepare.
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as pool:
            futures = [pool.submit(fn) for fn in functions]
            while not all(f.done() for f in futures):
                jobs = []
                while len(jobs) < min(self.batch_size, sum(not f.done() for f in futures)):
                    try:
                        jobs.append(self.queue.get(timeout=.05))
                    except queue.Empty:
                        continue
                if jobs:
                    try:
                        # Prefill and decode differ sharply in Q length.
                        # Do not pad every short decode to one long prompt.
                        groups = {}
                        for job in jobs:
                            # FlowBlock decode always has one of these two caches.
                            from bdlm_serving.methods.flowblock.prefix_cache import SparseTailCache
                            decode = isinstance(job["execution_cache"], (TailCache, SparseTailCache))
                            groups.setdefault("decode" if decode else "prefill", []).append(job)
                        for group in groups.values():
                            self.execute(group)
                    except BaseException as exc:
                        for job in jobs:
                            if not job["future"].done():
                                job["future"].set_exception(exc)
            return [f.result() for f in futures]

    @torch.inference_mode()
    def execute(self, jobs):
        self.observed = []
        start = time.perf_counter()
        qpad, batch = max(j["q"] for j in jobs), len(jobs)
        try:
            super().execute(jobs)
        except BaseException as exc:
            cid = len(self.physical_calls)
            self.physical_calls.append(dict(call_id=cid, batch=batch, qpad=qpad,
                kvpad=max(j["kv"] for j in jobs), actual_query_rows=batch*qpad,
                useful_query_rows=sum(j["q"] for j in jobs),
                requests=[j["owner"].request_id for j in jobs],
                failed=True, error=repr(exc), observed_layer_inputs=self.observed))
            for job in jobs:
                job["owner"].calls.append(dict(batch_call_id=cid, rows=job["q"],
                    physical_rows=qpad, batch=batch, failed=True, kv_rows=job["kv"]))
            if self.checkpoint is not None:
                self.checkpoint(self.physical_calls)
            raise
        expected = [[i, batch, qpad] for i in range(len(self.model.model.layers))]
        assert self.observed == expected, (self.observed, expected)
        self.physical_calls[-1].update(observed_layer_inputs=self.observed,
            observed_rows_over_layers=sum(b*q for _, b, q in self.observed),
            execution_wall_ms=(time.perf_counter()-start)*1000)
        if self.checkpoint is not None and len(self.physical_calls) % 50 == 0:
            self.checkpoint(self.physical_calls)
