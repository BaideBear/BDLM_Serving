"""Batch native forwards while preserving each controller's independent state."""
import concurrent.futures
import queue
import threading
import time
import types
import torch
from transformers.cache_utils import DynamicCache


class BatchCache(DynamicCache):
    def __init__(self, jobs, qpad, kpad):
        super().__init__()
        self.jobs, self.qpad, self.kpad = jobs, qpad, kpad

    def get_seq_length(self, layer_idx=0):
        return self.kpad-self.qpad

    def update(self, key, value, layer_idx, cache_kwargs=None):
        keys, values = [], []
        for i, job in enumerate(self.jobs):
            q = job["q"]
            k, v = job["execution_cache"].update(
                key[i:i+1, :, :q], value[i:i+1, :, :q], layer_idx, cache_kwargs)
            assert k.shape[-2] == job["kv"], (k.shape, job["kv"])
            keys.append(torch.nn.functional.pad(k, (0, 0, 0, self.kpad-k.shape[-2])))
            values.append(torch.nn.functional.pad(v, (0, 0, 0, self.kpad-v.shape[-2])))
        return torch.cat(keys), torch.cat(values)


class ForwardBatcher:
    def __init__(self, model, kind, batch_size):
        self.model, self.kind, self.batch_size = model, kind, batch_size
        self.queue = queue.Queue()
        self.physical_calls = []

    def submit(self, owner, args, kwargs):
        ids = args[0] if args else kwargs.get("input_ids")
        embeds = kwargs.get("inputs_embeds")
        inp = ids if ids is not None else embeds
        mask = kwargs["attention_mask"]
        while mask.ndim < 4:
            mask = mask.unsqueeze(1)
        future = concurrent.futures.Future()
        cache = kwargs["past_key_values"]
        execution_cache = cache
        if self.kind == "sdar" and not kwargs.get("store_kv", True):
            execution_cache = DynamicCache()
            for i, (k, v) in enumerate(cache.to_legacy_cache()):
                execution_cache.update(k, v, i)
        job = {"owner": owner, "ids": ids, "embeds": embeds, "q": inp.shape[1],
               "kv": mask.shape[-1], "mask": mask.bool(), "cache": cache,
               "execution_cache": execution_cache, "positions": kwargs["position_ids"],
               "future": future}
        self.queue.put(job)
        return future.result(timeout=300)

    @torch.inference_mode()
    def execute(self, jobs):
        qpad = max(j["q"] for j in jobs)
        kpad = max(j["kv"] for j in jobs)
        masks, embeddings, positions = [], [], []
        for job in jobs:
            inp = job["embeds"]
            if inp is None:
                inp = self.model.get_input_embeddings()(job["ids"])
            embeddings.append(torch.nn.functional.pad(inp, (0, 0, 0, qpad-job["q"])))
            positions.append(torch.nn.functional.pad(job["positions"], (0, qpad-job["q"])))
            mask = torch.nn.functional.pad(job["mask"],
                                           (0, kpad-job["kv"], 0, qpad-job["q"]))
            if qpad > job["q"]:
                mask[:, :, job["q"]:, 0] = True
            masks.append(mask)
        cache = BatchCache(jobs, qpad, kpad)
        kw = {"store_kv": True} if self.kind == "sdar" else {"output_attentions": True}
        result = self.model(inputs_embeds=torch.cat(embeddings),
                            position_ids=torch.cat(positions),
                            attention_mask=torch.cat(masks), past_key_values=cache,
                            use_cache=True, **kw)
        call_id = len(self.physical_calls)
        self.physical_calls.append({"call_id": call_id, "batch": len(jobs),
            "qpad": qpad, "kvpad": kpad, "actual_query_rows": len(jobs)*qpad,
            "useful_query_rows": sum(j["q"] for j in jobs),
            "requests": [j["owner"].request_id for j in jobs]})
        for i, job in enumerate(jobs):
            job["owner"].calls.append({"batch_call_id": call_id, "rows": job["q"],
                                      "physical_rows": qpad, "batch": len(jobs),
                                      "kv_rows": job["kv"]})
            job["future"].set_result(types.SimpleNamespace(
                logits=result.logits[i:i+1, :job["q"]], past_key_values=job["cache"]))

    def run(self, functions):
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as pool:
            futures = [pool.submit(fn) for fn in functions]
            while not all(f.done() for f in futures):
                try:
                    first = self.queue.get(timeout=.05)
                except queue.Empty:
                    continue
                jobs = [first]
                deadline = time.monotonic()+.003 if self.batch_size > 1 else time.monotonic()
                while len(jobs) < self.batch_size:
                    try:
                        jobs.append(self.queue.get(timeout=max(.00001, deadline-time.monotonic())))
                    except queue.Empty:
                        break
                    if time.monotonic() >= deadline:
                        break
                try:
                    self.execute(jobs)
                except BaseException as exc:
                    for job in jobs:
                        if not job["future"].done():
                            job["future"].set_exception(exc)
                    # Other queued requests are drained by subsequent iterations.
            results = [f.result() for f in futures]
        return results


class ModelFacade:
    def __init__(self, batcher, request_id, seed):
        self.batcher, self.native = batcher, batcher.model
        self.request_id, self.calls = request_id, []
        self.generator = torch.Generator(device=self.native.device).manual_seed(seed)
        self.model = self.native.model
        if batcher.kind == "sdar":
            self.model = types.SimpleNamespace(
                word_embeddings=self.native.get_input_embeddings(),
                layers=self.native.model.layers)

    def __getattr__(self, name):
        return getattr(self.native, name)

    def __call__(self, *args, **kwargs):
        return self.batcher.submit(self, args, kwargs)

    def sdar_sample(self, logits):
        from bdlm_serving.runtime.sampling import top_k_logits, top_p_logits
        shape = logits.shape[:-1]
        flat = logits.reshape(-1, logits.shape[-1])/.6
        flat = top_p_logits(top_k_logits(flat, 20), .95)
        probability = flat.softmax(-1)
        token = torch.multinomial(probability, 1, generator=self.generator)
        confidence = probability.gather(-1, token)
        return token.view(shape), confidence.view(shape)

    def llada20_sample(self, logits):
        # Checkpoint temperature=0 bypasses scaling, not multinomial sampling.
        shape = logits.shape[:-1]
        prob = logits.reshape(-1, logits.shape[-1]).softmax(-1)
        token = torch.multinomial(prob, 1, generator=self.generator)
        return token.view(shape), prob.gather(-1, token).view(shape)

    def _sample_with_temperature_topk_topp(self, logits, **kwargs):
        if self.batcher.kind == "llada20":
            # This entry point is FlowBlock's temperature-zero top-1 control.
            ids = logits.argmax(-1)
            p = logits.float().softmax(-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
            return ids, p
        if self.batcher.kind == "sdar":
            return self.sdar_sample(logits)
        return self.native._sample_with_temperature_topk_topp(logits, **kwargs)


class DirectModelFacade(ModelFacade):
    """Qualification reference: native call, without packing or BatchCache."""
    def __call__(self, *args, **kwargs):
        if self.batcher.kind == "sdar":
            kwargs.setdefault("store_kv", True)
        inp = args[0] if args else kwargs.get("input_ids")
        if inp is None:
            inp = kwargs["inputs_embeds"]
        output = self.native(*args, **kwargs)
        self.calls.append({"batch_call_id": None, "rows": inp.shape[1],
                           "physical_rows": inp.shape[1], "batch": 1,
                           "kv_rows": kwargs["attention_mask"].shape[-1]})
        return output
