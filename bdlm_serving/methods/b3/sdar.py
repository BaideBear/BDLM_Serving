import torch
from transformers.cache_utils import DynamicCache
from bdlm_serving.methods.b3.gate import B3Gate, logits_entropy


class SDARB3:
    """Single B32 paper-derived B3 configuration, using SDAR's native sampler."""
    def __init__(self, model, sample, *, threshold=.9, kind="sdar"):
        self.model, self.sample = model, sample
        self.threshold, self.kind = threshold, kind

    @torch.inference_mode()
    def run(self, prompt, gen_length, seed, mask_id, eos_ids):
        p = prompt.shape[1]
        total = ((p+gen_length+31)//32)*32
        x = torch.full((1, total), mask_id, dtype=torch.long, device=prompt.device)
        x[:, :p] = prompt
        start = p//32*32
        cache = DynamicCache()
        events = []
        gate = B3Gate(reset_previous_on_promotion=True)
        opened = set()

        def forward(a, b, phase, commit):
            prefix = cache.get_seq_length()
            assert prefix == a
            positions = torch.arange(a, b, device=x.device)
            keys = torch.arange(b, device=x.device)
            if self.kind == "llada20":
                kw = {"output_attentions": True}
            else:
                kw = {"store_kv": commit}
            logits = self.model(
                x[:, a:b], position_ids=positions[None],
                attention_mask=(positions[:, None]//32 >= keys[None, :]//32)[None, None],
                past_key_values=cache, use_cache=True, **kw).logits
            if self.kind == "llada20" and not commit:
                cache.crop(prefix)
            events.append({"phase": phase, "q_tokens": b-a, "physical_forwards": 1,
                           "absolute_positions": positions.tolist()})
            return logits

        if start:
            forward(0, start, "prefill", True)
        while start < total:
            block = start//32
            for step in range(33):
                if bool((x[:, start:start+32] != mask_id).all()):
                    forward(start, start+32, "commit", True)
                    break
                joint = block in opened and start+64 <= total
                logits = forward(start, start+(64 if joint else 32), "b3", False)
                accepted_positions = []
                for j in range(2 if joint else 1):
                    a = start+32*j
                    cur = x[:, a:a+32]
                    masked = cur == mask_id
                    if not bool(masked.any()):
                        continue
                    proposed, confidence = self.sample(logits[:, j*32:(j+1)*32])
                    take = masked & (confidence > self.threshold)
                    if not bool(take.any()):
                        take[0, confidence.masked_fill(~masked, -torch.inf)[0].argmax()] = True
                    cur[take] = proposed[take]
                    accepted_positions += (take[0].nonzero().flatten()+a).tolist()
                current_logits = logits[:, :32]
                predicts_eos = any(bool((current_logits.argmax(-1) == e).any()) for e in eos_ids)
                decision = gate.observe(block, logits_entropy(current_logits)[0],
                    accepted_positions, joint, predicts_eos,
                    logits_entropy(logits[:, 32:])[0] if joint else None)
                if decision["open_on_next_pass"]:
                    opened.add(block)
                events[-1].update(b3_gate=decision, accepted=len(accepted_positions),
                                  edited=0, future_active=joint)
            else:
                raise RuntimeError("SDAR B3 progress bound exceeded")
            if any(bool((x[:, p:start+32] == e).any()) for e in eos_ids):
                break
            start += 32
        return x, events, {"scope": f"paper-derived B32 B3 adapted to native {self.kind} without T2T",
                           "official_implementation": False}
