"""Experiment-local RNG injection for one serial, non-overlapping Q256 request."""
from unittest.mock import patch


def sampler_class(module, seed, continuous, inits, fingerprint):
    """Keep the pinned initializer (including its single real canvas draw) intact.

    Override its module-local torch binding only during __init__, not torch's
    public functions. The sampler always stores a real private Generator.
    This adapter is for a serial qualification worker, not a serving API.
    """
    torch = module.torch
    previous = None

    class Sampler(module.SelectedRowSampler):
        def __init__(self, **kw):
            nonlocal previous
            ordinal = len(inits)
            if kw["cap"] != 256 or kw["seed"] != (seed + ordinal) % (1 << 64):
                raise ValueError("Not the expected sequential single-block initialization")
            draws = []
            carried = continuous and ordinal != 0

            class Factory:
                def manual_seed(self, value):
                    if value != kw["seed"]:
                        raise ValueError("Unexpected sampler seed")
                    if carried:
                        if previous is None or previous.device != kw["device"]:
                            raise ValueError("Missing same-device previous generator")
                        return previous
                    return torch.Generator(device=kw["device"]).manual_seed(value)

            class InitTorch:
                def __getattr__(self, k):
                    return getattr(torch, k)

                def Generator(self, *, device):
                    if device != kw["device"]:
                        raise ValueError("Unexpected generator device")
                    return Factory()

                def randint(self, high, size, *, generator, device):
                    if draws or size != (1, 256) or high != kw["vocab_size"]:
                        raise ValueError("Unexpected canvas draw")
                    before = fingerprint(generator.get_state())
                    ids = torch.randint(high, size, generator=generator, device=device)
                    draws.append(dict(before=before, after=fingerprint(generator.get_state()),
                                      ids=fingerprint(ids)))
                    return ids

            with patch.object(module, "torch", InitTorch()):
                super().__init__(**kw)
            if len(draws) != 1 or (previous is self.rng) != carried:
                raise ValueError("Invalid canvas draw or generator lineage")
            inits.append(dict(ordinal=ordinal, carried=carried, **draws[0]))
            previous = self.rng

    return Sampler
