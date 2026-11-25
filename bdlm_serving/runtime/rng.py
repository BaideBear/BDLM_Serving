"""Shared RNG exclusion for the reference executor's request lifecycle.

torch.random.fork_rng swaps process-global generators. Request initialization
and sampling must use the same reentrant lock, including callers that create a
request while ManagedRequest is executing another. This intentionally
serializes the reference implementation; it is not a GPU concurrency design.
"""
import threading


REFERENCE_RNG_LOCK = threading.RLock()
