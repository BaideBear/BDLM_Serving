"""Explicit process-local RMSNorm launch protocol shared by every SDAR policy."""
import hashlib
from pathlib import Path

EXPECTED_LAYER_NORM_SHA256 = "53e13d970888ee0354afe44933eeb07e4687ad594a691187fdafb18c891ec8ba"


def configure_sdar_rmsnorm(warps=None):
    import flash_attn.ops.triton.layer_norm as layer_norm
    import triton
    path = Path(layer_norm.__file__)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if source_hash != EXPECTED_LAYER_NORM_SHA256:
        raise RuntimeError("SDAR RMSNorm implementation changed; requalify numerical protocol")
    tuner = layer_norm._layer_norm_fwd_1pass_kernel
    if warps is not None:
        if warps != 4:
            raise ValueError("Only the independently qualified fixed4 protocol may be requested")
        tuner.configs = [triton.Config({}, num_warps=warps)]
        tuner.cache.clear()

    def snapshot():
        return {
            "protocol": "native_autotune" if warps is None else "native_rmsnorm_fixed4",
            "fixed_rms_warps": warps,
            "candidate_configs": [str(c) for c in tuner.configs],
            "chosen_configs": {str(k): str(v) for k, v in tuner.cache.items()},
            "source_sha256": source_hash,
            "scope": "Same native kernel source; common numerical control, not ERQW speed innovation",
        }
    return snapshot
