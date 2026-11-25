"""Preserve a bounded per-request failure without changing the decode policy."""


def capture_optimus_progress_failure(error, engine):
    """Read the failing frame; no retry, token edit, or additional forward."""
    if type(error) is not RuntimeError or str(error) != "Optimus progress bound exceeded":
        raise error
    trace = error.__traceback__
    while trace is not None:
        state = trace.tb_frame.f_locals
        if (state.get("self") is engine and trace.tb_frame.f_code.co_name == "run"
                and all(k in state for k in ("x", "events", "start", "step", "width", "ready"))):
            width = state["width"]
            assert state["step"] + 1 == 2 * width + 4 and state["ready"] < 32
            failure = {
                "type": "optimus_progress_limit",
                "message": str(error), "block": state["start"] // 32,
                "steps_in_block": state["step"] + 1,
                "step_limit": 2 * width + 4,
                "cached_prefix": state["ready"],
                "delivered_output_tokens": 0,
                "scope": "no complete response returned; all attempted forwards and Q charged",
            }
            extra = {
                "mode": engine.mode, "fixed_chunk": engine.chunk, "window_switches": 0,
                "source_hashes": engine.hashes, "m2t_threshold": .9,
                "t2t_threshold": .5 if engine.kind == "llada21" else None,
                "t2t_cache_rule": "edited input must be re-encoded before prefix freeze",
                "request_failure": failure,
            }
            return state["x"], state["events"], extra, failure
        trace = trace.tb_next
    raise RuntimeError("Cannot locate Optimus failure state; refuse to manufacture a record") from error
