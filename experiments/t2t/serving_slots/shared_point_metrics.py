"""Compact fixed-budget freeze/summary audit for one frozen model/offer point.

Failure semantics: ``valid_algorithm_failure`` is a retained, incorrect request.
It stays in every cohort denominator and contributes its billed F/Q; excluding
it would inflate both macro TPF and pooled TE. Macro TPF is the per-request
mean of N/F; pooled TE is sum(N)/sum(Q). The two ratios are not interchangeable.
"""
from tools.frozen90_six_shards import OFFERS, verify_complete

BLOCK = {"llada21": 32, "gemma": 256}
STATUSES = ("complete", "valid_algorithm_failure")


def _int(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _receipt(rec, model, offer):
    if rec.get("status") not in STATUSES:
        raise ValueError(f"Unregistered status: {rec.get('status')!r}")
    n = _int(rec.get("N"), "N")
    if n > 2048:
        raise ValueError(f"N exceeds the generation cap: {n} > 2048")
    f = _int(rec.get("F"), "F", minimum=1)
    q = _int(rec.get("Q"), "Q", minimum=1)
    df = _int(rec.get("decoder_F"), "decoder_F", minimum=1)
    ef = _int(rec.get("encoder_F"), "encoder_F")
    if model == "llada21" and ef:
        raise ValueError("LLaDA has no post-prefill encoder forward")
    if f != df + ef:
        raise ValueError(f"F must equal decoder_F + post-prefill encoder_F: {f} != {df}+{ef}")
    if q < df:
        raise ValueError(f"Q cannot be below one row per decoder forward: {q} < {df}")
    peak = offer
    if "max_decoder_Q" in rec:
        peak = _int(rec["max_decoder_Q"], "max_decoder_Q", minimum=1)
        if peak > offer:
            raise ValueError(f"Invalid max decoder width: {peak} > offer {offer}")
    if q > df * peak:
        raise ValueError(f"Q exceeds the decoder width cap: {q} > {df}*{peak}")
    if "decoder_Q" in rec and _int(rec["decoder_Q"], "decoder_Q", minimum=1) != q:
        raise ValueError("Q must equal the actual decoder_Q")
    if type(rec.get("correct")) is not bool:
        raise ValueError("correct must be a boolean")
    if rec["status"] == "valid_algorithm_failure" and (n != 0 or rec["correct"]):
        raise ValueError("A valid algorithm failure must have N=0 and correct=False")
    return dict(index=rec["index"], status=rec["status"], N=n, F=f, Q=q,
                decoder_F=df, encoder_F=ef, correct=rec["correct"])


def _cohort(rows):
    if not rows:
        return dict(requests=0, complete=0, failures=0, correct=0, N=0, F=0, Q=0,
                    macro_TPF=None, pooled_TE=None)
    return dict(
        requests=len(rows),
        complete=sum(r["status"] == "complete" for r in rows),
        failures=sum(r["status"] == "valid_algorithm_failure" for r in rows),
        correct=sum(r["correct"] for r in rows),
        N=sum(r["N"] for r in rows),
        F=sum(r["F"] for r in rows),
        Q=sum(r["Q"] for r in rows),
        macro_TPF=sum(r["N"] / r["F"] for r in rows) / len(rows),
        pooled_TE=sum(r["N"] for r in rows) / sum(r["Q"] for r in rows),
    )


def summarize(samples, receipts, model, offer, version):
    """Verify completeness and bills, then aggregate all90/original30/new60.

    ``verify_complete`` owns identity/board/version/status checks. This function
    adds numerical bill validation and the fixed-budget ratios.
    """
    if model not in OFFERS or type(offer) is not int or offer not in OFFERS[model]:
        raise ValueError(f"Unregistered model/offer: {model}/{offer}")
    if not version or not isinstance(version, str):
        raise ValueError("A non-empty version is required")
    receipts = list(receipts)
    if verify_complete(samples, receipts, model=model, offer=offer, version=version) != 90:
        raise ValueError("Expected exactly 90 frozen receipts")
    rows = [_receipt(rec, model, offer) for rec in receipts]
    by_index = {r["index"]: r for r in rows}
    if len(by_index) != len(rows):
        raise ValueError("Duplicate frozen index")
    return dict(
        model=model, offer=offer, version=version,
        requests=len(rows),
        complete=sum(r["status"] == "complete" for r in rows),
        failures=sum(r["status"] == "valid_algorithm_failure" for r in rows),
        correct=sum(r["correct"] for r in rows),
        cohorts={
            "all90": _cohort(rows),
            "original30": _cohort([r for r in rows if r["index"] < 30]),
            "new60": _cohort([r for r in rows if r["index"] >= 30]),
        },
    )
