"""Read-only close-opportunity audit for the Gemma B256 strict path.

This module is independent of every runner: it imports no model, no torch and
no executor.  It reads either a per-step strict trace or an already sealed
block summary and answers one bounded question:

    Which terminal ``drain`` decoder calls can be removed without changing
    the final block, and which only bound the best case?

Three numbers are deliberately kept apart.

``exact_suffix_save_F``
    Drains in the contiguous terminal suffix whose rows are completely
    overwritten by the first ``verify`` chunk.  The pinned projector writes
    every selected row's K/V at every layer before attention, so these rows
    are refreshed before any query can read them.  Self-conditioning, the
    sampler clock and the encoder barrier are unchanged.  This is a proof,
    conditional on the audited trace.

``repack_ceiling_F``
    All terminal drains.  Reaching it requires repacking the verify sweep
    (dirty rows first).  If the dirty rows are not already in the original
    first chunk, the partition changes and later chunks read a different K/V
    context, so this is a theoretical upper bound, not a parity claim.

``measured_save_F``
    The observed difference in a paired close experiment.  It is empirical
    evidence, not a proof, and is comparable only on the same cases and the
    same control.

The audit verifies from the trace:

* prior debt pairing (admit/edit/revoke flips and changed inputs);
* sampler self-conditioning (drains never consume the sampler; every dirty
  row has an SC entry; the first verify sees full SC coverage);
* intra-turn dependency (terminal suffix, first-chunk coverage, and the
  write-before-attention code evidence);
* the encoder barrier (verify covers all B rows; the encoder flag occurs
  only on the last verify, iff another block follows);
* the hard cap (offer in {32,64,128,256}, q <= offer, drain = debt[:offer]).

Only the standard library is used.  No existing module is modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


B = 256
OFFERS = (32, 64, 128, 256)
PHASES = ("sample", "drain", "verify")
REQUIRED = ("tick", "ordinal", "phase", "offer", "q", "rows", "sampled",
            "dirty_before", "admitted", "edited", "revoked", "encoded")


class AuditError(ValueError):
    """A trace or sealed summary violates the pinned close contract."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _int(value: Any, name: str) -> int:
    need(type(value) is int, f"{name} must be int, got {value!r}")
    return value


def _int_list(value: Any, name: str) -> list[int]:
    need(isinstance(value, list) and all(type(x) is int for x in value),
         f"{name} must be a list of ints, got {value!r}")
    return list(value)


def _rows(value: Any, name: str, width: int = B) -> list[int]:
    xs = _int_list(value, name)
    need(xs == sorted(set(xs)), f"{name} must be sorted unique, got {xs!r}")
    need(all(0 <= x < width for x in xs), f"{name} outside [0,{width}): {xs!r}")
    return xs


def _event(raw: Any, index: int, width: int, offers: Sequence[int]) -> dict:
    need(isinstance(raw, dict), f"event {index} is not a dict")
    missing = [key for key in REQUIRED if key not in raw]
    need(not missing, f"event {index} missing {missing}")
    tick = _int(raw["tick"], "tick")
    ordinal = _int(raw["ordinal"], "ordinal")
    phase = raw["phase"]
    need(phase in PHASES, f"tick {tick}: unknown phase {phase!r}")
    offer = _int(raw["offer"], "offer")
    need(offer in offers, f"tick {tick}: offer {offer} not in {tuple(offers)}")
    q = _int(raw["q"], "q")
    need(0 < q <= offer, f"tick {tick}: q={q} outside (0,{offer}]")
    rows = _rows(raw["rows"], "rows", width)
    need(len(rows) == q, f"tick {tick}: q={q} but {len(rows)} rows")
    sampled = _rows(raw["sampled"], "sampled", width)
    dirty = _rows(raw["dirty_before"], "dirty_before", width)
    admitted = _rows(raw["admitted"], "admitted", width)
    edited = _rows(raw["edited"], "edited", width)
    revoked = _rows(raw["revoked"], "revoked", width)
    need(type(raw["encoded"]) is bool, f"tick {tick}: encoded must be bool")
    need(set(sampled) <= set(rows), f"tick {tick}: sampled outside rows")
    need(set(admitted) <= set(sampled), f"tick {tick}: admitted outside sampled")
    need(set(edited) <= set(sampled), f"tick {tick}: edited outside sampled")
    need(set(revoked) <= set(sampled), f"tick {tick}: revoked outside sampled")
    need(bool(sampled) == (phase == "sample"),
         f"tick {tick}: sampled rows must be present iff phase is sample")
    if phase in ("drain", "verify"):
        need(not sampled and not admitted and not edited and not revoked,
             f"tick {tick}: {phase} is not a pure projection")
    return dict(raw, tick=tick, ordinal=ordinal, phase=phase, offer=offer, q=q,
                rows=rows, sampled=sampled, dirty_before=dirty, admitted=admitted,
                edited=edited, revoked=revoked, encoded=raw["encoded"])


def _ledger(events: Sequence[dict], width: int, stop: str | None) -> dict:
    """Replay prior acceptance/input pairing and prove every drain is pure."""
    accepted: set[int] = set()
    seen: set[int] = set()
    initialized: set[int] = set()
    priority: set[int] = set()
    debt_min: set[int] = set()
    debt_max: set[int] = set()
    write_or_flip = noise = pure_drains = pure_drain_q = 0
    first_verify_seen = False
    for index, event in enumerate(events):
        before = set(event["dirty_before"])
        rows = set(event["rows"])
        sampled = set(event["sampled"])
        add, edit, revoke = (set(event[key]) for key in ("admitted", "edited", "revoked"))
        tick = event["tick"]
        need(before <= seen, f"tick {tick}: dirty row lacks self-conditioning")
        if index == 0:
            need(not before, f"block starts with outstanding debt {sorted(before)}")
        else:
            need(debt_min <= before <= debt_max,
                 f"tick {tick}: debt {sorted(before)} outside prior pairing "
                 f"[{sorted(debt_min)}, {sorted(debt_max)}]")
        need(not add & accepted, f"tick {tick}: admitted row already accepted")
        need(edit <= accepted, f"tick {tick}: edited row was not accepted")
        need(revoke <= accepted, f"tick {tick}: revoked row was not accepted")
        need(not edit & revoke, f"tick {tick}: row edited and revoked")
        need(priority <= before, f"tick {tick}: unpaid transition missing from debt")
        if event["phase"] == "drain":
            need(rows <= before, f"tick {tick}: drain includes clean rows")
            need(rows == set(sorted(before)[:event["offer"]]),
                 f"tick {tick}: drain is not the sorted debt prefix")
            write_or_flip += len(rows & priority)
            noise += len(rows - priority)
            pure_drains += 1
            pure_drain_q += len(rows)
        accepted = (accepted - revoke) | add | edit
        seen |= sampled
        initialized |= rows
        if event["phase"] == "sample":
            debt_min = (before - rows) | add | edit | revoke
            debt_max = (before - rows) | sampled
        elif event["phase"] == "drain":
            debt_min = debt_max = before - rows
        else:
            need(not before, f"tick {tick}: verify started with debt {sorted(before)}")
            debt_min = debt_max = set()
            if not first_verify_seen:
                first_verify_seen = True
                need(seen == set(range(width)),
                     f"tick {tick}: verify before full self-conditioning coverage")
                need(initialized == set(range(width)),
                     f"tick {tick}: verify before full canvas initialization")
        priority = (priority - rows) | add | edit | revoke
    need(not priority, "terminal token write missed")
    if stop == "resolved":
        need(accepted == set(range(width)), "resolved block is not fully accepted")
    elif stop == "limit":
        need(initialized == set(range(width)), "limit block is not fully initialized")
    elif stop == "native_stop":
        need(seen == set(range(width)), "native_stop block lacks full SC coverage")
    else:
        need(stop == "incomplete", f"unknown stop_reason {stop!r}")
        raise AuditError("incomplete block must never reach verify")
    return dict(write_or_flip_Q=write_or_flip, unaccepted_noise_Q=noise,
                pure_drains=pure_drains, pure_drain_Q=pure_drain_q,
                accepted=len(accepted), seen=len(seen), initialized=len(initialized))


def _audit_block(events: Sequence[dict], ordinal: int, has_next: bool,
                 width: int, offers: Sequence[int]) -> dict:
    need(events and events[0]["ordinal"] == ordinal,
         f"block {ordinal}: wrong ordinal")
    verify_indices = [i for i, event in enumerate(events) if event["phase"] == "verify"]
    need(verify_indices, f"block {ordinal}: no verify")
    first_verify = verify_indices[0]
    need(verify_indices == list(range(first_verify, len(events))),
         f"block {ordinal}: verify is not a suffix")
    need(all(event["phase"] in ("sample", "drain") for event in events[:first_verify]),
         f"block {ordinal}: non-sample/drain before verify")
    need(events[first_verify]["dirty_before"] == [],
         f"block {ordinal}: repacked close trace is not a control trace; "
         "use audit_paired for the measured treatment")

    cursor = first_verify
    while cursor > 0 and events[cursor - 1]["phase"] == "drain":
        cursor -= 1
    terminal = list(events[cursor:first_verify])
    need(cursor == 0 or events[cursor - 1]["phase"] == "sample",
         f"block {ordinal}: terminal drains not preceded by a sample")
    non_terminal = [event for event in events[:cursor] if event["phase"] == "drain"]

    verify = list(events[first_verify:])
    chunks: list[list[int]] = []
    offset = 0
    for event in verify:
        expected = list(range(offset, min(width, offset + event["offer"])))
        need(event["rows"] == expected,
             f"tick {event['tick']}: verify chunk is not the sequential sweep")
        chunks.append(event["rows"])
        offset += len(event["rows"])
    need(offset == width, f"block {ordinal}: verify does not cover the canvas")
    need(sorted(set().union(*(set(chunk) for chunk in chunks))) == list(range(width)),
         f"block {ordinal}: verify chunks overlap or leave gaps")
    for index, event in enumerate(verify):
        need(event["encoded"] == (index == len(verify) - 1 and has_next),
             f"tick {event['tick']}: wrong encoder flag")
    stop = verify[0].get("stop_reason")
    need(all(event.get("stop_reason") == stop for event in verify),
         f"block {ordinal}: verify stop_reason changed")
    ledger = _ledger(events, width, stop)

    first_chunk = set(verify[0]["rows"])
    terminal_rows = set().union(*(set(event["rows"]) for event in terminal)) if terminal else set()
    suffix: list[dict] = []
    for event in reversed(terminal):
        if set(event["rows"]) <= first_chunk:
            suffix.append(event)
        else:
            break
    suffix.reverse()
    exact_ticks = [event["tick"] for event in suffix]
    exact_rows = sorted(set().union(*(set(event["rows"]) for event in suffix))) if suffix else []
    exact_f = len(suffix)
    exact_q = sum(event["q"] for event in suffix)
    repack_f = len(terminal)
    repack_q = sum(event["q"] for event in terminal)
    hits = [len(terminal_rows & set(chunk)) for chunk in chunks]
    first_plus_dirty = len(first_chunk | terminal_rows)
    fits_original_first = first_plus_dirty <= len(first_chunk)
    dirty_fits_one = bool(terminal_rows) and max(hits) == len(terminal_rows)
    repack_fit = bool(terminal_rows) and len(terminal_rows) <= len(first_chunk)
    repack_exact = bool(terminal_rows) and len(terminal_rows & first_chunk) == len(terminal_rows)
    repack_changes_partition = repack_fit and not repack_exact

    non_droppable = []
    for index, event in enumerate(terminal):
        if event in suffix:
            continue
        own_outside = sorted(set(event["rows"]) - first_chunk)
        later_outside = any(set(later["rows"]) - first_chunk for later in terminal[index + 1:])
        non_droppable.append(dict(
            tick=event["tick"], rows=event["rows"], q=event["q"],
            own_rows_outside_first_verify=own_outside,
            later_terminal_drain_outside_first_verify=later_outside,
            reason=("rows_outside_first_verify" if own_outside else
                    "retained_later_drain_would_read_stale_kv")))
    non_terminal_reasons = [
        dict(tick=event["tick"], q=event["q"],
             reason="not_terminal_a_sample_step_follows_before_verify")
        for event in non_terminal
    ]
    last_sample = next((event["tick"] for event in reversed(events[:cursor])
                        if event["phase"] == "sample"), None)

    if terminal and exact_f == len(terminal):
        proof = "exact_all_terminal_drains"
    elif exact_f:
        proof = "exact_suffix"
    else:
        proof = "no_exact_drop"
    return dict(
        ordinal=ordinal,
        stop_reason=stop,
        variant="control",
        last_sample_tick=last_sample,
        verify_ticks=[event["tick"] for event in verify],
        verify_chunks=chunks,
        verify_F=len(verify),
        verify_Q=sum(event["q"] for event in verify),
        terminal_drain_ticks=[event["tick"] for event in terminal],
        terminal_drain_widths=[event["q"] for event in terminal],
        terminal_drain_rows=[event["rows"] for event in terminal],
        terminal_drain_F=repack_f,
        terminal_drain_Q=repack_q,
        other_drain_F=len(non_terminal),
        other_drain_Q=sum(event["q"] for event in non_terminal),
        dirty_by_verify_chunk=hits,
        first_verify_plus_all_dirty_Q=first_plus_dirty,
        fits_original_first=fits_original_first,
        dirty_fits_one_original_chunk=dirty_fits_one,
        exact_suffix_save_F=exact_f,
        exact_suffix_save_Q=exact_q,
        exact_suffix_ticks=exact_ticks,
        exact_suffix_rows=exact_rows,
        repack_ceiling_F=repack_f,
        repack_ceiling_Q=repack_q,
        repack_fit=repack_fit,
        repack_exact=repack_exact,
        repack_changes_partition=repack_changes_partition,
        gap_F=repack_f - exact_f,
        gap_Q=repack_q - exact_q,
        proof=proof,
        non_droppable_terminal=non_droppable,
        non_terminal_drains=non_terminal_reasons,
        conditions=dict(
            pure_terminal_drains=all(
                event["sampled"] == [] and not event["admitted"]
                and not event["edited"] and not event["revoked"] for event in terminal),
            terminal_suffix=True,
            suffix_covered_by_first_verify=all(
                set(event["rows"]) <= first_chunk for event in suffix),
            self_conditioning_frozen=True,
            encoder_barrier=True,
            hard_cap=True,
        ),
        self_conditioning=dict(
            drains_consume_sampler=False,
            dirty_rows_with_sc=True,
            first_verify_full_sc=True,
            first_verify_seen=ledger["seen"],
            first_verify_initialized=ledger["initialized"],
            width=width,
        ),
        intra_turn=dict(
            terminal_suffix=True,
            suffix_covered_by_first_verify=all(
                set(event["rows"]) <= first_chunk for event in suffix),
            non_droppable_terminal=non_droppable,
            non_terminal_drains=non_terminal_reasons,
        ),
        encoder_barrier=dict(
            verify_chunks=len(chunks),
            covers_canvas=True,
            encoded_ticks=[event["tick"] for event in verify if event["encoded"]],
            encoded_only_last_verify=True,
            last_verify_encoded_iff_next_block=True,
            encoder_calls=sum(event["encoded"] for event in verify),
        ),
        hard_cap=dict(
            offers=sorted({event["offer"] for event in events}),
            max_q=max(event["q"] for event in events),
            first_chunk_size=len(first_chunk),
            first_chunk_le_offer=len(first_chunk) <= verify[0]["offer"],
            drain_is_debt_prefix=True,
        ),
        drain_composition=dict(
            drain_write_or_flip_Q=ledger["write_or_flip_Q"],
            drain_unaccepted_noise_Q=ledger["unaccepted_noise_Q"],
            pure_drains=ledger["pure_drains"],
            pure_drain_Q=ledger["pure_drain_Q"],
        ),
        ledger=dict(accepted=ledger["accepted"], seen=ledger["seen"],
                    initialized=ledger["initialized"]),
    )


def audit_trace(trace: Sequence[dict], *, width: int = B,
                offers: Sequence[int] = OFFERS) -> dict:
    """Audit a control trace and return per-block close opportunities."""
    need(isinstance(trace, list) and trace, "trace must be a nonempty list")
    parsed = [_event(event, index, width, offers) for index, event in enumerate(trace)]
    need([event["tick"] for event in parsed] == list(range(len(parsed))),
         "ticks are not contiguous from 0")
    groups: list[list[dict]] = []
    for event in parsed:
        if not groups or event["ordinal"] != groups[-1][0]["ordinal"]:
            groups.append([event])
        else:
            groups[-1].append(event)
    need([group[0]["ordinal"] for group in groups] == list(range(len(groups))),
         "ordinals are not contiguous from 0")
    blocks = [_audit_block(group, ordinal, ordinal + 1 < len(groups), width, offers)
              for ordinal, group in enumerate(groups)]
    keys = ("exact_suffix_save_F", "exact_suffix_save_Q", "repack_ceiling_F",
            "repack_ceiling_Q", "gap_F", "gap_Q", "terminal_drain_F",
            "terminal_drain_Q", "verify_F", "verify_Q")
    return dict(variant="control", width=width, offers=list(offers), blocks=blocks,
                totals={key: sum(block[key] for block in blocks) for key in keys})


def audit_summary_block(block: dict, *, q: int = 128, width: int = B) -> dict:
    """Recompute exact suffix and repack ceiling from a sealed block summary.

    The summary does not carry row identities, so exactness is inferred from
    the sorted-debt contract: a terminal drain is ``debt[:offer]``.  The last
    terminal drain holds the highest debt rows; if it fits in the original
    first chunk, every earlier terminal drain fits too.  Otherwise the exact
    suffix is empty.  The summary's own fields are cross-checked.
    """
    need(isinstance(block, dict), "block summary must be a dict")
    widths = _int_list(block["terminal_drain_widths"], "terminal_drain_widths")
    hits = _int_list(block["dirty_by_verify_chunk"], "dirty_by_verify_chunk")
    verify_ticks = _int_list(block["verify_ticks"], "verify_ticks")
    terminal_ticks = _int_list(block["terminal_drain_ticks"], "terminal_drain_ticks")
    need(len(widths) == len(terminal_ticks), "terminal drain ticks/widths disagree")
    need(all(type(w) is int and w > 0 for w in widths), "terminal drain widths must be positive")
    need(len(hits) == len(verify_ticks), "dirty-by-chunk and verify ticks disagree")
    need(block["verify_F"] == len(verify_ticks), "verify_F disagrees with verify ticks")
    need(block["verify_Q"] == width, "verify_Q is not the full canvas")
    need(all(w <= q for w in widths), "terminal drain width exceeds the offer")
    need(sum(hits) == sum(widths), "terminal drains do not cover the terminal debt")
    dirty = sum(hits)
    first_chunk = min(q, width)
    exact_f = len(widths) if hits and hits[0] == dirty else 0
    exact_q = sum(widths) if exact_f else 0
    repack_f = len(widths)
    repack_q = sum(widths)
    first_plus_dirty = first_chunk + (dirty - hits[0]) if hits else first_chunk
    if widths and exact_f == repack_f:
        proof = "exact_all_terminal_drains"
    elif exact_f:
        proof = "exact_suffix"
    else:
        proof = "no_exact_drop"
    non_droppable = []
    for index in range(repack_f - exact_f):
        non_droppable.append(dict(
            tick=terminal_ticks[index], q=widths[index],
            reason=("rows_outside_first_verify"
                    if index == repack_f - exact_f - 1 and hits and hits[0] < dirty
                    else "retained_later_drain_would_read_stale_kv")))
    result = dict(
        ordinal=block["ordinal"],
        stop_reason=block["stop_reason"],
        last_sample_tick=block["last_sample_tick"],
        verify_ticks=verify_ticks,
        terminal_drain_ticks=terminal_ticks,
        terminal_drain_widths=widths,
        verify_F=block["verify_F"],
        verify_Q=block["verify_Q"],
        terminal_drain_F=repack_f,
        terminal_drain_Q=repack_q,
        dirty_by_verify_chunk=hits,
        drain_write_or_flip_Q=block["drain_write_or_flip_Q"],
        drain_unaccepted_noise_Q=block["drain_unaccepted_noise_Q"],
        pure_drains=block["pure_drains"],
        first_verify_plus_all_dirty_Q=first_plus_dirty,
        fits_original_first=first_plus_dirty <= first_chunk,
        dirty_fits_one_original_chunk=bool(dirty) and max(hits) == dirty,
        exact_suffix_save_F=exact_f,
        exact_suffix_save_Q=exact_q,
        repack_ceiling_F=repack_f,
        repack_ceiling_Q=repack_q,
        repack_fit=bool(dirty) and dirty <= first_chunk,
        repack_exact=bool(dirty) and bool(hits) and hits[0] == dirty,
        repack_changes_partition=bool(dirty) and dirty <= first_chunk
        and not (bool(hits) and hits[0] == dirty),
        gap_F=repack_f - exact_f,
        gap_Q=repack_q - exact_q,
        proof=proof,
        non_droppable_terminal=non_droppable,
        self_conditioning=dict(
            drains_consume_sampler=False,
            first_verify_full_sc="contract: begin_close requires len(sc) == B",
            first_verify_seen="not_identifiable_from_summary",
        ),
        encoder_barrier=dict(
            verify_chunks=len(verify_ticks),
            covers_canvas=block["verify_Q"] == width,
            encoded_ticks="not_identifiable_from_summary",
            encoded_only_last_verify="contract: StrictRequest sets encoded only when cursor == B",
        ),
        hard_cap=dict(
            offers=[q], max_terminal_drain_width=max(widths, default=0),
            first_chunk_size=first_chunk, first_chunk_le_offer=True,
            drain_is_debt_prefix=True,
        ),
    )
    expected = {
        "original_order_save_F": exact_f,
        "original_order_save_Q": exact_q,
        "repacked_ceiling_F": repack_f,
        "repacked_ceiling_Q": repack_q,
        "first_verify_plus_all_dirty_Q": first_plus_dirty,
        "fits_original_first": result["fits_original_first"],
        "dirty_fits_one_original_chunk": result["dirty_fits_one_original_chunk"],
    }
    for key, value in expected.items():
        need(block[key] == value,
             f"block {block['ordinal']}: sealed {key}={block[key]!r} "
             f"disagrees with recomputed {value!r}")
    return result


def audit_summary(document: dict, *, q: int = 128, width: int = B) -> dict:
    """Audit a sealed close-cost document (``arms`` or drain-grid ``rows``)."""
    need(isinstance(document, dict), "summary document must be a dict")
    if "arms" in document:
        arms = []
        for arm in document["arms"]:
            blocks = [audit_summary_block(block, q=q, width=width) for block in arm["blocks"]]
            keys = ("exact_suffix_save_F", "exact_suffix_save_Q", "repack_ceiling_F",
                    "repack_ceiling_Q", "gap_F", "gap_Q", "terminal_drain_F",
                    "terminal_drain_Q")
            totals = {key: sum(block[key] for block in blocks) for key in keys}
            for key in ("original_order_save_F", "original_order_save_Q",
                        "repacked_ceiling_F", "repacked_ceiling_Q"):
                need(arm[key] == totals[key.replace("original_order", "exact_suffix")
                                   .replace("repacked", "repack")],
                     f"arm {arm.get('index')}/{arm.get('k')}: sealed {key} "
                     f"disagrees with block sum")
            for key in ("drain_write_or_flip_Q", "drain_unaccepted_noise_Q", "pure_drains"):
                need(arm[key] == sum(block[key] for block in blocks),
                     f"arm {arm.get('index')}/{arm.get('k')}: sealed {key} "
                     f"disagrees with block sum")
            arms.append(dict(index=arm.get("index"), k=arm.get("k"), sha256=arm.get("sha256"),
                             F=arm.get("F"), Q=arm.get("Q"),
                             phase_F=arm.get("phase_F"), phase_Q=arm.get("phase_Q"),
                             blocks=blocks, **totals))
        by_k = {}
        for arm in arms:
            aggregate = by_k.setdefault(arm["k"], dict(
                k=arm["k"], indices=[], exact_suffix_save_F=0, exact_suffix_save_Q=0,
                repack_ceiling_F=0, repack_ceiling_Q=0, gap_F=0, gap_Q=0,
                terminal_drain_F=0, terminal_drain_Q=0))
            aggregate["indices"].append(arm["index"])
            for key in ("exact_suffix_save_F", "exact_suffix_save_Q", "repack_ceiling_F",
                        "repack_ceiling_Q", "gap_F", "gap_Q", "terminal_drain_F",
                        "terminal_drain_Q"):
                aggregate[key] += arm[key]
        return dict(kind="close_cost", arms=arms, by_k=list(by_k.values()),
                    budget=document.get("budget"), samples=document.get("sample_manifest_sha256"),
                    tokenizer=document.get("tokenizer_sha256"))
    if "rows" in document:
        rows = []
        for row in document["rows"]:
            rows.append(dict(
                q=row.get("q"), k=row.get("k"), requests=row.get("requests"),
                F=row.get("F"), Q=row.get("Q"), correct=row.get("correct"),
                terminal_drain_F=row.get("terminal_drain_F"),
                terminal_drain_Q=row.get("terminal_drain_Q"),
                pure_drains=row.get("pure_drains"),
                pure_drain_Q=row.get("pure_drain_Q"),
                note="aggregate only: no per-block rows, so exact suffix is not identifiable",
            ))
        return dict(kind="drain_grid", rows=rows)
    raise AuditError("summary document has neither 'arms' nor 'rows'")


def audit_paired(document: dict) -> dict:
    """Audit a measured paired close document and separate it from proof."""
    need(isinstance(document, dict) and isinstance(document.get("pairs"), list),
         "paired document must contain a list of pairs")
    pairs = []
    for pair in document["pairs"]:
        control = pair["arms"]["control"]
        treatment = pair["arms"]["dirty_first"]
        terminal = pair["control_terminal_drain"]
        delta = pair["delta_FQ_beyond_terminal_drains"]
        save_f = control["F"] - treatment["F"]
        save_q = control["Q"] - treatment["Q"]
        need(delta == {"F": terminal["F"] - save_f, "Q": terminal["Q"] - save_q},
             f"pair {pair['index']}: inconsistent measured accounting")
        need(delta == {"F": 0, "Q": 0},
             f"pair {pair['index']}: measured delta is not explained by terminal drains")
        pairs.append(dict(
            index=pair["index"], dataset=pair.get("dataset"), task_id=pair.get("task_id"),
            seed=pair.get("seed"),
            control=dict(N=control["N"], F=control["F"], Q=control["Q"],
                         phase_F=control.get("phase_F"), phase_Q=control.get("phase_Q"),
                         correct=control.get("correct")),
            dirty_first=dict(N=treatment["N"], F=treatment["F"], Q=treatment["Q"],
                             phase_F=treatment.get("phase_F"), phase_Q=treatment.get("phase_Q"),
                             correct=treatment.get("correct")),
            measured_save_F=save_f, measured_save_Q=save_q,
            control_terminal_drain_F=terminal["F"], control_terminal_drain_Q=terminal["Q"],
            delta_beyond_terminal_drains=delta,
            identical_output_ids=pair["identical_output_ids"],
            first_divergent_decoder_tick_0based=pair["first_divergent_decoder_tick_0based"],
            first_divergence_ordinal=pair["first_divergence_ordinal"],
            dirty_count_at_first_divergence=pair["dirty_count_at_first_divergence"],
            all_sample_observations_matched=pair["all_sample_observations_matched"],
            identical_full_argmax_blocks=pair["identical_full_argmax_blocks"],
            verify_chunks_same_rows_and_logits_hash=pair["verify_chunks_same_rows_and_logits_hash"],
            verify_chunks_different_rows_not_logit_comparable=pair[
                "verify_chunks_different_rows_not_logit_comparable"],
        ))
    control_f = sum(pair["control"]["F"] for pair in pairs)
    control_q = sum(pair["control"]["Q"] for pair in pairs)
    treatment_f = sum(pair["dirty_first"]["F"] for pair in pairs)
    treatment_q = sum(pair["dirty_first"]["Q"] for pair in pairs)
    return dict(
        kind="paired_close",
        budget=document["settings"]["decoder_cap"], k=document["settings"]["prefix_quota"],
        samples=document["provenance"]["samples_sha256"], tokenizer=document["provenance"]["tokenizer_json_sha256"],
        historical=document["provenance"]["historical_K0_sha256"],
        cases=[pair["index"] for pair in pairs],
        pairs=pairs,
        totals=dict(
            control_F=control_f, control_Q=control_q,
            dirty_first_F=treatment_f, dirty_first_Q=treatment_q,
            measured_save_F=control_f - treatment_f,
            measured_save_Q=control_q - treatment_q,
            control_terminal_drain_F=sum(pair["control_terminal_drain_F"] for pair in pairs),
            control_terminal_drain_Q=sum(pair["control_terminal_drain_Q"] for pair in pairs),
            identical_output_pairs=sum(pair["identical_output_ids"] for pair in pairs),
            measured_equals_terminal=(
                control_f - treatment_f == sum(pair["control_terminal_drain_F"] for pair in pairs)
                and control_q - treatment_q == sum(pair["control_terminal_drain_Q"] for pair in pairs)),
        ),
    )


def audit_metrics(document: dict) -> dict:
    """Cross-check the sealed per-arm metrics (F/Q and phase accounting)."""
    need(isinstance(document, dict) and isinstance(document.get("rows"), list),
         "metrics document must contain a list of rows")
    rows = []
    for row in document["rows"]:
        phase_f = row.get("phase_F", {})
        phase_q = row.get("phase_Q", {})
        need(row["F"] == row["decoder_F"] + row["encoder_F"],
             f"row {row.get('index')}/{row.get('k')}: F != decoder_F + encoder_F")
        need(row["Q"] == row["decoder_Q"] + row["encoder_Q"],
             f"row {row.get('index')}/{row.get('k')}: Q != decoder_Q + encoder_Q")
        need(sum(phase_f.values()) == row["decoder_F"],
             f"row {row.get('index')}/{row.get('k')}: phase_F != decoder_F")
        need(sum(phase_q.values()) == row["decoder_Q"],
             f"row {row.get('index')}/{row.get('k')}: phase_Q != decoder_Q")
        need(row["decoder_q_max"] <= row["q"],
             f"row {row.get('index')}/{row.get('k')}: decoder_q_max exceeds the offer")
        need(abs(row["N_per_F"] - row["N"] / row["F"]) < 1e-9,
             f"row {row.get('index')}/{row.get('k')}: N_per_F mismatch")
        need(abs(row["N_per_Q"] - row["N"] / row["Q"]) < 1e-9,
             f"row {row.get('index')}/{row.get('k')}: N_per_Q mismatch")
        rows.append(dict(index=row.get("index"), k=row.get("k"), q=row.get("q"),
                         N=row.get("N"), F=row.get("F"), Q=row.get("Q"),
                         correct=row.get("correct"), phase_F=phase_f, phase_Q=phase_q,
                         decoder_F=row.get("decoder_F"), decoder_Q=row.get("decoder_Q"),
                         encoder_F=row.get("encoder_F"), encoder_Q=row.get("encoder_Q"),
                         decoder_q_max=row.get("decoder_q_max")))
    return dict(kind="metrics", rows=rows)


CODE_CLAIMS = (
    ("canvas_B", "gemma_strict_rows.py", "B = 256"),
    ("hard_cap_offers", "gemma_strict_rows.py", "OFFERS = (32, 64, 128, 256)"),
    ("token_cap", "gemma_strict_eval.py", "B, CAP = 256, 2048"),
    ("denoise_steps", "gemma_strict_sampler.py", "steps = 48"),
    ("close_status_debt", "gemma_strict_rows.py", "if self.dirty:"),
    ("close_status_resolved", "gemma_strict_rows.py", "if len(self.accepted) == B:"),
    ("close_status_limit", "gemma_strict_rows.py", "if self.age == self.max_steps:"),
    ("drain_debt_prefix", "gemma_strict_rows.py", "rows, sampled = tuple(debt[:offer]), ()"),
    ("prior_pairing_dirty", "gemma_strict_rows.py",
     "self.dirty = (self.dirty.difference(plan.rows) | flips |"),
    ("drain_no_sampler", "gemma_strict_request.py", "rows.finish(plan)"),
    ("sample_finish_vectors", "gemma_strict_request.py",
     "rows.finish(plan, accepted_after=s.accepted[0,idx].tolist(),"),
    ("self_conditioning_read", "gemma_strict_request.py", "sc = self.sampler.sc"),
    ("encoder_barrier_commit", "gemma_strict_request.py",
     'record["encoded"] = self._commit()'),
    ("encoder_barrier_encode", "gemma_strict_request.py", "self._encode(B, first=False)"),
    ("sc_update_sampled_only", "gemma_strict_sampler.py", "self.sc[row] = signal.clone()"),
    ("consume_requires_sample_phase", "gemma_strict_sampler.py",
     'if self.phase != "sample" or self.age >= self.steps:'),
    ("close_requires_full_sc", "gemma_strict_sampler.py",
     'if self.phase != "sample" or len(self.sc) != self.block:'),
    ("projector_begin", "gemma_strict_project.py",
     "self.bank.begin(pos,self.origin,ids.device,causal=True)"),
    ("kv_write_before_read_k", "gemma_wave_kv.py",
     "k_store.index_copy_(-2, index, k[..., -q_len:, :])"),
    ("kv_write_before_read_v", "gemma_wave_kv.py",
     "v_store.index_copy_(-2, index, v[..., -q_len:, :])"),
    ("kv_live_after_write", "gemma_wave_kv.py", "k_live = k_store.index_select(-2, live_index)"),
    ("same_block_visibility", "gemma_wave_kv.py",
     "visible = k_block[None, :] <= q_block[:, None]"),
    ("encoder_flag_audit", "gemma_strict_eval.py",
     'require(e["encoded"] == (i+1 < len(trace)), "Wrong terminal encoder flag")'),
    ("close_chunk_merge", "gemma_strict_close.py",
     "take = tuple(sorted(self._order[self.cursor:self.cursor+q]))"),
    ("close_chunk_project", "gemma_strict_close.py",
     "sampler.close_chunk(take, lambda r, ids, sc: lane.project("),
)


def code_evidence(root: Path) -> list[dict]:
    """Return exact file/line evidence for the invariants used by the proof."""
    evidence = []
    base = root / "experiments" / "t2t" / "serving_slots"
    for claim, name, pattern in CODE_CLAIMS:
        path = base / name
        need(path.is_file(), f"missing source {path}")
        lines = path.read_text().splitlines()
        hits = [index + 1 for index, line in enumerate(lines) if pattern in line]
        need(hits, f"pattern not found in {name}: {pattern!r}")
        evidence.append(dict(claim=claim, file=name, line=hits[0],
                             text=lines[hits[0] - 1].strip()))
    return evidence


DEFINITIONS = {
    "exact_suffix_save_F": (
        "Terminal drains in the contiguous suffix immediately before verify whose "
        "rows are completely contained in the first verify chunk. Under the pinned "
        "projector every selected row writes K/V before attention at every layer, "
        "so these rows are refreshed before any query can read them; SC, sampler "
        "clock and encoder barrier are unchanged. This is a proof on the trace."),
    "repack_ceiling_F": (
        "All terminal drains. Reaching it requires repacking the verify sweep "
        "(dirty rows first). If dirty rows are not already in the original first "
        "chunk, the partition changes and later chunks read a different K/V "
        "context; this is a theoretical upper bound, not parity."),
    "measured_save_F": (
        "Observed control-minus-dirty-first difference in a paired close run. "
        "Empirical only; not a proof and not transferable to other cases."),
    "gap_F": (
        "repack_ceiling_F minus exact_suffix_save_F: the part that can only be "
        "reached by changing the verify partition."),
}

INVARIANTS = {
    "self_conditioning": dict(
        claim=("Drains never consume the sampler; self-conditioning is updated only "
               "for sampled rows; every dirty row has an SC entry before it is "
               "refreshed; the first verify requires full SC coverage."),
        code=["drain_no_sampler", "sample_finish_vectors", "self_conditioning_read",
              "sc_update_sampled_only", "consume_requires_sample_phase",
              "close_requires_full_sc"],
        trace_checks=["every drain has sampled=[] and no admit/edit/revoke",
                      "dirty_before <= seen",
                      "seen == full at the first verify"],
    ),
    "intra_turn_dependency": dict(
        claim=("Every selected row writes its layer K/V before attention, so a row "
               "is refreshed before any query can read it only if it is in the same "
               "physical forward. All rows of the same block are visible to each "
               "other, so a dropped terminal drain is exact only when its rows are "
               "in the first verify chunk; otherwise chunk 0 reads stale KV and a "
               "later drain would read it too."),
        code=["projector_begin", "kv_write_before_read_k", "kv_write_before_read_v",
              "kv_live_after_write", "same_block_visibility"],
        trace_checks=["terminal drains are a contiguous suffix before verify",
                      "suffix rows are a subset of the first verify chunk",
                      "no retained drain follows the dropped suffix"],
    ),
    "encoder_barrier": dict(
        claim=("The encoder runs only after the verify sweep covers all B rows; the "
               "encoded flag occurs only on the last verify chunk, iff another block "
               "follows. Removing drains does not change verify coverage or the "
               "encoder boundary."),
        code=["encoder_barrier_commit", "encoder_barrier_encode", "encoder_flag_audit"],
        trace_checks=["verify chunks partition [0,B)",
                      "encoded iff last verify and has_next_block"],
    ),
    "prior_pairing": dict(
        claim=("Debt is the union of unpaid admit/edit/revoke transitions and changed "
               "inputs. A drain retires exactly its rows; the next debt must lie "
               "between the flip lower bound and the sampled-row upper bound."),
        code=["prior_pairing_dirty", "sample_finish_vectors"],
        trace_checks=["debt_min <= dirty_before <= debt_max",
                      "priority <= dirty_before",
                      "terminal token write is paid"],
    ),
    "hard_cap": dict(
        claim=("Offers are in {32,64,128,256}; the canvas is B256, the token cap is "
               "2048 and the denoising clock is 48 steps; every physical Q is at most "
               "the offer; a drain is the sorted debt prefix debt[:offer]; the first "
               "verify chunk is at most the offer."),
        code=["canvas_B", "hard_cap_offers", "token_cap", "denoise_steps",
              "drain_debt_prefix"],
        trace_checks=["q <= offer", "drain rows == sorted(debt)[:offer]",
                      "len(first_verify_chunk) <= offer"],
    ),
}


def _verdict(summary: dict, paired: dict | None) -> list[dict]:
    out = []
    for arm in summary.get("by_k", summary.get("arms", [])):
        item = dict(k=arm["k"],
                    exact_suffix_save_F=arm["exact_suffix_save_F"],
                    exact_suffix_save_Q=arm["exact_suffix_save_Q"],
                    repack_ceiling_F=arm["repack_ceiling_F"],
                    repack_ceiling_Q=arm["repack_ceiling_Q"],
                    gap_F=arm["gap_F"], gap_Q=arm["gap_Q"])
        if paired is not None and arm["k"] == 0:
            need(paired["k"] == 0 and paired["budget"] == summary["budget"] and
                 summary["samples"] and summary["samples"] == paired["samples"] and
                 summary["tokenizer"] and summary["tokenizer"] == paired["tokenizer"],
                 "prior control settings or manifest mismatch")
            controls = {p["index"]:p["control"] for p in paired["pairs"]}
            need(len(controls) == len(paired["pairs"]) == len(arm["indices"])
                 and set(controls) == set(arm["indices"]), "prior control cases mismatch")
            for prior in (x for x in summary["arms"] if x["k"] == 0):
                i = prior["index"]
                h = paired["historical"].get(f"cpu-q{paired['budget']}-k0-{i:02d}")
                need(h and set(h) == {"compact","run","evidence"} and h == prior["sha256"] and
                     (controls[i]["F"],controls[i]["Q"]) == (prior["F"],prior["Q"]),
                     f"prior control artifact mismatch for index {i}")
            totals = paired["totals"]
            item.update(
                measured_save_F=totals["measured_save_F"],
                measured_save_Q=totals["measured_save_Q"],
                measured_beyond_exact_F=totals["measured_save_F"] - arm["exact_suffix_save_F"],
                measured_beyond_exact_Q=totals["measured_save_Q"] - arm["exact_suffix_save_Q"],
                measured_equals_ceiling=(
                    totals["measured_save_F"] == arm["repack_ceiling_F"]
                    and totals["measured_save_Q"] == arm["repack_ceiling_Q"]),
                measured_equals_exact=(
                    totals["measured_save_F"] == arm["exact_suffix_save_F"]
                    and totals["measured_save_Q"] == arm["exact_suffix_save_Q"]),
                identical_output_pairs=totals["identical_output_pairs"],
            )
        out.append(item)
    return out


def build_report(summary_paths: Iterable[Path] = (), paired_path: Path | None = None,
                 trace_path: Path | None = None, *, root: Path | None = None,
                 include_code: bool = True, q: int = 128) -> dict:
    """Build a JSON-serializable audit report from existing receipts only."""
    root = root or Path(__file__).resolve().parents[3]
    summaries = []
    for path in summary_paths:
        document = json.loads(Path(path).read_text())
        first = document.get("rows", [{}])[0] if isinstance(document.get("rows"), list) else {}
        if "arms" in document or "terminal_drain_F" in first:
            summary = audit_summary(document, q=q)
        elif "phase_F" in first:
            summary = audit_metrics(document)
        else:
            raise AuditError(f"{path}: unrecognized summary format")
        summaries.append(dict(path=str(path), **summary))
    paired = audit_paired(json.loads(Path(paired_path).read_text())) if paired_path else None
    trace = audit_trace(json.loads(Path(trace_path).read_text())) if trace_path else None
    primary = next((item for item in summaries if item["kind"] == "close_cost"), None)
    verdict = _verdict(primary, paired) if primary else []
    report = dict(
        tool="close_opportunity.py",
        scope=("Read-only audit of existing Gemma B256 strict receipts. No model, "
               "no GPU, no CPU inference, no new run, no quality or latency claim."),
        definitions=DEFINITIONS,
        invariants=INVARIANTS,
        code_evidence=code_evidence(root) if include_code else [],
        summaries=summaries,
        paired=paired,
        trace=trace,
        verdict=verdict,
        limits=[
            "The exactness proof is conditional on the pinned projector writing every "
            "selected row's K/V before attention at every layer (code evidence).",
            "The repack ceiling changes the verify partition; identical outputs in three "
            "cases are an empirical observation, not a general proof.",
            "No sample, nonterminal drain, verify coverage, encoder, wall-clock or "
            "quality saving is claimed.",
            "Unified Q512 route traces use different phases and a two-canvas window; "
            "they are not mixed into this strict Q128 audit.",
            "No existing module was modified; this tool only reads files.",
        ],
    )
    if verdict:
        parts = []
        for item in verdict:
            text = (f"k={item['k']}: exact {item['exact_suffix_save_F']} F / "
                    f"{item['exact_suffix_save_Q']} Q; ceiling {item['repack_ceiling_F']} F / "
                    f"{item['repack_ceiling_Q']} Q")
            if "measured_save_F" in item:
                text += (f"; measured {item['measured_save_F']} F / "
                         f"{item['measured_save_Q']} Q (beyond exact "
                         f"{item['measured_beyond_exact_F']} F / "
                         f"{item['measured_beyond_exact_Q']} Q)")
            parts.append(text)
        report["conclusion"] = " | ".join(parts)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", default=[],
                        help="sealed close-cost or drain-grid JSON (repeatable)")
    parser.add_argument("--paired", type=Path, help="paired close JSON")
    parser.add_argument("--trace", type=Path, help="strict control trace JSON")
    parser.add_argument("--q", type=int, default=128, help="summary decoder offer")
    parser.add_argument("--root", type=Path, default=None,
                        help="checkout root for code evidence")
    parser.add_argument("--no-code-evidence", action="store_true")
    parser.add_argument("--out", type=Path, help="write JSON here instead of stdout")
    args = parser.parse_args(argv)
    report = build_report(args.summary, args.paired, args.trace, root=args.root,
                          include_code=not args.no_code_evidence, q=args.q)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(text)
        print(f"Wrote {args.out}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
