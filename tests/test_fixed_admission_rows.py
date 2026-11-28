from hashlib import sha256
import json

import pytest

from tools.fixed_admission_rows import case_digests, inspect_tick, MASK, output_ids, packed, tape_path


def tick(rows, ids, dirty=(), written=None, new=(), q=4, edits=()):
    changes = written or {}
    return dict(event=dict(rows=rows, dirty=sorted(set(dirty)&set(rows)), dirty_before=list(dirty),
                           dirty_after=list(set(dirty)-set(rows) | changes.keys()), admission_rows=list(new),
                           sampled=rows, prefix=[], written_absolute=list(changes), edited=list(edits), physical_q=len(rows),
                           decoder_Q=len(rows), history_Q=len(rows), layer_physical_q=[len(rows)]*20,
                           encoder_F=0, decoder_F=1, generation_F=1, offer=q,
                           admission_start=min(new) if new else None, over_offer_Q=max(0,len(rows)-q),
                           response_mode="full_block", nominal_offer=q, parent_plan_rows=rows[:q]),
                ids=ids, before=min(new) if new else 0, written=changes)


def test_disjoint_budget_and_crosscut_edit():
    c,d=inspect_tick(tick([1,2,3],[15,MASK,17],dirty=[1],written={1:145,2:120},edits=[1]),4,{1},1,2049)
    assert (c["writeback"],c["other_mask"],c["clean_decoded"],c["new_block"])==(1,1,1,0)
    assert c["t2t"]==c["t2t_on_writeback"]==1 and d=={1,2}


def test_optional_prefix_is_only_a_subset_of_clean_decoded():
    t=tick([4,5,6],[115,116,117],written={6:559},edits=[6])
    t["event"]["prefix"]=[6]
    c,_=inspect_tick(t,4,set(),0,2048)
    assert c["clean_decoded"]==3 and c["sampled_optional_prefix"]==c["t2t_on_optional_prefix"]==1


def test_initialization_is_not_regular_mask_or_an_arbitrary_overflow():
    rows=list(range(32,64))
    c,_=inspect_tick(tick(rows,[MASK]*32,new=rows),4,set(),3,2051)
    assert c["new_block"]==32 and c["other_mask"]==0 and c["over_offer_Q"]==28
    with pytest.raises(ValueError,match="new block"):
        inspect_tick(tick(rows[:8],[MASK]*8,new=rows[:8]),4,set(),3,2051)
    with pytest.raises(ValueError,match="overflow"):
        inspect_tick(tick(rows,[MASK]*32),4,set(),3,2051)
    with pytest.raises(ValueError,match="outstanding"):
        inspect_tick(tick([26],[MASK],dirty=[12]),4,{12},3,2051)


def test_transport_preserves_both_hashes_and_native_bytes_still_verify():
    orig = (packed(dict(widths={8: 20, 32: 2}, N=12)) + "\n").encode()
    row = json.loads(orig)
    received = (packed(row) + "\n").encode()
    producer_sha = sha256(orig).hexdigest()
    received_sha = sha256(received).hexdigest()
    assert producer_sha != received_sha
    assert case_digests(orig, row, producer_sha) == (producer_sha, producer_sha)
    with pytest.raises(ValueError, match="seal bytes"):
        case_digests(received, row, producer_sha)
    assert case_digests(received, row, producer_sha, transport=True) == (received_sha, producer_sha)


def test_transport_rejects_tampering_even_with_valid_receiver_serialization():
    orig = (packed(dict(widths={8: 20, 32: 2}, correct=True)) + "\n").encode()
    producer = sha256(orig).hexdigest()
    for patch, match in (
        ({"correct": False}, "seal digest"),
        ({"widths": {"8": 21, "32": 2}}, "seal digest"),
        ({"widths": {"08": 20, "32": 2}}, "noncanonical width keys"),
    ):
        altered = dict(json.loads(orig), **patch)
        with pytest.raises(ValueError, match=match):
            case_digests((packed(altered) + "\n").encode(), altered, producer, transport=True)
    row = json.loads(orig)
    with pytest.raises(ValueError, match="seal bytes"):
        case_digests((packed(row) + " \n").encode(), row, producer, transport=True)


def test_transport_raw_path_is_bound_to_attempt_and_index(tmp_path):
    name = "case-07.ndjson"
    row = {"raw": f"{tmp_path.name}/{name}"}
    assert tape_path(tmp_path, row, 7, transport=True) == tmp_path / name
    for wrong in (name, f"other/{name}", f"{tmp_path.name}/case-08.ndjson"):
        with pytest.raises(ValueError, match="transported raw path"):
            tape_path(tmp_path, {"raw": wrong}, 7, transport=True)
    (tmp_path / name).symlink_to(tmp_path / "other")
    with pytest.raises(ValueError, match="redirected raw path"):
        tape_path(tmp_path, row, 7, transport=True)


def test_output_reconstruction_checks_the_final_token_values():
    row = dict(prefill={"decoder_q": 17}, output_ids=2, N=1, end_reason="eos",
               ids_sha=sha256(b"[34,2]").hexdigest())
    assert output_ids(row, {16: 11, 17: 34, 18: 2, 19: 27}) == [34, 2]
    for writes in ({17: 34}, {17: 33, 18: 2}, {17: MASK, 18: 2}):
        with pytest.raises(ValueError, match="output token digest"):
            output_ids(row, writes)
