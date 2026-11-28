from types import SimpleNamespace

from bdlm_serving.methods.t2t.budget_family import Law
from experiments.t2t.serving_slots.gemma_full_admission import (
    FullAdmissionHandle,
    FullAdmissionRequest,
    admission_certificate,
)


def certificate(**changes):
    return admission_certificate(
        changes.get("initialized", set()),
        changes.get("dirty", set()),
        changes.get("accepted", set()),
        ordinal=changes.get("ordinal", 1),
        nominal_q=changes.get("nominal_q", 64),
        parent_rows=changes.get("parent_rows", list(range(64))))


def test_first_new_mask_block_over_parent_is_certified():
    item = certificate()
    assert item["physical_q"] == 256
    assert item["over_parent_rows"] == list(range(64, 256))
    assert item["explainable"]


def test_dirty_or_accepted_rows_are_not_eligible():
    assert not certificate(dirty={0})["explainable"]
    assert not certificate(accepted={0})["explainable"]


def test_first_canvas_or_parent_full_selection_is_not_admission():
    assert not certificate(ordinal=0)["explainable"]
    assert not certificate(parent_rows=list(range(256)))["explainable"]


def test_certificate_takes_block_local_rows_not_absolute_rows():
    try:
        admission_certificate(set(), set(), set(), ordinal=1, nominal_q=64,
                              parent_rows=list(range(64)),
                              full_rows=list(range(256, 512)))
    except ValueError:
        pass
    else:
        raise AssertionError("absolute admission rows were accepted")
    item = admission_certificate(set(), set(), set(), ordinal=1, nominal_q=64,
                                 parent_rows=list(range(64)))
    assert item["rows"] == list(range(256))


def first_projection_fixture():
    request = object.__new__(FullAdmissionRequest)
    request.origin = 2
    request.ordinal = 1
    request.cursor = 0
    request.tail = None
    request.nominal_offer = 64
    rows = SimpleNamespace(_pending=None, initialized=set(), dirty=set(),
                           accepted=set())
    canvas = SimpleNamespace(ordinal=1, rows=rows, can_sample=True,
                             sampler=SimpleNamespace(age=0, phase="sample"))
    # The parent plan addresses the fresh block in absolute physical space.
    return request, canvas, tuple(range(258, 322))


def test_first_projection_certifies_local_parent_and_projects_absolute_full():
    request, canvas, pos = first_projection_fixture()
    certificate, full = request._first_projection(
        (canvas,), pos, pos, pos, "sparse")
    base = request.origin + 256
    assert certificate["rows"] == list(range(256))
    assert certificate["parent_rows"] == list(range(64))
    assert certificate["over_parent_rows"] == list(range(64, 256))
    assert certificate["explainable"]
    assert full == tuple(range(base, base + 256))


def test_handle_binds_event_to_precommit_ordinal():
    handle = object.__new__(FullAdmissionHandle)
    request = SimpleNamespace(ordinal=0, status="ready", law=Law())
    event = {"q": 256, "ordinal": 0, "offer": 64,
             "decoder_Q": 256, "decoder_F": 1, "encoder_F": 1,
             "encoder_Q": 256, "full_admission": {"explainable": True,
                                                  "nominal_q": 64,
                                                  "physical_q": 256}}
    def step(q):
        # The successful nonterminal canvas is committed and promoted before
        # the handle observes the event.
        request.ordinal = 1
        return event
    request.step = step
    handle._lane = lambda: None
    handle._status = "ready"
    handle._request = request
    handle._offers = (64,)
    handle._laws = (Law(),)
    handle._block_size = 256
    handle._result = None
    out = handle.step(64)
    assert out["ordinal"] == 0
    assert request.status == "ready"
    assert handle._status == "ready"
