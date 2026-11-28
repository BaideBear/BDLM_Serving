from tools.audit_gemma_resident_horizon import measure


def test_retreat_counts_only_previously_paid_resolved_rows_as_clean():
    trace, decoder, states = [], [], []
    def call(pos, admitted, debt, sample, b=0):
        trace.append(dict(block=b, positions=pos, q=len(pos), visible_end=max(pos)+1,
                          admitted=admitted))
        decoder.append(dict(ids=[9]*len(pos), sample=sample))
        states.append(dict(debt=debt))
    call([0,1,2,3], [2,3], [2,3], {"rows":[0,1,2,3]})
    call([2,3], [], [], None)
    call([0], [], [], {"rows":[0]})
    call([4], [4], [4], {"rows":[0]}, b=1)
    arm=dict(trace=trace, decoder=decoder, states=states, bill={"D":len(trace)})
    counts, events = measure(arm,4,4)
    assert counts == {"sampling":3,"payment":1}
    assert events == [dict(tick=2,block=0,kind="sampling",rows=3,paid_resolved=2,
                           pending=0,unresolved=1,last_decoder=False)]
