import pytest
from types import SimpleNamespace as NS
from threading import Event

from experiments.t2t.serving_slots.fifo import Inbox, Pressure, arrival_trace, replay
from experiments.t2t.serving_slots.fifo_report import clock_quality, validate


def test_independent_arrivals_during_a_forward(monkeypatch):
    third,first = Event(),Event()
    put = Inbox.put
    def producer(self,i,e):
        if i>0:
            assert first.wait(5)
        put(self,i,e)
        if i==2: third.set()
    monkeypatch.setattr(Inbox,"put",producer)
    observed=[]
    def execute(i,offer):
        observed.append(offer(NS(tick=0)))
        if i==0:
            first.set()
            assert third.wait(5)
            observed.append(offer(NS(tick=1)))
        return i
    trace = [(0,0),(1,0),(2,0)]
    rows,times=replay(trace,"busy_low",(32,64),execute,lambda:None)
    assert rows==[0,1,2] and observed==[64,32,32,64]
    validate(times,trace,"busy_low",(32,64))
    times[0]["offers"][1]=(1,times[0]["offers"][1][1],0,64)
    with pytest.raises(ValueError):
        validate(times,trace,"busy_low",(32,64))


def test_same_arrivals_and_complementary_policy():
    cases=[dict(index=i) for i in range(90)]
    trace=arrival_trace(cases,80)
    assert trace==arrival_trace(cases[::-1],80)
    assert sorted(i for i,_ in trace)==list(range(90))
    assert [t for _,t in trace[:7]]==[0,10,20,30,40,50,640]
    assert [[p.offer(d) for d in (0,1,2,1,0)]
            for p in (Pressure("busy_low",(8,16)),Pressure("busy_high",(8,16)))]==[
                [16,16,8,8,16],[8,8,16,16,8]]


def test_arrival_jitter_cannot_disappear_inside_endpoint_span():
    trace=arrival_trace([dict(index=i) for i in range(90)],8000)
    def rows(delay):
        return [dict(planned=t,enqueued=t+(delay if j%2 else 0),finish=t+10000)
                for j,(_,t) in enumerate(trace)]
    assert clock_quality(rows(0),8000)["clock_pass"]
    assert not clock_quality(rows(500),8000)["clock_pass"]
