"""Offline, independent-request microbatches. No serving policy or model imports."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Condition


@dataclass
class Call:
    lane: int
    phase: str
    width: int
    data: object


def pack(calls, target=512):
    """Rectangular physical cost is batch * max(width), not sum(width).

    Separate incompatible phases. Descending widths and a 7/8 useful floor
    prevent a large build/prefill from absorbing many almost-empty lanes.
    Oversize singletons are not split: doing so would change a forward.
    """
    if type(target) is not int or target < 1:
        raise ValueError("positive row target required")
    groups = {}
    for c in calls:
        if type(c.width) is not int or c.width < 1:
            raise ValueError("invalid projection width")
        groups.setdefault(c.phase, []).append(c)
    for phase, group in groups.items():
        ordered = sorted(group, key=lambda c: (-c.width, c.lane))
        batch, useful = [], 0
        for c in ordered:
            paid = (len(batch)+1)*batch[0].width if batch else c.width
            if batch and (paid > target or 8*(useful+c.width) < 7*paid or phase == "encoder"):
                yield batch
                batch, useful = [], 0
            batch.append(c)
            useful += c.width
        if batch:
            yield batch


def waves(functions, execute, *, max_active, target=512):
    """Workers own their controllers. Only the caller executes model forwards.

    A function accepts call(phase, real_width, payload). A failed physical
    forward can already have written KV; abort every lane without replay.
    """
    if type(max_active) is not int or max_active < 1:
        raise ValueError("max_active must be a qualified positive resident limit")
    functions = list(functions)
    results = [None]*len(functions)
    cv, pending, replies, live, errors = Condition(), {}, {}, set(), []
    def submit(lane, phase, width, data):
        with cv:
            if errors:
                raise RuntimeError("wave aborted") from errors[0]
            pending[lane] = Call(lane, phase, width, data)
            cv.notify_all()
            cv.wait_for(lambda: errors or lane in replies)
            if errors:
                raise RuntimeError("wave aborted") from errors[0]
            return replies.pop(lane)

    def worker(i):
        try:
            results[i] = functions[i](lambda *args: submit(i, *args))
        except BaseException as exc:
            with cv:
                if not errors:
                    errors.append(exc)
                cv.notify_all()
        finally:
            with cv:
                live.discard(i)
                cv.notify_all()

    with ThreadPoolExecutor(max_workers=max_active) as pool:
        for off in range(0, len(functions), max_active):
            with cv:
                live.update(range(off, min(off+max_active, len(functions))))
                for i in sorted(live):
                    pool.submit(worker, i)
            try:
                while True:
                    with cv:
                        cv.wait_for(lambda: errors or len(pending) == len(live))
                        if errors:
                            raise errors[0]
                        if not live:
                            break
                        batch = list(pending.values())
                        pending.clear()
                    # Publish one wave atomically, after all shape groups complete.
                    out = {}
                    for group in pack(batch, target):
                        out.update(zip((c.lane for c in group), execute(group), strict=True))
                    with cv:
                        replies.update(out)
                        cv.notify_all()
            except BaseException as exc:
                with cv:
                    if not errors:
                        errors.append(exc)
                    cv.notify_all()
                raise
    return results
