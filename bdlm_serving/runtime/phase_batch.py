"""Keep full prompt prefill out of generation padding groups."""
import concurrent.futures
import queue
from bdlm_serving.runtime.timing import TimedAuditedBatcher


class PhaseBatcher(TimedAuditedBatcher):
    def run(self,functions):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as pool:
            futures=[pool.submit(fn) for fn in functions]
            while not all(f.done() for f in futures):
                jobs=[]
                while len(jobs)<min(self.batch_size,sum(not f.done() for f in futures)):
                    try:jobs.append(self.queue.get(timeout=.05))
                    except queue.Empty:continue
                if not jobs:continue
                try:
                    groups={}
                    for job in jobs:
                        phase=getattr(job["execution_cache"],"execution_phase",None)
                        if phase is None:
                            # Only qualification's legacy cache lacks the marker;
                            # it runs at batch1, so grouping cannot exceed its Q.
                            phase="legacy"
                        groups.setdefault("prefill" if phase=="prefill" else "decode",[]).append(job)
                    for group in groups.values():self.execute(group)
                except BaseException as exc:
                    for job in jobs:
                        if not job["future"].done():job["future"].set_exception(exc)
            return [f.result() for f in futures]
