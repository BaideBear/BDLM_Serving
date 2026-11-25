"""CUDA intervals bracket the model call before request futures are released."""
import torch
from bdlm_serving.runtime.audited_batch import AuditedBatcher


class TimedAuditedBatcher(AuditedBatcher):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model_interval=None
        def before(module,args):
            self.model_interval=(torch.cuda.Event(enable_timing=True),
                                 torch.cuda.Event(enable_timing=True))
            self.model_interval[0].record()
        def after(module,args,output):
            self.model_interval[1].record()
        self.handles.append(self.model.register_forward_pre_hook(before))
        self.handles.append(self.model.register_forward_hook(after))

    def execute(self,jobs):
        self.model_interval=None
        super().execute(jobs)
        begin,end=self.model_interval
        end.synchronize()
        self.physical_calls[-1]["model_cuda_interval_ms"]=begin.elapsed_time(end)
        self.physical_calls[-1]["timing_scope"]="CUDA event span bracketed around model; includes launch gaps and possibly other requests' shared-stream sampler work, not summed kernel-active time"
