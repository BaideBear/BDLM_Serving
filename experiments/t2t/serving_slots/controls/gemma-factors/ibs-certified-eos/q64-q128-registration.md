# Fixed Q64 and Q128 continuation

Q32 is sealed separately. These are two independent four-arm comparisons
on the same physical A800 0 board as Q32, subject to the same occupancy checks
and release protocol. There is no retuning. Q256 is not rerun.

For each tier, the launch must supply Q explicitly. The launcher, worker,
chunk size, cap tuple, and stopping/payment offer must all agree. Every
request and every arm must execute a first real decoder projection of that Q;
all subsequent decoder forwards, including payment, are bounded by that Q.
Use the same optimizer controls, bf16 checkpoint, grouped_mm, SDPA, thresholds,
seeds, frozen90 order, and gate indices as the Q32 registration. Six paired
passes are required separately at each tier before proceeding to its own
remaining 84. The second tier requires a fresh idle check and lock.

Within each tier, stock is unmodified hardcap IBS, and control observes the
same post-update stop rule while finishing stock for its old-work bill. The
same-stop comparison cuts at the identical first eligible certificate and
charges the paid variant's same real suffix-only forward when applicable.
This gives neither a new selection baseline nor a competing sampler. No
future trace, final response, answer, or expected savings participates in an
online decision. After inference, report exact full IDs and per-request N
and score invariance, correctness, mean N/F, and sum(N)/sum(decoder Q),
with original30/new60 separately. Forward and row savings include payment.
