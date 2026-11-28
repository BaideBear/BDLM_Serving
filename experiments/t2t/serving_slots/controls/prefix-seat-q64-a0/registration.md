# Does one omitted paid prefix seat displace the rightmost eligible warm MASK?

Before execution: DiffusionGemma, paid IBS, fixed Q64, physical A0 only, fixed
six indices 7,13,25,33,55,71, no replacements. Stock and observer must reproduce
each other bit for bit. The seat arm is a production arm, not a read-only probe.

Per physical block, at most one successful exchange. It may fire only on an
ordinary full-width sampling step (`len(rows)==64`) with unresolved rows. All
dirty rows in the plan are protected. The immutable paid generated prefix
(`ready>0`) must be resolved and debt-free. The exchange uses the shared
`choose_prefix_seat`: donor is the rightmost selected resident non-dirty MASK
that is neither the first actual unresolved MASK nor the original right edge;
incoming row is the rightmost omitted paid prefix row with resident K/V and
absolute position below both the paid prefix and the original right edge. At
least one selected unresolved MASK remains. The physical width, IDs and visible
right edge are unchanged. No cold MASK, prompt row, successor row, acceptance
threshold, RNG rule or cross-block causal boundary is touched.

The incoming resolved prefix row is projected as a query and receives fresh K/V;
because its prior is resolved, it is not passed to the MASK admission sampler.
Any token change elsewhere follows the existing native rule and is followed by a
real writeback in the normal paid debt ledger. Report N, paid F, selected decoder
Q, correctness, seat identities and the first trace fork. The six cases are only
a screening pair: no 90 expansion, profile default, TPF claim, or quality claim
without a separately registered full cohort.

## Launch gate after the failed instrumentation attempts

No prefix-seat case is audited as valid yet. The retained
`gemma-ibs-certified-eos-a0-q64-20261128T110532Z/raw.log` case 7 stock record is
an interface smoke only; it is not prefix-seat evidence. It was replayed through
the production `RECEIPT(kind=case).arms.stock ->
tools.run_gemma_ibs_certified_eos.reconcile` interface on n164 and validated
N=205, F=18, Q=1088, and `admissions == paid_rows == 256`. The compact receipt is
on n164 at
`results/gemma-q64-prefix-seat-a0-interface-smoke-v1/case-07-interface.json`.

Before another A0 launch, the worker must serialize this same production arm
record; the CPU reducer must report the retained smoke as an interface pass and
the prefix-seat reducer must require complete matched rows. Do not retry the
failed six solely to recover lost tickets. Any follow-up remains a fixed-budget,
one-physical-row selection exchange; do not modify confidence or acceptance
thresholds.
