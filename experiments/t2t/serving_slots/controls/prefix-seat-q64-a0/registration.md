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
