"""Causal dose for optional prefix rows on an ordinary single-head sampling plan."""


def progress_quota(previous, current, parent_quota):
    """Inputs are local paid-prefix lengths at sampling-plan entry, not post-step counts.

    A caller must forget previous on head/epoch change and must not update it on
    a drain or verification. The ordinary sampler and mandatory rows own their
    original selection and are not subject to this quota.
    """
    for value in (current, parent_quota, *(() if previous is None else (previous,))):
        if type(value) is not int or value < 0:
            raise ValueError("Expected non-negative integer length or quota")
    return 0 if previous is not None and current > previous else parent_quota
