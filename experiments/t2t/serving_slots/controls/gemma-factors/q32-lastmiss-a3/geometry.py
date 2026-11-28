"""The stock Q32 row allocator does not exclude a last-miss exchange."""
from bdlm_serving.methods.t2t.budget_family import Law, choose


def witness():
    masks = set(range(40, 100))
    previous_debt = set(range(100, 108))
    prefix = tuple(range(39, -1, -1))
    prev = choose(32, 256, previous_debt, masks, prefix, law=Law())
    # A rejected row can redraw its old ID. The sampler can also edit accepted
    # prefix/debt rows; changed IDs on those rows form the next mandatory set.
    current_debt = previous_debt | set(prev.prefix)
    cur = choose(32, 256, current_debt, masks,
                 tuple(p for p in prefix if p not in current_debt), law=Law())
    take = 60
    protected = {min(masks), max(cur.rows)} | current_debt | set(cur.prefix)
    drop = max(set(cur.mask) - protected)
    alt = (set(cur.rows)-{drop}) | {take}
    assert take in prev.mask and take not in cur.rows and take not in current_debt
    assert len(alt) == len(cur.rows) == 32 and protected <= alt and max(alt) == max(cur.rows)
    return dict(previous_rows=prev.rows, current_rows=cur.rows, prefix=cur.prefix,
                debt=sorted(current_debt), drop=drop, take=take, rightmost=max(alt))


if __name__ == "__main__":
    import json

    print(json.dumps(witness()))
