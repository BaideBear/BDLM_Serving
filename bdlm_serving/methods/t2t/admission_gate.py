"""First-approved LLaDA Q8 offer: score only the marginal base MASK."""
import json
import math
import random

SCHEMA = 'llada21-q8-victim-admission-v1'
CAL_SHA = 'd4457807244a5941cc43dc7ed3b4ba665eba962b8730c73babf8705cf1476478'
RANK_SHA = 'dbcaf8d1496bbd4867a9ad9f44e592bec520c14872c1cb6476f257521d03081d'
FEATURES = ('intercept', 'mask_ordinal', 'mask_population', 'fresh_population',
            'left_mask_gap', 'right_mask_gap', 'nearest_fresh', 'local_mask_fraction')
MODES = ('hard_admit_low', 'hard_admit_coin')


def vector(move):
    if move['reason'] != 'swap_mask':
        raise ValueError('Admission risk is defined only on a real crowded offer')
    v = move['victim']
    masks, fresh = sorted(move['masks']), move['fresh']
    if v not in masks or v in fresh or v == masks[0]:
        raise ValueError('Expected a non-protected valid MASK victim')
    idx = masks.index(v)
    gap = lambda d: min(d, 8) / 8
    return (1., idx / 7, len(masks) / 32, len(fresh) / 8,
            gap(v-masks[idx-1]), gap(masks[idx+1]-v if idx+1 < len(masks) else 8),
            gap(min((abs(j-v) for j in fresh), default=8)),
            sum(0 < abs(j-v) <= 3 for j in masks) / 6)


def risk(x, fit):
    z = sum(a*b for a,b in zip(x, fit['weights'], strict=True))
    return 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))


def read_admission_fit(path):
    if path is None:
        raise ValueError('Missing admission fit')
    from pathlib import Path
    fit = json.loads(Path(path).read_text())
    if (fit.get('schema') != SCHEMA or fit.get('calibration_sha256') != CAL_SHA
            or fit.get('ranker_sha256') != RANK_SHA or fit.get('features') != list(FEATURES)
            or len(fit['weights']) != len(FEATURES) or not all(map(math.isfinite, fit['weights']))
            or not (0 < fit['cutoff'] < 1 and 0 < fit['keep_probability'] < 1)
            or fit['fit_requests'] != 24):
        raise ValueError('Invalid admission fit or provenance')
    return fit


def admission_decision(move, fit, mode, seed, block, clock):
    if mode not in MODES:
        raise ValueError('Unknown timing arm')
    p = risk(vector(move), fit)
    if mode == 'hard_admit_low':
        return p <= fit['cutoff'], p
    u = random.Random(f'victim-admit-v1:{seed}:{block}:{clock}').random()
    return u < fit['keep_probability'], p
