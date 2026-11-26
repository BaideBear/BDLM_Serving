import random
from copy import deepcopy

import pytest

from bdlm_serving.methods.t2t import admission_gate as a
from experiments.t2t.calibrated.embedding_context import admission_study as study


def test_prestep_victim_only_and_inclusive_cutoff():
    offer = dict(reason='swap_mask', masks=[5, 6, 8, 12, 13, 22], base=[4,5,6,8],
                 victim=8, fresh=[1,4], entrant=0)
    x = a.vector(offer)
    assert x == (1., 2/7, 6/32, 2/8, 2/8, 4/8, 4/8, 2/6)
    changed = dict(offer, entrant=2, before_id=9, after_id=7, truth=True)
    assert a.vector(changed) == x
    fit = dict(weights=[0.]*8, cutoff=.5, keep_probability=.5)
    assert a.admission_decision(offer, fit, 'hard_admit_low', 1, 2, 3) == (True,.5)
    state = random.getstate()
    assert a.admission_decision(offer, fit, 'hard_admit_coin', 1, 2, 3) == (
        random.Random('victim-admit-v1:1:2:3').random() < .5,.5)
    assert random.getstate() == state
    for kw in [dict(reason='swap_old'), dict(victim=5), dict(victim=4)]:
        with pytest.raises(ValueError):
            a.vector(dict(offer, **kw))


def test_training_does_not_use_entrant_edit_label():
    x = (1., 0., 0., 0., 0., .4, .1, .2)
    z = (1., .5, .1, .2, .4, .2, .1, 0.)
    data = {i:[(x, 0, 0), (z, 0, 1)] for i in range(24)}
    fit = study.train(data)
    altered = deepcopy(data)
    altered[0] = [(x,1,0), (z,1,1)]
    assert study.train(altered) == fit
    assert fit['keep_probability'] == .5
    assert a.risk(x,fit) < fit['cutoff'] < a.risk(z,fit)
