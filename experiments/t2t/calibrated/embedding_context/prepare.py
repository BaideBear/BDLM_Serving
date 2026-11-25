"""Fit the preregistered 28 coefficients on calibration24; never read evaluation traces."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from bdlm_serving.evaluation.frozen import MANIFEST, ROOT, sha
from bdlm_serving.methods.t2t.embedding_context import CALIBRATION, DESIGN, Projector, context, input_table
from tools.edit_opportunity import fit, load
from tools.lexical_hazard import check_split


def prepare(cal, checkpoint, out):
    if out.resolve().is_relative_to(ROOT) or out.exists():
        raise ValueError('Choose a NEW private coefficient file outside the checkout')
    digest = sha(CALIBRATION)
    if sha(cal / 'weights.json') != digest:
        raise ValueError('Unexpected calibration24 manifest')
    train = json.loads(CALIBRATION.read_text())['samples']
    check_split(train, json.loads(MANIFEST.read_text())['samples'])
    with input_table(checkpoint) as table:
        projector = Projector(table)
        w = fit([load(cal / f"{t['index']:02d}.trace.jsonl.gz", t,
                      context_features=lambda *a: context(projector.unit, *a)) for t in train])
    if w.shape != (28,) or not np.isfinite(w).all():
        raise ValueError('Invalid context fit')
    payload = json.dumps(dict(**DESIGN, calibration_sha256=digest, weights=w.tolist()), allow_nan=False)
    out.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with os.fdopen(os.open(out, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), 'w') as f:
        f.write(payload + '\n')
    print('Private calibration24 coefficients:', out, 'SHA256:', sha(out))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('cal', 'checkpoint', 'out'):
        p.add_argument('--' + k, required=True, type=Path)
    prepare(**vars(p.parse_args()))
