"""Acquire native DiffusionGemma references on the same tasks with repeat seeds.

This is a thin entry point over the pinned native ``run_native`` path.  It does
not add a sparse runner, a supervisor, or any new sampling policy.  Each call
reuses the frozen task prompts and maps every seed by the predeclared
``REPEAT_SEED_OFFSET`` so the result is a same-task, different-seed reference in
the archived ``native256`` request format.
"""
from pathlib import Path
from types import SimpleNamespace

from ..evaluation.frozen import REPEAT_SEED_OFFSET
from .diffusiongemma import run_native


def acquire_native_reference(checkpoint, samples, output, *, prepare=None,
                             seed_offset=REPEAT_SEED_OFFSET):
    """Run native Gemma on the identical frozen tasks under repeat seeds.

    ``prepare(model)`` is called once after the checkpoint is loaded and before
    the model is moved to CUDA, matching ``run_native``'s capture hook.  The
    returned list contains per-request rows in the archived ``native256``
    format; the same rows are persisted as ``{index:02d}-native256.json`` under
    ``output``.
    """
    args = SimpleNamespace(checkpoint=Path(checkpoint), samples=Path(samples),
                           output=Path(output))
    return run_native(args, before_cuda=prepare, seed_offset=seed_offset)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "samples", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    rows = acquire_native_reference(args.checkpoint, args.samples, args.output)
    print(f"Wrote {len(rows)} native256 repeat-reference requests to {args.output}")
    return rows


if __name__ == "__main__":
    main()
