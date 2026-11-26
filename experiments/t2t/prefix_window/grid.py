"""Print the frozen list of runner arms for one model, experiment and budget."""
import argparse

from bdlm_serving.methods.t2t.protocol import Arm


MODELS = {"llada21": (32, (4, 8, 16, 32, 64), 8, 2),
          "gemma": (256, (16, 32, 64, 128, 256), 32, 4)}


def arms(model, q, study):
    block, budgets, pilot, pilot_k = MODELS[model]
    if q not in budgets:
        raise ValueError(f"{q} is not a registered {model} budget")
    ks = [1, pilot_k] if q == pilot else [1]
    if study == "matched":
        specs = [(0, "hard_prefix_zero")]
        modes = ("hard_prefix_frontier_right", "hard_prefix_random")
    elif study == "slack":
        # Reuse the same already-qualified zero arm for this Q when scoring.
        specs = []
        modes = ("hard_prefix_slack_right", "hard_prefix_slack_random")
    elif study == "oldest":
        if q != pilot:
            raise ValueError("First oldest-row pilot is restricted to the two registered budgets")
        specs = []
        modes = ("hard_prefix_slack_oldest",)
    elif study == "dose":
        if model != "gemma":
            raise ValueError("The cross-budget dose grid is registered for Gemma")
        ks = [2] if q == pilot else [2, 4]
        specs = []
        modes = ("hard_prefix_slack_right", "hard_prefix_slack_random")
    else:
        raise ValueError(f"Unknown study {study}")
    for k in ks:
        specs.extend((k, mode) for mode in modes)
    commands = [f"{q}:0:{q-k}:{mode}" for k, mode in specs]
    for s in commands:
        Arm.parse(s, block)
    return commands


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", choices=MODELS)
    p.add_argument("q", type=int)
    p.add_argument("study", choices=("matched", "slack", "oldest", "dose"))
    args = p.parse_args()
    print(" ".join(arms(args.model, args.q, args.study)))
