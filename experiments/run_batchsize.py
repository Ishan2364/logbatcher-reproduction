"""E5 (batch size, Figure 8): batch sizes 1, 5, 10, 15, 20 on the dataset subset.

    python -m experiments.run_batchsize --model claude-haiku-4.5 [--sizes 1 5 10] [--datasets ...]

Batch size 1 is the same request stream as "w/o batching" (E3) and 10 is the default, so those
runs come from the response cache when E3/E6 already ran.
"""
import argparse

from experiments.common import parse_cli_overrides, run_experiment, summarize
from experiments.run_ablation import model_overrides, slug
from experiments.run_extension import MODELS
from src.config import load_config


def run(model_key, sizes, datasets=None, extra=None):
    datasets = datasets or load_config()["subset"]
    base = {**model_overrides(model_key), **(extra or {})}
    results = {}
    for k in sorted(sizes, key=lambda k: k == 10):     # default size last (cached from E1/E6)
        results[f"batch_size={k}"] = run_experiment(f"E5_batchsize_{slug(model_key)}_bs{k}", datasets,
                                                    {**base, "logbatcher.batch_size": k})
    # summarize() reports % change against the first row; put the default (10) first if present
    order = sorted(results, key=lambda l: (l != "batch_size=10", int(l.split("=")[1])))
    return summarize({l: results[l] for l in order}, f"E5_batchsize_{slug(model_key)}",
                     cfg_note={"model": MODELS[model_key], "datasets": datasets, "sizes": sizes})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="gpt-oss-120b")
    ap.add_argument("--sizes", nargs="*", type=int, default=[1, 5, 10, 15, 20])
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    run(args.model, args.sizes, args.datasets, parse_cli_overrides(args.set))
