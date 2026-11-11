"""E3 (module ablation, Table 3) and E4 (sampling method, Table 4).

    python -m experiments.run_ablation --study ablation --model claude-haiku-4.5
    python -m experiments.run_ablation --study sampling --model claude-haiku-4.5 [--datasets ...]

Runs on config.yaml's `subset` unless --datasets is given. The "Full"/"DPP" row repeats the default
configuration, so it is served from the response cache when E1/E6 already ran that model.
"""
import argparse
import re

from experiments.common import parse_cli_overrides, run_experiment, summarize
from experiments.run_extension import MODELS
from src.config import load_config

STUDIES = {
    "ablation": ("E3_ablation", {
        "Full LogBatcher": {},
        "w/o partitioning": {"logbatcher.use_partitioning": False},
        "w/o caching": {"logbatcher.use_cache": False},
        "w/o batching": {"logbatcher.use_batching": False},
    }),
    "sampling": ("E4_sampling", {
        "DPP (default)": {},
        "Random sampling": {"logbatcher.sampler": "random"},
        "Similarity sampling": {"logbatcher.sampler": "similarity"},
    }),
}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def model_overrides(model_key):
    m = MODELS[model_key]
    return {"llm.provider": m["provider"], "llm.model": m["model"], "llm.max_wait_seconds": 3600}


def run_study(study, model_key, datasets=None, extra=None):
    experiment, variants = STUDIES[study]
    datasets = datasets or load_config()["subset"]
    base = {**model_overrides(model_key), **(extra or {})}
    results = {}
    # default configuration last: it is the same request stream as E1/E6, so by then it is cached
    for label in sorted(variants, key=lambda l: variants[l] == {}):
        results[label] = run_experiment(f"{experiment}_{slug(model_key)}_{slug(label)}", datasets,
                                        {**base, **variants[label]})
    results = {l: results[l] for l in variants}
    return summarize(results, f"{experiment}_{slug(model_key)}",
                     cfg_note={"model": MODELS[model_key], "datasets": datasets, "variants": variants})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=list(STUDIES), required=True)
    ap.add_argument("--model", choices=list(MODELS), default="gpt-oss-120b")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--set", nargs="*", default=[], help="extra config overrides")
    args = ap.parse_args()
    run_study(args.study, args.model, args.datasets, parse_cli_overrides(args.set))
