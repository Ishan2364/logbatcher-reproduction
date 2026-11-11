"""E6 (extension, Option A): accuracy vs cost across LLMs.

    python -m experiments.run_extension --run gpt-oss-20b claude-haiku-4.5
    python -m experiments.run_extension --summarize

gpt-oss-120b is the main model, so its numbers come from E1 (results/E1_logbatcher.csv) rather
than a second run. Cost = provider-reported input/output tokens x the per-1M prices in config.yaml.
"""
import argparse

import pandas as pd

from experiments.common import run_experiment
from src.config import load_config, project_path

MODELS = {
    "gpt-oss-120b": {"experiment": "E1_logbatcher", "provider": "groq", "model": "openai/gpt-oss-120b",
                     "label": "gpt-oss-120b (Groq)"},
    "gpt-oss-20b": {"experiment": "E6_gpt-oss-20b", "provider": "groq", "model": "openai/gpt-oss-20b",
                    "label": "gpt-oss-20b (Groq)"},
    "claude-haiku-4.5": {"experiment": "E6_claude-haiku-4.5", "provider": "bedrock",
                         "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                         "label": "Claude Haiku 4.5 (Bedrock)"},
}


def run(names, datasets=None, extra_overrides=None):
    for name in names:
        m = MODELS[name]
        if m["experiment"] == "E1_logbatcher":
            print(f"{name} is the E1 model; run experiments.run_main instead")
            continue
        overrides = {"llm.provider": m["provider"], "llm.model": m["model"], **(extra_overrides or {})}
        run_experiment(m["experiment"], datasets, overrides)


def summarize():
    cfg = load_config()
    res = project_path(cfg["paths"]["results_dir"])
    rows, per_ds = [], []
    for name, m in MODELS.items():
        path = res / f"{m['experiment']}.csv"
        if not path.exists():
            print(f"(skip {name}: {path.name} not found yet)")
            continue
        df = pd.read_csv(path)
        df = df[df["Dataset"] != "Average"].copy()
        p_in, p_out = cfg["llm"]["pricing"].get(m["model"], [None, None])
        cost = None if p_in is None else (df["T_input"].sum() * p_in + df["T_output"].sum() * p_out) / 1e6
        rows.append({
            "Model": m["label"], "Provider": m["provider"], "Datasets": len(df),
            "GA": df["GA"].mean(), "MLA": df["MLA"].mean(), "ED": df["ED"].mean(),
            "LLM_calls": int(df["LLM_calls"].sum()), "T_total": int(df["T_total"].sum()),
            "T_invoc": df["T_total"].sum() / max(df["LLM_calls"].sum(), 1),
            "T_output": int(df["T_output"].sum()),
            "Cost_USD": cost, "Latency_per_call_s": df["LLM_latency_s"].sum() / max(df["LLM_calls"].sum(), 1),
            "Fallbacks": int(df["fallbacks"].fillna(0).sum()) if "fallbacks" in df else 0,
        })
        per_ds.append(df[["Dataset", "GA", "MLA", "ED", "LLM_calls", "T_total"]].assign(Model=m["label"]))
    out = pd.DataFrame(rows)
    out.to_csv(res / "E6_extension.csv", index=False, float_format="%.4f")
    if per_ds:
        pd.concat(per_ds).to_csv(res / "E6_extension_per_dataset.csv", index=False, float_format="%.4f")
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", choices=list(MODELS), default=[])
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.run:
        run(args.run, args.datasets, {"llm.max_wait_seconds": 3600})
    if args.summarize or not args.run:
        summarize()
