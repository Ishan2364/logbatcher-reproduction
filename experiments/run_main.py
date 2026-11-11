"""E1 (main comparison, Table 1) and E2 (token efficiency, Table 2).

    python -m experiments.run_main [--datasets HDFS Apache] [--set llm.model=openai/gpt-oss-20b]

Writes results/E1_logbatcher.csv (raw per-dataset run), results/E1_main.csv (side-by-side with
Drain and the paper's reported numbers) and results/E2_efficiency.csv.
"""
import argparse

import pandas as pd

from experiments.common import parse_cli_overrides, run_experiment
from src.baseline_drain import run as run_drain
from src.config import load_config, project_path


def build_tables(lb, results_dir):
    paper = pd.read_csv(results_dir / "paper_reported.csv")
    drain_path = results_dir / "drain_pp.csv"
    if not drain_path.exists():
        run_drain(postprocess=True)
    drain = pd.read_csv(drain_path)
    lb = lb[lb["Dataset"] != "Average"]
    t = (drain[["Dataset", "GA", "MLA", "ED"]].rename(columns=lambda c: c if c == "Dataset" else f"Drain_{c}")
         .merge(lb[["Dataset", "GA", "MLA", "ED"]].rename(columns=lambda c: c if c == "Dataset" else f"Ours_{c}"), on="Dataset")
         .merge(paper[["Dataset", "LB_GA", "LB_MLA", "LB_ED"]].rename(columns=lambda c: c.replace("LB_", "Paper_")), on="Dataset"))
    t = pd.concat([t, pd.DataFrame([{"Dataset": "Average", **t.mean(numeric_only=True)}])], ignore_index=True)
    t.to_csv(results_dir / "E1_main.csv", index=False, float_format="%.3f")

    e = (lb[["Dataset", "LLM_calls", "T_total", "T_invoc", "T_total_tiktoken", "T_invoc_tiktoken", "LLM_latency_s"]]
         .merge(paper[["Dataset", "LB_T_total", "LB_T_invoc", "LILAC_T_total", "LILAC_T_invoc"]]
                .rename(columns={"LB_T_total": "Paper_T_total", "LB_T_invoc": "Paper_T_invoc"}), on="Dataset"))
    e = pd.concat([e, pd.DataFrame([{"Dataset": "Average", **e.mean(numeric_only=True)}])], ignore_index=True)
    e.to_csv(results_dir / "E2_efficiency.csv", index=False, float_format="%.1f")
    return t, e


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. llm.max_wait_seconds=3600")
    ap.add_argument("--experiment", default="E1_logbatcher")
    args = ap.parse_args()
    lb = run_experiment(args.experiment, args.datasets, parse_cli_overrides(args.set))
    if args.experiment == "E1_logbatcher" and not args.datasets:
        t, e = build_tables(lb, project_path(load_config()["paths"]["results_dir"]))
        print(t.to_string(index=False))
        print(e.to_string(index=False))
