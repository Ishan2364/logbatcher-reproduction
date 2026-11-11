"""Sanity checks and diagnostics quoted in the paper. Everything here is free: it uses mock LLMs,
the response cache, or no LLM at all.

    python -m experiments.run_checks

Writes:
  results/E0_oracle.csv            pipeline with a perfect LLM, default matching
  results/E0_oracle_paper_match.csv  same, but with the paper's '(.*?)' placeholder matching
  results/E0_mask_control.csv      no LLM at all: templates are regex-masked logs (control)
  results/E0_prompt_v1_HDFS.csv    first prompt wording ("the single template shared by these logs")
  results/E2_call_stats.csv        logs per call, prompt/answer tokens, reasoning tokens (E1 replay)
  results/cluster_purity.csv       templates mixed in one cluster, cosine vs euclidean DBSCAN
"""
import numpy as np
import pandas as pd

import src.llm_client as llm
from experiments.common import apply_overrides, run_experiment
from src.config import dataset_names, load_config, project_path
from src.data_loader import load_dataset
from src.llm_client import INSTRUCTION, count_tokens, make_client
from src.logbatcher import run_dataset
from src.partition import partition

RES = project_path(load_config()["paths"]["results_dir"])

PROMPT_V1 = (
    "You are given a batch of log messages, one per line. They were printed by the same "
    "logging statement. Find the parts that change at run time (identifiers, numbers, "
    "addresses, paths, names, sizes, and similar) and replace each one with a {placeholder}; "
    "keep every constant word and symbol exactly as it appears. A log may contain no "
    "variables at all. Reply with the single template shared by these logs, wrapped in backticks."
)
# The wording shown in Figure 5 of the paper, to compare instruction length.
PAPER_INSTRUCTION = (
    "You will be provided with some log messages separated by line breaks. You must abstract "
    "variables with {placeholders} to extract the corresponding template. There might be no "
    "variables in the log message. Print the input log's template delimited by backticks."
)


def oracle_checks():
    run_experiment("E0_oracle", overrides={"llm.provider": "mock_oracle"})
    run_experiment("E0_oracle_paper_match", overrides={"llm.provider": "mock_oracle",
                                                        "logbatcher.match_mode": "paper"})
    run_experiment("E0_mask_control", overrides={"llm.provider": "mock_mask"})


def prompt_v1_check():
    current = llm.INSTRUCTION
    llm.INSTRUCTION = PROMPT_V1
    try:
        run_experiment("E0_prompt_v1_HDFS", ["HDFS"])
    finally:
        llm.INSTRUCTION = current


def call_stats():
    """Replay E1 from the response cache and describe the calls."""
    cfg = load_config()
    client = make_client(cfg)
    rows = []
    for name in dataset_names(cfg):
        start = len(client.calls)
        run_dataset(name, cfg, client=client)
        c = client.calls[start:]
        rows.append({"Dataset": name, "Calls": len(c),
                     "Mean_logs_per_call": np.mean([x["batch_size"] for x in c]),
                     "Prompt_tiktoken": np.mean([x["tt_input"] for x in c]),
                     "Answer_tiktoken": np.mean([x["tt_output"] for x in c]),
                     "Provider_input": np.mean([x["input_tokens"] for x in c]),
                     "Provider_output": np.mean([x["output_tokens"] for x in c])})
    c = client.calls
    rows.append({"Dataset": "All calls", "Calls": len(c),
                 "Mean_logs_per_call": np.mean([x["batch_size"] for x in c]),
                 "Prompt_tiktoken": np.mean([x["tt_input"] for x in c]),
                 "Answer_tiktoken": np.mean([x["tt_output"] for x in c]),
                 "Provider_input": np.mean([x["input_tokens"] for x in c]),
                 "Provider_output": np.mean([x["output_tokens"] for x in c])})
    d = pd.DataFrame(rows)
    d["Instruction_tokens_ours"] = count_tokens(INSTRUCTION)
    d["Instruction_tokens_paper"] = count_tokens(PAPER_INSTRUCTION)
    d.to_csv(RES / "E2_call_stats.csv", index=False, float_format="%.2f")
    print(d.tail(1).to_string(index=False))


def cluster_purity():
    """No LLM needed: how often does a cluster mix several labelled templates?"""
    rows = []
    for name in dataset_names():
        df = load_dataset(name)
        contents = df["Content"].tolist()
        template = dict(zip(df["Content"], df["EventTemplate"]))
        counts = df["Content"].value_counts()
        row = {"Dataset": name}
        for metric in ("cosine", "euclidean"):
            clusters, labels = partition(contents, 0.5, 5, metric)
            real = clusters[:-1] if (labels == -1).any() else clusters
            mixed = [g for g in real if len({template[x] for x in g}) > 1]
            row[f"{metric}_clusters"] = len(clusters)
            row[f"{metric}_outlier_lines"] = int((labels == -1).sum())
            row[f"{metric}_mixed_clusters"] = len(mixed)
            row[f"{metric}_lines_in_mixed"] = int(sum(counts[x] for g in mixed for x in g))
        rows.append(row)
    d = pd.DataFrame(rows)
    d.loc[len(d)] = {"Dataset": "Total", **d.drop(columns="Dataset").sum().to_dict()}
    d.to_csv(RES / "cluster_purity.csv", index=False)
    print(d.tail(1).to_string(index=False))


if __name__ == "__main__":
    oracle_checks()
    prompt_v1_check()
    call_stats()
    cluster_purity()
