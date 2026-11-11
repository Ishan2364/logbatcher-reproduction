"""Phase 2: parsing metrics (GA, MLA, ED), token metrics, and result reporting.

All metric functions take two equal-length sequences of template strings, one per log line,
in the same line order: predicted templates and ground-truth templates.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import regex as re
from rapidfuzz.distance import Levenshtein

from src.config import load_config, project_path


def normalize_template(t):
    """Whitespace normalisation used before any string comparison."""
    return re.sub(r"\s+", " ", str(t)).strip()


def group_accuracy(pred, gt):
    """GA: a log is correct iff the set of logs sharing its predicted template equals the set
    of logs sharing its ground-truth template."""
    pred = pd.Series([normalize_template(p) for p in pred])
    gt = pd.Series([normalize_template(g) for g in gt])
    if len(pred) != len(gt):
        raise ValueError("pred and gt must have the same length")
    pred_groups = pred.groupby(pred).groups      # template -> index of logs
    gt_size = gt.value_counts()
    correct = 0
    for idx in pred_groups.values():
        true_labels = gt.loc[idx].unique()
        # predicted group must be pure and must contain the whole true group
        if len(true_labels) == 1 and gt_size[true_labels[0]] == len(idx):
            correct += len(idx)
    return correct / len(gt)


def message_level_accuracy(pred, gt):
    """MLA / parsing accuracy: exact template match per log, after whitespace normalisation."""
    ok = [normalize_template(p) == normalize_template(g) for p, g in zip(pred, gt)]
    return float(np.mean(ok))


def edit_distance_score(pred, gt):
    """ED: 1 - mean normalised (character-level) Levenshtein distance; higher is better."""
    d = [Levenshtein.normalized_distance(normalize_template(p), normalize_template(g)) for p, g in zip(pred, gt)]
    return 1.0 - float(np.mean(d))


def token_metrics(calls):
    """calls: list of dicts with input_tokens / output_tokens per LLM invocation (cached or not)."""
    n = len(calls)
    total = sum(c["input_tokens"] + c["output_tokens"] for c in calls)
    return {
        "LLM_calls": n,
        "T_total": int(total),
        "T_invoc": round(total / n, 1) if n else 0.0,
        "T_input": int(sum(c["input_tokens"] for c in calls)),
        "T_output": int(sum(c["output_tokens"] for c in calls)),
    }


def evaluate(pred, gt):
    return {
        "GA": group_accuracy(pred, gt),
        "MLA": message_level_accuracy(pred, gt),
        "ED": edit_distance_score(pred, gt),
        "PredTemplates": len({normalize_template(p) for p in pred}),
        "TrueTemplates": len({normalize_template(g) for g in gt}),
    }


def report(rows, experiment, config=None, results_dir=None):
    """Save one row per dataset to results/<experiment>.csv (plus an Average row) and the run
    config to results/<experiment>.json. Returns the DataFrame."""
    results_dir = Path(results_dir or project_path(load_config()["paths"]["results_dir"]))
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    num = df.select_dtypes("number").columns
    avg = {c: df[c].mean() for c in num}
    avg["Dataset"] = "Average"
    for c in df.columns:
        if c not in avg:
            avg[c] = ""
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    df.to_csv(results_dir / f"{experiment}.csv", index=False, float_format="%.4f")
    if config is not None:
        meta = {"experiment": experiment, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "config": config}
        with open(results_dir / f"{experiment}.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
    return df
