"""Shared experiment runner: config overrides, per-dataset predictions, resumable result files.

Every LLM response is cached, so an interrupted experiment (e.g. Groq daily token cap) is
resumed by simply running the same command again.
"""
import json
import sys
import time

import pandas as pd

from src.config import dataset_names, load_config, project_path
from src.evaluate import report
from src.llm_client import BudgetExhausted, make_client
from src.logbatcher import run_dataset


def apply_overrides(cfg, overrides):
    """overrides: {"logbatcher.batch_size": 5, "llm.model": "..."}"""
    for dotted, value in (overrides or {}).items():
        node = cfg
        *path, last = dotted.split(".")
        for p in path:
            node = node[p]
        node[last] = value
    return cfg


def parse_cli_overrides(pairs):
    """['llm.model=openai/gpt-oss-20b', 'logbatcher.batch_size=5'] -> dict with YAML-typed values."""
    import yaml
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def run_experiment(experiment, datasets=None, overrides=None, extra_cols=None):
    """Run LogBatcher on `datasets` with `overrides`; writes results/<experiment>.csv/.json and
    results/predictions/<experiment>/<dataset>.csv. Returns the result DataFrame."""
    cfg = apply_overrides(load_config(), overrides)
    datasets = datasets or dataset_names(cfg)
    res_dir = project_path(cfg["paths"]["results_dir"])
    pred_dir = res_dir / "predictions" / experiment
    pred_dir.mkdir(parents=True, exist_ok=True)
    client = make_client(cfg)
    rows = []
    t0 = time.time()
    print(f"### {experiment}: {len(datasets)} datasets, model={cfg['llm']['model']}, overrides={overrides}", flush=True)
    for name in datasets:
        try:
            row, preds = run_dataset(name, cfg, client=client)
        except BudgetExhausted as e:
            print(f"!!! {experiment}: stopped at {name}: {e}\n    Re-run the same command later; finished work is cached.", flush=True)
            if rows:
                report(rows, experiment + "_partial", config=cfg)
            sys.exit(2)
        row = {**(extra_cols or {}), **row}
        rows.append(row)
        preds.to_csv(pred_dir / f"{name}.csv", index=False)
        print(f"  {name:12s} GA={row['GA']:.3f} MLA={row['MLA']:.3f} ED={row['ED']:.3f} "
              f"calls={row['LLM_calls']:4d} T_total={row['T_total']:6d} billed={row['Tokens_billed_this_run']:6d} "
              f"({row['seconds']:.0f}s)", flush=True)
    df = report(rows, experiment, config=cfg)
    partial = res_dir / f"{experiment}_partial.csv"
    if partial.exists():
        partial.unlink()
        (res_dir / f"{experiment}_partial.json").unlink(missing_ok=True)
    avg = df.iloc[-1]
    print(f"### {experiment} done in {time.time() - t0:.0f}s: GA={avg['GA']:.3f} MLA={avg['MLA']:.3f} "
          f"ED={avg['ED']:.3f} T_total(avg)={avg['T_total']:.0f}", flush=True)
    return df


def summarize(variants, experiment, cfg_note=None):
    """variants: {label: per-dataset DataFrame (with Average row)} -> one row per variant."""
    rows = []
    base = None
    for label, df in variants.items():
        avg = df[df["Dataset"] == "Average"].iloc[0]
        r = {"Variant": label}
        for k in ["GA", "MLA", "ED", "LLM_calls", "T_total", "T_invoc", "T_total_tiktoken", "LLM_latency_s"]:
            r[k] = float(avg[k])
        if base is None:
            base = r
        for k in ["GA", "MLA", "ED"]:
            r[f"{k}_change_%"] = 100.0 * (r[k] - base[k]) / base[k]
        rows.append(r)
    out = pd.DataFrame(rows)
    path = project_path(load_config()["paths"]["results_dir"]) / f"{experiment}.csv"
    out.to_csv(path, index=False, float_format="%.4f")
    if cfg_note:
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(cfg_note, f, indent=2, default=str)
    print(out.to_string(index=False))
    return out
