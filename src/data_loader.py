"""Phase 1: load Loghub-2k datasets (Khan et al. corrected labels) into one consistent format.

load_dataset(name) -> DataFrame[LineId, Content, EventTemplate]
"""
import argparse

import pandas as pd
import regex as re

from src.config import dataset_names, load_config, project_path


def log_format_to_regex(log_format):
    """Turn a loghub format string such as '<Date> <Time> <Level>: <Content>' into a regex.

    Literal spaces in the format match any run of whitespace; each <Field> becomes a lazy group.
    """
    headers = []
    parts = re.split(r"(<[^<>]+>)", log_format)
    pattern = ""
    for i, part in enumerate(parts):
        if i % 2 == 0:
            pattern += re.sub(" +", r"\\s+", part)
        else:
            header = part.strip("<>")
            pattern += f"(?P<{header}>.*?)"
            headers.append(header)
    return headers, re.compile("^" + pattern + "$")


def parse_raw_log(name, cfg=None):
    """Split every line of the raw *_2k.log file into header fields + Content using the log format."""
    cfg = cfg or load_config()
    headers, rx = log_format_to_regex(cfg["datasets"][name]["log_format"])
    path = project_path(cfg["paths"]["data_dir"]) / name / f"{name}_2k.log"
    rows, skipped = [], 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = rx.search(line.strip())
            if m is None:
                skipped += 1
                continue
            rows.append([m.group(h) for h in headers])
    df = pd.DataFrame(rows, columns=headers)
    df.insert(0, "LineId", range(1, len(df) + 1))
    df.attrs["skipped"] = skipped
    return df


def load_dataset(name, cfg=None):
    """Return the ground-truth DataFrame for one system: LineId, Content, EventTemplate."""
    cfg = cfg or load_config()
    if name not in cfg["datasets"]:
        raise ValueError(f"Unknown dataset {name!r}; choose from {dataset_names(cfg)}")
    path = project_path(cfg["paths"]["data_dir"]) / name / f"{name}_2k.log_structured_corrected.csv"
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["LineId"] = df["LineId"].astype(int)
    return df[["LineId", "Content", "EventTemplate"]].sort_values("LineId").reset_index(drop=True)


def dataset_stats(cfg=None):
    """Sanity table: #logs, #unique templates, empty contents, and agreement with the raw-log parse."""
    cfg = cfg or load_config()
    rows = []
    for name in dataset_names(cfg):
        df = load_dataset(name, cfg)
        raw = parse_raw_log(name, cfg)
        same_len = len(raw) == len(df)
        if same_len:
            content_match = (raw["Content"].str.strip().values == df["Content"].str.strip().values).mean()
        else:
            content_match = float("nan")
        rows.append({
            "Dataset": name,
            "Logs": len(df),
            "Templates": df["EventTemplate"].nunique(),
            "EmptyContent": int((df["Content"].str.strip() == "").sum()),
            "RawLinesParsed": len(raw),
            "RawContentMatch": round(float(content_match), 4),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--stats", action="store_true", help="print the dataset table for all 16 systems")
    args = ap.parse_args()
    if args.stats:
        stats = dataset_stats()
        print(stats.to_string(index=False))
        out = project_path(load_config()["paths"]["results_dir"]) / "dataset_stats.csv"
        stats.to_csv(out, index=False)
        print(f"saved {out}")
    elif args.dataset:
        print(load_dataset(args.dataset).head(10).to_string())
