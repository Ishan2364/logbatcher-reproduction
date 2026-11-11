"""Phase 3: Drain baseline on all 16 datasets.

Default implementation is loghub's benchmark Drain (`logparser3` package, He et al. 2017) with the
standard per-dataset settings from config.yaml. `--impl drain3` runs the streaming drain3 package
instead, for comparison.
"""
import argparse
import contextlib
import io
import time

import pandas as pd
import regex as re
from logparser.Drain import LogParser as BenchmarkDrain

from src.config import dataset_names, load_config, project_path
from src.data_loader import load_dataset
from src.evaluate import evaluate, report
from src.postprocess import normalize_template


class _InMemoryDrain(BenchmarkDrain):
    """Benchmark Drain fed from our DataFrame; templates are kept in memory instead of written to disk."""

    def __init__(self, df, **kw):
        super().__init__(log_format="<Content>", indir=".", outdir=".", **kw)
        self._df = df

    def load_data(self):
        self.df_log = self._df[["LineId", "Content"]].copy()

    def outputResult(self, logClustL):
        templates = [None] * len(self.df_log)
        for clust in logClustL:
            t = " ".join(clust.logTemplate)
            for line_id in clust.logIDL:
                templates[line_id - 1] = t
        self.df_log["EventTemplate"] = templates


def drain_benchmark(df, settings):
    parser = _InMemoryDrain(df, depth=settings["depth"], st=settings["st"], rex=settings["regex"])
    with contextlib.redirect_stdout(io.StringIO()):
        parser.parse("in-memory")
    return parser.df_log["EventTemplate"].tolist()


def drain3_parse(df, settings):
    from drain3 import TemplateMiner
    from drain3.masking import MaskingInstruction
    from drain3.template_miner_config import TemplateMinerConfig

    cfg = TemplateMinerConfig()
    cfg.drain_sim_th = settings["st"]
    cfg.drain_depth = settings["depth"]
    cfg.masking_instructions = [MaskingInstruction(rx, "*") for rx in settings["regex"]]
    cfg.mask_prefix, cfg.mask_suffix = "<", ">"
    cfg.parametrize_numeric_tokens = True
    miner = TemplateMiner(config=cfg)
    cluster_ids = [miner.add_log_message(c)["cluster_id"] for c in df["Content"]]
    final = {c.cluster_id: c.get_template() for c in miner.drain.clusters}
    return [final[i] for i in cluster_ids]


def run(impl="benchmark", datasets=None, experiment=None, postprocess=False):
    cfg = load_config()
    datasets = datasets or dataset_names(cfg)
    experiment = experiment or ("drain" if impl == "benchmark" else "drain3") + ("_pp" if postprocess else "")
    pred_dir = project_path(cfg["paths"]["results_dir"]) / "predictions" / experiment
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in datasets:
        df = load_dataset(name, cfg)
        settings = cfg["datasets"][name]["drain"]
        t0 = time.time()
        pred = drain_benchmark(df, settings) if impl == "benchmark" else drain3_parse(df, settings)
        if postprocess:   # same Khan-style template rules LogBatcher applies to LLM output
            pred = [normalize_template(t) for t in pred]
        elapsed = time.time() - t0
        m = evaluate(pred, df["EventTemplate"].tolist())
        rows.append({"Dataset": name, **m, "Seconds": round(elapsed, 2)})
        pd.DataFrame({"LineId": df["LineId"], "Content": df["Content"], "Predicted": pred,
                      "Truth": df["EventTemplate"]}).to_csv(pred_dir / f"{name}.csv", index=False)
        print(f"{name:12s} GA={m['GA']:.3f} MLA={m['MLA']:.3f} ED={m['ED']:.3f}  ({elapsed:.1f}s)")
    out = report(rows, experiment, config={"impl": impl, "postprocess": postprocess, "drain_settings": {d: cfg["datasets"][d]["drain"] for d in datasets}})
    print(out.iloc[-1][["GA", "MLA", "ED"]].to_string())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["benchmark", "drain3"], default="benchmark")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--postprocess", action="store_true", help="apply Khan-style template normalisation")
    args = ap.parse_args()
    run(args.impl, args.datasets, postprocess=args.postprocess)
