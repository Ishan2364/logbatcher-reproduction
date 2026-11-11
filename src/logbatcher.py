"""Phase 4f: the LogBatcher pipeline (partition -> cache -> batch & query -> match & prune).

    python -m src.logbatcher --dataset HDFS [--provider groq|mock_oracle|mock_mask] [--verbose]
"""
import argparse
import time
from collections import Counter, deque

import pandas as pd

from src.cache import TemplateCache
from src.config import load_config, project_path
from src.data_loader import load_dataset
from src.evaluate import evaluate, token_metrics
from src.llm_client import make_client
from src.matcher import is_usable, match_and_prune, restore_constants
from src.partition import chunk_partition, partition
from src.postprocess import normalize_template, postprocess
from src.preprocess import masked_template
from src.sampler import sample


class LogBatcher:
    def __init__(self, cfg, client, verbose=False):
        self.cfg = cfg
        self.lb = cfg["logbatcher"]
        self.client = client
        self.verbose = verbose

    def _partition(self, contents):
        if self.lb["use_partitioning"]:
            clusters, _ = partition(contents, **self.lb["dbscan"])
        else:
            clusters = chunk_partition(contents, self.lb["no_partition_chunk_size"])
        return clusters

    def parse(self, contents):
        """contents: list of log messages in line order. Returns (templates per line, stats)."""
        t0 = time.time()
        counts = Counter(contents)
        batch_size = self.lb["batch_size"] if self.lb["use_batching"] else 1
        cache = TemplateCache(length_check=self.lb.get("cache_length_check", True)) if self.lb["use_cache"] else None
        clusters = self._partition(contents)
        n_initial = len(clusters)
        queue = deque(clusters)
        assigned = {}
        stats = Counter()

        while queue:
            cluster = [c for c in queue.popleft() if c not in assigned]
            if cache is not None:
                rest = []
                for log in cluster:
                    t = cache.lookup(log, weight=counts[log])
                    if t is None:
                        rest.append(log)
                    else:
                        assigned[log] = t
                        stats["cache_hit_lines"] += counts[log]
                cluster = rest
            if not cluster:
                continue

            batch = sample(cluster, batch_size, self.lb["sampler"], self.lb.get("seed", 0))
            resp = self.client.query(batch)
            ext = self.lb.get("extended_rules", False)
            template = postprocess(resp["text"], ext)
            if self.lb.get("variability_check", False) and is_usable(template):
                template = normalize_template(restore_constants(template, batch), ext)
            mm = self.lb.get("match_mode", "strict_first")
            if is_usable(template):
                matched, unmatched, mode = match_and_prune(cluster, template, mm)
            else:
                matched, unmatched, mode = [], cluster, "loose"

            if not matched:
                # LLM template unusable or matches nothing: fall back to a masked version of the
                # first batch log, which is guaranteed to match at least that log.
                stats["fallbacks"] += 1
                template = normalize_template(masked_template(batch[0]), ext)
                matched, unmatched, mode = match_and_prune(cluster, template, mm)
                if batch[0] not in matched:
                    template = normalize_template(batch[0], ext)
                    matched, mode = [batch[0]], "loose"
                    unmatched = [c for c in cluster if c != batch[0]]
            if unmatched:
                stats["pruned_requeues"] += 1
                queue.append(unmatched)

            for log in matched:
                assigned[log] = template
            if cache is not None:
                cache.add(template, matched[0], match_count=sum(counts[l] for l in matched), mode=mode)

            if self.verbose:
                print("-" * 100)
                for b in batch:
                    print("  >", b[:160])
                print(f"  LLM: {resp['text'].strip()[:200]!r}")
                print(f"  template: {template!r}  matched={len(matched)} unmatched={len(unmatched)}"
                      f"  tokens={resp['input_tokens']}+{resp['output_tokens']} cached={resp['cached']}")

        templates = [assigned[c] for c in contents]
        for k in ("cache_hit_lines", "fallbacks", "pruned_requeues"):
            stats.setdefault(k, 0)
        stats.update({"initial_clusters": n_initial, "seconds": round(time.time() - t0, 2)})
        return templates, dict(stats)


def run_dataset(name, cfg, verbose=False, client=None):
    """Parse one dataset with the given config; returns (metrics row, predictions DataFrame)."""
    df = load_dataset(name, cfg)
    oracle = dict(zip(df["Content"], df["EventTemplate"]))
    if client is None:
        client = make_client(cfg, oracle=oracle)
    else:
        client.oracle = oracle
    n_before = len(client.calls)
    parser = LogBatcher(cfg, client, verbose=verbose)
    templates, stats = parser.parse(df["Content"].tolist())
    calls = client.calls[n_before:]
    m = evaluate(templates, df["EventTemplate"].tolist())
    tok = token_metrics(calls)
    tt = token_metrics([{"input_tokens": c["tt_input"], "output_tokens": c["tt_output"]} for c in calls])
    row = {"Dataset": name, **m, **tok,
           "T_total_tiktoken": tt["T_total"], "T_invoc_tiktoken": tt["T_invoc"],
           "Tokens_billed_this_run": sum(c["input_tokens"] + c["output_tokens"] for c in calls if not c["cached"]),
           "LLM_latency_s": round(sum(c["latency"] for c in calls), 2),
           **stats}
    preds = pd.DataFrame({"LineId": df["LineId"], "Content": df["Content"],
                          "Predicted": templates, "Truth": df["EventTemplate"]})
    return row, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--provider", help="override llm.provider")
    ap.add_argument("--model", help="override llm.model")
    ap.add_argument("--verbose", action="store_true", help="print every prompt batch and response")
    args = ap.parse_args()
    cfg = load_config()
    if args.provider:
        cfg["llm"]["provider"] = args.provider
    if args.model:
        cfg["llm"]["model"] = args.model
    row, preds = run_dataset(args.dataset, cfg, verbose=args.verbose)
    out = project_path(cfg["paths"]["results_dir"]) / "predictions" / "single_runs"
    out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out / f"{args.dataset}_{cfg['llm']['provider']}.csv", index=False)
    for k, v in row.items():
        print(f"{k:24s} {v:.4f}" if isinstance(v, float) else f"{k:24s} {v}")


if __name__ == "__main__":
    main()
