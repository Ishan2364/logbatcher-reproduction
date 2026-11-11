import numpy as np

from src.cache import TemplateCache, matches, template_to_regex
from src.config import load_config
from src.logbatcher import run_dataset
from src.matcher import is_usable, match_and_prune
from src.partition import partition
from src.postprocess import extract_template, normalize_template, postprocess
from src.preprocess import clustering_tokens, masked_template
from src.sampler import dpp_greedy, sample


# 4a preprocessing
def test_tokens_mask_variables_but_keep_urls_whole():
    assert clustering_tokens("PacketResponder 1 for block blk_-123 terminating") == \
        ["PacketResponder", "<*>", "for", "block", "<*>", "terminating"]
    assert clustering_tokens("after trim url = https://www.google.com/search?q=test") == \
        ["after", "trim", "url", "=", "<*>"]
    assert clustering_tokens("START: tftp pid=16563 from=10.100.4.251")[-3:] == ["from", "=", "<*>"]


def test_masked_template():
    assert masked_template("Running task 26.0 in stage 26.0 (TID 1226)") == "Running task <*> in stage <*> (TID <*>)"


# 4b partitioning
def test_partition_orders_by_size_and_dedups():
    logs = ["open file a%d" % i for i in range(12)] + ["close socket %d" % i for i in range(6)] + ["odd one"]
    clusters, labels = partition(logs)
    assert len(clusters[0]) == 12 and len(clusters[1]) == 6
    assert clusters[-1] == ["odd one"]                    # outlier cluster last
    clusters, _ = partition(["same line"] * 7)
    assert clusters == [["same line"]]                    # duplicates removed


# 4c cache
def test_regex_conversion_and_modes():
    rx = template_to_regex("Receiving block <*> src: <*>")
    assert matches(rx, "Receiving block blk_1 src:   /1.2.3.4:5")
    assert not matches(template_to_regex("x (<*>) y"), "x () y")                 # no empty variables
    assert matches(template_to_regex("x (<*>) y", "paper"), "x () y")             # paper's (.*?)
    assert not matches(template_to_regex("rhost=<*>", "strict"), "rhost=1.2.3.4 user=root")
    assert matches(template_to_regex("rhost=<*>", "loose"), "rhost=1.2.3.4 user=root")


def test_cache_lookup_frequency_and_length_check():
    c = TemplateCache(length_check=True)
    c.add("Got task <*>", "Got task 1", match_count=1)
    c.add("Got <*>", "Got x", match_count=5)
    assert c.lookup("Got task 7") == "Got task <*>"       # 'Got <*>' fails the token-count check
    assert c.lookup("Got y") == "Got <*>"
    assert c.lookup("Nothing here") is None
    assert c.entries[0].template == "Got <*>"             # most frequent first


# 4d sampling
def test_dpp_prefers_diverse_items():
    # items 0 and 1 are near-identical; DPP should not pick both before item 2
    K = np.array([[1, .99, 0], [.99, 1, 0], [0, 0, 1.]])
    sel = dpp_greedy(K, 2)
    assert set(sel) in ({0, 2}, {1, 2})


def test_sample_sizes_and_methods():
    logs = [f"user u{i} logged in from host h{i % 3}" for i in range(30)]
    for m in ["dpp", "random", "similarity", "farthest"]:
        b = sample(logs, 10, m)
        assert len(b) == 10 and len(set(b)) == 10
    assert sample(logs[:4], 10) == logs[:4]


# 4e post-processing / matching
def test_extract_template_variants():
    assert extract_template("Log template: `Running task {task_id} in stage {id}`") == "Running task <*> in stage <*>"
    assert extract_template("```\nOpen {file}\n```") == "Open <*>"
    assert extract_template("Open {file}") == "Open <*>"
    assert extract_template("") is None


def test_normalize_template_rules():
    assert normalize_template("Receiving  block <*> src: /<*>:<*>") == "Receiving block <*> src: <*>"
    assert normalize_template("size 1024 at /var/log/x") == "size <*> at <*>"
    assert normalize_template("flag true id 0x1f3a") == "flag <*> id <*>"
    assert postprocess("`a {x}{y} b`") == "a <*> b"


def test_match_and_prune_strict_first():
    cluster = ["fail rhost=1.2.3.4", "fail rhost=5.6.7.8 user=root"]
    m, u, mode = match_and_prune(cluster, "fail rhost=<*>")
    assert m == ["fail rhost=1.2.3.4"] and u == ["fail rhost=5.6.7.8 user=root"] and mode == "strict"
    m, u, mode = match_and_prune(["at Mon Jul 10"], "at <*>")
    assert m == ["at Mon Jul 10"] and mode == "loose"     # falls back when strict matches nothing
    assert not is_usable("<*> <*>") and is_usable("open <*>")


# 4f pipeline (no API calls)
def test_pipeline_with_oracle_llm():
    cfg = load_config()
    cfg["llm"]["provider"] = "mock_oracle"
    row, preds = run_dataset("HDFS", cfg)
    assert row["GA"] == 1.0 and row["MLA"] == 1.0
    assert row["LLM_calls"] <= 20 and len(preds) == 2000


def test_ablation_switches_run():
    cfg = load_config()
    cfg["llm"]["provider"] = "mock_mask"
    for key in ["use_partitioning", "use_cache", "use_batching"]:
        c = load_config()
        c["llm"]["provider"] = "mock_mask"
        c["logbatcher"][key] = False
        row, _ = run_dataset("Apache", c)
        assert row["GA"] > 0.9, key


def test_restore_constants():
    from src.matcher import restore_constants
    batch = ["a:1 open through proxy p:2 HTTPS", "b:9 open through proxy q:3 HTTPS"]
    assert restore_constants("<*> open through proxy <*> <*>", batch) == "<*> open through proxy <*> HTTPS"
    batch = ["x 403 bytes sent", "y 12 bytes sent"]
    assert restore_constants("<*> <*> <*> sent", batch) == "<*> <*> bytes sent"
    assert restore_constants("<*> <*> <*> sent", batch[:1]) == "<*> <*> <*> sent"      # no evidence
    batch = ["user alice in", "user bob in"]
    assert restore_constants("user <*> in", batch) == "user <*> in"                      # varies


def test_restore_constants_does_not_hang_on_many_placeholders():
    import time
    from src.matcher import restore_constants
    t = "ask <*> to delete " + " ".join(["<*>"] * 100)
    batch = ["ask a to delete " + " ".join(f"blk_{i}" for i in range(60))] * 3
    t0 = time.time()
    restore_constants(t, batch)
    assert time.time() - t0 < 2


def test_extended_rules_are_opt_in():
    t = '<*> "GET <*> HTTP/1.1" status: <*>'
    assert normalize_template(t) == '<*> "GET <*> HTTP/1.<*>" status: <*>'
    assert normalize_template(t, extended=True) == '<*> "GET <*> <*>" status: <*>'
    assert normalize_template("<*> close, <*> bytes (<*> KB) sent", extended=True) == "<*> close, <*> bytes (<*>) sent"
    assert normalize_template("proxy.cse.cuhk.edu.hk:<*> close", extended=True) == "<*> close"
    assert normalize_template("proxy.cse.cuhk.edu.hk:<*> close") == "proxy.cse.cuhk.edu.hk:<*> close"


def test_unknown_provider_is_an_error():
    import pytest
    cfg = load_config()
    cfg["llm"]["provider"] = "bedrock llm.model=typo"
    with pytest.raises(ValueError):
        run_dataset("Apache", cfg)
