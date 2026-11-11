import pandas as pd
import pytest

from src.data_loader import load_dataset
from src.evaluate import (edit_distance_score, evaluate, group_accuracy,
                          message_level_accuracy, report, token_metrics)


def test_ground_truth_scores_perfectly():
    gt = load_dataset("HDFS")["EventTemplate"].tolist()
    m = evaluate(gt, gt)
    assert m["GA"] == m["MLA"] == m["ED"] == 1.0


def test_every_log_its_own_template_scores_low():
    df = load_dataset("HDFS")
    pred = df["Content"].tolist()           # raw message as template -> mostly singleton groups
    m = evaluate(pred, df["EventTemplate"].tolist())
    assert m["GA"] < 0.1
    assert m["MLA"] < 0.1


def test_group_accuracy_toy():
    gt = ["A", "A", "B", "B", "C"]
    assert group_accuracy(["x", "x", "y", "y", "z"], gt) == 1.0     # names don't matter
    assert group_accuracy(["x", "x", "x", "x", "z"], gt) == 0.2     # A+B merged -> only C correct
    assert group_accuracy(["x", "w", "y", "y", "z"], gt) == 0.6     # A split -> B and C correct


def test_mla_whitespace_insensitive():
    assert message_level_accuracy(["a  <*> b "], ["a <*> b"]) == 1.0
    assert message_level_accuracy(["a <*>"], ["a b"]) == 0.0


def test_edit_distance_bounds():
    assert edit_distance_score(["abc"], ["abc"]) == 1.0
    assert edit_distance_score(["abc"], ["xyz"]) == 0.0
    assert 0 < edit_distance_score(["abcd"], ["abcx"]) < 1


def test_token_metrics():
    calls = [{"input_tokens": 100, "output_tokens": 20}, {"input_tokens": 50, "output_tokens": 10}]
    t = token_metrics(calls)
    assert t["T_total"] == 180 and t["T_invoc"] == 90.0 and t["LLM_calls"] == 2
    assert token_metrics([])["T_total"] == 0


def test_report_writes_csv(tmp_path):
    df = report([{"Dataset": "A", "GA": 1.0}, {"Dataset": "B", "GA": 0.5}], "unit", config={"k": 1}, results_dir=tmp_path)
    assert (tmp_path / "unit.csv").exists() and (tmp_path / "unit.json").exists()
    assert df.iloc[-1]["Dataset"] == "Average" and df.iloc[-1]["GA"] == pytest.approx(0.75)
