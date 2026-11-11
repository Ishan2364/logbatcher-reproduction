"""E6 add-on (Option D): classify every wrongly parsed log in an experiment's predictions.

    python -m experiments.error_analysis --experiments E1_logbatcher E6_claude-haiku-4.5 drain_pp

Template errors (log's predicted template != ground truth), one category per log:
  label_style   - equal after both sides get the same Khan-style normalisation: the labels themselves
                  are inconsistent, not the parser
  over_parsing  - some constant text was turned into <*>
  under_parsing - some variable was left as constant text
  mixed         - both of the above
Grouping errors (the log's predicted group != its true group):
  merged        - its predicted group also contains logs of other true templates
  split         - its true template's logs are spread over several predicted templates
"""
import argparse
import difflib

import pandas as pd

from src.config import load_config, project_path
from src.evaluate import normalize_template as ws
from src.postprocess import normalize_template as khan

VAR = "<*>"


def _tokens(t):
    return ws(t).split(" ")


def template_error(pred, truth):
    """Return None if correct, else one of label_style / over_parsing / under_parsing / mixed."""
    if ws(pred) == ws(truth):
        return None
    if khan(pred) == khan(truth):
        return "label_style"
    over = under = False
    p, t = _tokens(pred), _tokens(truth)
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=p, b=t, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        ps, ts = " ".join(p[i1:i2]), " ".join(t[j1:j2])
        p_var, t_var = ps.count(VAR), ts.count(VAR)
        if p_var > t_var:
            over = True
        elif t_var > p_var:
            under = True
        else:   # same number of variables but different text around them
            p_const = ps.replace(VAR, "").strip()
            t_const = ts.replace(VAR, "").strip()
            if len(p_const) < len(t_const):
                over = True
            elif len(p_const) > len(t_const):
                under = True
            else:
                over = under = True
    return "mixed" if over and under else ("over_parsing" if over else "under_parsing")


def analyse(preds):
    preds = preds.copy()
    preds["PredN"] = preds["Predicted"].map(ws)
    preds["TruthN"] = preds["Truth"].map(ws)
    truths_per_pred = preds.groupby("PredN")["TruthN"].nunique()
    preds_per_truth = preds.groupby("TruthN")["PredN"].nunique()
    rows = []
    for _, r in preds.iterrows():
        merged = truths_per_pred[r.PredN] > 1
        split = preds_per_truth[r.TruthN] > 1
        rows.append({"template_error": template_error(r.Predicted, r.Truth),
                     "group_error": "merged" if merged else ("split" if split else None)})
    return pd.concat([preds.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def summarize_experiment(experiment):
    cfg = load_config()
    pred_dir = project_path(cfg["paths"]["results_dir"]) / "predictions" / experiment
    out, examples = [], []
    for f in sorted(pred_dir.glob("*.csv")):
        a = analyse(pd.read_csv(f, dtype=str, keep_default_na=False))
        te = a["template_error"].value_counts()
        ge = a["group_error"].value_counts()
        out.append({"Experiment": experiment, "Dataset": f.stem, "Logs": len(a),
                    "Template_errors": int(a["template_error"].notna().sum()),
                    **{k: int(te.get(k, 0)) for k in ["over_parsing", "under_parsing", "mixed", "label_style"]},
                    "Group_errors": int(a["group_error"].notna().sum()),
                    **{k: int(ge.get(k, 0)) for k in ["merged", "split"]}})
        bad = a[a["template_error"].notna()].drop_duplicates(["PredN", "TruthN"])
        for _, r in bad.groupby("template_error").head(3).iterrows():
            examples.append({"Experiment": experiment, "Dataset": f.stem, "Category": r.template_error,
                             "Predicted": r.Predicted, "Truth": r.Truth, "Example_log": r.Content})
    return pd.DataFrame(out), pd.DataFrame(examples)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", required=True)
    args = ap.parse_args()
    res = project_path(load_config()["paths"]["results_dir"])
    tables, exs = zip(*(summarize_experiment(e) for e in args.experiments))
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(res / "E6_error_analysis.csv", index=False)
    pd.concat(exs, ignore_index=True).to_csv(res / "E6_error_examples.csv", index=False)
    totals = table.groupby("Experiment")[[c for c in table.columns if c not in ("Experiment", "Dataset")]].sum()
    print(table.to_string(index=False))
    print("\nTotals:\n" + totals.to_string())
