"""Phase 8: build the paper's tables and quoted numbers from results/.

    python -m experiments.make_paper_assets

Writes paper/generated/numbers.tex (one \\newcommand per number quoted in the prose) and
paper/generated/tab_*.tex, and copies the figures the paper uses into paper/figures/. No number in
paper/main.tex is typed by hand, so each one traces to a CSV through this file.
"""
import shutil

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from experiments.make_figures import fig_batch_size, fig_cost_accuracy, fig_errors, fig_robustness
from src.config import load_config, project_path

cfg = load_config()
RES = project_path(cfg["paths"]["results_dir"])
FIG = project_path(cfg["paths"]["figures_dir"])
OUT = project_path("paper/generated")
PAPER_FIG = project_path("paper/figures")
nums = {}


def num(name, value, fmt="{:.3f}"):
    nums[name] = fmt.format(value)


def count(name, value):
    num(name, value, "{:,.0f}")


def read(name):
    return pd.read_csv(RES / name)


def per_system(df):
    return df[~df["Dataset"].isin(["Average", "All calls", "Total"])].copy()


def tex_table(name, text):
    (OUT / f"tab_{name}.tex").write_text(text)


def pct(new, base):
    return 100.0 * (new - base) / base


def fmt_pct(x):
    return f"{x:+.1f}\\%".replace("-", "$-$").replace("+", "$+$")


# ---------------------------------------------------------------------------- E1: main comparison
def main_table():
    t = read("E1_main.csv")
    d, avg = per_system(t), t[t["Dataset"] == "Average"].iloc[0]
    stats = read("dataset_stats.csv").set_index("Dataset")
    paper = per_system(read("paper_reported.csv"))
    rows = []
    for _, r in d.iterrows():
        cells = [r["Dataset"], str(stats.loc[r["Dataset"], "Templates"])]
        for m in ("GA", "MLA", "ED"):
            vals = [r[f"Drain_{m}"], r[f"Ours_{m}"], r[f"Paper_{m}"]]
            best = max(vals)
            cells_m = [f"\\textbf{{{v:.3f}}}" if abs(v - best) < 5e-4 else f"{v:.3f}" for v in vals]
            cells += cells_m
        rows.append(" & ".join(cells) + r" \\")
    a = ["Average", ""]
    for m in ("GA", "MLA", "ED"):
        a += [f"{avg[f'Drain_{m}']:.3f}", f"{avg[f'Ours_{m}']:.3f}", f"{avg[f'Paper_{m}']:.3f}"]
    body = "\n".join(rows)
    tex_table("main", rf"""\begin{{table*}}[!t]
\caption{{Parsing accuracy per system: Drain and our LogBatcher (gpt-oss-120b) are our runs; ``Reported'' is copied from Table~1 of~\cite{{xiao2024}}. Best value per row and metric in bold.}}
\label{{tab:main}}
\centering\footnotesize\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lr ccc ccc ccc}}
\toprule
& & \multicolumn{{3}}{{c}}{{Group Accuracy}} & \multicolumn{{3}}{{c}}{{Message-Level Accuracy}} & \multicolumn{{3}}{{c}}{{Edit-Distance Score}} \\
\cmidrule(lr){{3-5}}\cmidrule(lr){{6-8}}\cmidrule(lr){{9-11}}
System & \#T & Drain & Ours & Reported & Drain & Ours & Reported & Drain & Ours & Reported \\
\midrule
{body}
\midrule
{" & ".join(a)} \\
\bottomrule
\end{{tabular}}
\end{{table*}}
""")
    for m in ("GA", "MLA", "ED"):
        num(f"Drain{m}", avg[f"Drain_{m}"])
        num(f"Ours{m}", avg[f"Ours_{m}"])
        num(f"Rep{m}", avg[f"Paper_{m}"])
        num(f"RepDrain{m}", paper[f"Drain_{m}"].mean())
        num(f"OursMed{m}", d[f"Ours_{m}"].median())
        num(f"RepMed{m}", d[f"Paper_{m}"].median())
        ex = d[~d["Dataset"].isin(["Proxifier", "OpenStack"])]
        num(f"OursX{m}", ex[f"Ours_{m}"].mean())
        num(f"RepX{m}", ex[f"Paper_{m}"].mean())
        num(f"pDrain{m}", wilcoxon(d[f"Ours_{m}"], d[f"Drain_{m}"]).pvalue)
        num(f"pRep{m}", wilcoxon(d[f"Ours_{m}"], d[f"Paper_{m}"]).pvalue)
    idx = d.set_index("Dataset")
    for sys_, tag in (("Proxifier", "Prox"), ("OpenStack", "Stack"), ("Linux", "Linux"), ("Hadoop", "Hadoop"),
                      ("HealthApp", "Health"), ("Thunderbird", "Thunder"), ("Windows", "Windows"), ("Mac", "Mac")):
        for m in ("GA", "MLA", "ED"):
            num(f"{tag}Ours{m}", idx.loc[sys_, f"Ours_{m}"])
            num(f"{tag}Rep{m}", idx.loc[sys_, f"Paper_{m}"])
    num("LinuxDrainGA", idx.loc["Linux", "Drain_GA"])
    count("NMatchGA", (d["Ours_GA"] >= d["Paper_GA"] - 0.005).sum())
    count("NMatchMLA", (d["Ours_MLA"] >= d["Paper_MLA"] - 0.005).sum())
    count("NBeatDrainGA", (d["Ours_GA"] > d["Drain_GA"]).sum())
    count("NBeatDrainMLA", (d["Ours_MLA"] > d["Drain_MLA"]).sum())
    count("TotalTemplates", per_system(read("dataset_stats.csv"))["Templates"].sum())


# ---------------------------------------------------------------------------- E0: checks
def checks():
    o = read("E0_oracle.csv")
    o = o[o["Dataset"] == "Average"].iloc[0]
    num("OracleGA", o["GA"])
    num("OracleMLA", o["MLA"])
    p = read("E0_oracle_paper_match.csv").set_index("Dataset")
    num("OraclePaperLinuxGA", p.loc["Linux", "GA"])
    num("OraclePaperGA", p.loc["Average", "GA"])
    num("OracleLinuxGA", read("E0_oracle.csv").set_index("Dataset").loc["Linux", "GA"])
    v1 = read("E0_prompt_v1_HDFS.csv").set_index("Dataset").loc["HDFS"]
    num("PromptVOneHDFS", v1["GA"])
    e1 = read("E1_logbatcher.csv").set_index("Dataset")
    num("HDFSGA", e1.loc["HDFS", "GA"])
    c = read("E2_call_stats.csv")
    a = c[c["Dataset"] == "All calls"].iloc[0]
    num("LogsPerCall", a["Mean_logs_per_call"], "{:.1f}")
    count("PromptTok", a["Prompt_tiktoken"])
    count("AnswerTok", a["Answer_tiktoken"])
    count("ProvOutTok", a["Provider_output"])
    count("ReasonTok", a["Provider_output"] - a["Answer_tiktoken"])
    count("InstrOurs", a["Instruction_tokens_ours"])
    count("InstrPaper", a["Instruction_tokens_paper"])
    pur = read("cluster_purity.csv").set_index("Dataset")
    tot = pur.loc["Total"]
    count("MixedCos", tot["cosine_lines_in_mixed"])
    count("MixedEuc", tot["euclidean_lines_in_mixed"])
    num("MixedCosPct", 100 * tot["cosine_lines_in_mixed"] / 32000, "{:.0f}")
    num("MixedEucPct", 100 * tot["euclidean_lines_in_mixed"] / 32000, "{:.0f}")
    count("ClustersCos", tot["cosine_clusters"])
    count("ClustersEuc", tot["euclidean_clusters"])
    count("MixedClustersCos", tot["cosine_mixed_clusters"])
    count("MixedClustersEuc", tot["euclidean_mixed_clusters"])
    for s, tag in (("Proxifier", "Prox"), ("OpenStack", "Stack")):
        count(f"{tag}MixedCos", pur.loc[s, "cosine_lines_in_mixed"])
        count(f"{tag}MixedEuc", pur.loc[s, "euclidean_lines_in_mixed"])
    hadoop = pd.read_csv(RES / "predictions/E1_logbatcher/Hadoop.csv", dtype=str, keep_default_na=False)
    count("DFSLines", hadoop["Predicted"].str.contains("DFSClient_NONMAPREDUCE_1537864556_1", regex=False).sum())
    # recorded once: re-running E1 from the response cache takes seconds, so this cannot be recomputed
    num("EOneHours", read("wallclock.csv").set_index("Experiment").loc["E1_logbatcher", "Seconds"] / 3600, "{:.0f}")
    # cache activity in the main run
    e = per_system(read("E1_logbatcher.csv"))
    count("CacheLines", e["cache_hit_lines"].fillna(0).sum())
    num("CachePct", 100 * e["cache_hit_lines"].fillna(0).sum() / 32000, "{:.0f}")
    count("Fallbacks", e["fallbacks"].fillna(0).sum())
    count("TotalCalls", e["LLM_calls"].sum())
    full = per_system(read("E3_ablation_claude-haiku-4-5_w-o-partitioning.csv"))
    count("CacheLinesNoPart", full["cache_hit_lines"].fillna(0).sum())


# ---------------------------------------------------------------------------- E2: tokens
def token_table():
    e = read("E2_efficiency.csv")
    d, a = per_system(e), e[e["Dataset"] == "Average"].iloc[0]

    def row(r, label):
        return (f"{label} & {r['LLM_calls']:.0f} & {r['T_total']:,.0f} & {r['T_total_tiktoken']:,.0f} & "
                f"{r['T_invoc_tiktoken']:.0f} & {r['Paper_T_total']:,.0f} & {r['Paper_T_invoc']:.0f} \\\\")

    body = "\n".join(row(r, r["Dataset"]) for _, r in d.iterrows())
    tex_table("tokens", rf"""\begin{{table}}[!t]
\caption{{Token use of our LogBatcher (gpt-oss-120b) per system. ``Prov.'' is the provider's count, ``tik.'' a tiktoken count of prompt and answer text; ``Reported'' is copied from Table~2 of~\cite{{xiao2024}}.}}
\label{{tab:tokens}}
\centering
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrr}}
\toprule
& & \multicolumn{{3}}{{c}}{{Ours}} & \multicolumn{{2}}{{c}}{{Reported}} \\
\cmidrule(lr){{3-5}}\cmidrule(lr){{6-7}}
System & Calls & $T_{{total}}$ prov. & $T_{{total}}$ tik. & $T_{{invoc}}$ tik. & $T_{{total}}$ & $T_{{invoc}}$ \\
\midrule
{body}
\midrule
{row(a, "Average")}
\bottomrule
\end{{tabular}}}}
\end{{table}}
""")
    num("CallsAvg", a["LLM_calls"], "{:.0f}")
    count("TtotProv", a["T_total"])
    count("TtotTik", a["T_total_tiktoken"])
    count("TinvTik", a["T_invoc_tiktoken"])
    count("TtotRep", a["Paper_T_total"])
    count("TinvRep", a["Paper_T_invoc"])
    count("TtotLilac", a["LILAC_T_total"])
    count("TinvLilac", a["LILAC_T_invoc"])
    count("ImpliedCalls", a["Paper_T_total"] / a["Paper_T_invoc"])


# ---------------------------------------------------------------------------- E3/E4/E5
def variant_table():
    rep = read("paper_reported_tables.csv")
    ab = read("E3_ablation_claude-haiku-4-5.csv")
    sm = read("E4_sampling_claude-haiku-4-5.csv")

    def rows(ours, table):
        r_ = rep[rep["Table"] == table].set_index("Variant")
        base = ours.iloc[0]
        rb = r_.iloc[0]
        lines = []
        for _, r in ours.iterrows():
            cells = [r["Variant"]]
            for m in ("GA", "MLA", "ED"):
                cells.append(f"{r[m]:.3f}" + ("" if r["Variant"] == base["Variant"] else f" ({fmt_pct(pct(r[m], base[m]))})"))
            q = r_.loc[r["Variant"]]
            for m in ("GA", "MLA", "ED"):
                cells.append(f"{q[m]:.3f}" + ("" if r["Variant"] == base["Variant"] else f" ({fmt_pct(pct(q[m], rb[m]))})"))
            lines.append(" & ".join(cells) + r" \\")
        return "\n".join(lines)

    tex_table("variants", rf"""\begin{{table*}}[!t]
\caption{{Module ablation (RQ2) and sampling method (RQ3). ``Ours'' uses Claude Haiku~4.5 on six systems; ``Reported'' is copied from Tables~3 and~4 of~\cite{{xiao2024}} (GPT-3.5-Turbo, 16 systems). All percentages are recomputed here relative to the first row of each block; \cite{{xiao2024}} prints some of them relative to the reduced score.}}
\label{{tab:variants}}
\centering\footnotesize
\begin{{tabular}}{{l ccc ccc}}
\toprule
& \multicolumn{{3}}{{c}}{{Ours}} & \multicolumn{{3}}{{c}}{{Reported}} \\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}
Variant & GA & MLA & ED & GA & MLA & ED \\
\midrule
{rows(ab, "Table3")}
\midrule
{rows(sm, "Table4")}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")
    a = ab.set_index("Variant")
    s = sm.set_index("Variant")
    for k, v in (("Full", "Full LogBatcher"), ("NoPart", "w/o partitioning"), ("NoCache", "w/o caching"), ("NoBatch", "w/o batching")):
        num(f"Abl{k}GA", a.loc[v, "GA"])
        num(f"Abl{k}MLA", a.loc[v, "MLA"])
    for k, v in (("Dpp", "DPP (default)"), ("Rand", "Random sampling"), ("Sim", "Similarity sampling")):
        num(f"Smp{k}GA", s.loc[v, "GA"])
        num(f"Smp{k}MLA", s.loc[v, "MLA"])
    b = read("E5_batchsize_claude-haiku-4-5.csv")
    b["k"] = b["Variant"].str.split("=").str[1].astype(int)
    b = b.set_index("k")
    for k, tag in ((1, "One"), (5, "Five"), (10, "Ten"), (15, "Fifteen"), (20, "Twenty")):
        num(f"BS{tag}GA", b.loc[k, "GA"])
        num(f"BS{tag}MLA", b.loc[k, "MLA"])
        count(f"BS{tag}Tok", b.loc[k, "T_invoc"])
    d = per_system(read("E4_sampling_claude-haiku-4-5_random-sampling.csv")).set_index("Dataset")
    f = per_system(read("E4_sampling_claude-haiku-4-5_dpp-default.csv")).set_index("Dataset")
    num("StackDppGA", f.loc["OpenStack", "GA"])
    num("StackRandGA", d.loc["OpenStack", "GA"])


# ---------------------------------------------------------------------------- E6: LLM comparison
def llm_table():
    e = read("E6_extension.csv")
    import sqlite3
    db = sqlite3.connect(project_path(cfg["paths"]["llm_cache"]))
    ids = {"gpt-oss-120b (Groq)": "openai/gpt-oss-120b", "gpt-oss-20b (Groq)": "openai/gpt-oss-20b",
           "Claude Haiku 4.5 (Bedrock)": "us.anthropic.claude-haiku-4-5-20251001-v1:0"}
    lines = []
    for _, r in e.iterrows():
        lat = [x[0] for x in db.execute("select latency from responses where model=?", (ids[r["Model"]],))]
        lines.append(f"{r['Model'].split(' (')[0].replace('Claude ', '')} & {r['GA']:.3f} & {r['MLA']:.3f} & {r['ED']:.3f} & "
                     f"{r['LLM_calls']:,.0f} & {r['T_invoc']:.0f} & \\${r['Cost_USD']:.3f} & {np.median(lat):.2f} \\\\")
    body = "\n".join(lines)
    tex_table("llms", rf"""\begin{{table}}[!t]
\caption{{LLM comparison on all 16 systems (provider token counts). Cost is list price for parsing all 32,000 lines; latency is the median per call.}}
\label{{tab:llms}}
\centering
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lccccccc}}
\toprule
Model & GA & MLA & ED & Calls & Tok/call & Cost & Lat.\ (s) \\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
""")
    i = e.set_index("Provider")
    m = e.set_index("Model")
    for tag, k in (("Big", "gpt-oss-120b (Groq)"), ("Small", "gpt-oss-20b (Groq)"), ("Haiku", "Claude Haiku 4.5 (Bedrock)")):
        for x in ("GA", "MLA", "ED"):
            num(f"{tag}{x}", m.loc[k, x])
        num(f"{tag}Cost", m.loc[k, "Cost_USD"])
        count(f"{tag}TokCall", m.loc[k, "T_invoc"])
    num("CostRatio", m.loc["Claude Haiku 4.5 (Bedrock)", "Cost_USD"] / m.loc["gpt-oss-20b (Groq)", "Cost_USD"], "{:.0f}")
    del i


# ---------------------------------------------------------------------------- E7: variability check
def vc_table():
    lines = []
    for tag, label, a, b in (("Haiku", "Haiku 4.5", "E6_claude-haiku-4.5", "E7_vcc_claude-haiku-4.5"),
                             ("Small", "gpt-oss-20b", "E6_gpt-oss-20b", "E7_vcc_gpt-oss-20b"),
                             ("Big", "gpt-oss-120b", "E1_logbatcher", "E7_vcc_gpt-oss-120b")):
        A, B = read(f"{a}.csv"), read(f"{b}.csv")
        pa, pb = per_system(A).set_index("Dataset"), per_system(B).set_index("Dataset")
        ma, mb = pa.mean(numeric_only=True), pb.mean(numeric_only=True)
        dl = ((pb["MLA"] - pa["MLA"]) * 2000).round() + ((pb["GA"] - pa["GA"]) * 2000).round()
        up = int(((((pb["MLA"] - pa["MLA"]) * 2000).round() > 0) | (((pb["GA"] - pa["GA"]) * 2000).round() > 0)).sum())
        down = int(((((pb["MLA"] - pa["MLA"]) * 2000).round() < 0) | (((pb["GA"] - pa["GA"]) * 2000).round() < 0)).sum())
        calls = int(pb["LLM_calls"].sum() - pa["LLM_calls"].sum())
        toks = int(pb["T_total"].sum() - pa["T_total"].sum())
        lines.append(f"{label} & {ma['GA']:.3f} $\\to$ {mb['GA']:.3f} & {ma['MLA']:.3f} $\\to$ {mb['MLA']:.3f} & "
                     f"{ma['ED']:.3f} $\\to$ {mb['ED']:.3f} & {up} / {down} & {calls:+d} \\\\")
        for x in ("GA", "MLA", "ED"):
            num(f"VC{tag}Off{x}", ma[x])
            num(f"VC{tag}On{x}", mb[x])
        count(f"VC{tag}Up", up)
        count(f"VC{tag}Down", down)
        count(f"VC{tag}ExtraCalls", calls)
        count(f"VC{tag}ExtraTok", toks)
        for s, st in (("Windows", "Windows"), ("Linux", "Linux"), ("OpenSSH", "OpenSSH"), ("Proxifier", "Prox"), ("Hadoop", "Hadoop")):
            for x in ("GA", "MLA"):
                num(f"VC{tag}{st}Off{x}", pa.loc[s, x])
                num(f"VC{tag}{st}On{x}", pb.loc[s, x])
        del dl
    body = "\n".join(lines)
    tex_table("vc", rf"""\begin{{table}}[!t]
\caption{{Effect of the variability check on all 16 systems (off $\to$ on). ``Sys.'' counts systems with more / fewer correct lines; ``Calls'' is the change in LLM calls.}}
\label{{tab:vc}}
\centering
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lcccrr}}
\toprule
Model & GA & MLA & ED & Sys. & Calls \\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
""")


# ---------------------------------------------------------------------------- error analysis
def error_table():
    e = read("E6_error_analysis.csv")
    labels = [("drain_pp", "Drain"), ("E1_logbatcher", "gpt-oss-120b"), ("E6_gpt-oss-20b", "gpt-oss-20b"),
              ("E6_claude-haiku-4.5", "Haiku 4.5")]
    tot = e.groupby("Experiment").sum(numeric_only=True)
    lines = []
    for k, lab in labels:
        r = tot.loc[k]
        lines.append(f"{lab} & {r['Template_errors']:,.0f} & {r['over_parsing']:,.0f} & {r['under_parsing']:,.0f} & "
                     f"{r['mixed']:,.0f} & {r['label_style']:,.0f} & {r['merged']:,.0f} & {r['split']:,.0f} \\\\")
    body = "\n".join(lines)
    tex_table("errors", rf"""\begin{{table}}[!t]
\caption{{Errors over all 32,000 lines. Template errors split into over-parsing, under-parsing, mixed and label style; group errors into merged and split.}}
\label{{tab:errors}}
\centering
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrrr}}
\toprule
Parser & Errors & Over & Under & Mixed & Label & Merged & Split \\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
""")
    for k, tag in (("drain_pp", "Drain"), ("E1_logbatcher", "Big"), ("E6_gpt-oss-20b", "Small"), ("E6_claude-haiku-4.5", "Haiku")):
        count(f"Err{tag}", tot.loc[k, "Template_errors"])
        count(f"Over{tag}", tot.loc[k, "over_parsing"])
        count(f"Under{tag}", tot.loc[k, "under_parsing"])
        count(f"Merged{tag}", tot.loc[k, "merged"])
        count(f"Group{tag}", tot.loc[k, "Group_errors"])
    count("LabelMax", tot.loc[[k for k, _ in labels], "label_style"].max())


# ---------------------------------------------------------------------------- E8/E9: unstated choices
def sensitivity_table():
    base = per_system(read("E1_logbatcher.csv")).set_index("Dataset")
    rules = per_system(read("E8_rules_gpt-oss-120b.csv")).set_index("Dataset")
    haiku = per_system(read("E6_claude-haiku-4.5.csv")).set_index("Dataset")
    euc_h = per_system(read("E9_euclid_claude-haiku-4.5.csv")).set_index("Dataset")
    euc_b = per_system(read("E9_euclid_gpt-oss-120b.csv")).set_index("Dataset")
    both_h = per_system(read("E10_euclid_rules_claude-haiku-4.5.csv")).set_index("Dataset")
    both_b = per_system(read("E10_euclid_rules_gpt-oss-120b.csv")).set_index("Dataset")
    nocache = per_system(read("E9_euclid_nocache_claude-haiku-4.5.csv")).set_index("Dataset")
    control = per_system(read("E0_mask_control.csv")).set_index("Dataset")
    six = list(euc_b.index)

    def fmt(s):
        return f"{s['GA']:.3f} & {s['MLA']:.3f} & {s['ED']:.3f}"

    def mean(df, idx=None):
        return (df if idx is None else df.loc[idx]).mean(numeric_only=True)

    lines = [
        f"No LLM (control), 16 & cosine & ours & {fmt(mean(control))} \\\\",
        r"\midrule",
        f"gpt-oss-120b, 16 & cosine & ours & {fmt(mean(base))} \\\\",
        f"gpt-oss-120b, 16 & cosine & ours+3 & {fmt(mean(rules))} \\\\",
        r"\midrule",
        f"gpt-oss-120b, {len(six)} & cosine & ours & {fmt(mean(base, six))} \\\\",
        f"gpt-oss-120b, {len(six)} & Euclidean & ours & {fmt(mean(euc_b))} \\\\",
        f"gpt-oss-120b, {len(six)} & Euclidean & ours+3 & {fmt(mean(both_b))} \\\\",
        r"\midrule",
        f"Haiku 4.5, 16 & cosine & ours & {fmt(mean(haiku))} \\\\",
        f"Haiku 4.5, 16 & Euclidean & ours & {fmt(mean(euc_h))} \\\\",
        f"Haiku 4.5, 16 & Euclidean & ours+3 & {fmt(mean(both_h))} \\\\",
    ]
    body = "\n".join(lines)
    tex_table("sensitivity", rf"""\begin{{table}}[!t]
\caption{{Sensitivity to settings the source paper leaves unstated. ``Runs'' gives the model and number of systems; ``ours+3'' adds three post-processing rules read from the authors' public code. The control sends nothing to an LLM.}}
\label{{tab:sensitivity}}
\centering
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lllccc}}
\toprule
Runs & Distance & Rules & GA & MLA & ED \\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
""")
    frames = {"Rules": rules, "CtrlAll": control, "EucHaiku": euc_h, "BothHaiku": both_h, "EucBigSix": euc_b,
              "BothBigSix": both_b, "NoCacheEuc": nocache}
    for tag, df in frames.items():
        m = mean(df)
        for x in ("GA", "MLA", "ED"):
            num(f"{tag}{x}", m[x])
    m = mean(base, six)
    for x in ("GA", "MLA", "ED"):
        num(f"CosBigSix{x}", m[x])
        num(f"RulesProx{x}", rules.loc["Proxifier", x])
        num(f"RulesStack{x}", rules.loc["OpenStack", x])
        num(f"EucBigProx{x}", euc_b.loc["Proxifier", x])
        num(f"EucBigStack{x}", euc_b.loc["OpenStack", x])
        num(f"EucBigAndroid{x}", euc_b.loc["Android", x])
        num(f"BothBigProx{x}", both_b.loc["Proxifier", x])
        num(f"BothBigStack{x}", both_b.loc["OpenStack", x])
        num(f"EucHaikuProx{x}", euc_h.loc["Proxifier", x])
        num(f"EucHaikuStack{x}", euc_h.loc["OpenStack", x])
        num(f"EucHaikuBGL{x}", euc_h.loc["BGL", x])
        num(f"HaikuProx{x}", haiku.loc["Proxifier", x])
        num(f"HaikuStack{x}", haiku.loc["OpenStack", x])
        num(f"HaikuBGL{x}", haiku.loc["BGL", x])
        num(f"BigAndroid{x}", base.loc["Android", x])
        num(f"CtrlLinux{x}", control.loc["Linux", x])
    count("EucCallsHaiku", euc_h["LLM_calls"].sum())
    count("CosCallsHaiku", haiku["LLM_calls"].sum())
    count("NoCacheCallsHaiku", nocache["LLM_calls"].sum())
    for tag, col in (("GA", "GA"), ("MLA", "MLA")):
        num(f"NoCachePct{tag}", abs(pct(nocache[col].mean(), euc_h[col].mean())), "{:.1f}")
    num("NoCacheProxGA", nocache.loc["Proxifier", "GA"])
    num("NoCacheThunderGA", nocache.loc["Thunderbird", "GA"])
    num("EucThunderGA", euc_h.loc["Thunderbird", "GA"])
    cos_nc = per_system(read("E6_nocache_claude-haiku-4.5.csv")).set_index("Dataset")
    num("CosNoCacheGA", cos_nc["GA"].mean())
    count("CosCacheHits", haiku["cache_hit_lines"].fillna(0).sum())
    count("CosCacheProx", haiku.loc["Proxifier", "cache_hit_lines"])
    subset = cfg["subset"]
    num("SixEucCacheGA", euc_h.loc[subset, "GA"].mean())
    num("SixEucNoCacheGA", nocache.loc[subset, "GA"].mean())
    count("EucCacheHaiku", euc_h["cache_hit_lines"].fillna(0).sum())
    num("EucCachePct", 100 * euc_h["cache_hit_lines"].fillna(0).sum() / 32000, "{:.0f}")
    count("EucCacheBigSix", euc_b["cache_hit_lines"].fillna(0).sum())
    count("NSix", len(six))
    num("NoCacheExtraPct", 100 * (nocache["LLM_calls"].sum() / euc_h["LLM_calls"].sum() - 1), "{:.0f}")
    up = int(((euc_h["GA"] - haiku["GA"]) * 2000).round().gt(0).sum())
    down = int(((euc_h["GA"] - haiku["GA"]) * 2000).round().lt(0).sum())
    count("EucHaikuUp", up)
    count("EucHaikuDown", down)
    # labelled templates behind the group errors that remain with the extra rules
    pr = pd.read_csv(RES / "predictions/E8_rules_gpt-oss-120b/Proxifier.csv", dtype=str, keep_default_na=False)
    st = pd.read_csv(RES / "predictions/E8_rules_gpt-oss-120b/OpenStack.csv", dtype=str, keep_default_na=False)
    a = int(pr["Predicted"].eq("<*> open through proxy <*> <*>").sum())
    b = int(st["Truth"].str.contains('"<*> <*> <*>"', regex=False).sum())
    count("HttpsLines", a)
    count("SocksLines", pr["Truth"].str.endswith("SOCKS5").sum())
    count("StackLines", b)
    num("TwoDecisions", (a + b) / 32000)
    num("RemainingGap", read("E1_main.csv").set_index("Dataset").loc["Average", "Paper_GA"] - rules["GA"].mean())


def figures():
    fig_robustness()
    fig_batch_size()
    fig_cost_accuracy()
    fig_errors()
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    for f in ("fig_robustness_boxplots", "fig_batch_size_claude-haiku-4-5", "fig_cost_vs_accuracy",
              "fig_error_categories_E1_logbatcher"):
        shutil.copy(FIG / f"{f}.pdf", PAPER_FIG / f"{f}.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main_table()
    checks()
    token_table()
    variant_table()
    llm_table()
    vc_table()
    error_table()
    sensitivity_table()
    figures()
    with open(OUT / "numbers.tex", "w") as f:
        f.write("% Generated by experiments/make_paper_assets.py -- do not edit by hand.\n")
        for k, v in sorted(nums.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"wrote {len(nums)} numbers and {len(list(OUT.glob('tab_*.tex')))} tables to paper/generated/")
