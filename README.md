# LogBatcher reproduction and extension

An independent re-implementation of **LogBatcher** (Xiao, Le & Zhang, *Stronger, Cheaper and
Demonstration-Free Log Parsing with LLMs*, arXiv:2406.06156), evaluated on the 16 Loghub-2k
datasets, with an LLM accuracy-vs-cost comparison and an error analysis as the extension.

LogBatcher turns raw log messages into templates (`Received block <*> of size <*> from <*>`)
without labelled examples: it clusters similar logs, sends a small, diverse batch from each
cluster to an LLM, and caches the resulting templates so most logs never reach the LLM.

## Layout

```
config.yaml              every setting used in the experiments (datasets, DBSCAN, batch size, LLMs, prices)
src/
  data_loader.py         Phase 1  load_dataset(name) -> LineId, Content, EventTemplate
  evaluate.py            Phase 2  GA, MLA, ED, token metrics, report()
  baseline_drain.py      Phase 3  loghub benchmark Drain (+ optional template post-processing)
  preprocess.py          Phase 4a tokenisation and variable masking (clustering only)
  partition.py           Phase 4b TF-IDF + DBSCAN partitioning, outliers last
  cache.py               Phase 4c template cache, regex matching
  sampler.py             Phase 4d DPP / random / similarity / farthest-point sampling
  llm_client.py          Phase 4d prompt, Groq client, mocks, SQLite response cache
  llm_anthropic.py       Phase 4d Claude on Amazon Bedrock
  postprocess.py         Phase 4e template extraction + Khan et al. style normalisation
  matcher.py             Phase 4e matching & pruning
  logbatcher.py          Phase 4f the pipeline (CLI)
experiments/
  run_main.py            E1 main comparison, E2 token efficiency
  run_ablation.py        E3 module ablation, E4 sampling method
  run_batchsize.py       E5 batch size
  run_extension.py       E6 LLM accuracy vs cost
  error_analysis.py      E6 add-on: error categories
  run_checks.py          free sanity checks and diagnostics quoted in the paper
  make_figures.py        Phase 7 figures and tables
  make_paper_assets.py   Phase 8 LaTeX tables and quoted numbers, generated from results/
  check_paper.py         Phase 9 static checks on the paper (works without LaTeX)
tests/                   unit tests (no API calls)
results/                 CSV + JSON per experiment, predictions/ per dataset
figures/                 PNG (300 dpi) + PDF
data/loghub-2k/          datasets (included; the download commands below rebuild them)
cache/llm_cache.sqlite   every LLM response, so all results re-run with no API key
paper/                   main.tex, generated/, figures/, findings notes, one-page summary
```

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env          # then put your GROQ_API_KEY in .env
.venv/bin/python -m pytest tests -q
```

API keys are only needed to make *new* LLM calls: everything in `results/` re-runs from the included
response cache. Claude on Bedrock uses the normal AWS credential chain (`~/.aws/credentials`, region in
`config.yaml` -> `llm.aws_region`). The IAM user needs `bedrock:InvokeModel` on the Claude
inference profile.

**Data.** Loghub-2k with the corrected labels of Khan et al. (ICSE 2022), as distributed by the
LogBatcher authors (they also re-calibrated Proxifier):

```bash
BASE=https://raw.githubusercontent.com/LogIntelligence/LogBatcher/master/datasets/loghub-2k
for d in HDFS Hadoop Spark Zookeeper BGL HPC Thunderbird Windows Linux Android HealthApp Apache Proxifier OpenSSH OpenStack Mac; do
  mkdir -p data/loghub-2k/$d
  for f in ${d}_2k.log ${d}_2k.log_structured_corrected.csv ${d}_2k.log_templates_corrected.csv; do
    curl -sfL -o data/loghub-2k/$d/$f "$BASE/$d/$f"; done; done
```

## Running

```bash
# one dataset, printing every prompt and response
.venv/bin/python -m src.logbatcher --dataset HDFS --verbose
.venv/bin/python -m src.logbatcher --dataset Apache --provider mock_oracle      # no API calls

.venv/bin/python -m src.data_loader --stats                                    # dataset table
.venv/bin/python -m src.baseline_drain --postprocess                           # Drain baseline
.venv/bin/python -m experiments.run_main                                       # E1 + E2 (gpt-oss-120b)
.venv/bin/python -m experiments.run_extension --run gpt-oss-20b claude-haiku-4.5 --summarize   # E6
.venv/bin/python -m experiments.run_ablation --study ablation --model claude-haiku-4.5          # E3
.venv/bin/python -m experiments.run_ablation --study sampling --model claude-haiku-4.5          # E4
.venv/bin/python -m experiments.run_batchsize --model claude-haiku-4.5                          # E5
.venv/bin/python -m experiments.error_analysis --experiments drain_pp E1_logbatcher E6_claude-haiku-4.5
.venv/bin/python -m experiments.make_figures

# unstated-setting sensitivity (E8-E10)
.venv/bin/python -m experiments.run_main --experiment E8_rules_gpt-oss-120b --set logbatcher.extended_rules=true
.venv/bin/python -m experiments.run_main --experiment E9_euclid_claude-haiku-4.5 --set llm.provider=bedrock llm.model=us.anthropic.claude-haiku-4-5-20251001-v1:0 logbatcher.dbscan.metric=euclidean
.venv/bin/python -m experiments.run_main --experiment E9_euclid_gpt-oss-120b --datasets Proxifier OpenStack HDFS OpenSSH Linux Android --set logbatcher.dbscan.metric=euclidean

# paper
.venv/bin/python -m experiments.run_checks
.venv/bin/python -m experiments.make_paper_assets
.venv/bin/python -m experiments.check_paper
```

Every LLM response is cached in `cache/llm_cache.sqlite`, keyed by a hash of the full request.
Re-running any command reproduces the same numbers without new API calls; an interrupted run
(e.g. Groq's free-tier daily token cap) resumes where it stopped.

## Demo (for a viva)

Runs offline from the response cache, in about a second, and prints each batch, the LLM's answer and
the template it became:

```bash
.venv/bin/python -m src.logbatcher --dataset Apache --verbose
```

## Building the paper

`paper/main.tex` uses the IEEEtran class. Upload the whole `paper/` folder to Overleaf, or run
`pdflatex main.tex` twice inside `paper/`. All numbers and tables come from `paper/generated/`; after
changing results, re-run `make_paper_assets` rather than editing them.

## Implementation notes and deviations from the paper

- **LLMs.** Main model: `openai/gpt-oss-120b` on Groq (temperature 0, reasoning effort low). The
  paper's GPT-3.5-Turbo is not used. Extension: `openai/gpt-oss-20b` (Groq) and Claude Haiku 4.5
  (Bedrock, temperature 0 via `extra_body`, because SDK 1.x removed the sampling keyword).
  E3-E5 use Claude Haiku 4.5 because of Groq free-tier limits.
- **Placeholder matching.** A `<*>` must match non-empty text, and is first matched as a single
  token; multi-token matching is used only when single-token matching matches nothing in the
  cluster (`logbatcher.match_mode`). The paper's `(.*?)` let `connection from <*> (<*>) at <*>`
  swallow the separately labelled `connection from <*> () at <*>` and let a trailing `<*>` absorb
  extra fields (Linux GA 0.30 even with a perfect oracle LLM; 1.00 after the change).
- **Prompt.** Own wording, demonstration-free, instruction + batch. It tells the model to template
  only the first log when the batch mixes statements; without that, gpt-oss produced
  over-general templates for mixed clusters (HDFS GA 0.85 -> 0.998).
- **Post-processing.** Khan et al. rules (DS, PS, BL, US, DG, WV, DV, CV) plus hexadecimal values;
  boolean/default-string matching is case-sensitive. The same rules are applied to Drain output,
  which reproduces the paper's Drain numbers (avg GA/MLA/ED 0.840/0.545/0.902 vs 0.840/0.544/0.902).
- **Tokens.** Reported twice: provider-reported (`T_total`, includes gpt-oss reasoning and chat
  formatting) and tiktoken `cl100k_base` over prompt + answer text (`T_total_tiktoken`, closer to the
  paper's accounting).
- **No partitioning (ablation).** Consecutive 100-line windows instead of DBSCAN clusters.

## Extension: variability check (not in the paper, off by default)

`logbatcher.variability_check: true` re-checks each LLM template against the batch it came from:
a `<*>` that captures the *same* digit-free word in every batch log shows no variability, so it is
put back as constant text (`src/matcher.py:restore_constants`). It adds no LLM calls. Designed
after inspecting Haiku 4.5 errors on Windows/Proxifier, so it is evaluated on all 16 datasets
(`results/E7_vcc_*.csv`). Run: `python -m experiments.run_main --experiment E7_vcc_<model> --set logbatcher.variability_check=true ...`

## Sources consulted

The datasets come from the LogBatcher authors' repository, and Drain is the loghub `logparser3`
implementation. The pipeline was written from the paper. After E1 finished we read the authors'
public code (https://github.com/LogIntelligence/LogBatcher, state of October 2026, which is later than
the preprint) to diagnose the Proxifier and OpenStack failures. That reading gave two things: the
Euclidean DBSCAN distance (`logbatcher.dbscan.metric`) and three template rules, re-implemented behind
`logbatcher.extended_rules`. Both are off by default, so E1-E7 are unaffected; they are evaluated
separately as E8-E10. None of the authors' code was copied or executed.
