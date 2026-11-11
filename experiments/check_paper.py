"""Phase 9: static checks on paper/main.tex (usable without a LaTeX installation).

    python -m experiments.check_paper

Checks environments and braces, \\ref/\\label and \\cite/\\bibitem pairs, figure files, number macros
and generated tables, flags words missing from the system dictionary, and estimates the page count.
"""
import re
import sys
from pathlib import Path

from src.config import project_path

PAPER = project_path("paper")
KNOWN_WORDS = {
    "logbatcher", "loghub", "drain", "tokenise", "tokenisation", "tokenised", "ablation", "llm", "llms", "dbscan", "tf",
    "idf", "dpp", "hdfs", "hadoop", "zookeeper", "bgl", "hpc", "openssh", "openstack", "proxifier", "healthapp",
    "android", "groq", "bedrock", "tiktoken", "gpt", "oss", "haiku", "claude", "chatgpt", "lilac", "divlog", "logppt",
    "uniparser", "logram", "ael", "aiops", "levenshtein", "ga", "mla", "ed", "api", "url", "urls", "uuids", "ip",
    "sqlite", "regex", "github", "arxiv", "vectorised", "normalisation", "normalised", "labelled", "labelling",
    "parsers", "parser", "datasets", "dataset", "misclassified", "hyperparameter", "hyperparameters", "ieee", "acm",
}


def strip_comments(text):
    return re.sub(r"(?<!\\)%.*", "", text)


def check_environments(text, name, errors):
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", text):
        kind, env = m.groups()
        if kind == "begin":
            stack.append(env)
        elif not stack or stack.pop() != env:
            errors.append(f"{name}: \\end{{{env}}} does not match")
    if stack:
        errors.append(f"{name}: unclosed environments {stack}")


def check_braces(text, name, errors):
    depth = 0
    for i, ch in enumerate(text):
        if ch in "{}" and text[i - 1] != "\\":
            depth += 1 if ch == "{" else -1
            if depth < 0:
                errors.append(f"{name}: unmatched }} near: {text[max(0, i - 50):i + 10]!r}")
                return
    if depth:
        errors.append(f"{name}: {depth} unclosed brace(s)")


def main():
    errors, warnings = [], []
    main_tex = strip_comments((PAPER / "main.tex").read_text())
    numbers_file = PAPER / "generated/numbers.tex"
    numbers = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", numbers_file.read_text()))

    inputs = re.findall(r"\\input\{([^}]+)\}", main_tex)
    full = main_tex
    for rel in inputs:
        path = PAPER / (rel + ".tex")
        if not path.exists():
            errors.append(f"missing input: {rel}.tex")
        elif "numbers" not in rel:
            tab = path.read_text()
            check_environments(tab, rel, errors)
            check_braces(tab, rel, errors)
            full += "\n" + tab
    check_environments(main_tex, "main.tex", errors)
    check_braces(main_tex, "main.tex", errors)

    used = set(re.findall(r"\\([A-Za-z]+)", full.replace("\\\\", " ")))
    builtin = {
        "documentclass", "usepackage", "usetikzlibrary", "graphicspath", "input", "begin", "end", "title", "author",
        "IEEEauthorblockN", "IEEEauthorblockA", "maketitle", "section", "subsection", "label", "ref", "cite", "emph",
        "texttt", "textit", "textbf", "item", "centering", "caption", "includegraphics", "columnwidth", "textwidth",
        "resizebox", "toprule", "midrule", "bottomrule", "cmidrule", "multicolumn", "footnotesize", "small",
        "scriptsize", "setlength", "tabcolsep", "varepsilon", "det", "propto", "to", "node", "draw", "bibitem",
        "newblock", "url", "IEEEkeywords", "appendix", "thanks", "bibliographystyle", "bibliography", "linewidth",
        "ldots", "times", "approx", "hline", "newcommand", "IEEEtriggeratref", "section*", "ge", "le",
    }
    for cmd in sorted(used):
        if cmd not in numbers and cmd not in builtin:
            errors.append(f"unknown command \\{cmd}")
    unused = sorted(set(numbers) - used)
    if unused:
        warnings.append(f"{len(unused)} generated numbers are not used in the text (harmless)")

    labels = set(re.findall(r"\\label\{([^}]+)\}", full))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", full))
    errors += [f"\\ref to missing label {r}" for r in sorted(refs - labels)]
    warnings += [f"label never referenced: {r}" for r in sorted(labels - refs)]

    cites = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", full):
        cites |= {c.strip() for c in group.split(",")}
    items = set(re.findall(r"\\bibitem\{([^}]+)\}", full))
    errors += [f"\\cite without \\bibitem: {c}" for c in sorted(cites - items)]
    errors += [f"reference never cited: {c}" for c in sorted(items - cites)]
    print(f"references: {len(items)}")

    for fig in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", full):
        if not any((PAPER / "figures" / f"{fig}{ext}").exists() for ext in ("", ".pdf", ".png")):
            errors.append(f"missing figure: {fig}")

    # Plain text with numbers filled in, for word count, spelling and reading.
    body = main_tex.split(r"\begin{document}", 1)[1]
    plain = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", body, flags=re.S)
    plain = re.sub(r"\\([A-Za-z]+)\{\}", lambda m: numbers.get(m.group(1), " "), plain)
    plain = re.sub(r"\\(?:cite|ref|label|input|includegraphics(?:\[[^\]]*\])?)\{[^}]*\}", " ", plain)
    plain = re.sub(r"\\texttt\{[^}]*\}", " ", plain)
    plain = re.sub(r"\$[^$]*\$", " ", plain)
    plain = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " ", plain, flags=re.S)
    plain = re.sub(r"\\(begin|end)\{[^}]*\}(\[[^\]]*\])?", " ", plain)
    plain = re.sub(r"\\[A-Za-z]+\*?", " ", plain)
    plain = re.sub(r"[{}~]", " ", plain)
    (PAPER / "generated/plain_text.txt").write_text(plain)

    refs_start = plain.find("Reference") if "References" in plain else -1
    words = len(re.findall(r"[A-Za-z0-9][\w.,'-]*", plain))
    tables = len(re.findall(r"\\input\{generated/tab_", main_tex))
    wide = sum("table*" in (PAPER / (r + ".tex")).read_text() for r in inputs if "tab_" in r)
    figs = main_tex.count(r"\begin{figure}") ; figs_wide = main_tex.count(r"\begin{figure*}")
    # IEEE conference: about 950 words of body text per page; floats estimated in page fractions.
    pages = words / 950 + wide * 0.28 + (tables - wide) * 0.09 + figs_wide * 0.25 + figs * 0.14
    print(f"words (with references): {words}; tables {tables} ({wide} wide); figures {figs + figs_wide}; "
          f"estimated pages {pages:.1f} (plan: 6-8)")

    dict_path = Path("/usr/share/dict/words")
    if dict_path.exists():
        vocab = {w.strip().lower() for w in dict_path.open()}
        tokens = re.findall(r"[A-Za-z][a-z]+", plain)

        def known(w):
            w = w.lower()
            stems = [w, w[:-1], w[:-2], w[:-3], w.replace("ise", "ize"), w.replace("isation", "ization"),
                     w.replace("our", "or"), w[:-3] + "e", w[:-2] + "e"]
            return w in KNOWN_WORDS or any(s in vocab or s in KNOWN_WORDS for s in stems)

        odd = sorted({t for t in tokens if not known(t)})
        print(f"words not in dictionary ({len(odd)}): {odd}")

    for w in warnings:
        print("warning:", w)
    if errors:
        print(f"\n{len(errors)} ERROR(S):")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print("all structural checks passed")


if __name__ == "__main__":
    main()
