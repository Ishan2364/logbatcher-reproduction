"""Phase 4e (part 1): turn a raw LLM answer into a normalised template.

extract_template(): pull the template out of the LLM text (backticks, {placeholders} -> <*>).
normalize_template(): template-style rules following Khan et al. (ICSE'22) guidelines, plus
hexadecimal masking as LogBatcher suggests. The same rules are used to post-process baseline
(Drain) templates so every parser is scored under one labelling style.
"""
import regex as re

PLACEHOLDER = "<*>"

_BOOL_AND_DEFAULT = {"true", "false", "null", "root", "admin"}
# Delimiters for spotting path-like strings ('/' is not one of them, '.' and '-' are kept).
_PATH_DELIMS = r"(\s|,|!|;|:|=|\||\"|'|\[|\]|\(|\)|\{|\})"
# Delimiters for the remaining token-level rules.
_TOKEN_DELIMS = r"(\s|,|!|;|:|=|\||\"|'|\[|\]|\(|\)|\{|\}|\.|-|\+|@|#|\$|%|&)"
_PATH = re.compile(r"^(/[^/]+)+/?$")
_DIGITS = re.compile(r"^\d+$")
_HEX = re.compile(r"^(0[xX][0-9a-fA-F]+|(?=[0-9a-fA-F]*\d)(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{8,})$")
_WORD_WITH_VAR = re.compile(r"^[^\s/]*<\*>[^\s/]*$")
# Optional rules, re-implemented after reading the LogBatcher authors' public post-processing:
_SLASH_THEN_DOT = re.compile(r".*/.*\..*")                  # HTTP/1.1, /path/to/file.txt
_LONG_HOSTNAME = re.compile(r"^([a-zA-Z0-9-]+\.){3,}[a-z]+$")  # proxy.cse.cuhk.edu.hk
_SIZE_UNIT = re.compile(r"<\*> [KGTM]?B\b")                  # <*> KB -> <*>


def _merge_repeated(template):
    """Collapse variables glued together by punctuation, e.g. '<*>:<*>' -> '<*>'."""
    pairs = ["<*>.<*>", "<*>:<*>", "<*>#<*>", "<*>/<*>", "<*>@<*>", "<*><*>"]
    changed = True
    while changed:
        changed = False
        for p in pairs:
            if p in template:
                template = template.replace(p, PLACEHOLDER)
                changed = True
        for wrapped, plain in ((" #<*>#", " <*>"), (" #<*> #", " <*>")):
            if wrapped in template:
                template = template.replace(wrapped, plain)
                changed = True
    return template


def _is_path_like(token, extended):
    if _PATH.match(token):
        return True
    return extended and bool(_SLASH_THEN_DOT.match(token) or _LONG_HOSTNAME.match(token))


def normalize_template(template, extended=False):
    t = re.sub(r"\s+", " ", template).strip()                        # DS: double spaces
    parts = re.split(_PATH_DELIMS, t)                                 # PS: path-like strings
    t = "".join(PLACEHOLDER if _is_path_like(p, extended) else p for p in parts)
    out = []
    for tok in re.split(_TOKEN_DELIMS, t):
        if tok in _BOOL_AND_DEFAULT:                                  # BL / US (lower-case only)
            tok = PLACEHOLDER
        elif _DIGITS.match(tok) or _HEX.match(tok):                   # DG + hex
            tok = PLACEHOLDER
        elif PLACEHOLDER in tok and tok != PLACEHOLDER and _WORD_WITH_VAR.match(tok):
            tok = PLACEHOLDER                                         # WV: word glued to a variable
        out.append(tok)
    t = _merge_repeated("".join(out))                                 # DV / CV
    return _SIZE_UNIT.sub(PLACEHOLDER, t) if extended else t


_FENCE = re.compile(r"```(?:[a-zA-Z]*\n)?(.*?)```", re.S)
_TICK = re.compile(r"`([^`\n]+)`")
_BRACES = re.compile(r"\{[^{}]*\}")


def extract_template(text):
    """Return the template text from an LLM answer, or None if nothing usable is found."""
    if text is None:
        return None
    text = text.strip()
    m = _FENCE.search(text)
    if m:
        lines = [l.strip() for l in m.group(1).strip().splitlines() if l.strip()]
        cand = lines[0] if lines else None
    else:
        m = _TICK.search(text)
        if m:
            cand = m.group(1)
        else:
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            cand = lines[-1] if lines else None
            if cand:
                cand = re.sub(r"^(log\s*)?template\s*:\s*", "", cand, flags=re.I)
    if not cand:
        return None
    cand = re.sub(r"^(log\s*)?template\s*:\s*", "", cand.strip("`").strip(), flags=re.I)
    return _BRACES.sub(PLACEHOLDER, cand)


def postprocess(text, extended=False):
    """LLM text -> normalised template (or None)."""
    t = extract_template(text)
    if t is None:
        return None
    t = normalize_template(t, extended)
    return t or None
