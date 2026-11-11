"""Phase 4a: tokenisation and variable masking, used only for clustering.

The LLM always sees the original log text; these tokens just decide which logs look alike.
"""
import regex as re

MASK = "<*>"

# Whole-message masks applied before tokenising (order matters: URLs before paths/IPs).
_MASKS = [
    re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s,;\"'()\[\]]+"),                       # URL
    re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),  # UUID
    re.compile(r"blk_-?\d+"),                                                         # HDFS block id
    re.compile(r"(?<![\w.])(\d{1,3}\.){3}\d{1,3}(:\d+)?(?![\w.])"),                   # IPv4[:port]
    re.compile(r"(?<![\w/])(/[\w.\-@%+~]+){2,}/?"),                                   # file path (2+ segments)
    re.compile(r"\b0[xX][0-9a-fA-F]+\b"),                                             # 0x-hex
    re.compile(r"\b(?=[0-9a-fA-F]*\d)(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{8,}\b"),     # long hex
    re.compile(r"(?<![A-Za-z])[-+]?\d+(\.\d+)?(?![A-Za-z])"),                         # plain number
]

# Second-level delimiters; '/' and '.' are left alone so paths and dotted names stay whole.
_DELIMS = re.compile(r"([=:,()\[\]{}\"'|;])")
_HAS_DIGIT = re.compile(r"\d")


def mask(content):
    """Replace likely variables with <*> in the message string, keeping the surrounding text."""
    for rx in _MASKS:
        content = rx.sub(MASK, content)
    return content


def tokenize(content):
    """Whitespace split, then split each token on code-ish delimiters (keeping URLs/paths intact,
    because they were already masked to a single <*>)."""
    tokens = []
    for tok in mask(content).split():
        for piece in _DELIMS.split(tok):
            if piece and not piece.isspace():
                tokens.append(piece)
    return tokens


def clustering_tokens(content):
    """Tokens for TF-IDF: delimiter-split, and any token still containing a digit becomes <*>."""
    return [MASK if (MASK in t or _HAS_DIGIT.search(t)) else t for t in tokenize(content)]


def masked_template(content):
    """Heuristic template of a single log (fallback when the LLM answer is unusable)."""
    out = []
    for tok in mask(content).split():
        pieces = _DELIMS.split(tok)
        out.append("".join(MASK if (_HAS_DIGIT.search(p) and MASK not in p) else p for p in pieces))
    return " ".join(out)
