"""Phase 4e (part 2): matching & pruning (Algorithm 1 of the paper)."""
from src.cache import matches, template_to_regex


def match_and_prune(cluster, template, match_mode="strict_first"):
    """Split a cluster (unique contents) into logs the template matches and logs it doesn't.

    match_mode:
      strict_first - single-token variables first; allow multi-token variables only if that
                     matches nothing (stops a trailing <*> from absorbing extra tokens)
      loose        - always allow multi-token, non-empty variables
      paper        - the paper's '(.*?)' (variables may also be empty)
    Returns (matched, unmatched, mode_used).
    """
    modes = ["strict", "loose"] if match_mode == "strict_first" else [match_mode]
    for mode in modes:
        rx = template_to_regex(template, mode)
        matched, unmatched = [], []
        for log in cluster:
            (matched if matches(rx, log) else unmatched).append(log)
        if matched:
            return matched, unmatched, mode
    return [], list(cluster), modes[-1]


def is_usable(template):
    """Reject templates with no constant text at all (e.g. '<*>' or '<*> <*>'), which would match
    almost anything and swallow other templates' logs."""
    return template is not None and any(ch.isalnum() for ch in template.replace("<*>", ""))


def restore_constants(template, batch):
    """Variability check: a <*> that captures the same purely alphabetic word in every batch log
    shows no variability, so it is put back as constant text (words/identifiers without digits only).
    Needs >= 2 matching batch logs."""
    import regex as re
    from src.cache import _REGEX_TIMEOUT, template_to_regex

    def _match(rx, log):
        try:
            return rx.match(log.strip(), timeout=_REGEX_TIMEOUT)
        except TimeoutError:
            return None

    for mode in ("strict", "loose"):
        rx = template_to_regex(template, mode)
        found = [m for m in (_match(rx, b) for b in batch) if m is not None]
        if found:
            break
    if len(found) < 2:
        return template
    parts = template.split("<*>")
    out = [parts[0]]
    for i, rest in enumerate(parts[1:]):
        values = {m.group(i + 1) for m in found}
        v = values.pop() if len(values) == 1 else None
        out.append(v if v is not None and re.fullmatch(r"[A-Za-z][A-Za-z_]*", v) else "<*>")
        out.append(rest)
    return "".join(out)
