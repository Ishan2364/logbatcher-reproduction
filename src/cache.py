"""Phase 4c: template cache with regex matching, most-frequent-first lookup.

Matching modes for a <*> placeholder:
  strict - one or more non-space characters (a variable is a single token)
  loose  - any non-empty text, spaces included (the paper's '(.*?)', minus the empty match)
Strict is tried first and loose only when strict matches nothing; see matcher.match_and_prune.
"""
from dataclasses import dataclass

import regex as re

_REGEX_TIMEOUT = 0.05   # seconds; a pathological template must not hang the run
VAR_PATTERNS = {"strict": r"(\S+?)", "loose": r"(.+?)", "paper": r"(.*?)"}


def template_to_regex(template, mode="loose"):
    """Escape the template, turn each <*> into a lazy group and any whitespace into \\s+, anchor it."""
    parts = template.split("<*>")
    body = VAR_PATTERNS[mode].join(re.sub(r"\\\s+|\s+", r"\\s+", re.escape(p)) for p in parts)
    return re.compile("^" + body + "$", re.S)


def matches(rx, log):
    try:
        return rx.match(log.strip(), timeout=_REGEX_TIMEOUT) is not None
    except TimeoutError:
        return False


@dataclass
class CacheEntry:
    template: str
    regex: object
    mode: str
    reference_log: str
    match_count: int


class TemplateCache:
    def __init__(self, length_check=True):
        self.entries = []
        self.length_check = length_check
        self.hits = 0

    def __len__(self):
        return len(self.entries)

    def lookup(self, log, weight=1):
        """Return the template of the most frequent entry that fully matches `log`, else None."""
        n_tok = len(log.split())
        for e in self.entries:
            if self.length_check and len(e.reference_log.split()) != n_tok:
                continue
            if matches(e.regex, log):
                e.match_count += weight
                self.hits += weight
                self._resort()
                return e.template
        return None

    def add(self, template, reference_log, match_count=1, mode="loose"):
        for e in self.entries:
            if e.template == template:
                e.match_count += match_count
                if mode == "loose" and e.mode != "loose":     # widen if this template needed it
                    e.mode, e.regex = mode, template_to_regex(template, mode)
                self._resort()
                return e
        e = CacheEntry(template, template_to_regex(template, mode), mode, reference_log, match_count)
        self.entries.append(e)
        self._resort()
        return e

    def _resort(self):
        self.entries.sort(key=lambda e: e.match_count, reverse=True)
