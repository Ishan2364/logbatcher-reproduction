"""Phase 4d (part 2): prompt construction and a pluggable LLM client.

Every raw response is cached in SQLite keyed by a hash of (provider, model, request), so
re-running an experiment is free and reproducible. Token counts are recorded twice: the
provider's own usage numbers and a tiktoken (cl100k_base) count, which is closer to the
paper's GPT-3.5 accounting.

Providers:
  groq         - Groq chat completions (needs GROQ_API_KEY in .env)
  bedrock      - Claude on Amazon Bedrock (AWS credentials; see src/llm_anthropic.py)
  mock_oracle  - returns the ground-truth template of the first log in the batch (pipeline test)
  mock_mask    - returns a regex-masked version of the first log (no LLM at all)
"""
import hashlib
import json
import os
import random
import sqlite3
import time

import tiktoken
from dotenv import load_dotenv

from src.config import PROJECT_ROOT, project_path

INSTRUCTION = (
    "You will see a batch of log messages, one per line. Logs printed by the same logging "
    "statement share their constant text and differ only in run-time values. Replace every "
    "run-time value (IDs, numbers, IP addresses, ports, paths, URLs, names, sizes, times and "
    "similar) with a {placeholder}. A placeholder stands for one value: never let it cover "
    "constant words. A log may contain no variables. If the logs do not all come from the same "
    "statement, give the template of the first log only. Reply with one template wrapped in backticks."
)

_enc = tiktoken.get_encoding("cl100k_base")
REAL_PROVIDERS = ("groq", "bedrock")
MOCK_PROVIDERS = ("mock_oracle", "mock_mask")


def count_tokens(text):
    return len(_enc.encode(text))


def build_messages(batch):
    return [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": "\n".join(batch)},
    ]


class BudgetExhausted(RuntimeError):
    """Provider asked us to wait longer than max_wait_seconds (e.g. daily token cap)."""


class ResponseCache:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")      # several experiment processes may share it
        self.db.execute("""CREATE TABLE IF NOT EXISTS responses (
            key TEXT PRIMARY KEY, provider TEXT, model TEXT, request TEXT, text TEXT,
            input_tokens INTEGER, output_tokens INTEGER, latency REAL, created REAL)""")
        self.db.commit()

    def get(self, key):
        row = self.db.execute("SELECT text, input_tokens, output_tokens, latency FROM responses WHERE key=?",
                              (key,)).fetchone()
        if row is None:
            return None
        return {"text": row[0], "input_tokens": row[1], "output_tokens": row[2], "latency": row[3]}

    def put(self, key, provider, model, request, resp):
        self.db.execute("INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?,?,?)",
                        (key, provider, model, json.dumps(request), resp["text"], resp["input_tokens"],
                         resp["output_tokens"], resp["latency"], time.time()))
        self.db.commit()


class LLMClient:
    def __init__(self, llm_cfg, cache_path, oracle=None):
        self.cfg = llm_cfg
        self.provider = llm_cfg["provider"]
        if self.provider not in REAL_PROVIDERS + MOCK_PROVIDERS:
            raise ValueError(f"unknown LLM provider {self.provider!r}")
        self.model = llm_cfg["model"] if self.provider in REAL_PROVIDERS else self.provider
        self.extra = dict(llm_cfg.get("model_params", {}).get(self.model, {}))
        self.cache = ResponseCache(cache_path)
        self.oracle = oracle or {}
        self.calls = []          # one record per invocation (cached or not), in order
        self._client = None

    # -- provider calls ---------------------------------------------------------------
    def _groq(self, messages):
        if self._client is None:
            load_dotenv(PROJECT_ROOT / ".env")
            from groq import Groq
            key_var = self.cfg.get("api_key_env", "GROQ_API_KEY")
            key = os.environ.get(key_var)
            if not key:
                raise RuntimeError(f"{key_var} is not set; add it to logbatcher-project/.env")
            self._client = Groq(api_key=key, max_retries=0, timeout=120)
        import groq
        for attempt in range(self.cfg.get("max_retries", 8) + 1):
            t0 = time.time()
            try:
                r = self._client.chat.completions.create(
                    model=self.model, messages=messages, temperature=self.cfg.get("temperature", 0),
                    max_tokens=self.cfg.get("max_tokens", 512), **self.extra)
                return {"text": r.choices[0].message.content or "", "input_tokens": r.usage.prompt_tokens,
                        "output_tokens": r.usage.completion_tokens, "latency": time.time() - t0}
            except (groq.RateLimitError, groq.APIConnectionError, groq.InternalServerError) as e:
                wait = self._retry_after(e, attempt)
                if wait > self.cfg.get("max_wait_seconds", 900):
                    raise BudgetExhausted(f"{self.model}: provider asked to wait {wait:.0f}s ({e})") from e
                print(f"  [groq] {type(e).__name__}; sleeping {wait:.1f}s (attempt {attempt + 1})", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"{self.model}: giving up after {attempt + 1} attempts")

    @staticmethod
    def _retry_after(err, attempt):
        resp = getattr(err, "response", None)
        if resp is not None:
            h = resp.headers.get("retry-after")
            if h:
                try:
                    return float(h) + 0.5
                except ValueError:
                    pass
        return min(60.0, 2 ** attempt) + random.random()

    def _bedrock(self, messages):
        if self._client is None:
            from src.llm_anthropic import BedrockClaude
            self._client = BedrockClaude(self.cfg)
        return self._client.chat(self.model, messages, self.extra)

    def _mock(self, messages):
        from src.preprocess import masked_template
        first = messages[-1]["content"].split("\n")[0]
        if self.provider == "mock_oracle":
            tmpl = self.oracle.get(first, masked_template(first))
        else:
            tmpl = masked_template(first)
        text = "`" + tmpl.replace("<*>", "{var}") + "`"
        return {"text": text, "input_tokens": sum(count_tokens(m["content"]) for m in messages),
                "output_tokens": count_tokens(text), "latency": 0.0}

    # -- public API ---------------------------------------------------------------------
    def query(self, batch):
        """Send one batch; returns dict(text, input_tokens, output_tokens, latency, cached, ...)."""
        messages = build_messages(batch)
        request = {"provider": self.provider, "model": self.model, "messages": messages,
                   "temperature": self.cfg.get("temperature", 0), "max_tokens": self.cfg.get("max_tokens", 512),
                   "extra": self.extra}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        real = self.provider in REAL_PROVIDERS
        hit = self.cache.get(key) if real else None
        if hit is not None:
            resp, cached = hit, True
        else:
            if self.provider == "groq":
                resp = self._groq(messages)
            elif self.provider == "bedrock":
                resp = self._bedrock(messages)
            else:
                resp = self._mock(messages)
            cached = False
            if real:
                self.cache.put(key, self.provider, self.model, request, resp)
        prompt_text = "\n".join(m["content"] for m in messages)
        rec = {**resp, "cached": cached, "batch_size": len(batch),
               "tt_input": count_tokens(prompt_text), "tt_output": count_tokens(resp["text"])}
        self.calls.append(rec)
        return rec


def make_client(cfg, oracle=None):
    return LLMClient(cfg["llm"], project_path(cfg["paths"]["llm_cache"]), oracle=oracle)
