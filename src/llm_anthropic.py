"""Claude on Amazon Bedrock (Anthropic Python SDK, `anthropic[bedrock]`).

Uses the Bedrock runtime client (`AnthropicBedrock`, InvokeModel) with a cross-region inference
profile ID such as `us.anthropic.claude-haiku-4-5-20251001-v1:0`. Credentials come from the
standard AWS chain (~/.aws/credentials). Temperature is sent through `extra_body` because SDK 1.x
removed the sampling keywords; models from the 4.5/4.6 line (e.g. Haiku 4.5) still honour it.
"""
import time

import anthropic


class BedrockClaude:
    def __init__(self, llm_cfg):
        self.cfg = llm_cfg
        self.client = anthropic.AnthropicBedrock(aws_region=llm_cfg.get("aws_region", "us-east-2"),
                                                 max_retries=llm_cfg.get("max_retries", 8), timeout=120)

    def chat(self, model, messages, extra=None):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        body = dict(extra or {})
        if self.cfg.get("temperature") is not None:
            body["temperature"] = self.cfg["temperature"]
        t0 = time.time()
        r = self.client.messages.create(model=model, max_tokens=self.cfg.get("max_tokens", 1024),
                                        system=system, messages=turns, extra_body=body)
        text = "" if r.stop_reason == "refusal" else "".join(b.text for b in r.content if b.type == "text")
        return {"text": text, "input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens,
                "latency": time.time() - t0}
