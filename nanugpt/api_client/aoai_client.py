from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import AzureOpenAI

@dataclass
class ChatResult:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class AzureConfig:
    api_key: str
    endpoint: str
    api_version: str
    deployment: str

    @staticmethod
    def from_env() -> "AzureConfig":
        key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
        version = os.environ.get("OPENAI_API_VERSION", "").strip()
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
        missing = [
            name
            for name, val in [
                ("AZURE_OPENAI_API_KEY", key),
                ("AZURE_OPENAI_ENDPOINT", endpoint),
                ("OPENAI_API_VERSION", version),
                ("AZURE_OPENAI_DEPLOYMENT", deployment),
            ]
            if not val
        ]
        if missing:
            raise RuntimeError(
                f"Missing required environment variables for Azure OpenAI: {', '.join(missing)}"
            )
        return AzureConfig(api_key=key, endpoint=endpoint, api_version=version, deployment=deployment)


def ensure_text_parts(content: Any, part_type: str) -> List[Dict[str, str]]:
    if isinstance(content, str):
        return [{"type": part_type, "text": content}]
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                item_type = item.get("type", part_type)
                if item_type in {"input_text", "output_text"}:
                    parts.append({"type": item_type, "text": item["text"]})
                elif item_type == "text":
                    parts.append({"type": part_type, "text": item["text"]})
        if parts:
            return parts
    return [{"type": part_type, "text": str(content)}]


def messages_to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        part_type = "output_text" if role == "assistant" else "input_text"
        formatted.append({"role": role, "content": ensure_text_parts(content, part_type)})
    return formatted


def extract_response_text(resp: Any) -> str:
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text
    texts: List[str] = []
    for item in getattr(resp, "output", []) or []:
        for part in getattr(item, "content", []) or []:
            text = getattr(part, "text", None)
            if text:
                texts.append(text)
    if texts:
        return "".join(texts)
    if isinstance(resp, dict):
        output = resp.get("output") or []
        for item in output:
            for part in item.get("content", []):
                text = part.get("text")
                if text:
                    texts.append(text)
        if texts:
            return "".join(texts)
    return ""


class AOAIClient:
    """
    Thin wrapper around Azure OpenAI Responses API with retries.
    """

    def __init__(
        self,
        cfg: AzureConfig,
        timeout: float = 300.0,
        max_retries: int = 8,
        backoff_base: float = 2.0,
        backoff_jitter: Tuple[float, float] = (0.2, 0.6),
    ) -> None:
        self.client = AzureOpenAI(
            azure_endpoint=cfg.endpoint,
            api_key=cfg.api_key,
            api_version=cfg.api_version,
            timeout=timeout,
        )
        self.deployment = cfg.deployment
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_jitter = backoff_jitter

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        on_retry: Optional[Callable[[int, Exception], None]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> ChatResult:
        attempt = 0
        while True:
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.deployment,
                    "input": messages_to_responses_input(messages),
                }
                if temperature is not None:
                    kwargs["temperature"] = temperature
                if max_completion_tokens is not None:
                    kwargs["max_output_tokens"] = max_completion_tokens
                if reasoning_effort is not None:
                    kwargs["reasoning"] = {"effort": reasoning_effort}
                resp = self.client.responses.create(**kwargs)
                content = extract_response_text(resp)
                usage = getattr(resp, "usage", None)
                input_tokens = getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "output_tokens", 0) or 0
                total_tokens = getattr(usage, "total_tokens", 0) or 0
                return ChatResult(
                    content=content.strip(),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                )
            except Exception as e:
                attempt += 1
                if attempt > self.max_retries:
                    if on_retry is not None:
                        try:
                            on_retry(attempt, e)
                        except Exception:
                            pass
                    raise
                if on_retry is not None:
                    try:
                        on_retry(attempt, e)
                    except Exception:
                        pass
                sleep_s = (self.backoff_base ** (attempt - 1)) + random.uniform(*self.backoff_jitter)
                time.sleep(min(60.0, sleep_s))


