import asyncio
import os
import threading
import time
from typing import List

import pytest

from nanugpt.api_client.aoai_client import AOAIClient, AzureConfig, ChatResult
from nanugpt.api_client.prompt_exec import PromptExecutor, PromptRequest


def test_prompt_request_to_messages_both_roles():
    req = PromptRequest(system_prompt="sys", user_prompt="user")
    messages = req.to_messages()
    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]


def test_prompt_request_to_messages_single_role_system_only():
    req = PromptRequest(system_prompt="sys", user_prompt="")
    messages = req.to_messages()
    assert messages == [{"role": "system", "content": "sys"}]


def test_prompt_request_to_messages_single_role_user_only():
    req = PromptRequest(system_prompt="", user_prompt="user")
    messages = req.to_messages()
    assert messages == [{"role": "user", "content": "user"}]


def test_prompt_request_to_messages_requires_content():
    req = PromptRequest(system_prompt="  ", user_prompt="\n")
    with pytest.raises(ValueError):
        req.to_messages()


class ConcurrencyTrackingClient:
    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self._lock = threading.Lock()
        self._current = 0
        self.max_inflight = 0
        self.calls: List[List[dict]] = []

    def chat(self, messages, **kwargs):
        with self._lock:
            self._current += 1
            if self._current > self.max_inflight:
                self.max_inflight = self._current
        try:
            time.sleep(self.delay)
            self.calls.append(messages)
            return ChatResult(content="ok")
        finally:
            with self._lock:
                self._current -= 1


def test_run_prompts_respects_concurrency_and_order():
    client = ConcurrencyTrackingClient(delay=0.05)
    executor = PromptExecutor(client, max_concurrency=4)
    prompts = [
        PromptRequest(system_prompt=f"sys {i}", user_prompt=f"user {i}") for i in range(4)
    ]

    results = executor.run_prompts(prompts, concurrency=2)

    assert [res.index for res in results] == [0, 1, 2, 3]
    assert all(res.succeeded for res in results)
    assert client.max_inflight == 2
    assert len(client.calls) == 4


class RetryRecordingClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        on_retry = kwargs.get("on_retry")
        if on_retry:
            on_retry(1, RuntimeError("transient error"))
        return ChatResult(content=f"response-{self.calls}")


def test_run_prompts_invokes_hooks_and_user_retry():
    client = RetryRecordingClient()
    executor = PromptExecutor(client, max_concurrency=1)

    start_calls = []
    complete_calls = []
    retry_calls = []
    user_retry_calls = []

    async def on_start(index, request, result):
        start_calls.append(index)

    async def on_complete(index, request, chat_result, result):
        complete_calls.append((index, chat_result.content))

    def on_retry(index, request, attempt, exc):
        retry_calls.append((index, attempt, str(exc)))

    def user_retry(attempt, exc):
        user_retry_calls.append((attempt, str(exc)))

    prompt = PromptRequest(
        system_prompt="sys",
        user_prompt="user",
        extra_kwargs={"on_retry": user_retry},
        metadata={"tracking_id": "abc123"},
    )

    results = executor.run_prompts(
        [prompt],
        on_start=on_start,
        on_complete=on_complete,
        on_retry=on_retry,
    )

    result = results[0]
    assert result.succeeded
    assert result.attempts == 2
    assert start_calls == [0]
    assert complete_calls == [(0, "response-1")]
    assert retry_calls == [(0, 1, "transient error")]
    assert user_retry_calls == [(1, "transient error")]
    assert result.request.metadata["tracking_id"] == "abc123"


def test_run_prompts_raises_for_non_prompt_request():
    client = ConcurrencyTrackingClient()
    executor = PromptExecutor(client, max_concurrency=1)
    with pytest.raises(TypeError):
        executor.run_prompts([PromptRequest(system_prompt="sys", user_prompt="ok"), object()])


def test_run_prompts_validates_concurrency():
    client = ConcurrencyTrackingClient()
    executor = PromptExecutor(client, max_concurrency=1)
    with pytest.raises(ValueError):
        executor.run_prompts([PromptRequest(system_prompt="sys", user_prompt="ok")], concurrency=0)


@pytest.mark.integration
def test_prompt_executor_end_to_end_real_api():
    try:
        cfg = AzureConfig.from_env()
    except RuntimeError:
        pytest.skip("Azure OpenAI environment variables are not set.")

    client = AOAIClient(cfg, timeout=60.0, max_retries=3)
    executor = PromptExecutor(client, max_concurrency=2)

    prompts = [
        PromptRequest(
            system_prompt="You are a concise assistant.",
            user_prompt="Respond with the word 'hello' once.",
        ),
        PromptRequest(
            system_prompt="You are a concise assistant.",
            user_prompt="Respond with the word 'world' once.",
        ),
    ]

    results = executor.run_prompts(prompts, concurrency=2)
    assert len(results) == 2
    assert all(res.succeeded for res in results)
    for res in results:
        assert res.chat_result is not None
        assert res.chat_result.content
