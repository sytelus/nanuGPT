from __future__ import annotations

"""
Async prompt execution utilities built on top of AOAIClient.

Hooks provided to PromptExecutor are expected to be light-weight callables; if they
return a coroutine it will be scheduled on the executor's event loop.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .aoai_client import AOAIClient, ChatResult

logger = logging.getLogger(__name__)

StartHook = Callable[[int, "PromptRequest", "PromptResult"], Any]
CompleteHook = Callable[[int, "PromptRequest", ChatResult, "PromptResult"], Any]
ErrorHook = Callable[[int, "PromptRequest", BaseException, "PromptResult"], Any]
RetryHook = Callable[[int, "PromptRequest", int, Exception], Any]

PromptTuple = Tuple[
    str,
    str,
    Optional[float],
    Optional[int],
    Optional[str],
]
PromptInput = Union["PromptRequest", PromptTuple]


@dataclass
class PromptRequest:
    system_prompt: str
    user_prompt: str
    temperature: Optional[float] = None
    max_completion_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    messages: Optional[Sequence[Dict[str, Any]]] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_messages(self) -> List[Dict[str, Any]]:
        if self.messages is not None:
            return [dict(msg) for msg in self.messages]
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]


@dataclass
class PromptResult:
    index: int
    request: PromptRequest
    chat_result: Optional[ChatResult] = None
    error: Optional[BaseException] = None
    attempts: int = 0
    started_at: float = 0.0
    ended_at: float = 0.0

    @property
    def duration(self) -> float:
        if self.ended_at <= self.started_at:
            return 0.0
        return self.ended_at - self.started_at

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.chat_result is not None


class PromptExecutor:
    """
    Orchestrates batched prompt execution against AOAIClient with bounded parallelism.
    """

    def __init__(
        self,
        client: AOAIClient,
        *,
        max_concurrency: int = 1,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        on_start: Optional[StartHook] = None,
        on_complete: Optional[CompleteHook] = None,
        on_error: Optional[ErrorHook] = None,
        on_retry: Optional[RetryHook] = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.client = client
        self.max_concurrency = max_concurrency
        self._loop = loop
        self._on_start = on_start
        self._on_complete = on_complete
        self._on_error = on_error
        self._on_retry = on_retry

    async def run_prompts(
        self,
        prompts: Sequence[PromptInput],
        *,
        concurrency: Optional[int] = None,
        on_start: Optional[StartHook] = None,
        on_complete: Optional[CompleteHook] = None,
        on_error: Optional[ErrorHook] = None,
        on_retry: Optional[RetryHook] = None,
    ) -> List[PromptResult]:
        if not prompts:
            return []

        loop = asyncio.get_running_loop()
        self._loop = loop

        max_concurrency = concurrency or self.max_concurrency
        if max_concurrency < 1:
            raise ValueError("concurrency must be >= 1")

        normalized: List[PromptRequest] = [self._normalize_prompt(p) for p in prompts]
        semaphore = asyncio.Semaphore(max_concurrency)

        tasks = [
            asyncio.create_task(
                self._run_single(
                    index=i,
                    request=req,
                    semaphore=semaphore,
                    hooks=(
                        on_start or self._on_start,
                        on_complete or self._on_complete,
                        on_error or self._on_error,
                        on_retry or self._on_retry,
                    ),
                )
            )
            for i, req in enumerate(normalized)
        ]

        results = await asyncio.gather(*tasks)
        return results

    def _normalize_prompt(self, value: PromptInput) -> PromptRequest:
        if isinstance(value, PromptRequest):
            return value
        if not isinstance(value, tuple):
            raise TypeError(f"Unsupported prompt spec type: {type(value)!r}")
        if not (2 <= len(value) <= 5):
            raise ValueError(
                "Prompt tuple must have between 2 and 5 elements: "
                "(system_prompt, user_prompt, [temperature], [max_completion_tokens], [reasoning_effort])"
            )
        system_prompt = value[0]
        user_prompt = value[1]
        temperature = value[2] if len(value) > 2 else None
        max_completion_tokens = value[3] if len(value) > 3 else None
        reasoning_effort = value[4] if len(value) > 4 else None
        return PromptRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
        )

    async def _run_single(
        self,
        *,
        index: int,
        request: PromptRequest,
        semaphore: asyncio.Semaphore,
        hooks: Tuple[
            Optional[StartHook],
            Optional[CompleteHook],
            Optional[ErrorHook],
            Optional[RetryHook],
        ],
    ) -> PromptResult:
        start_hook, complete_hook, error_hook, retry_hook = hooks
        async with semaphore:
            started_at = time.perf_counter()
            result = PromptResult(index=index, request=request, started_at=started_at)
            self._schedule_hook(start_hook, index, request, result)
            # Run the synchronous AOAIClient call in a background thread
            result = await asyncio.to_thread(
                self._execute_prompt_sync,
                index,
                request,
                result,
                retry_hook,
            )
            result.ended_at = time.perf_counter()
            if result.error is not None:
                self._schedule_hook(error_hook, index, request, result.error, result)
            else:
                assert result.chat_result is not None
                self._schedule_hook(complete_hook, index, request, result.chat_result, result)
            return result

    def _execute_prompt_sync(
        self,
        index: int,
        request: PromptRequest,
        result: PromptResult,
        retry_hook: Optional[RetryHook],
    ) -> PromptResult:
        retry_calls = 0

        def _handle_retry(attempt: int, exc: Exception) -> None:
            nonlocal retry_calls
            retry_calls += 1
            self._schedule_hook(retry_hook, index, request, attempt, exc)
            if user_retry is not None:
                try:
                    user_retry(attempt, exc)
                except Exception:
                    logger.exception("User-provided on_retry hook failed")

        kwargs: Dict[str, Any] = dict(request.extra_kwargs) if request.extra_kwargs else {}
        if request.temperature is not None:
            kwargs.setdefault("temperature", request.temperature)
        if request.max_completion_tokens is not None:
            kwargs.setdefault("max_completion_tokens", request.max_completion_tokens)
        if request.reasoning_effort is not None:
            kwargs.setdefault("reasoning_effort", request.reasoning_effort)
        user_retry = kwargs.get("on_retry")
        kwargs["on_retry"] = _handle_retry

        messages = request.to_messages()
        try:
            chat_res = self.client.chat(messages, **kwargs)
            result.chat_result = chat_res
            attempts = retry_calls + 1
            result.attempts = attempts
        except Exception as exc:
            result.error = exc
            attempts = retry_calls or 1
            result.attempts = attempts
        return result

    def _schedule_hook(self, hook: Optional[Callable[..., Any]], *args: Any) -> None:
        if hook is None:
            return

        def runner() -> None:
            try:
                outcome = hook(*args)
                if asyncio.iscoroutine(outcome):
                    try:
                        current_loop = asyncio.get_running_loop()
                        current_loop.create_task(outcome)
                    except RuntimeError:
                        asyncio.run(outcome)
            except Exception:
                logger.exception("PromptExecutor hook failed")

        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(runner)
        else:
            runner()

