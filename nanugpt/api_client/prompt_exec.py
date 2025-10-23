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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .aoai_client import AOAIClient, ChatResult

logger = logging.getLogger(__name__)

StartHook = Callable[[int, "PromptRequest", "PromptResult"], Any]
CompleteHook = Callable[[int, "PromptRequest", ChatResult, "PromptResult"], Any]
ErrorHook = Callable[[int, "PromptRequest", BaseException, "PromptResult"], Any]
RetryHook = Callable[[int, "PromptRequest", int, Exception], Any]

@dataclass
class PromptRequest:
    """Prompt configuration plus extra settings for `AOAIClient.chat`.

    `extra_kwargs` passes options straight to the client (e.g. `tools`, `seed`, `on_retry`).
    `metadata` is caller bookkeeping available via `PromptResult.request.metadata`.
    """
    system_prompt: str
    user_prompt: str
    temperature: Optional[float] = None
    max_completion_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_messages(self) -> List[Dict[str, Any]]:
        """Create chat messages, omitting whichever role prompt is blank."""
        messages: List[Dict[str, Any]] = []
        if self.system_prompt and self.system_prompt.strip():
            messages.append({"role": "system", "content": self.system_prompt})
        if self.user_prompt and self.user_prompt.strip():
            messages.append({"role": "user", "content": self.user_prompt})
        if messages:
            return messages
        raise ValueError("PromptRequest requires a non-empty system_prompt or user_prompt")


@dataclass
class PromptResult:
    """Stores execution details for a prompt; caller metadata lives on `request.metadata`."""
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
    """Runs prompts in parallel batches using AOAIClient."""

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
        self._on_start = on_start      # Fired right before dispatching a prompt.
        self._on_complete = on_complete  # Fired after a prompt succeeds.
        self._on_error = on_error      # Fired after a prompt fails without recovery.
        self._on_retry = on_retry      # Fired before retrying a prompt after an error.

    async def run_prompts(
        self,
        prompts: Sequence[PromptRequest],
        *,  # Keyword-only from here so callers name the optional overrides.
        concurrency: Optional[int] = None,
        on_start: Optional[StartHook] = None,
        on_complete: Optional[CompleteHook] = None,
        on_error: Optional[ErrorHook] = None,
        on_retry: Optional[RetryHook] = None,
    ) -> List[PromptResult]:
        """Run `PromptRequest` objects with the chosen concurrency."""
        if not prompts:
            return []

        loop = asyncio.get_running_loop()
        self._loop = loop

        max_concurrency = concurrency or self.max_concurrency
        if max_concurrency < 1:
            raise ValueError("concurrency must be >= 1")

        normalized: List[PromptRequest] = []
        for i, prompt in enumerate(prompts):
            if not isinstance(prompt, PromptRequest):
                raise TypeError(f"Prompt at index {i} must be a PromptRequest, got {type(prompt)!r}")
            normalized.append(prompt)
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
        """Coordinate one prompt execution, invoking hooks and collecting timing."""
        start_hook, complete_hook, error_hook, retry_hook = hooks
        async with semaphore:
            started_at = time.perf_counter()
            result = PromptResult(index=index, request=request, started_at=started_at)
            # Invoke on_start before the prompt leaves the executor.
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
                # Invoke on_error once the final attempt fails.
                self._schedule_hook(error_hook, index, request, result.error, result)
            else:
                assert result.chat_result is not None
                # Invoke on_complete after a successful response.
                self._schedule_hook(complete_hook, index, request, result.chat_result, result)
            return result

    def _execute_prompt_sync(
        self,
        index: int,
        request: PromptRequest,
        result: PromptResult,
        retry_hook: Optional[RetryHook],
    ) -> PromptResult:
        """
        Run the synchronous AOAI client call on a worker thread.

        Any retry callable in `extra_kwargs["on_retry"]` is wrapped so executor hooks
        still fire while honoring user-provided behavior.
        """
        retry_calls = 0

        def _handle_retry(attempt: int, exc: Exception) -> None:
            nonlocal retry_calls
            retry_calls += 1
            # Invoke on_retry right before we try the prompt again.
            self._schedule_hook(retry_hook, index, request, attempt, exc)
            if user_retry is not None:
                try:
                    user_retry(attempt, exc)
                except Exception:
                    logger.exception("User-provided on_retry hook failed")

        # Shallow copy extra_kwargs so the caller can reuse the same PromptRequest safely.
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
        """Run hook callables safely on the event loop, tolerating sync or async returns."""
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
