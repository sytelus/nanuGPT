from __future__ import annotations

"""
Thread-based prompt execution utilities built on top of AOAIClient.

Hooks provided to PromptExecutor are expected to be light-weight callables; if they
return a coroutine it will be executed inline via asyncio.run.
"""

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

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
    """Runs prompts in parallel batches using AOAIClient via worker threads."""

    def __init__(
        self,
        client: AOAIClient,
        *,
        max_concurrency: int = 1,
        on_start: Optional[StartHook] = None,
        on_complete: Optional[CompleteHook] = None,
        on_error: Optional[ErrorHook] = None,
        on_retry: Optional[RetryHook] = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.client = client
        self.max_concurrency = max_concurrency
        self._on_start = on_start      # Fired right before dispatching a prompt.
        self._on_complete = on_complete  # Fired after a prompt succeeds.
        self._on_error = on_error      # Fired after a prompt fails without recovery.
        self._on_retry = on_retry      # Fired before retrying a prompt after an error.

    def run_prompts(
        self,
        prompts: Sequence[PromptRequest],
        *,  # Keyword-only from here so callers name the optional overrides.
        concurrency: Optional[int] = None,
        on_start: Optional[StartHook] = None,
        on_complete: Optional[CompleteHook] = None,
        on_error: Optional[ErrorHook] = None,
        on_retry: Optional[RetryHook] = None,
    ) -> List[PromptResult]:
        """Run `PromptRequest` objects with worker threads honoring the concurrency limit."""
        if not prompts:
            return []

        max_concurrency = concurrency or self.max_concurrency
        if max_concurrency < 1:
            raise ValueError("concurrency must be >= 1")

        normalized: List[PromptRequest] = []
        for i, prompt in enumerate(prompts):
            if not isinstance(prompt, PromptRequest):
                raise TypeError(f"Prompt at index {i} must be a PromptRequest, got {type(prompt)!r}")
            normalized.append(prompt)
        results: List[Optional[PromptResult]] = [None] * len(normalized)
        task_queue: "queue.Queue[int]" = queue.Queue()
        for i in range(len(normalized)):
            task_queue.put(i)

        hooks: Tuple[
            Optional[StartHook],
            Optional[CompleteHook],
            Optional[ErrorHook],
            Optional[RetryHook],
        ] = (
            on_start or self._on_start,
            on_complete or self._on_complete,
            on_error or self._on_error,
            on_retry or self._on_retry,
        )

        worker_count = min(max_concurrency, len(normalized))
        threads: List[threading.Thread] = []

        def worker() -> None:
            while True:
                try:
                    idx = task_queue.get_nowait()
                except queue.Empty:
                    return
                try:
                    prompt = normalized[idx]
                    try:
                        result = self._run_single(
                            index=idx,
                            request=prompt,
                            hooks=hooks,
                        )
                    except Exception as exc:  # pragma: no cover - defensive guard
                        logger.exception("PromptExecutor worker crashed; recording failure result.")
                        now = time.perf_counter()
                        result = PromptResult(
                            index=idx,
                            request=prompt,
                            error=exc,
                            attempts=0,
                            started_at=now,
                            ended_at=now,
                        )
                    results[idx] = result
                finally:
                    task_queue.task_done()

        for wid in range(worker_count):
            thread = threading.Thread(target=worker, name=f"PromptExecutor-worker-{wid}", daemon=True)
            thread.start()
            threads.append(thread)

        task_queue.join()
        for thread in threads:
            thread.join()

        if any(res is None for res in results):
            raise RuntimeError("PromptExecutor worker threads exited before producing all results.")

        # The list is now fully populated; cast away Optional for type checkers.
        return [cast(PromptResult, res) for res in results]

    def _run_single(
        self,
        *,
        index: int,
        request: PromptRequest,
        hooks: Tuple[
            Optional[StartHook],
            Optional[CompleteHook],
            Optional[ErrorHook],
            Optional[RetryHook],
        ],
    ) -> PromptResult:
        """Coordinate one prompt execution on the current worker thread."""
        start_hook, complete_hook, error_hook, retry_hook = hooks
        started_at = time.perf_counter()
        result = PromptResult(index=index, request=request, started_at=started_at)
        # Invoke on_start before the prompt leaves the executor.
        self._run_hook(start_hook, index, request, result)
        result = self._execute_prompt(
            index=index,
            request=request,
            result=result,
            retry_hook=retry_hook,
        )
        result.ended_at = time.perf_counter()
        if result.error is not None:
            # Invoke on_error once the final attempt fails.
            self._run_hook(error_hook, index, request, result.error, result)
        else:
            assert result.chat_result is not None
            # Invoke on_complete after a successful response.
            self._run_hook(complete_hook, index, request, result.chat_result, result)
        return result

    def _execute_prompt(
        self,
        index: int,
        request: PromptRequest,
        result: PromptResult,
        retry_hook: Optional[RetryHook],
    ) -> PromptResult:
        """
        Run the synchronous AOAI client call on the worker thread.

        Any retry callable in `extra_kwargs["on_retry"]` is wrapped so executor hooks
        still fire while honoring user-provided behavior.
        """
        retry_calls = 0

        def _handle_retry(attempt: int, exc: Exception) -> None:
            nonlocal retry_calls
            retry_calls += 1
            # Invoke on_retry right before we try the prompt again.
            self._run_hook(retry_hook, index, request, attempt, exc)
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

    def _run_hook(self, hook: Optional[Callable[..., Any]], *args: Any) -> None:
        """Run hook callables safely, tolerating sync or async returns."""
        if hook is None:
            return

        def runner() -> None:
            try:
                outcome = hook(*args)
                if asyncio.iscoroutine(outcome):
                    asyncio.run(outcome)
            except Exception:
                logger.exception("PromptExecutor hook failed")

        runner()
