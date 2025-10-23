from __future__ import annotations

import argparse
import random
from typing import List

from rich.console import Console

from nanugpt.api_client.prompt_exec import PromptRequest
from nanugpt.api_client.runner import run as run_prompt_executor


def build_requests(total: int, rng: random.Random) -> List[PromptRequest]:
    numbers = list(range(1, total + 1))
    rng.shuffle(numbers)
    requests: List[PromptRequest] = []
    for idx, rnd in enumerate(numbers, start=1):
        prompt = PromptRequest(
            system_prompt="You are a witty assistant.",
            user_prompt=f"Tell me a joke {rnd}",
            metadata={"id": idx, "rnd": rnd},
        )
        requests.append(prompt)
    return requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate win_use prompts via Azure OpenAI.")
    parser.add_argument("--total", type=int, default=1000, help="Total number of prompts to run.")
    parser.add_argument("--workers", type=int, default=8, help="Maximum worker threads.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for shuffling prompts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    console = Console()

    requests = build_requests(args.total, rng)
    run_prompt_executor(
        requests,
        workers=args.workers,
        console=console,
        output_subdir="prompt_entropy",
    )


if __name__ == "__main__":
    main()
