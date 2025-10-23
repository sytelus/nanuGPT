from __future__ import annotations

import argparse
import random
from typing import List

from rich.console import Console

from nanugpt.api_client.prompt_exec import PromptRequest
from nanugpt.api_client.runner import run as run_prompt_executor


def random_abbreviation(rng: random.Random) -> str:
    letters = 'abcdefghijklmnopqrstuvwxyz'
    return ''.join(rng.choices(letters, k=3))

def build_requests(total: int, rng: random.Random) -> List[PromptRequest]:
    requests: List[PromptRequest] = []
    for idx in range(1, total + 1):
        abbr = random_abbreviation(rng)
        prompt = PromptRequest(
            system_prompt="",
            user_prompt=f"""Imagine you know a lots of useful prompts that non-technical users on Windows will provide to a computer use agent and that the agent can solve by generating and executing code in either PowerShell, Python, C#, C++ or Rust assuming the dependencies you need are already installed. All of these prompts are very different from each other. None of them contains any specific data or any technical terms that a typical Windows user might not know. These prompts are simple, not too long and are fairly general to be directly usable by all of the users. Each of these prompts are uniquely represented by exactly 3 letters that is derived from prompt's content in some manner. Give me the prompt corresponding to {abbr}. Follow this up with a complete and working code in appropriate language inside ``` block. Do not mention {abbr} in your output, only generate plain text and do not output anything else.""",
            metadata={"id": idx, "abbr": abbr},
        )
        requests.append(prompt)
    return requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate win_use prompts via Azure OpenAI.")
    parser.add_argument("--total", type=int, default=1000, help="Total number of prompts to run.")
    parser.add_argument("--workers", type=int, default=16, help="Maximum worker threads.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling prompts.")
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
        output_subdir="prompt_entropy_win_usecases2",
    )


if __name__ == "__main__":
    main()
