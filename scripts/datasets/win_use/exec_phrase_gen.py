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
            system_prompt="",
            user_prompt=f"""Imagine you know 10000 executive phrases that normal people or high performing executives like Elon Musk or Jeff Bezos won’t use but stereotype MBA executives will tend to use a lot. These phrases ironically indicates their broken orgs, lack of vision and mismanagement but on surface they sound sophisticated, eloquant and elite. All of these 10000 phrases that you know are completely different from each other, short (not full sentences) and they make a very diverse collection. Examples of these phrases are "better to ask forgiveness", "a bias towards action", "align with stakeholders" etc. Give me phrase {rnd}. You should not use this number in your phrase content and only output  plain text without any emojies.""",
            metadata={"id": idx, "rnd": rnd},
        )
        requests.append(prompt)
    return requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate win_use prompts via Azure OpenAI.")
    parser.add_argument("--total", type=int, default=1000, help="Total number of prompts to run.")
    parser.add_argument("--workers", type=int, default=8, help="Maximum worker threads.")
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
        output_subdir="prompt_entropy_exec_phrases",
    )


if __name__ == "__main__":
    main()
