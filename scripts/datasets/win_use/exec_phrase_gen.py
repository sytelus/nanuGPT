from __future__ import annotations

import argparse
import random
from typing import List

from rich.console import Console

from nanugpt.api_client.prompt_exec import PromptRequest
from nanugpt.api_client.runner import run as run_prompt_executor

def random_id(rng: random.Random) -> str:
    letters = 'abcdefghijklmnopqrstuvwxyz'
    return ''.join(rng.choices(letters, k=5)) + '-0.' + str(rng.randint(0, 99)).zfill(2)

def build_requests(total: int, rng: random.Random) -> List[PromptRequest]:
    requests: List[PromptRequest] = []
    for i in range(1, total + 1):
        id = random_id(rng)
        prompt = PromptRequest(
            system_prompt="",
            user_prompt=f"""Imagine you know the collection executive phrases that normal people or high performing executives would not use but mediocre or stereotype executives will tend to use a lot. These phrases ironically indicates their broken orgs, lack of vision and mismanagement but on surface they sound sophisticated, eloquant and elite. All of these phrases are completely different from each other and short (not full sentences) making your collection a very diverse and comprehensive collection. Examples of these phrases might be "better to ask forgiveness", "a bias towards action", "align with stakeholders" etc.

            I have been asking you to give me 5 items at a time from your collection by giving you this prompt over and over in a loop so I can collect all the items from your list. However, you have no memory of what you returned to me in previous calls. As I want you to give me unique items in each call and avoid duplicate with previous calls, I ask you return me items from your list in decreasing order of probability that we may encounter in practice. I will give you the index of the call which increments by 1 after each call so you can know how far down the list we are. Currently, this is the call number {i}. Please give me 5 unique executive phrase from your collection that you have not given me in previous calls. Only generate plain text of the phrase and do not output anything else.""",
            metadata={"index": i, "id": id},
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
        output_subdir="prompt_entropy_exec_phrases_i5",
    )


if __name__ == "__main__":
    main()
