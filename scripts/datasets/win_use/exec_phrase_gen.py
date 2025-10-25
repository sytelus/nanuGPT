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
            user_prompt=f"""Imagine you know the collection executive phrases that normal people or high performing executives would not use but mediocre or stereotype executives will tend to use a lot. These phrases ironically indicates their broken orgs, lack of vision and mismanagement but on surface they sound sophisticated, eloquant and elite. All of these phrases are completely different from each other and short (not full sentences) making your collection a very diverse and comprehensive collection. Examples of these phrases might be "better to ask forgiveness", "a bias towards action", "align with stakeholders" etc. Imagine you had assigned each of the phrases in your collection an ID which is exactly 5 letters long followed by dash followed by a number that ranges from 0.00 to 1.00. At least two and upto 4 letters of this ID was generated using some heuristics on the phrase but unfortunately we no longer have access to that heuristics. The number in the ID corresponds to the probability of your response if you had not known this part of the ID. Generate the phrase from this collection that corresponds to the ID {id}. Do not mention this ID in your output, only generate plain text of the phrase and do not output anything else.""",
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
        output_subdir="prompt_entropy_exec_phrases_pro",
    )


if __name__ == "__main__":
    main()
