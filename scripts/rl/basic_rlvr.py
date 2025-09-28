# Code originally from https://github.com/aburkov/theLMbook/blob/main/GRPO.py

import copy
import random
import re
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

CompletionMessage = Dict[str, str]
CompletionList = List[CompletionMessage]
DatasetExample = Dict[str, str]
BatchSample = Union[DatasetExample, Tuple[str, str]]
RewardFn = Callable[[Sequence[str], Sequence[CompletionList], Sequence[Optional[str]]], Sequence[float]]
PolicyModel = Union[PreTrainedModel, nn.DataParallel]


class RolloutData(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    old_log_probs: torch.Tensor
    ref_log_probs: torch.Tensor
    formatted_completions: List[CompletionList]
    repeated_prompts: List[str]
    repeated_answers: List[Optional[str]]
    logits_to_keep: int
    batch_size: int
    num_generations: int

def set_random_seed(seed: int = 42) -> None:
    # Set the seed for Python's built-in random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(42)

MODEL_MODE = "completion"  # "chat" or "completion"

SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <reasoning> </reasoning> and <answer> </answer> tags, respectively, i.e., <reasoning> reasoning process here </reasoning> <answer> answer here </answer>.""" \
if MODEL_MODE == "chat" else \
"""
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_answer_from_model_output(text: str) -> Optional[str]:
    # Split on <answer> and take everything after the last occurrence
    parts = text.split("<answer>")
    if len(parts) < 2:  # No <answer> tag found
        return None

    last_part = parts[-1]

    # Extract content up to </answer>
    if "</answer>" not in last_part:
        return None

    answer = last_part.split("</answer>")[0].strip()
    return None if answer == "..." else answer

def extract_answer_from_dataset(text: str) -> Optional[str]:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def prepare_dataset(
    tokenizer: PreTrainedTokenizerBase,
    split: str = "train",
) -> List[DatasetExample]:
    data = load_dataset('openai/gsm8k', 'main')[split]
    formatted_data: List[DatasetExample] = []

    for raw in data:
        example: DatasetExample = cast(DatasetExample, raw)
        # Convert list of messages to a single string prompt.

        if MODEL_MODE == "chat":
            prompt_str = build_prompt([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
                {"role":"assistant", "content": "<reasoning>\n"}
            ], tokenizer)
        else:
            prompt_str = build_prompt([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]}
            ], tokenizer)

        formatted_example = {
            "prompt": prompt_str,  # Now a string rather than a list.
            "answer": extract_answer_from_dataset(example["answer"])
        }
        formatted_data.append(formatted_example)

    return formatted_data

def build_prompt(
    messages: Sequence[CompletionMessage],
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    if MODEL_MODE == "chat":
        # messages = [{"role":"system","content":...}, {"role":"user","content":...}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        return "\n".join(msg["content"].strip() for msg in messages)

    return prompt

def _extract_last_number(text: str) -> Optional[float]:
    """Extract the trailing numeric token from ``text`` when present.

    Example:
        >>> _extract_last_number('cost = 7.5 dollars')
        7.5
    """

    # Remove $ and % signs
    text = text.replace('$', '').replace('%', '')

    # Look for numbers that are:
    # - preceded by space or = or start of string (via \b or ^)
    # - followed by end of string or space
    pattern = r'(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$'
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def _extract_single_number(text: str) -> Optional[float]:
    """Return the numeric value if ``text`` contains exactly one number.

    Example:
        >>> _extract_single_number('answer: -3.2')
        -3.2
    """

    numbers = re.findall(r'-?\d*\.?\d+', text)
    return float(numbers[0]) if len(numbers) == 1 else None


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    eval_examples: Sequence[DatasetExample],
    device: torch.device,
    batch_size: int = 32,
) -> float:
    """Run greedy generation on ``eval_examples`` and report accuracy."""

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    if pad_token_id is None or eos_token_id is None:
        raise ValueError("Tokenizer must define both pad_token_id and eos_token_id for evaluation")

    was_training = model.training
    model.eval()
    correct = 0
    total = len(eval_examples)

    print("Evaluating model on ", total, "examples...")

    if total == 0:
        print("No evaluation examples provided; skipping evaluation.")
        return 0.0

    effective_batch_size = min(batch_size, total)

    try:
        for start in range(0, total, effective_batch_size):
            batch_examples = eval_examples[start:start + effective_batch_size]
            prompts = [example["prompt"] for example in batch_examples]
            expected_answers = [example.get("answer") for example in batch_examples]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                truncation=False,
            )

            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=512,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    forced_eos_token_id=eos_token_id,
                    early_stopping=False,
                ) # type: ignore

            responses = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for example, expected, response in zip(batch_examples, expected_answers, responses):
                full_prompt = example["prompt"]

                try:
                    predicted = extract_answer_from_model_output(response)

                    is_correct = False
                    if predicted == expected:
                        is_correct = True
                    else:
                        pred_num = _extract_single_number(str(predicted))
                        exp_num = _extract_single_number(str(expected))
                        if pred_num is not None and exp_num is not None and pred_num == exp_num:
                            is_correct = True
                        else:
                            pred_num = _extract_last_number(str(predicted))
                            exp_num = _extract_last_number(str(expected))
                            is_correct = (
                                pred_num is not None and
                                exp_num is not None and
                                pred_num == exp_num
                            )

                    if is_correct:
                        correct += 1

                except Exception as e:
                    print("\nFailed to parse model output for prompt:")
                    print(full_prompt)
                    print("Error:", e)
                    print("-"*50)
    finally:
        if was_training:
            model.train()

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    print("="*50)

    return accuracy

def correctness_reward(
    prompts: Sequence[str],
    completions: Sequence[CompletionList],
    answer: Sequence[Optional[str]],
    **kwargs: object,
) -> List[float]:
    """
    Assigns a reward based on the correctness of the model's answer.

    Args:
        prompts (list[str]): List of prompt texts.
        completions (list[list[dict]]): List of completion dictionaries.
        answer (list[str]): List of expected answers.
        **kwargs: Additional keyword arguments.

    Returns:
        list[float]: Reward scores based on answer correctness.

    Example:
        >>> correctness_reward(
        ...     prompts=["What is 1+1?"],
        ...     completions=[[{"content": "<answer>2</answer>"}]],
        ...     answer=["2"],
        ... )
        [2.0]

    Explanation:
        1. Extracts the text content from each completion.
        2. Processes each response to extract the answer portion.
        3. Compares extracted answers with expected answers using two methods:
           - Exact string matching (2.0 points)
           - Numeric equivalence check (1.5 points)
        4. Returns a list of reward scores.
    """

    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_answer_from_model_output(r) for r in responses]
    rewards = []

    for r, a in zip(extracted, answer):
        if r == a:  # Exact match case
            rewards.append(2.0)
        else:
            # Try numeric equivalence
            r_num = _extract_single_number(str(r))
            a_num = _extract_single_number(str(a))
            if r_num is not None and a_num is not None and r_num == a_num:
                rewards.append(1.5)
            else:
                rewards.append(0.0)

    # TODO: log completion lengths
    completion_lengths = [len(response.split()) for response in responses]
    return rewards


def format_reward(
    completions: Sequence[CompletionList],
    **kwargs: object,
) -> List[float]:
    """
    Assigns a reward for adhering to the desired XML format.

    Args:
        completions (list[list[dict]]): List of completion dictionaries.
        **kwargs: Additional keyword arguments.

    Returns:
        list[float]: Reward scores based on format compliance.

    Example:
        >>> format_reward([[{"content": "<reasoning>...</reasoning><answer>1</answer>"}]])
        [0.8]

    Explanation:
        1. Extracts the text content from each completion.
        2. Assigns points based on the presence of required XML tags:
           - 0.2 points for opening <reasoning> tag
           - 0.2 points for closing </reasoning> tag
           - 0.2 points for opening <answer> tag
           - 0.2 points for closing </answer> tag
        3. Returns a list of format compliance scores.
    """
    # Extract the content from each completion's first element
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    format_scores = []

    for response in responses:
        score = 0.0
        if "<reasoning>" in response: score += 0.2
        if "</reasoning>" in response: score += 0.2
        if "<answer>" in response: score += 0.2
        if "</answer>" in response: score += 0.2
        rewards.append(score)
        format_scores.append(score)

    return rewards


def combined_reward(
    prompts: Sequence[str],
    completions: Sequence[CompletionList],
    answer: Sequence[Optional[str]],
) -> List[float]:
    """
    Combines correctness and format rewards to provide a comprehensive evaluation.

    Args:
        prompts (list[str]): List of prompt texts.
        completions (list[list[dict]]): List of completion dictionaries.
        answer (list[str]): List of expected answers.

    Returns:
        list[float]: Combined rewards for each prompt-completion pair.

    Example:
        >>> combined_reward(
        ...     prompts=["What is 2+2?"],
        ...     completions=[[{"content": "<answer>4</answer>"}]],
        ...     answer=["4"],
        ... )
        [2.8]

    Explanation:
        1. Calculates individual reward components:
           - Correctness rewards (range: 0.0 to 2.0)
           - Format rewards (range: 0.0 to 0.8)
        2. Combines the rewards by adding them together.
        3. Returns the combined scores with total range of 0.0 to 2.8.
    """
    # Get individual rewards
    correctness_scores = correctness_reward(prompts=prompts, completions=completions, answer=answer)
    format_scores = format_reward(completions=completions)

    # Combine rewards - correctness is weighted more heavily
    combined_rewards = []
    for c_score, f_score in zip(correctness_scores, format_scores):
        # Correctness score range: 0.0 to 2.0
        # Format score range: 0.0 to 0.8
        # Total range: 0.0 to 2.8
        combined_rewards.append(c_score + f_score)

    return combined_rewards

def selective_log_softmax(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute the log probabilities for the tokens specified in input_ids using a selective log-softmax.

    Args:
        logits (torch.Tensor): A tensor of shape (batch_size, seq_len, vocab_size) containing raw logits from the model.
        input_ids (torch.Tensor): A tensor of shape (batch_size, seq_len) containing the token indices for which we want the log probabilities.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, seq_len) where each element is the log probability
                      corresponding to the token in input_ids at that position.

    Explanation:
        1. F.log_softmax is applied along the vocabulary dimension (dim=-1) to convert logits into log probabilities.
        2. The tensor input_ids is reshaped (via unsqueeze) to have an extra dimension so that we can use it as indices
           in the log_probs tensor.
        3. torch.gather collects the log probability at the index specified in input_ids for each position.
        4. Finally, squeeze(-1) removes the extra dimension, returning a tensor with the same shape as input_ids.
    """
    # Convert raw logits into log probabilities along the vocabulary axis.
    log_probs = F.log_softmax(logits, dim=-1)  # (batch, seq_len, vocab)

    # Gather the log probability for each token id in the sequence.
    # input_ids.unsqueeze(-1) -> (batch, seq_len, 1)
    selected_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1))

    # Remove the trailing singleton dimension so the output matches input_ids.
    return selected_log_probs.squeeze(-1)

def compute_log_probs(
    model: PolicyModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_to_keep: int,
) -> torch.Tensor:
    """
    Compute per-token log probabilities for a subset of tokens (typically the completion tokens).

    Args:
        model: The language model to use.
        input_ids (torch.Tensor): Tensor of shape (batch_size, total_seq_len) containing token ids
                                  for both prompt and completion.
        attention_mask (torch.Tensor): Tensor of shape (batch_size, total_seq_len) indicating which tokens are real (1) or padding (0).
        logits_to_keep (int): Number of tokens (from the completion part) for which we need log probabilities.

    Returns:
        torch.Tensor: Log probabilities for the last `logits_to_keep` tokens of each sequence. Shape: (batch_size, logits_to_keep).

    Explanation:
        1. We obtain the full logits for the prompt+completion sequence from the model.
        2. We slice off the last logit along the sequence dimension because it does not correspond to any input token.
        3. We then restrict both the input_ids and logits to the last logits_to_keep tokens, which should
           correspond to the generated completion portion.
        4. Finally, we use the selective_log_softmax to compute log probabilities only for those tokens.
    """
    # Run the model forward pass and obtain logits.
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits  # Shape: (batch_size, total_seq_len, vocab_size)

    # Remove the last logit as it does not have a corresponding target token.
    logits = logits[:, :-1, :]  # New shape: (batch_size, total_seq_len - 1, vocab_size)

    # Slice the input_ids to keep only the last logits_to_keep tokens.
    # This corresponds to the generated completion tokens.
    input_ids = input_ids[:, -logits_to_keep:]  # Shape: (batch_size, logits_to_keep)

    # Also slice the logits to keep only those corresponding to the completion tokens.
    logits = logits[:, -logits_to_keep:, :]  # Shape: (batch_size, logits_to_keep, vocab_size)

    # Compute and return the log probabilities for the selected tokens.
    return selective_log_softmax(logits, input_ids)

def create_completion_mask(completion_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """
    Create a binary mask for the generated completion tokens so that tokens after the first EOS are ignored.

    Args:
        completion_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) with generated token ids.
        eos_token_id (int): The token id representing the end-of-sequence.

    Returns:
        torch.Tensor: A mask tensor of shape (batch_size, seq_len) with 1s for tokens up to and including the first EOS
                      and 0s for tokens following the first EOS.

    Explanation:
        1. First, a boolean mask (is_eos) is created indicating where in the sequence the EOS token appears.
        2. An index tensor (eos_idx) is initialized, assuming that no EOS is found (defaulting to the sequence length).
        3. For sequences where EOS exists, eos_idx is updated to the position (index) of the first EOS.
        4. A sequence index tensor is created that contains indices for each position in the sequence.
        5. The final mask is computed by comparing the sequence indices to eos_idx (after adding a dimension).
    """
    # Determine which positions in each sequence equal the EOS token.
    is_eos = completion_ids == eos_token_id  # Boolean tensor of shape (batch_size, seq_len)

    # Initialize a tensor to store the index of the first EOS for each sequence.
    # If no EOS is found, default to the full sequence length (is_eos.size(1)).
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)

    # Identify sequences that contain at least one EOS.
    mask_exists = is_eos.any(dim=1)
    # For sequences with an EOS, update eos_idx to the index of the first occurrence.
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]

    # Create a tensor of indices [0, 1, 2, ..., seq_len-1] and replicate it for each sequence in the batch.
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)

    # Build the mask: positions with an index less than or equal to the first EOS index are marked as 1.
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    return completion_mask

def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    num_generations: int = 4,
    max_completion_length: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For given prompts, generate the specified number of completions for specified maximum length using the model.

    Args:
        model: The language model used for generation.
        tokenizer: The tokenizer to process the prompts and decode the outputs.
        prompts (list of str): List of input prompt strings.
        num_generations (int): Number of completions to generate per prompt.
        max_completion_length (int): Maximum number of new tokens to generate for the completion.

    Returns:
        tuple: Contains the following tensors. Shapes assume ``B`` prompts and ``G`` generations:
            - prompt_ids: (B * G, prompt_seq_len)
            - prompt_mask: (B * G, prompt_seq_len)
            - completion_ids: (B * G, completion_seq_len)
            - completion_mask: (B * G, completion_seq_len)

    Example:
        >>> p_ids, p_mask, c_ids, c_mask = generate_completions(model, tokenizer, ["Q?"], 2, 5)
        >>> p_ids.shape, c_ids.shape
        (torch.Size([2, 3]), torch.Size([2, 5]))

    Explanation:
        1. The prompts are tokenized and padded (with padding added to the left).
        2. Each prompt is repeated num_generations times so that multiple completions are generated per prompt.
        3. The model.generate() function is called to generate new tokens.
        4. The generated output contains the prompt followed by the completion; we remove the prompt part to get the completions.
        5. A mask is created (via create_completion_mask) so that only tokens up to the first EOS are considered.
    """
    device = next(model.parameters()).device

    # Tokenize the list of prompts with padding. The padding_side="left" ensures alignment on the right.
    tokenizer.padding_side  = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    prompt_ids = inputs["input_ids"].to(device)      # Shape: (batch_size, prompt_seq_len)
    prompt_mask = inputs["attention_mask"].to(device)  # Shape: (batch_size, prompt_seq_len)
    prompt_seq_len = prompt_ids.size(1)  # Save the prompt length to later separate prompt from completion.

    # Repeat each prompt num_generations times.
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)   # New shape: (batch_size*num_generations, prompt_seq_len)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0) # New shape: (batch_size*num_generations, prompt_seq_len)

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    if pad_token_id is None or eos_token_id is None:
        raise ValueError("Tokenizer must define both pad_token_id and eos_token_id for generation")

    # Generate new tokens for each prompt. The output includes the original prompt and the generated tokens.
    outputs = model.generate(
        prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,           # nucleus sampling for structure
        repetition_penalty=1.05,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        early_stopping=False
    ) # type: ignore

    print(f"Output batch size: {outputs.size(0)}, Device after model: {outputs.device}")

    assert torch.equal(outputs[:, :prompt_seq_len].to(prompt_ids.device), prompt_ids)

    completion_ids = outputs[:, prompt_seq_len:]

    #print(tokenizer.decode(completion_ids[0], skip_special_tokens=True)[:200])

    # Create a binary mask that ignores tokens beyond the first EOS token.
    completion_mask = create_completion_mask(completion_ids, eos_token_id)

    return prompt_ids, prompt_mask, completion_ids, completion_mask

def generate_rollout_data(
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_samples: Sequence[BatchSample],
    num_generations: int,
    max_completion_length: int,
) -> RolloutData:
    """
    Generate rollouts and compute static log probabilities for both the old policy (current model)
    and the reference model. Gradients are disabled so that these remain fixed.

    Args:
        model: The current model (policy) used to generate rollouts.
        ref_model: The static reference model.
        tokenizer: The tokenizer.
        batch_samples: List of training samples.
        num_generations: Number of completions to generate per prompt.
        max_completion_length: Maximum completion length.

    Returns:
        A dictionary with rollout data including both old and reference log probabilities.

    Example:
        >>> rollout = generate_rollout_data(model, model, tokenizer, samples, 2, 16)
        >>> rollout["old_log_probs"].shape
        torch.Size([len(samples) * 2, 16])
    """
    tokenizer.padding_side  = "left"

    # Extract prompts and answers.
    prompts = [sample["prompt"] if isinstance(sample, dict) else sample[0] for sample in batch_samples]
    answers = [sample["answer"] if isinstance(sample, dict) else sample[1] for sample in batch_samples]

    # Generate completions and associated masks.
    # We generate once, and then use the same completions to compute both sets of log probabilities.
    with torch.no_grad():
        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
            model, tokenizer, prompts, num_generations, max_completion_length
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        # Shape: (batch_size * num_generations, prompt_seq_len + completion_seq_len)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Compute old_log_probs from the current model, with gradients disabled.
        old_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)

        # Compute ref_log_probs from the reference model, which remains static.
        ref_log_probs = compute_log_probs(ref_model, input_ids, attention_mask, logits_to_keep)

    formatted_completions: List[CompletionList] = [
        [{'content': tokenizer.decode(ids, skip_special_tokens=True)}]
        for ids in completion_ids
    ]
    repeated_prompts: List[str] = [p for p in prompts for _ in range(num_generations)]
    repeated_answers: List[Optional[str]] = [a for a in answers for _ in range(num_generations)]

    return RolloutData(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
        old_log_probs=old_log_probs,   # Static log probs from the current model (old policy)
        ref_log_probs=ref_log_probs,     # Static log probs from the reference model
        formatted_completions=formatted_completions,
        repeated_prompts=repeated_prompts,
        repeated_answers=repeated_answers,
        logits_to_keep=logits_to_keep,
        batch_size=len(prompts),
        num_generations=num_generations
    )

def grpo_loss(
    model: PolicyModel,
    rollout_data: RolloutData,
    reward_function: RewardFn,
    beta: float,
    epsilon: float,
) -> Tuple[torch.Tensor, float]:
    """
    Update the policy model by maximizing the GRPO objective.

    Args:
        model: The current policy model.
        ref_model: The reference model.
        rollout_data: Dictionary containing rollout data.
        tokenizer: The tokenizer.
        reward_function: Function to compute rewards.
        optimizer: The optimizer.
        beta (float): KL penalty coefficient.
        epsilon (float): Clipping parameter.

    Returns:
        float: The loss value.

    Example:
        >>> isinstance(
        ...     grpo_loss(model, model, rollout, tokenizer, combined_reward, optimizer, 0.1, 0.2),
        ...     float,
        ... )
        True
    """
    # Extract data from rollout
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]
    logits_to_keep = rollout_data["logits_to_keep"]
    batch_size = int(rollout_data["batch_size"])
    num_generations = int(rollout_data["num_generations"])

    # Compute current log probabilities
    current_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
    # Shape note: log probabilities are (batch_size * num_generations, completion_seq_len).

    # Compute policy ratio
    ratio = torch.exp(current_log_probs - old_log_probs)  # Same shape as current_log_probs.

    # Get rewards data
    formatted_completions = rollout_data["formatted_completions"]
    repeated_prompts = rollout_data["repeated_prompts"]
    repeated_answers = rollout_data["repeated_answers"]

    # Compute rewards
    rewards = torch.tensor(
        reward_function(
            prompts=repeated_prompts,
            completions=formatted_completions,
            answer=repeated_answers,
        ),
        dtype=torch.float32,
        device=current_log_probs.device,
    )
    rewards_matrix = rewards.view(batch_size, num_generations)
    avg_reward = rewards_matrix.mean().item()
    print(f"Average Reward: {avg_reward:.4f}")

    mean_rewards = rewards_matrix.mean(dim=1).repeat_interleave(num_generations)
    std_rewards = rewards_matrix.std(dim=1).repeat_interleave(num_generations)
    advantages = ((rewards_matrix.view(-1) - mean_rewards) / (std_rewards + 1e-4)).unsqueeze(1)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    surrogate_loss = torch.min(surr1, surr2)
    kl = torch.exp(ref_log_probs - current_log_probs) - (ref_log_probs - current_log_probs) - 1
    per_token_loss = surrogate_loss - beta * kl
    loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    return loss, avg_reward


def train_with_grpo(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_data: Sequence[DatasetExample],
    num_iterations: int,
    steps_per_iteration: int,
    batch_size: int,
    num_generations: int,
    max_completion_length: int,
    beta: float,
    learning_rate: float,
    mu: int,
    epsilon: float,
    reward_function: RewardFn,
    device_ids: List[int],
) -> PreTrainedModel:
    """
    Iterative Group Relative Policy Optimization algorithm.

    Args:
        model: The initial policy model to be fine-tuned.
        tokenizer: The tokenizer used for encoding prompts and decoding completions.
        train_data (list): List of training samples with "prompt" and "answer" fields.
        num_iterations (int): Number of outer iterations (reward model updates).
        steps_per_iteration (int): Number of policy update steps per iteration.
        batch_size (int): Number of prompt samples per batch.
        num_generations (int): Number of completions to generate per prompt.
        max_completion_length (int): Maximum token length for completions.
        beta (float): KL-divergence penalty coefficient.
        learning_rate (float): Learning rate for optimizer.
        mu (int): Number of GRPO updates per batch of generations.
        epsilon (float): Clipping parameter for surrogate objective.
        reward_function: Function that evaluates completions and returns rewards.

    Returns:
        The fine-tuned policy model.
    """
    if not device_ids:
        raise ValueError("GRPO training requires at least one CUDA device.")

    primary_device = torch.device(f"cuda:{device_ids[0]}")
    model = model.to(primary_device)

    policy_model: PolicyModel
    # Use DataParallel only when >1 GPU is present. This keeps single-GPU runs simple
    # while letting the heavy GRPO forward/backward passes fan out across multiple
    # devices automatically (matching the original notebook’s behaviour).
    if len(device_ids) > 1:
        policy_model = nn.DataParallel(model, device_ids=device_ids)
    else:
        policy_model = model

    def unwrap(module_like: PolicyModel) -> PreTrainedModel:
        if isinstance(module_like, nn.DataParallel):
            return cast(PreTrainedModel, module_like.module)
        return cast(PreTrainedModel, module_like)

    # Outer loop for iterations with reward model updates
    for iteration in range(1, num_iterations + 1):
        print(f"\nStarting iteration {iteration}/{num_iterations}")

        # Create reference model for KL constraint
        # Clone the (possibly DataParallel-wrapped) policy onto the primary device so
        # KL comparisons always run locally and avoid redundant multi-GPU copies.
        reference_model = copy.deepcopy(unwrap(policy_model)).to(primary_device)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False

        optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
        policy_model.train()

        # Inner loop for policy updates
        for step in range(1, steps_per_iteration + 1):
            # Sample batch of prompts
            batch_samples = random.sample(train_data, batch_size)

            # Set old policy for this step
            with torch.no_grad():
                policy_module = unwrap(policy_model)

                # Save and switch modes just for generation
                was_training = policy_module.training
                prev_cache   = getattr(policy_module.config, "use_cache", False)
                policy_module.eval()
                policy_module.config.use_cache = True

                # Generate completions and compute log probs
                rollout_data = generate_rollout_data(
                    policy_module, reference_model, tokenizer,
                    batch_samples, num_generations, max_completion_length
                )

                # Restore training state for the update step
                policy_module.config.use_cache = prev_cache
                if was_training:
                    policy_module.train()

                # TODO: remove forward hoot from the reference model for grads

            # Multiple GRPO updates per batch of generations
            for grpo_iter in range(1, mu + 1):
                loss, avg_reward = grpo_loss(
                    policy_model, rollout_data,
                    reward_function, beta, epsilon
                )
                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=0.1)
                optimizer.step()
                print(f"Iteration {iteration}/{num_iterations}, Step {step}/{steps_per_iteration}, "
                      f"GRPO update {grpo_iter}/{mu}, Loss: {loss.item():.4f}, Avg Reward: {avg_reward:.4f}")

        # Optional: Update reward model here if using reward model training
        # This is not implemented in the original code but present in the pseudocode
        print(f"Completed iteration {iteration}. Reward model update would happen here.")

    return unwrap(policy_model)

def enable_grads(model: PreTrainedModel) -> PreTrainedModel:
    model.train()

    # Disable caching for gradient checkpointing
    model.config.use_cache = False
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Enable input gradients properly
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    return model

def main():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("GRPO training requires at least one CUDA device.")

    primary_device = torch.device("cuda:0")
    device = primary_device
    print(f"Using device: {device}")

    # 0.5B model is not smart to generate the <reasoning> and <answer> tags
    # so several iterations of SFT to teach it these tags are recommended before RL
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir = "math_solver_model"

    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, # use bf16 for performance
        #attn_implementation="flash_attention_2",
        device_map=None,
        trust_remote_code=True,
    )
    model = model.to(primary_device)
    print("Downloaded model")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    # fix tokenizer padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    print(f"Detected {num_gpus} GPUs")
    device_ids = list(range(num_gpus))

    # create a list of all dataset examples as prompts that will be given to models and answers we expect
    # [{'prompt': 'Respond in the following format:\n\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?', 'answer': '72'},...]
    all_data = prepare_dataset(tokenizer, "train")
    random.shuffle(all_data)
    # reserve some examples for test set
    num_eval_examples = 30
    train_data = all_data[num_eval_examples:]
    eval_data = all_data[:num_eval_examples]

    pre_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
    print(f"Pre-GRPO Accuracy: {pre_grpo_accuracy:.2f}%")

    model = enable_grads(model)

    print("\nStarting RL finetuning using GRPO...")
    training_config = {
        'num_iterations' : 1,
        'steps_per_iteration': 500,                    # Total number of RL training steps.
        'batch_size': 7,                     # Number of samples per training step.
        'num_generations': 12,                # Number of completions generated per prompt.
        'max_completion_length': 400,        # Maximum token length for each generated completion.
        'beta': 0.04,                         # KL divergence penalty coefficient.
        'learning_rate': 5e-6,                # Learning rate for RL fine-tuning.
        'mu': 1,
        'epsilon': 0.1,
        'reward_function': combined_reward
    }

    # Fine-tune the model using GRPO RL training.
    model = train_with_grpo(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        device_ids=device_ids,
        **training_config
    )

    # Step 2: FINAL EVALUATION & SAVING
    print("\nFinal model evaluation after GRPO RL finetuning:")
    # Evaluate the final model performance using the evaluation dataset.
    post_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
    print(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}%")
    print(f"Total Improvement: {post_grpo_accuracy - pre_grpo_accuracy:.2f}%")

    print("\nSaving GRPO finetuned model...")
    # Save the final finetuned model and tokenizer to disk.
    model.save_pretrained("grpo_finetuned_model")
    tokenizer.save_pretrained("grpo_finetuned_model")

if __name__ == "__main__":
    main()
