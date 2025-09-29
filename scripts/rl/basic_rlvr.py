"""
GRPO (Group Relative Policy Optimization) training script
========================================================

Goal
----
Fine-tune a causal LM to solve GSM8K math questions *and* emit results in a
tagged XML format:
    <reasoning> ... </reasoning><answer> ... </answer>

Approach (high level)
---------------------
1) Rollouts (no grad): For each prompt, sample `num_generations` completions.
   Build a mask that keeps tokens up to the first EOS per sample.
2) Cache static per-token log-probs from:
     (a) "old policy" (the current model snapshot used to sample) and
     (b) a frozen reference model for KL regularization.
3) Policy updates: compute advantages by normalizing rewards within each
   prompt’s group of generations; update the policy with a PPO-like clipped
   objective + reverse-KL penalty to the reference.

Design notes
------------
- **Multi-GPU**: We use `nn.DataParallel` for update steps when >1 GPU.
  Rollout generation + log-prob caching runs on the *primary* device to avoid
  cross-device overhead during `generate()`.
- **Generation config**: We temporarily enable `use_cache=True` only for
  generation to speed up decoding, then restore it for training
  (needed for gradient checkpointing).
- **Precision & perf**:
  * Model params run in bfloat16.
  * TF32 is enabled for faster float32 matmuls on Ampere+ GPUs.
  * Fused AdamW is used when available.
  * `torch.autocast` in the policy forward reduces cast overhead in updates.
  * Batch decode completions when computing rewards.
- **Determinism**: A fixed seed is set. Note that RL uses sampling, so full
  bitwise determinism is not guaranteed.

Inputs / outputs
----------------
- Dataset:  `openai/gsm8k`, split `"train"` (we sample a small eval slice).
- Model:    Hugging Face `AutoModelForCausalLM` (Qwen2.5-1.5B-Instruct by default).
- Saves:    Final weights + tokenizer at `grpo_finetuned_model/`.

Known limitations & recommendations
-----------------------------------
- For serious training, prefer DDP over `DataParallel`.
- The reward is rule-based (format + correctness). A trained reward model can
  improve quality but is out of scope here.
"""

import argparse
import copy
import random
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def build_system_prompt(model_mode: str) -> str:
    if model_mode == "chat":
        return (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
            "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
            "The reasoning process and answer are enclosed within <reasoning> </reasoning> and <answer> </answer> tags, "
            "respectively, i.e., <reasoning> reasoning process here </reasoning> <answer> answer here </answer>."
        )
    return (
        "First, think about the reasoning process in the mind and then provide the answer. "
        "The reasoning process and answer are enclosed within <reasoning> </reasoning> and <answer> </answer> tags, "
        "respectively, i.e., <reasoning> reasoning process here </reasoning> <answer> answer here </answer>."
    )


def extract_answer_from_model_output(text: str) -> Optional[str]:
    parts = text.split("<answer>")
    if len(parts) < 2:
        return None
    last_part = parts[-1]
    if "</answer>" not in last_part:
        return None
    answer = last_part.split("</answer>")[0].strip()
    return None if answer == "..." else answer

def extract_answer_from_dataset(text: str) -> Optional[str]:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def build_prompt(
    messages: Sequence[CompletionMessage],
    tokenizer: PreTrainedTokenizerBase,
    model_mode: str,
) -> str:
    if model_mode == "chat":
        # Use the model's chat template to serialize the conversation.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    # For "completion" mode, we concatenate message contents.
    return "\n".join(msg["content"].strip() for msg in messages)


def _extract_last_number(text: str) -> Optional[float]:
    """Extract the trailing numeric token if present.
        'cost = 7.5 dollars' -> 7.5
    """
    text = text.replace("$", "").replace("%", "")
    match = re.search(r"(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$", text)
    return float(match.group(1)) if match else None


def _extract_single_number(text: str) -> Optional[float]:
    """Return the numeric value if the text contains exactly one number.
       'answer: -3.2' -> -3.2
    """
    numbers = re.findall(r"-?\d*\.?\d+", text)
    return float(numbers[0]) if len(numbers) == 1 else None

def prepare_dataset(
    tokenizer: PreTrainedTokenizerBase,
    model_mode: str,
    system_prompt: str,
    split: str,
) -> List[DatasetExample]:
    data = load_dataset("openai/gsm8k", "main")[split]
    formatted_data: List[DatasetExample] = []

    for raw in data:
        example: DatasetExample = cast(DatasetExample, raw)
        if model_mode == "chat":
            # Prime the assistant to continue with tags; Qwen benefits from this.
            prompt_str = build_prompt(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example["question"]},
                    {"role": "assistant", "content": "<reasoning>\n"},
                ],
                tokenizer,
                model_mode,
            )
        else:
            prompt_str = build_prompt(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example["question"]},
                ],
                tokenizer,
                model_mode,
            )

        formatted_data.append(
            {
                "prompt": prompt_str,
                "answer": extract_answer_from_dataset(example["answer"]),
            }
        )
    return formatted_data

def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    eval_examples: Sequence[DatasetExample],
    device: torch.device,
    batch_size: int,
    greedy_eval: bool,
    max_new_tokens: int,
    sampling_temperature: Optional[float],
) -> float:
    """Greedy generation over `eval_examples`; returns accuracy percent."""
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not greedy_eval and sampling_temperature is None:
        raise ValueError("sampling_temperature must be provided when greedy_eval is False")
    num_return_sequences = 1  # Keep one completion per prompt for accuracy alignment.
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    if pad_token_id is None or eos_token_id is None:
        raise ValueError("Tokenizer must define both pad_token_id and eos_token_id for evaluation")

    was_training = model.training
    model.eval()
    correct = 0
    total = len(eval_examples)
    print("Evaluating model on", total, "examples...")
    if total == 0:
        print("No evaluation examples provided; skipping evaluation.")
        return 0.0

    effective_batch_size = min(batch_size, total)

    try:
        for start in range(0, total, effective_batch_size):
            batch_examples = eval_examples[start : start + effective_batch_size]
            prompts = [ex["prompt"] for ex in batch_examples]
            expected_answers = [ex.get("answer") for ex in batch_examples]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            input_ids = inputs["input_ids"].to(device, non_blocking=True)
            attention_mask = inputs["attention_mask"].to(device, non_blocking=True)

            with torch.no_grad():
                if greedy_eval:
                    generated = model.generate(  # type: ignore
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        forced_eos_token_id=eos_token_id,
                        early_stopping=False,
                        do_sample=False,
                        num_return_sequences=num_return_sequences,
                    )
                else:
                    if sampling_temperature is None:
                        raise AssertionError("sampling_temperature must be set when using sampled evaluation")
                    generated = model.generate(  # type: ignore
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        forced_eos_token_id=eos_token_id,
                        early_stopping=False,
                        do_sample=True,
                        temperature=float(sampling_temperature),
                        num_return_sequences=num_return_sequences,
                    )

            # Decode only the *new* tokens (avoid the prompt leaking into parsing).
            new_tokens = generated[:, input_ids.size(1) :]
            responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for example, expected, response in zip(batch_examples, expected_answers, responses):
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
                                pred_num is not None and exp_num is not None and pred_num == exp_num
                            )
                    if is_correct:
                        correct += 1
                except Exception as e:
                    print("\nFailed to parse model output for prompt:")
                    print(example["prompt"])
                    print("Error:", e)
                    print("-" * 50)
    finally:
        if was_training:
            model.train()

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


# -------------------------
# Reward functions
# -------------------------
def correctness_reward(
    prompts: Sequence[str],
    completions: Sequence[CompletionList],
    answer: Sequence[Optional[str]],
) -> List[float]:
    """2.0 for exact match; 1.5 for numeric-equivalent; else 0.0."""
    responses = [completion[0]["content"] for completion in completions]
    extracted = [extract_answer_from_model_output(r) for r in responses]
    rewards: List[float] = []
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
) -> List[float]:
    """Up to 0.8 for using required XML tags."""
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for response in responses:
        score = 0.0
        if "<reasoning>" in response:
            score += 0.2
        if "</reasoning>" in response:
            score += 0.2
        if "<answer>" in response:
            score += 0.2
        if "</answer>" in response:
            score += 0.2
        rewards.append(score)
    return rewards


def combined_reward(
    prompts: Sequence[str],
    completions: Sequence[CompletionList],
    answer: Sequence[Optional[str]],
) -> List[float]:
    """Correctness (0..2.0) + format (0..0.8)."""
    correctness_scores = correctness_reward(prompts=prompts, completions=completions, answer=answer)
    format_scores = format_reward(completions=completions)
    return [c + f for c, f in zip(correctness_scores, format_scores)]


REWARD_FUNCTIONS: Dict[str, RewardFn] = {
    "combined": combined_reward,
    "correctness": correctness_reward,
    "format": format_reward,
}

def selective_log_softmax(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Return log P(token_t) for each position t in `input_ids`, using logits.
    """
    log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


def compute_log_probs(
    model: PolicyModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_to_keep: int,
) -> torch.Tensor:
    """
    Per-token log-probs for the `logits_to_keep` tokens at the end of the sequence.
    """
    # Autocast for faster forward (model params are bfloat16).
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # (B, T, V)
    logits = logits[:, :-1, :]                            # align next-token targets
    toks = input_ids[:, -logits_to_keep:]                 # (B, K)
    logits = logits[:, -logits_to_keep:, :]               # (B, K, V)
    return selective_log_softmax(logits, toks)


def create_completion_mask(completion_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """
    Mask = 1 for tokens up to and including the first EOS; 0 after.
    Returned dtype is int to remain compatible with HF attention_mask.
    """
    is_eos = completion_ids == eos_token_id                      # (B, Tc)
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    mask_exists = is_eos.any(dim=1)
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    seq_idx = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    return (seq_idx <= eos_idx.unsqueeze(1)).int()

def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    num_generations: int = 4,
    max_completion_length: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate `num_generations` completions per prompt (sampling).
    Returns: prompt_ids, prompt_mask, completion_ids, completion_mask
    with shapes (B*G, ...).
    """
    device = next(model.parameters()).device

    # Left padding keeps the "new tokens" aligned to the right for easy slicing.
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    prompt_ids = inputs["input_ids"].to(device, non_blocking=True)
    prompt_mask = inputs["attention_mask"].to(device, non_blocking=True)
    prompt_seq_len = prompt_ids.size(1)

    # Repeat each prompt `G` times for G completions.
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    if pad_token_id is None or eos_token_id is None:
        raise ValueError("Tokenizer must define both pad_token_id and eos_token_id for generation")

    outputs = model.generate(
        prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        early_stopping=False,
    )  # type: ignore

    # Generated tensor includes the prompt prefix.
    # Some HF models may insert additional special tokens; this assert protects
    # us from mismatched slicing assumptions.
    assert torch.equal(outputs[:, :prompt_seq_len].to(prompt_ids.device), prompt_ids), \
        "Model.generate() did not return input prefix as-is; cannot slice completions safely."

    completion_ids = outputs[:, prompt_seq_len:]
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
    Generate completions and cache log-probs from:
      - the current model snapshot ("old policy" during sampling), and
      - the frozen reference model.
    """
    prompts = [s["prompt"] if isinstance(s, dict) else s[0] for s in batch_samples]
    answers = [s["answer"] if isinstance(s, dict) else s[1] for s in batch_samples]

    with torch.no_grad():
        p_ids, p_mask, c_ids, c_mask = generate_completions(
            model, tokenizer, prompts, num_generations, max_completion_length
        )
        input_ids = torch.cat([p_ids, c_ids], dim=1)
        attention_mask = torch.cat([p_mask, c_mask], dim=1)  # type: ignore[arg-type]
        logits_to_keep = c_ids.size(1)

        old_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
        ref_log_probs = compute_log_probs(ref_model, input_ids, attention_mask, logits_to_keep)

    # PERF: batch decode instead of per-sample loop
    decoded = tokenizer.batch_decode(c_ids, skip_special_tokens=True)
    formatted_completions: List[CompletionList] = [[{"content": s}] for s in decoded]
    repeated_prompts: List[str] = [p for p in prompts for _ in range(num_generations)]
    repeated_answers: List[Optional[str]] = [a for a in answers for _ in range(num_generations)]

    return RolloutData(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=c_mask,
        old_log_probs=old_log_probs,
        ref_log_probs=ref_log_probs,
        formatted_completions=formatted_completions,
        repeated_prompts=repeated_prompts,
        repeated_answers=repeated_answers,
        logits_to_keep=logits_to_keep,
        batch_size=len(prompts),
        num_generations=num_generations,
    )


def grpo_loss(
    model: PolicyModel,
    rollout_data: RolloutData,
    reward_function: RewardFn,
    beta: float,
    epsilon: float,
) -> Tuple[torch.Tensor, float]:
    """
    GRPO objective with PPO-style clipping and reverse-KL penalty.
    """
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]
    logits_to_keep = rollout_data["logits_to_keep"]
    batch_size = int(rollout_data["batch_size"])
    num_generations = int(rollout_data["num_generations"])

    current_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
    ratio = torch.exp(current_log_probs - old_log_probs)

    rewards = torch.tensor(
        reward_function(
            prompts=rollout_data["repeated_prompts"],
            completions=rollout_data["formatted_completions"],
            answer=rollout_data["repeated_answers"],
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
    surrogate = torch.min(surr1, surr2)

    # Reverse-KL approximation (stable around 0): exp(Δ) - Δ - 1, where Δ = ref - current.
    delta = ref_log_probs - current_log_probs
    kl = torch.exp(delta) - delta - 1.0

    per_token = surrogate - beta * kl

    # Convert mask to float once; avoid divide-by-zero on empty completions.
    cmask = completion_mask.to(per_token.dtype)
    token_counts = cmask.sum(dim=1).clamp_min(1)  # FIX: guard against division by zero
    loss = -((per_token * cmask).sum(dim=1) / token_counts).mean()

    return loss, avg_reward


# -------------------------
# Training loop
# -------------------------
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
    Iterative GRPO training with a frozen reference model per iteration.
    """
    if not device_ids:
        raise ValueError("GRPO training requires at least one CUDA device.")

    primary_device = torch.device(f"cuda:{device_ids[0]}")
    model = model.to(primary_device)

    policy_model: PolicyModel = nn.DataParallel(model, device_ids=device_ids) if len(device_ids) > 1 else model

    def unwrap(module_like: PolicyModel) -> PreTrainedModel:
        return cast(PreTrainedModel, module_like.module) if isinstance(module_like, nn.DataParallel) else cast(PreTrainedModel, module_like)

    for iteration in range(1, num_iterations + 1):
        print(f"\nStarting iteration {iteration}/{num_iterations}")

        # Snapshot policy -> reference (no grads).
        reference_model = copy.deepcopy(unwrap(policy_model)).to(primary_device)
        reference_model.eval()
        for p in reference_model.parameters():
            p.requires_grad = False

        # PERF: fused AdamW when available (safe fallback if not).
        try:
            optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate, fused=True)  # type: ignore[arg-type]
        except TypeError:
            optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)

        policy_model.train()

        for step in range(1, steps_per_iteration + 1):
            batch_samples = random.sample(train_data, batch_size)

            # Generate with current policy snapshot on the primary device.
            with torch.no_grad():
                policy_module = unwrap(policy_model)
                # Temporarily switch to eval + enable KV cache for faster decoding.
                was_training = policy_module.training
                prev_cache = getattr(policy_module.config, "use_cache", False)
                policy_module.eval()
                policy_module.config.use_cache = True

                rollout_data = generate_rollout_data(
                    policy_module, reference_model, tokenizer, batch_samples, num_generations, max_completion_length
                )

                # Restore training state.
                policy_module.config.use_cache = prev_cache
                if was_training:
                    policy_module.train()

            # Do `mu` GRPO steps per rollout batch.
            for grpo_iter in range(1, mu + 1):
                loss, avg_reward = grpo_loss(policy_model, rollout_data, reward_function, beta, epsilon)
                optimizer.zero_grad(set_to_none=True)  # PERF: reduce allocator pressure
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=0.1)
                optimizer.step()
                print(
                    f"Iteration {iteration}/{num_iterations}, Step {step}/{steps_per_iteration}, "
                    f"GRPO update {grpo_iter}/{mu}, Loss: {loss.item():.4f}, Avg Reward: {avg_reward:.4f}"
                )

        print(f"Completed iteration {iteration}. (Placeholder: reward model update would happen here.)")

    return unwrap(policy_model)


# -------------------------
# Model init utilities
# -------------------------
def enable_grads(model: PreTrainedModel) -> PreTrainedModel:
    """
    Prepare model for training with gradient checkpointing and disabled KV cache.
    """
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Some HF models expose a helper; if not, attach a small hook.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, inp, out):
            out.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    return model

def run_grpo_training(
    seed: int = 42,  # Sets deterministic seeds for Python, NumPy, and PyTorch.
    model_mode: str = "completion",  # "chat" => use tokenizer chat template; "completion" => plain text prompt.
    greedy_eval: bool = False,  # If true, evaluation uses deterministic decoding instead of sampling.
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",  # Hugging Face model identifier to load.
    train_split: str = "train",  # Dataset split to sample prompts from.
    num_eval_examples: int = 30,  # Number of held-out examples used for evaluation.
    eval_batch_size: int = 32,  # Batch size for evaluation forward passes.
    eval_max_new_tokens: int = 512,  # Hard cap on generated tokens during evaluation.
    eval_temperature: float = 0.7,  # Sampling temperature for evaluation when not greedy.
    num_iterations: int = 1,  # Outer GRPO iterations; each snapshots a reference policy.
    steps_per_iteration: int = 500,  # Policy update steps per iteration before resampling.
    batch_size: int = 7,  # Prompts drawn per rollout batch during training.
    num_generations: int = 12,  # Sampled completions per prompt in each rollout.
    max_completion_length: int = 400,  # Maximum tokens generated per sampled completion.
    beta: float = 0.04,  # Reverse-KL penalty coefficient to keep policy near reference.
    learning_rate: float = 5e-6,  # Learning rate for the optimizer driving GRPO updates.
    mu: int = 1,  # Number of GRPO optimizer steps performed per rollout batch.
    epsilon: float = 0.1,  # PPO-style clipping threshold for the policy ratio.
    reward_function: RewardFn = combined_reward,  # Callable scoring completions used during training.
) -> None:
    """Run GRPO RL fine-tuning end-to-end."""
    system_prompt = build_system_prompt(model_mode)

    set_random_seed(seed)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("GRPO training requires at least one CUDA device.")
    primary_device = torch.device("cuda:0")
    print(f"Using device: {primary_device} | #GPUs detected: {num_gpus}")

    print(f"Downloading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
    ).to(primary_device)
    print("Downloaded model.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"  # centralized padding side
    # Ensure padding/eos ids are set consistently for decoder-only models.
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    device_ids = list(range(num_gpus))

    # Prepare data: a shuffled train split, with a small held-out eval slice.
    all_data = prepare_dataset(tokenizer, model_mode, system_prompt, train_split)
    random.shuffle(all_data)
    if num_eval_examples > len(all_data):
        num_eval_examples = len(all_data)
    train_data = all_data[num_eval_examples:]
    eval_data = all_data[:num_eval_examples]

    pre_grpo_accuracy = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        eval_examples=eval_data,
        device=primary_device,
        batch_size=eval_batch_size,
        greedy_eval=greedy_eval,
        max_new_tokens=eval_max_new_tokens,
        sampling_temperature=None if greedy_eval else eval_temperature,
    )
    print(f"Pre-GRPO Accuracy: {pre_grpo_accuracy:.2f}%")

    model = enable_grads(model)

    print("\nStarting RL finetuning using GRPO...")
    model = train_with_grpo(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        num_iterations=num_iterations,
        steps_per_iteration=steps_per_iteration,
        batch_size=batch_size,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        beta=beta,
        learning_rate=learning_rate,
        mu=mu,
        epsilon=epsilon,
        reward_function=reward_function,
        device_ids=device_ids,
    )

    print("\nFinal model evaluation after GRPO RL finetuning:")
    post_grpo_accuracy = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        eval_examples=eval_data,
        device=primary_device,
        batch_size=eval_batch_size,
        greedy_eval=greedy_eval,
        max_new_tokens=eval_max_new_tokens,
        sampling_temperature=None if greedy_eval else eval_temperature,
    )
    print(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}%")
    print(f"Total Improvement: {post_grpo_accuracy - pre_grpo_accuracy:.2f}%")

    print("\nSaving GRPO finetuned model...")
    model.save_pretrained("grpo_finetuned_model")
    tokenizer.save_pretrained("grpo_finetuned_model")


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training for GSM8K with XML-formatted answers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for all libraries")
    parser.add_argument("--model-mode", type=str, default="completion", choices=("completion", "chat"), help="Tokenizer prompt formatting mode")
    parser.add_argument("--greedy-eval", action="store_true", help="Use greedy decoding for evaluation")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="HF model id")
    parser.add_argument("--train-split", type=str, default="train", help="Dataset split to load")
    parser.add_argument("--num-eval-examples", type=int, default=30, help="Number of examples reserved for evaluation")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--eval-max-new-tokens", type=int, default=512, help="Max tokens generated during evaluation")
    parser.add_argument("--eval-temperature", type=float, default=0.7, help="Sampling temperature for evaluation when not greedy")
    parser.add_argument("--num-iterations", type=int, default=1, help="Number of GRPO outer iterations")
    parser.add_argument("--steps-per-iteration", type=int, default=500, help="Policy update steps per iteration")
    parser.add_argument("--batch-size", type=int, default=7, help="Training batch size (number of prompts)")
    parser.add_argument("--num-generations", type=int, default=12, help="Number of rollouts per prompt")
    parser.add_argument("--max-completion-length", type=int, default=400, help="Maximum tokens generated per completion during rollouts")
    parser.add_argument("--beta", type=float, default=0.04, help="Reverse-KL penalty coefficient")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Optimizer learning rate")
    parser.add_argument("--mu", type=int, default=1, help="Number of GRPO updates per rollout batch")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Clipping threshold for PPO objective")
    parser.add_argument("--reward-function", type=str, default="combined", choices=tuple(REWARD_FUNCTIONS.keys()), help="Reward function to use during training")

    args = parser.parse_args()

    reward_function = REWARD_FUNCTIONS[args.reward_function]

    run_grpo_training(
        seed=args.seed,
        model_mode=args.model_mode,
        greedy_eval=args.greedy_eval,
        model_name=args.model_name,
        train_split=args.train_split,
        num_eval_examples=args.num_eval_examples,
        eval_batch_size=args.eval_batch_size,
        eval_max_new_tokens=args.eval_max_new_tokens,
        eval_temperature=args.eval_temperature,
        num_iterations=args.num_iterations,
        steps_per_iteration=args.steps_per_iteration,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        learning_rate=args.learning_rate,
        mu=args.mu,
        epsilon=args.epsilon,
        reward_function=reward_function,
    )


if __name__ == "__main__":
    main()
