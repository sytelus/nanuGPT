"""
GRPO (Group Relative Policy Optimization) training script
========================================================

Perform RL using verifiable rewards on datasets like GSM8K and pre-trained
language models like Qwen2.5-1.5B-Instruct.

The script can work on single GPU or multiple GPUs, however, currently only data
parallelism is supported. Each rank holds policy model and reference model. We generate rollouts
and compare with reference on that rank to get reward per rank and generate loss
for that replica to generate gradients for that replica. Gradients are averaged
across replicas using DDP. After each iteration, the policy model is copied to
the reference model and the process is repeated.
"""

import argparse
import copy
import importlib
import json
import logging
import os
import random
import re
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, TypedDict, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    StateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

FSDP_SHARDING_MAP: Dict[str, ShardingStrategy] = {
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    "no_shard": ShardingStrategy.NO_SHARD,
}
BACKWARD_PREFETCH_MAP: Dict[str, Optional[BackwardPrefetch]] = {
    "pre": BackwardPrefetch.BACKWARD_PRE,
    "post": BackwardPrefetch.BACKWARD_POST,
    "none": None,
}

DEFAULT_ACTIVATION_CHECKPOINT_THRESHOLD = 14_000_000_000
LOG_PROB_CHUNK_SIZE = 8


logger = logging.getLogger("nano_rlvr")


CompletionMessage = Dict[str, str]
CompletionList = List[CompletionMessage]
DatasetExample = Dict[str, str]
BatchSample = Union[DatasetExample, Tuple[str, str]]
RewardFn = Callable[[Sequence[str], Sequence[CompletionList], Sequence[Optional[str]]], Sequence[float]]
PolicyModel = Union[PreTrainedModel, FSDP]

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


def unwrap_model(module_like: PolicyModel) -> PreTrainedModel:
    """Return the underlying HF model from an FSDP wrapper or plain module."""
    base = getattr(module_like, "module", module_like)
    return cast(PreTrainedModel, base)


def is_fsdp_model(module_like: PolicyModel) -> bool:
    return isinstance(module_like, FSDP)


def infer_transformer_layer_cls(model: nn.Module) -> Set[type[nn.Module]]:
    """Infer transformer block classes for auto-wrapping.

    We first look at `_no_split_modules` provided by HF models, then fall back to
    inspecting common container attributes. The heuristic keeps the default
    compact but lets us wrap block-wise when possible.
    """
    inferred: Set[type[nn.Module]] = set()

    no_split = getattr(model, "_no_split_modules", None)
    if isinstance(no_split, (list, tuple)) and model.__class__.__module__:
        try:
            module = importlib.import_module(model.__class__.__module__)
        except ModuleNotFoundError:
            module = None
        if module is not None:
            for name in no_split:
                candidate = getattr(module, name, None)
                if isinstance(candidate, type) and issubclass(candidate, nn.Module):
                    inferred.add(candidate)

    container_candidates = (
        ("model", "layers"),
        ("model", "h"),
        ("transformer", "layers"),
        ("transformer", "h"),
        ("backbone", "layers"),
    )
    for parent_attr, child_attr in container_candidates:
        parent = getattr(model, parent_attr, None)
        if parent is None:
            continue
        children = getattr(parent, child_attr, None)
        if isinstance(children, (nn.ModuleList, list, tuple)) and children:
            inferred.add(type(children[0]))

    return inferred


def build_fsdp_model(model: PreTrainedModel, device: torch.device, cfg: Dict[str, Any]) -> PolicyModel:
    model.to(device)

    auto_wrap_policy = None
    if cfg.get("auto_wrap", True):
        layer_classes = infer_transformer_layer_cls(model)
        if layer_classes and transformer_auto_wrap_policy is not None:
            auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_classes)

    sharding_key = cfg.get("sharding_strategy", "full_shard")
    sharding = FSDP_SHARDING_MAP.get(sharding_key)
    if sharding is None:
        raise ValueError(f"Unsupported FSDP sharding strategy '{sharding_key}'. Available: {sorted(FSDP_SHARDING_MAP)}")

    precision = cfg.get("mixed_precision", "bf16")
    if precision not in ("bf16", "fp16", "fp32"):
        raise ValueError("--fsdp-precision must be one of 'bf16', 'fp16', 'fp32'")
    if precision == "fp32" or MixedPrecision is None:
        mixed_precision = None
    else:
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        mixed_precision = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

    cpu_offload = CPUOffload(offload_params=True) if cfg.get("offload_params") else None

    backward_prefetch_key = cfg.get("backward_prefetch", "pre")
    backward_prefetch = BACKWARD_PREFETCH_MAP.get(backward_prefetch_key)
    if backward_prefetch_key not in BACKWARD_PREFETCH_MAP:
        raise ValueError(f"Unsupported backward prefetch option '{backward_prefetch_key}'.")

    limit_all_gathers = bool(cfg.get("limit_all_gathers", False))
    forward_prefetch = bool(cfg.get("forward_prefetch", True))

    policy_model: PolicyModel = FSDP(
        model,
        sharding_strategy=sharding,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        cpu_offload=cpu_offload,
        backward_prefetch=backward_prefetch,
        forward_prefetch=forward_prefetch,
        limit_all_gathers=limit_all_gathers,
        device_id=device if device.type == "cuda" else None,
        use_orig_params=True,
        sync_module_states=True,
    )

    return policy_model


def prepare_policy_model(model: PreTrainedModel, device: torch.device, cfg: Dict[str, Any]) -> PolicyModel:
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    fsdp_enabled = cfg.get("enabled", True) and torch.cuda.is_available() and world_size > 1
    if fsdp_enabled:
        return build_fsdp_model(model, device, cfg)
    model.to(device)
    return model

def init_torch(seed) -> Tuple[str, int]:
    device_name, device_id = 'cpu', -1

    if not ("RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ):
        logger.warning("Not all distributed environment variables detected; defaulting to single-process.")
        os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", LOCAL_WORLD_SIZE="1", MASTER_ADDR="localhost", MASTER_PORT="12355")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            assert gpu_count-1 >= local_rank, f'LOCAL_RANK={local_rank} is greater than available GPUs={gpu_count}'
            torch.cuda.set_device(local_rank)
            device_name, device_id = f'cuda:{local_rank}', local_rank
        elif gpu_count == 1:
            torch.cuda.set_device(0)
            device_name,device_id = 'cuda:0', 0
        # for deterministic training, reset GPU after setting the device
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # let other values be controlled by env vars
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        torch.set_float32_matmul_precision('high')

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://",
                            device_id=torch.device(device_name))  # type: ignore[arg-type]
    logger.info("Distributed: rank=%d world_size=%d local_rank=%d", dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"]))

    random.seed(seed+dist.get_rank())
    np.random.seed(seed+dist.get_rank())
    torch.manual_seed(seed+dist.get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed+dist.get_rank())

    torch.distributed.barrier()

    return device_name, device_id

def configure_logging(run_dir: Path) -> None:
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    class _Formatter(logging.Formatter):
        def __init__(self) -> None:
            super().__init__("%(asctime)s [rank=%(rank)s] %(levelname)s %(message)s", "%H:%M:%S")

        def format(self, record: logging.LogRecord) -> str:
            record.rank = rank
            return super().format(record)

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    level = logging.INFO if rank == 0 else logging.WARNING
    logger.setLevel(level)
    logger.propagate = False

    handlers = [
        logging.FileHandler(run_dir / "training.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
    for handler in handlers:
        handler.setFormatter(_Formatter())
        handler.setLevel(level)
        logger.addHandler(handler)

    logger.info("Run directory: %s", run_dir)

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

def build_prompt(messages: Sequence[CompletionMessage], tokenizer: PreTrainedTokenizerBase, model_mode: str) -> str:
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
    tokenizer: PreTrainedTokenizerBase, model_mode: str, system_prompt: str, dataset_cfg: Mapping[str, str], split: str,
) -> List[DatasetExample]:
    args: List[str] = [dataset_cfg["path"]]
    subset = dataset_cfg.get("subset")
    if subset:
        args.append(subset)
    data = load_dataset(*args, split=split)
    formatted_data: List[DatasetExample] = []

    for raw in data:
        example: DatasetExample = cast(DatasetExample, raw)
        if model_mode == "chat":
            # Prime the assistant to continue with tags; Qwen benefits from this.
            prompt_str = build_prompt([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": "<reasoning>\n"},
            ], tokenizer, model_mode)
        else:
            prompt_str = build_prompt([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["question"]},
            ], tokenizer, model_mode)

        formatted_data.append({"prompt": prompt_str, "answer": extract_answer_from_dataset(example["answer"])})
    return formatted_data

def evaluate_model(
    model: PolicyModel, tokenizer: PreTrainedTokenizerBase, eval_examples: Sequence[DatasetExample],
    batch_size: int, greedy_eval: bool, max_new_tokens: int, sampling_temperature: Optional[float],
) -> float:
    if not greedy_eval and sampling_temperature is None:
        raise ValueError("sampling_temperature must be provided when greedy_eval is False")

    module = cast(nn.Module, model)
    base_model = unwrap_model(model)
    gather_ctx = nullcontext()
    if is_fsdp_model(module):
        sharding_strategy = getattr(module, "sharding_strategy", None)
        if sharding_strategy != ShardingStrategy.NO_SHARD:
            gather_ctx = FSDP.summon_full_params(module, recurse=True, offload_to_cpu=False)

    device = next(base_model.parameters()).device
    num_return_sequences = 1  # Keep one completion per prompt for accuracy alignment.

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    if pad_token_id is None or eos_token_id is None:
        raise ValueError("Tokenizer must define both pad_token_id and eos_token_id for evaluation")

    was_training = module.training
    module.eval()
    correct = 0
    total = len(eval_examples)
    logger.info("Evaluating model on %d examples...", total)
    if total == 0:
        logger.warning("No evaluation examples provided; skipping evaluation.")
        return 0.0

    effective_batch_size = min(batch_size, total)

    try:
        with gather_ctx:
            for start in range(0, total, effective_batch_size):
                batch_examples = eval_examples[start : start + effective_batch_size]
                prompts = [ex["prompt"] for ex in batch_examples]
                expected_answers = [ex.get("answer") for ex in batch_examples]

                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
                input_ids = inputs["input_ids"].to(device, non_blocking=True)
                attention_mask = inputs["attention_mask"].to(device, non_blocking=True)

                with torch.no_grad():
                    if greedy_eval:
                        generated = base_model.generate(  # type: ignore
                            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, pad_token_id=pad_token_id,
                            eos_token_id=eos_token_id, forced_eos_token_id=eos_token_id, early_stopping=False, do_sample=False,
                            num_return_sequences=num_return_sequences,
                        )
                    else:
                        if sampling_temperature is None:
                            raise AssertionError("sampling_temperature must be set when using sampled evaluation")
                        generated = base_model.generate(  # type: ignore
                            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, pad_token_id=pad_token_id,
                            eos_token_id=eos_token_id, forced_eos_token_id=eos_token_id, early_stopping=False, do_sample=True,
                            temperature=float(sampling_temperature), num_return_sequences=num_return_sequences,
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
                    except Exception as exc:
                        logger.warning(
                            "Failed to parse model output for prompt '%s': %s",
                            example["prompt"],
                            exc,
                        )
    finally:
        if was_training:
            module.train()

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    logger.info("Accuracy: %.2f%% (%d/%d)", accuracy, correct, total)

    # complete eval on all ranks before returning
    torch.distributed.barrier()

    return accuracy


# -------------------------
# Reward functions
# -------------------------
def correctness_reward(prompts: Sequence[str], completions: Sequence[CompletionList], answer: Sequence[Optional[str]]) -> List[float]:
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


def format_reward(completions: Sequence[CompletionList]) -> List[float]:
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


def combined_reward(prompts: Sequence[str], completions: Sequence[CompletionList], answer: Sequence[Optional[str]]) -> List[float]:
    """Correctness (0..2.0) + format (0..0.8)."""
    correctness_scores = correctness_reward(prompts=prompts, completions=completions, answer=answer)
    format_scores = format_reward(completions=completions)
    return [c + f for c, f in zip(correctness_scores, format_scores)]


REWARD_FUNCTIONS: Dict[str, RewardFn] = {
    "combined": combined_reward,
    "correctness": correctness_reward,
    "format": format_reward,
}

KNOWN_DATASETS: Dict[str, Dict[str, str]] = {
    "gsm8k": {"path": "openai/gsm8k", "subset": "main", "train_split": "train", "test_split": "test"},
}

def selective_log_softmax(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Return log P(token_t) for each position t in `input_ids`, using logits.
    """
    log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


def compute_log_probs(
    model: PolicyModel, input_ids: torch.Tensor, attention_mask: torch.Tensor, logits_to_keep: int, chunk_size: int = LOG_PROB_CHUNK_SIZE,
) -> torch.Tensor:
    """
    Per-token log-probs for the `logits_to_keep` tokens at the end of the sequence.
    """
    total = input_ids.size(0)
    chunk_size = max(1, int(chunk_size))
    outputs: List[torch.Tensor] = []
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        ids_chunk = input_ids[start:end]
        mask_chunk = attention_mask[start:end]

        # `torch.autocast` in the policy forward reduces cast overhead in updates
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids_chunk, attention_mask=mask_chunk).logits  # (C, T, V)
        logits = logits[:, :-1, :]                      # align next-token targets
        toks = ids_chunk[:, -logits_to_keep:]           # (C, K)
        logits = logits[:, -logits_to_keep:, :]         # (C, K, V)
        outputs.append(selective_log_softmax(logits, toks))

    return torch.cat(outputs, dim=0)


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
    model: nn.Module, tokenizer: PreTrainedTokenizerBase, prompts: Sequence[str], num_generations: int = 4,
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

    outputs = model.generate(  # type: ignore
        prompt_ids, attention_mask=prompt_mask, max_new_tokens=max_completion_length, do_sample=True, temperature=0.7, top_p=0.9,
        repetition_penalty=1.05, pad_token_id=pad_token_id, eos_token_id=eos_token_id, early_stopping=False,
    )

    # Generated tensor includes the prompt prefix.
    # Some HF models may insert additional special tokens; this assert protects
    # us from mismatched slicing assumptions.
    assert torch.equal(outputs[:, :prompt_seq_len].to(prompt_ids.device), prompt_ids), \
        "Model.generate() did not return input prefix as-is; cannot slice completions safely."

    completion_ids = outputs[:, prompt_seq_len:]
    completion_mask = create_completion_mask(completion_ids, eos_token_id)
    return prompt_ids, prompt_mask, completion_ids, completion_mask

def generate_rollout_data(
    model: PolicyModel, ref_model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, batch_samples: Sequence[BatchSample],
    num_generations: int, max_completion_length: int,
) -> RolloutData:
    """
    Generate completions and cache log-probs from:
      - the current model snapshot ("old policy" during sampling), and
      - the frozen reference model.
    """
    prompts = [s["prompt"] if isinstance(s, dict) else s[0] for s in batch_samples]
    answers = [s["answer"] if isinstance(s, dict) else s[1] for s in batch_samples]

    with torch.no_grad():
        p_ids, p_mask, c_ids, c_mask = generate_completions(model, tokenizer, prompts, num_generations, max_completion_length)
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
        input_ids=input_ids, attention_mask=attention_mask, completion_mask=c_mask, old_log_probs=old_log_probs,
        ref_log_probs=ref_log_probs, formatted_completions=formatted_completions, repeated_prompts=repeated_prompts,
        repeated_answers=repeated_answers, logits_to_keep=logits_to_keep, batch_size=len(prompts), num_generations=num_generations,
    )


def grpo_loss(model: PolicyModel, rollout_data: RolloutData, reward_function: RewardFn, beta: float, epsilon: float) -> Tuple[torch.Tensor, float]:
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
            prompts=rollout_data["repeated_prompts"], completions=rollout_data["formatted_completions"],
            answer=rollout_data["repeated_answers"],
        ),
        dtype=torch.float32,
        device=current_log_probs.device,
    )

    rewards_matrix = rewards.view(batch_size, num_generations)
    avg_reward = rewards_matrix.mean().item()
    logger.debug("Average reward (pre-normalization): %.4f", avg_reward)

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
    policy_model: PolicyModel, device: torch.device, tokenizer: PreTrainedTokenizerBase,
    train_data: Sequence[DatasetExample], num_iterations: int, steps_per_iteration: int, batch_size: int,
    num_generations: int, max_completion_length: int, beta: float, learning_rate: float, mu: int, epsilon: float,
    reward_function: RewardFn,
) -> PolicyModel:
    """Train the policy with GRPO, supporting both plain and FSDP-wrapped models."""

    base_policy = unwrap_model(policy_model)

    sharding_strategy = getattr(policy_model, "sharding_strategy", None)

    for iteration in range(1, num_iterations + 1):
        logger.info("Starting iteration %d/%d", iteration, num_iterations)

        # Snapshot the current policy. Under FSDP we temporarily materialize full
        # parameters (offloaded to CPU) before cloning so the reference model
        # stays unsharded but recomp  only once per outer iteration.
        if is_fsdp_model(policy_model) and sharding_strategy != ShardingStrategy.NO_SHARD:
            clone_ctx = FSDP.summon_full_params(policy_model, recurse=True, offload_to_cpu=True)
            with clone_ctx:
                state_dict = {k: v.detach().cpu() for k, v in base_policy.state_dict().items()}

            base_dtype = next(base_policy.parameters()).dtype
            try:
                reference_model = AutoModelForCausalLM.from_config(base_policy.config, trust_remote_code=True)
            except TypeError:
                reference_model = AutoModelForCausalLM.from_config(base_policy.config)
            reference_model.to(dtype=base_dtype)
            reference_model.load_state_dict(state_dict)
            reference_model.to(device=device, dtype=base_dtype)
            del state_dict
        else:
            reference_model = copy.deepcopy(base_policy).to(device)
        reference_model.eval()
        for p in reference_model.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(
            policy_model.parameters(), lr=learning_rate, fused=torch.cuda.is_available()
        )  # type: ignore[arg-type]

        policy_model.train()

        for step in range(1, steps_per_iteration + 1):
            batch_samples = random.sample(train_data, batch_size)

            with torch.no_grad():
                was_training = policy_model.training
                prev_cache = getattr(base_policy.config, "use_cache", False)
                policy_model.eval()
                base_policy.config.use_cache = True

                rollout_data = generate_rollout_data(
                    policy_model, reference_model, tokenizer, batch_samples, num_generations, max_completion_length,
                )

                base_policy.config.use_cache = prev_cache
                if was_training:
                    policy_model.train()

            for grpo_iter in range(1, mu + 1):
                loss, avg_reward = grpo_loss(policy_model, rollout_data, reward_function, beta, epsilon)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=0.1)
                optimizer.step()
                logger.info(
                    "Iteration %d/%d, Step %d/%d, GRPO update %d/%d, Loss: %.4f, Avg reward: %.4f",
                    iteration, num_iterations, step, steps_per_iteration, grpo_iter, mu, loss.item(), avg_reward,
                )

        logger.info("Completed iteration %d. (Placeholder: reward model update would happen here.)", iteration)
        del reference_model

    return policy_model


# -------------------------
# Model init utilities
# -------------------------
def enable_activation_checkpointing(model: PreTrainedModel) -> PreTrainedModel:
    """
    Prepare model for training with gradient checkpointing and disabled KV cache.

    Must be invoked *before* wrapping the module with FSDP so shard metadata
    captures the checkpointed blocks correctly.
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
    device_name: str, device_id: int, model_mode: str = "completion", greedy_eval: bool = False,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", dataset: str = "gsm8k", num_eval_examples: int = 30,
    eval_batch_size: int = 32, eval_max_new_tokens: int = 512, eval_temperature: float = 0.7, num_iterations: int = 1,
    steps_per_iteration: int = 500, batch_size: int = 7, num_generations: int = 12, max_completion_length: int = 400,
    beta: float = 0.04, learning_rate: float = 5e-6, mu: int = 1, epsilon: float = 0.1,
    reward_function: RewardFn = combined_reward, out_dir: Optional[str] = None,
    fsdp_config: Optional[Dict[str, Any]] = None,
) -> dict:
    """Run GRPO RL fine-tuning end-to-end."""

    fsdp_cfg: Dict[str, Any] = dict(fsdp_config or {})
    fsdp_cfg.setdefault("enabled", True)
    fsdp_cfg.setdefault("sharding_strategy", "full_shard")
    fsdp_cfg.setdefault("mixed_precision", "bf16")
    fsdp_cfg.setdefault("offload_params", False)
    fsdp_cfg.setdefault("limit_all_gathers", False)
    fsdp_cfg.setdefault("forward_prefetch", True)
    fsdp_cfg.setdefault("backward_prefetch", "pre")
    fsdp_cfg.setdefault("auto_wrap", True)
    fsdp_cfg.setdefault("activation_checkpoint_threshold", DEFAULT_ACTIVATION_CHECKPOINT_THRESHOLD)

    device = torch.device(device_name)
    logger.info("Primary training device: %s (id=%s)", device, device_id)

    logger.info("Downloading model %s...", model_name)
    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    logger.info("Model download complete.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # validate and fix tokenizer
    if tokenizer.eos_token is None:
        raise ValueError("`tokenizer.eos_token` must be provided, but got None.")

    logger.info("Model max length: %s", str(getattr(model.config, "n_positions", None)))
    logger.info("Tokenizer max length: %s", str(tokenizer.model_max_length))
    if tokenizer.model_max_length != getattr(model.config, "n_positions", None):
        logger.warning(
            "Tokenizer max length (%s) != model max length (%s); using tokenizer's value.",
            str(tokenizer.model_max_length), str(getattr(model.config, "n_positions", None)),
        )
    logger.info("Tokenizer vocab size: %d", tokenizer.vocab_size)
    logger.info("Tokenizer pad token: %s (id=%s)", repr(tokenizer.pad_token), tokenizer.pad_token_id)
    logger.info("Tokenizer eos token: %s (id=%s)", repr(tokenizer.eos_token), tokenizer.eos_token_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning("Setting `tokenizer.pad_token` to `tokenizer.eos_token` (%s).", repr(tokenizer.eos_token))
    tokenizer.padding_side = "left"  # centralized padding side

    if dataset not in KNOWN_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Known datasets: {', '.join(sorted(KNOWN_DATASETS))}")
    dataset_cfg = KNOWN_DATASETS[dataset]

    system_prompt = build_system_prompt(model_mode)
    logger.info("System prompt:\n%s", system_prompt)

    train_data = prepare_dataset(tokenizer, model_mode, system_prompt, dataset_cfg, dataset_cfg["train_split"])
    random.shuffle(train_data)
    eval_split_name = dataset_cfg.get("test_split")
    if eval_split_name:
        eval_data = prepare_dataset(tokenizer, model_mode, system_prompt, dataset_cfg, eval_split_name)
        total_eval_examples = len(eval_data)
        if num_eval_examples >= 0:
            effective_eval_examples = min(num_eval_examples, total_eval_examples)
            if effective_eval_examples < num_eval_examples:
                logger.warning(
                    "Requested %d eval examples but only %d available; using %d instead.",
                    num_eval_examples, total_eval_examples, effective_eval_examples,
                )
        else:
            effective_eval_examples = total_eval_examples
        eval_data = eval_data[:effective_eval_examples]
        eval_label = eval_split_name if effective_eval_examples == total_eval_examples else f"{eval_split_name}[:{effective_eval_examples}]"
    else:
        total_examples = len(train_data)
        effective_eval_examples = min(num_eval_examples if num_eval_examples >= 0 else total_examples, total_examples)
        if effective_eval_examples < (num_eval_examples if num_eval_examples >= 0 else total_examples):
            logger.warning(
                "Requested %d eval examples but only %d available; using %d instead.",
                num_eval_examples, total_examples, effective_eval_examples,
            )
        eval_data = train_data[:effective_eval_examples]
        train_data = train_data[effective_eval_examples:]
        eval_label = f"{dataset_cfg['train_split']}[:{effective_eval_examples}]"

    logger.info(
        "Dataset '%s': train=%d (split=%s) eval=%d (split=%s)",
        dataset, len(train_data), dataset_cfg["train_split"], len(eval_data), eval_label,
    )

    params = sum(p.numel() for p in model.parameters())
    logger.info("Model has %d parameters.", params)
    threshold = int(fsdp_cfg.get("activation_checkpoint_threshold", DEFAULT_ACTIVATION_CHECKPOINT_THRESHOLD))
    if params >= threshold and torch.cuda.is_available():
        logger.info("Enabling activation checkpointing for memory efficiency...")
        enable_activation_checkpointing(model)

    policy_model = prepare_policy_model(model, device, fsdp_cfg)

    logger.info("Pre-GRPO evaluation...")
    pre_grpo_accuracy = evaluate_model(
        model=policy_model, tokenizer=tokenizer, eval_examples=eval_data, batch_size=eval_batch_size, greedy_eval=greedy_eval,
        max_new_tokens=eval_max_new_tokens, sampling_temperature=None if greedy_eval else eval_temperature,
    )
    logger.info("Pre-GRPO accuracy: %.2f%%", pre_grpo_accuracy)

    logger.info("Starting RL finetuning using GRPO...")
    policy_model = train_with_grpo(
        policy_model=policy_model, device=device, tokenizer=tokenizer, train_data=train_data, num_iterations=num_iterations,
        steps_per_iteration=steps_per_iteration, batch_size=batch_size, num_generations=num_generations,
        max_completion_length=max_completion_length, beta=beta, learning_rate=learning_rate, mu=mu, epsilon=epsilon,
        reward_function=reward_function,
    )

    logger.info("Final model evaluation after GRPO RL finetuning...")
    post_grpo_accuracy = evaluate_model(
        model=policy_model, tokenizer=tokenizer, eval_examples=eval_data, batch_size=eval_batch_size, greedy_eval=greedy_eval,
        max_new_tokens=eval_max_new_tokens, sampling_temperature=None if greedy_eval else eval_temperature,
    )
    improvement = post_grpo_accuracy - pre_grpo_accuracy
    logger.info("Post-GRPO accuracy: %.2f%%", post_grpo_accuracy)
    logger.info("Total improvement: %.2f%%", improvement)

    artifact_dir = run_dir / "grpo_finetuned_model"
    model_to_save = unwrap_model(policy_model)
    if is_fsdp_model(policy_model) and getattr(policy_model, "sharding_strategy", None) != ShardingStrategy.NO_SHARD:
        state_cfg = StateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(policy_model, StateDictType.FULL_STATE_DICT, state_cfg):  # type: ignore[arg-type]
            state_dict = policy_model.state_dict()
        if dist.get_rank() == 0:
            model_to_save.save_pretrained(artifact_dir, state_dict=state_dict)
            tokenizer.save_pretrained(artifact_dir)
            logger.info("Saved GRPO fine-tuned model artifacts to %s", artifact_dir)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    else:
        model_to_save.save_pretrained(artifact_dir)
        tokenizer.save_pretrained(artifact_dir)
        logger.info("Saved GRPO fine-tuned model artifacts to %s", artifact_dir)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    metrics = {
        "pre_grpo_accuracy": pre_grpo_accuracy,
        "post_grpo_accuracy": post_grpo_accuracy,
        "total_improvement": improvement,
        "eval_examples": len(eval_data),
        "train_examples": len(train_data),
    }
    return metrics

def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training with XML-formatted answers")
    fsdp_sharding_choices = tuple(sorted(FSDP_SHARDING_MAP.keys())) if FSDP_SHARDING_MAP else ("full_shard",)
    fsdp_backward_choices = tuple(sorted(BACKWARD_PREFETCH_MAP.keys())) if BACKWARD_PREFETCH_MAP else ("pre", "post", "none")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for all libraries")
    parser.add_argument("--model-mode", type=str, default="completion", choices=("completion", "chat"), help="Tokenizer prompt formatting mode")
    parser.add_argument("--greedy-eval", action="store_true", help="Use greedy decoding for evaluation")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="HF model id")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=tuple(KNOWN_DATASETS.keys()), help="Dataset to use")
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
    parser.add_argument("--reward-function", type=str, default="combined", choices=tuple(REWARD_FUNCTIONS.keys()),
                        help="Reward function to use during training")
    parser.add_argument("--out-dir", type=str, default=None, help="Base directory for outputs (overrides OUT_DIR environment variable)")
    parser.add_argument("--no-fsdp", action="store_true", help="Disable FSDP and keep parameters replicated per device")
    parser.add_argument("--fsdp-sharding-strategy", type=str, default="full_shard", choices=fsdp_sharding_choices,
                        help="FSDP sharding strategy (default: full_shard)")
    parser.add_argument("--fsdp-precision", type=str, default="bf16", choices=("bf16", "fp16", "fp32"),
                        help="Mixed precision dtype for FSDP parameters and gradients")
    parser.add_argument("--fsdp-offload-params", action="store_true", help="Offload parameter shards to CPU between iterations")
    parser.add_argument("--fsdp-limit-all-gathers", action="store_true", help="Enable FSDP's limit_all_gathers flag")
    parser.add_argument("--no-fsdp-forward-prefetch", action="store_true", help="Disable forward prefetching in FSDP")
    parser.add_argument("--fsdp-backward-prefetch", type=str, default="pre", choices=fsdp_backward_choices,
                        help="Backward prefetch schedule for FSDP overlap")
    parser.add_argument("--no-fsdp-auto-wrap", action="store_true", help="Skip transformer auto-wrapping for FSDP")
    parser.add_argument("--activation-checkpoint-threshold", type=int, default=DEFAULT_ACTIVATION_CHECKPOINT_THRESHOLD,
                        help="Enable activation checkpointing when parameter count exceeds this threshold")

    args = parser.parse_args()

    reward_function = REWARD_FUNCTIONS[args.reward_function]

    # output dir
    base_dir = Path(args.out_dir or os.environ.get("OUT_DIR") or "~/out_dir").expanduser().resolve()
    run_dir = base_dir / "nano_rlvr" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True, exist_ok=False)

    configure_logging(run_dir)

    device_name, device_id = init_torch(seed=args.seed)

    fsdp_cli_config = {
        "enabled": not args.no_fsdp,
        "sharding_strategy": args.fsdp_sharding_strategy,
        "mixed_precision": args.fsdp_precision,
        "offload_params": args.fsdp_offload_params,
        "limit_all_gathers": args.fsdp_limit_all_gathers,
        "forward_prefetch": not args.no_fsdp_forward_prefetch,
        "backward_prefetch": args.fsdp_backward_prefetch,
        "auto_wrap": not args.no_fsdp_auto_wrap,
        "activation_checkpoint_threshold": args.activation_checkpoint_threshold,
    }
    train_params = {
        "device_name": device_name, "device_id": device_id, "model_mode": args.model_mode, "greedy_eval": args.greedy_eval,
        "model_name": args.model_name, "dataset": args.dataset, "num_eval_examples": args.num_eval_examples,
        "eval_batch_size": args.eval_batch_size, "eval_max_new_tokens": args.eval_max_new_tokens, "eval_temperature": args.eval_temperature,
        "num_iterations": args.num_iterations, "steps_per_iteration": args.steps_per_iteration, "batch_size": args.batch_size,
        "num_generations": args.num_generations, "max_completion_length": args.max_completion_length, "beta": args.beta,
        "learning_rate": args.learning_rate, "mu": args.mu, "epsilon": args.epsilon, "out_dir": str(run_dir),
        "fsdp_config": fsdp_cli_config,
    }

    config = {
        "seed": args.seed, "world_size": dist.get_world_size(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    config.update(train_params)

    # save configuration
    if dist.get_rank() == 0:
        config_path = run_dir / f"config.json"
        with config_path.open("w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=2)
        logger.info("Configuration saved to %s", config_path)

    metrics = run_grpo_training(**train_params, reward_function=reward_function)

    if dist.get_rank() == 0:
        metrics_path = run_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)
        logger.info("Metrics saved to %s", metrics_path)
    logger.info("Run complete.")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
