"""
RL with Verifiable Rewards on datasets like GSM8K and pre-trained
language models like Qwen2.5-1.5B-Instruct using GRPO or Dr GRPO.

Script can work with single 80GB GPU with defaults or multi-GPU with data
parallelism. Each rank holds policy model and reference model. Policy rollouts
are generated and compared with reference on same rank to get reward and loss
for that rank. Gradients are averaged across ranks using DDP. The policy model
is copied once to initialize the reference model.
"""

from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TypedDict, Union, Any, cast, get_args, get_origin
from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import MISSING, asdict, dataclass, field, fields as dataclass_fields, replace
import copy
import json
import logging
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger("nano_rlvr")

_wandb_run = None  # init later if wandb is available and env vars are set

# some type aliases
CompletionMessage = Dict[str, str]
CompletionList = List[CompletionMessage]
DatasetExample = dict  # {"prompt": str, "answer": Optional[str]}
BatchSample = Union[DatasetExample, Tuple[str, str]]
# prompts, completions, answers -> reward
RewardFn = Callable[[Sequence[str], Sequence[CompletionList], Sequence[Optional[str]]], Sequence[float]]
PolicyModel = Union[PreTrainedModel, torch.nn.parallel.DistributedDataParallel]

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
    n_generations: int
    policy_duration: float
    reference_duration: float

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

def combined_reward(prompts: Sequence[str],
                    completions: Sequence[CompletionList],
                    answer: Sequence[Optional[str]]) -> List[float]:
    """Correctness (0..2.0) + format (0..0.8)."""
    correctness_scores = correctness_reward(prompts=prompts, completions=completions, answer=answer)
    format_scores = format_reward(completions=completions)
    return [c + f for c, f in zip(correctness_scores, format_scores)]

def gsm8k_prompt(model_mode: str) -> str:
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

dataset_configs: Dict[str, Dict[str, Any]] = {
    "gsm8k": {"path": "openai/gsm8k", "subset": "main",
              "train_split": "train", "test_split": "test",
              "prompt_builder": gsm8k_prompt,
              "reward_fn": combined_reward,
            },
}
@dataclass(frozen=True)
class Config:
    seed: int = 42
    model_mode: str = field(default="completion", metadata={"choices": ("completion", "chat")})
    greedy_eval: bool = False
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset: str = field(default="gsm8k", metadata={"choices": tuple(dataset_configs.keys())})
    n_eval: int = 30
    eval_batch_size: int = 32
    max_new_tokens: int = 512
    eval_temperature: float = 0.7
    steps_per_iteration: int = 500
    batch_size: int = 7
    n_generations: int = 12
    max_completion_length: int = 400
    beta: float = 0.04
    learning_rate: float = 5e-6
    mu: int = 1
    epsilon: float = 0.1
    grpo_variant: str = field(default="grpo", metadata={"choices": ("grpo", "dr_grpo")})
    out_dir: Optional[str] = None
    run_name: Optional[str] = None
    description: Optional[str] = field(default=None, metadata={"help": "Optional run notes logged to Weights & Biases"})
    lr_warmup_steps: int = 0
    lr_cooldown_frac: float = 1.0
    device_name: str = field(default="cpu", metadata={"cli": False})
    device_id: int = field(default=-1, metadata={"cli": False})
    run_dir: str = field(default="", metadata={"cli": False})

def log_to_wandb(data: Dict[str, Any], *, summary: bool = False) -> None:
    if _wandb_run is not None:
        if summary:
            _wandb_run.summary.update(data)
    else:
        _wandb_run.log(data)

def log_message(message: str, level: int = logging.INFO) -> None:
    logger.log(level, message)
    log_to_wandb({'level': logging.getLevelName(level), 'message': message})

def log_metrics(values: Dict[str, float]) -> None:
    logger.info(", ".join(f"{k}={v}" for k, v in values.items()))
    log_to_wandb(values)


def log_summary(values: Dict[str, Any]) -> None:
    log_to_wandb(values, summary=True)
    logger.info(", ".join(f"{k}={v}" for k, v in values.items()))

def init_wandb(config: Mapping[str, object]) -> None:
    global _wandb_run
    if _wandb_run is None and (os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_HOST")):
        try:
            import wandb
        except ImportError:
            logger.warning("wandb not installed; skipping wandb logging.")
            return
        wandb_host = os.environ.get('WANDB_HOST', None)
        wandb.login(host=wandb_host) # use API key from WANDB_API_KEY env variable
        _wandb_run = wandb.init(  # type: ignore[arg-type]
            project="nano-rlvr",
            name=config["run_name"],
            config=dict(config),
            notes=config.get("description"),
            save_code=True,)

def configure_logging(run_dir: Path, config: Config) -> None:
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

    logger.info(f"Run directory: {run_dir}")

    init_wandb(asdict(config))

def init_torch(seed) -> Tuple[str, int]:
    device_name, device_id = 'cpu', -1

    # if env vars for distributed setup are not set, assume single-process
    if not ("RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ):
        log_message("distributed environment variables not set; defaulting to single-process.", level=logging.WARNING)
        os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", LOCAL_WORLD_SIZE="1", MASTER_ADDR="localhost", MASTER_PORT="12355")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_name, device_id = f'cuda:{local_rank}', local_rank
        # for deterministic training, reset GPU after setting the device
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://",
                            device_id=torch.device(device_name))  # type: ignore[arg-type]
    log_summary({"distributed/rank": float(dist.get_rank()), "distributed/world_size": float(dist.get_world_size()),"distributed/local_rank": float(int(os.environ["LOCAL_RANK"]))})

    random.seed(seed+dist.get_rank())
    np.random.seed(seed+dist.get_rank())
    torch.manual_seed(seed+dist.get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed+dist.get_rank())
    torch.distributed.barrier()

    return device_name, device_id

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
            messages, tokenize=False, add_generation_prompt=False  #type: ignore[call-arg]
        ) # type: ignore[call-arg]
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
    tokenizer: PreTrainedTokenizerBase, model_mode: str, dataset_cfg: Mapping[str, str], split: str,
) -> List[DatasetExample]:
    args: List[str] = [dataset_cfg["path"]]
    subset = dataset_cfg.get("subset")
    if subset:
        args.append(subset)
    data = load_dataset(*args, split=split)
    formatted_data: List[DatasetExample] = []
    system_prompt = dataset_cfg["prompt_builder"](model_mode) # type: ignore[arg-type]

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
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, eval_examples: Sequence[DatasetExample],
    config: Config,
) -> Dict[str, float]:
    if not config.greedy_eval and config.eval_temperature is None:
        raise ValueError("eval_temperature must be provided when greedy_eval is False")

    device = next(model.parameters()).device
    num_return_sequences = 1  # Keep one completion per prompt for accuracy alignment.
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    if pad_token_id is None or eos_token_id is None:
        raise ValueError("Tokenizer must define both pad_token_id and eos_token_id for evaluation")

    was_training = model.training
    model.eval()
    correct = 0
    total_examples = len(eval_examples)

    shard_indices = list(range(rank, total_examples, world_size))
    local_examples: List[DatasetExample] = [eval_examples[idx] for idx in shard_indices]
    local_total = len(local_examples)

    effective_batch_size = min(config.batch_size, local_total) if local_total > 0 else config.batch_size

    completion_lengths: List[int] = []
    try:
        for start in range(0, local_total, effective_batch_size):
            batch_examples = local_examples[start : start + effective_batch_size]
            prompts = [ex["prompt"] for ex in batch_examples]
            expected_answers = [ex.get("answer") for ex in batch_examples]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
            input_ids = inputs["input_ids"].to(device, non_blocking=True) # type: ignore[arg-type]
            attention_mask = inputs["attention_mask"].to(device, non_blocking=True) # type: ignore[arg-type]

            with torch.no_grad():
                if config.greedy_eval:
                    generated = model.generate(  # type: ignore
                        input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=config.max_new_tokens, pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id, forced_eos_token_id=eos_token_id, early_stopping=False, do_sample=False,
                        num_return_sequences=num_return_sequences,
                    )
                else:
                    if config.eval_temperature is None:
                        raise AssertionError("eval_temperature must be set when using sampled evaluation")
                    generated = model.generate(  # type: ignore
                        input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=config.max_new_tokens, pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id, forced_eos_token_id=eos_token_id, early_stopping=False, do_sample=True,
                        temperature=float(config.eval_temperature), num_return_sequences=num_return_sequences,
                    )

            # Decode only the *new* tokens (avoid the prompt leaking into parsing).
            new_tokens = generated[:, input_ids.size(1) :]
            responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for example, expected, response in zip(batch_examples, expected_answers, responses):
                completion_lengths.append(len(response.split()))
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
            model.train()


    stats = torch.tensor([correct, local_total], dtype=torch.long, device=device)
    torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
    global_correct = int(stats[0].item())
    global_total = int(stats[1].item())

    accuracy = (global_correct / global_total) * 100 if global_total > 0 else 0.0
    avg_completion_len = float(np.mean(completion_lengths)) if completion_lengths else 0.0

    # complete eval on all ranks before returning
    torch.distributed.barrier()

    return {"accuracy": accuracy, "avg_completion_length": avg_completion_len, "correct": global_correct, "total": global_total}

def _resolve_cli_type(field_type):
    origin = get_origin(field_type)
    if origin is Union:
        args = [arg for arg in get_args(field_type) if arg is not type(None)]
        if len(args) == 1:
            return args[0], True
    return field_type, False

def parse_dc(cls):
    parser = ArgumentParser(prog=cls.__name__, description=cls.__doc__)
    for f in dataclass_fields(cls):
        if not f.init or f.metadata.get("cli", True) is False:
            continue

        option = "--" + f.name.replace("_", "-")
        has_default = not (f.default is MISSING and f.default_factory is MISSING)
        default_value = None
        if f.default is not MISSING:
            default_value = f.default
        elif f.default_factory is not MISSING:  # type: ignore[misc]
            default_value = f.default_factory()  # type: ignore[call-arg]

        required = not has_default
        arg_type, _ = _resolve_cli_type(f.type)

        if arg_type is bool:
            parser.add_argument(
                option,
                dest=f.name,
                action=BooleanOptionalAction,
                required=required,
                default=(None if required else default_value),
            )
            continue

        kwargs: Dict[str, object] = {"dest": f.name, "required": required}
        if arg_type is not None:
            kwargs["type"] = arg_type  # type: ignore[assignment]
        if not required:
            kwargs["default"] = default_value
        choices = f.metadata.get("choices")
        if choices:
            kwargs["choices"] = choices
        help_text = f.metadata.get("help")
        if help_text:
            kwargs["help"] = help_text

        parser.add_argument(option, **kwargs)  # type: ignore[arg-type]

    parsed = vars(parser.parse_args())
    return cls(**parsed)

def selective_log_softmax(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Return log P(token_t) for each position t in `input_ids`, using logits.
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (B, T, V)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


def compute_log_probs(model: PolicyModel, input_ids: torch.Tensor, attention_mask: torch.Tensor, logits_to_keep: int) -> torch.Tensor:
    """
    Per-token log-probs for the `logits_to_keep` tokens at the end of the sequence.
    """
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if input_ids.is_cuda else nullcontext()
    with autocast_ctx:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # (B, T, V)
    logits = logits[:, :-1, :]                            # align next-token targets
    toks = input_ids[:, -logits_to_keep:]                 # (B, K)
    logits = logits[:, -logits_to_keep:, :]               # (B, K, V)
    return selective_log_softmax(logits, toks)

def create_completion_mask(completion_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """
    Mask = 1 for tokens up to and including the first EOS; 0 after. Returned dtype is int to remain compatible with HF attention_mask.
    """
    is_eos = completion_ids == eos_token_id                      # (B, Tc)
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    mask_exists = is_eos.any(dim=1)
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    seq_idx = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    return (seq_idx <= eos_idx.unsqueeze(1)).int()

def gen_completions(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompts: Sequence[str],
    n_generations: int, max_completion_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: # prompt_ids, prompt_mask, completion_ids, completion_mask

    device = next(model.parameters()).device

    # Left padding keeps the "new tokens" aligned to the right for easy slicing.
    inputs = tokenizer(prompts, return_tensors="pt", padding=True) # type: ignore[arg-type]
    prompt_ids = inputs["input_ids"].to(device, non_blocking=True) # type: ignore[arg-type]
    prompt_mask = inputs["attention_mask"].to(device, non_blocking=True) # type: ignore[arg-type]
    prompt_seq_len = prompt_ids.size(1)

    # Repeat each prompt `G` times for G completions.
    prompt_ids = prompt_ids.repeat_interleave(n_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(n_generations, dim=0)

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
    completion_mask = create_completion_mask(completion_ids, int(eos_token_id)) # type: ignore[arg-type]
    return prompt_ids, prompt_mask, completion_ids, completion_mask

def gen_rollouts(
    model: PreTrainedModel, ref_model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
    batch: Sequence[BatchSample], config: Config,
) -> RolloutData:

    prompts = [s["prompt"] if isinstance(s, dict) else s[0] for s in batch]
    answers = [s["answer"] if isinstance(s, dict) else s[1] for s in batch]

    with torch.no_grad():
        policy_start = time.perf_counter()
        p_ids, p_mask, c_ids, c_mask = gen_completions(model, tokenizer, prompts, config.n_generations, config.max_completion_length)
        input_ids = torch.cat([p_ids, c_ids], dim=1)
        attention_mask = torch.cat([p_mask, c_mask], dim=1)  # type: ignore[arg-type]
        logits_to_keep = c_ids.size(1)

        old_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
        policy_duration = time.perf_counter() - policy_start

        ref_start = time.perf_counter()
        ref_log_probs = compute_log_probs(ref_model, input_ids, attention_mask, logits_to_keep)
        reference_duration = time.perf_counter() - ref_start

    # PERF: batch decode instead of per-sample loop
    decoded = tokenizer.batch_decode(c_ids, skip_special_tokens=True)
    formatted_completions: List[CompletionList] = [[{"content": s}] for s in decoded]
    repeated_prompts: List[str] = [p for p in prompts for _ in range(config.n_generations)]
    repeated_answers: List[Optional[str]] = [a for a in answers for _ in range(config.n_generations)]

    return RolloutData(
        input_ids=input_ids, attention_mask=attention_mask, completion_mask=c_mask, old_log_probs=old_log_probs,
        ref_log_probs=ref_log_probs, formatted_completions=formatted_completions, repeated_prompts=repeated_prompts,
        repeated_answers=repeated_answers, logits_to_keep=logits_to_keep, batch_size=len(prompts), n_generations=config.n_generations,
        policy_duration=policy_duration, reference_duration=reference_duration,
    )

def grpo_loss(
    model: PolicyModel, rollouts: RolloutData, reward_function: RewardFn,
    beta: float, epsilon: float, algo_name: str,max_completion_length: int,
) -> Tuple[torch.Tensor, float]:

    current_log_probs = compute_log_probs(model,
                                          input_ids=rollouts["input_ids"],
                                          attention_mask=rollouts["attention_mask"],
                                          logits_to_keep=rollouts["logits_to_keep"])
    ratio = torch.exp(current_log_probs - rollouts["old_log_probs"])

    rewards = torch.tensor(
        reward_function(rollouts["repeated_prompts"],
                        rollouts["formatted_completions"],
                        rollouts["repeated_answers"]),
        dtype=torch.float32,
        device=current_log_probs.device,
    )

    rewards_matrix = rewards.view(rollouts["batch_size"], rollouts["n_generations"])
    avg_reward = rewards_matrix.mean().item()

    mean_rewards = rewards_matrix.mean(dim=1).repeat_interleave(rollouts["n_generations"])
    if algo_name == "dr_grpo":
        # Dr.GRPO: remove std normalization; use only (r - mean) as the advantage for all tokens.
        advantages = (rewards_matrix.view(-1) - mean_rewards).unsqueeze(1)
    else:
        std_rewards = rewards_matrix.std(dim=1).repeat_interleave(rollouts["n_generations"])
        advantages = ((rewards_matrix.view(-1) - mean_rewards) / (std_rewards + 1e-4)).unsqueeze(1)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    surrogate = torch.min(surr1, surr2)

    # Reverse-KL approximation (stable around 0): exp(Δ) - Δ - 1, where Δ = ref - current.
    delta = rollouts["ref_log_probs"] - current_log_probs
    kl = torch.exp(delta) - delta - 1.0

    per_token = surrogate - beta * kl

    cmask = rollouts["completion_mask"].to(per_token.dtype)

    if algo_name == "grpo":
        token_counts = cmask.sum(dim=1).clamp_min(1)
        loss = -((per_token * cmask).sum(dim=1) / token_counts).mean()
    elif algo_name == "dr_grpo":
        # Dr.GRPO: remove sequence-length normalization. Aggregate at token-level without dividing by per-sample token counts.
        denom = max(per_token.size(0), 1)
        denom_tensor = per_token.new_tensor(float(denom))
        loss = -(per_token * cmask).sum() / denom_tensor
    else:
        raise ValueError(f"Unknown algo_name '{algo_name}' (expected 'grpo' or 'dr_grpo').")

    return loss, avg_reward

def unwrap(module_like: PolicyModel) -> PreTrainedModel:
    return cast(PreTrainedModel, module_like.module)

def get_lr(step: int, config: Config) -> float:
    total_updates = config.steps_per_iteration * config.mu
    warmup_steps = config.lr_warmup_steps
    cooldown_start_step = int(total_updates * config.lr_cooldown_frac)

    if warmup_steps > 0 and step <= warmup_steps:
        return (step / float(warmup_steps)) * config.learning_rate
    if step < cooldown_start_step or cooldown_start_step >= total_updates:
        return config.learning_rate
    progress = (step - cooldown_start_step) / float(total_updates - cooldown_start_step)
    return (1.0 - progress) * config.learning_rate

def train_iterations(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, train_data: Sequence[DatasetExample], config: Config,
) -> PreTrainedModel:
    train_start, total_tokens = time.perf_counter(), 0
    policy_rollout_time_total, reference_rollout_time_total, policy_update_time_total = 0, 0, 0

    reward_fn = dataset_configs[config.dataset]["reward_fn"]
    policy_model: PolicyModel = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[config.device_id], gradient_as_bucket_view=False, #TODO: check if gradient_as_bucket_view=True works
    )  # grads are kept in reducer buckets avoiding 2x memory usage

    update_count = 0

    # Snapshot policy -> reference (no grads).
    ref = copy.deepcopy(unwrap(policy_model)).to(torch.device(config.device_name)) # type: ignore[arg-type]
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.learning_rate, fused=torch.cuda.is_available())  # type: ignore[arg-type]

    policy_model.train()

    for step in range(1, config.steps_per_iteration + 1):  # number of policy updates
        batch = random.sample(train_data, config.batch_size)
        # rollout with policy
        with torch.no_grad():
            policy = unwrap(policy_model)

            # Temporarily switch to eval + enable KV cache for faster decoding.
            was_training = policy.training
            prev_cache = getattr(policy.config, "use_cache", False)
            policy.eval()
            policy.config.use_cache = True

            rollouts = gen_rollouts(policy, ref, tokenizer, batch, config)

            # Restore training state.
            policy.config.use_cache = prev_cache
            if was_training:
                policy.train()

        for update_i in range(1, config.mu + 1):  # number of GRPO updates per batch
            update_count += 1
            lr = get_lr(update_count, config=config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            update_start = time.perf_counter()
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()
            with autocast_ctx:
                loss, avg_reward = grpo_loss(
                    policy_model, rollouts, reward_fn, config.beta, config.epsilon, config.grpo_variant,
                    config.max_completion_length,
                )
            optimizer.zero_grad()  # TODO: check if this would work because gradient_as_bucket_view=True
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=0.1)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # TODO: check if this would work because gradient_as_bucket_view=True
            update_duration = time.perf_counter() - update_start

            update_tokens = rollouts["completion_mask"].sum().item()
            total_tokens += update_tokens
            policy_update_time_total += update_duration
            policy_rollout_time_total += rollouts["policy_duration"]
            reference_rollout_time_total += rollouts["reference_duration"]
            train_elapsed = time.perf_counter() - train_start

            log_metrics({
                "train/step": step,
                "train/update": update_count,
                "train/update_i ": update_i,
                "train/batch_size": rollouts["batch_size"],
                "train/n_generations": rollouts["n_generations"],
                "train/avg_completion_tokens": rollouts["completion_mask"].sum(dim=1).float().mean().item(),
                "train/loss": loss.item(),
                "train/reward": avg_reward,
                "train/grad_norm": total_norm.item(),
                "train/lr": lr,
                "train/update_count": update_count,
                "train/update_tokens": update_tokens,
                "train/tokens_throughput": update_tokens / max(update_duration, 1e-9),
                "train/policy_rollout_time_s": rollouts["policy_duration"],
                "train/reference_rollout_time_s": rollouts["reference_duration"],
                "train/policy_update_time_s": update_duration,
                "train/train_elapsed_s": train_elapsed,
                "train/total_tokens": total_tokens,
                "train/policy_rollout_time_total_s": policy_rollout_time_total,
                "train/reference_rollout_time_total_s": reference_rollout_time_total,
                "train/policy_update_time_total_s": policy_update_time_total,
            })

    return unwrap(policy_model)

def enable_activation_checkpointing(model: PreTrainedModel) -> PreTrainedModel:
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if hasattr(model, "enable_input_require_grads"): # Some HF models expose a helper; if not, attach a small hook
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, inp, out):
            out.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    return model

def train(config: Config) -> None:
    log_message(f"Downloading model {config.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, dtype="auto", device_map={"": config.device_id}, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = enable_activation_checkpointing(model) # needed for > 1.5B models on 80GB GPUs
    log_message(f"Model download complete. dtype is {str(next(model.parameters()).dtype)}")

    # validate and fix tokenizer
    if tokenizer.eos_token is None:
        raise ValueError("`tokenizer.eos_token` must be provided, but got None.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        log_message(f"Forcing tokenizer.pad_token=tokenizer.eos_token", level=logging.WARNING)
    tokenizer.padding_side = "left"  # centralized padding side

    # prepare datasets
    dataset_cfg = dataset_configs[config.dataset]
    train_data = prepare_dataset(tokenizer, config.model_mode, dataset_cfg, dataset_cfg["train_split"])
    random.shuffle(train_data)
    eval_data = prepare_dataset(tokenizer, config.model_mode, dataset_cfg, dataset_cfg["test_split"])
    eval_data = eval_data[:config.n_eval] if config.n_eval > 0 else eval_data
    log_summary({"dataset/train_examples": len(train_data), "dataset/eval_examples": len(eval_data)})

    pre_rl_metrics = evaluate_model(model=model, tokenizer=tokenizer, eval_examples=eval_data, config=config)
    log_summary({'pre_rl/'+k: v for k, v in pre_rl_metrics.items()})

    model = train_iterations(model=model, tokenizer=tokenizer, train_data=train_data, config=config)

    post_rl_metrics = evaluate_model(model=model, tokenizer=tokenizer, eval_examples=eval_data, config=config)
    log_summary({'post_rl/'+k: v for k, v in post_rl_metrics.items()})

    artifact_dir = Path(config.run_dir) / "finetuned_model"
    model.save_pretrained(artifact_dir)
    tokenizer.save_pretrained(artifact_dir)
    log_message(f"Saved fine-tuned model to {artifact_dir}")

def main() -> None:
    config = parse_dc(Config)

    base_dir = Path(config.out_dir or os.environ.get("OUT_DIR") or "~/out_dir").expanduser().resolve()
    run_name = config.run_name or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = (base_dir / "nano_rlvr" / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    config = replace(config, out_dir=str(base_dir), run_name=run_name, run_dir=str(run_dir))

    configure_logging(run_dir, config)

    device_name, device_id = init_torch(seed=config.seed)
    config = replace(config, device_name=device_name, device_id=device_id)

    if dist.get_rank() == 0:
        config_path = run_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as config_file:
            json.dump(asdict(config), config_file, indent=2)
        log_message(f"Configuration saved to {config_path}")

    train(config)

    global _wandb_run
    if _wandb_run:
        try:
            _wandb_run.finish()
            _wandb_run = None
        except Exception as exc:  # pragma: no cover
            logger.debug("wandb finish failed: %s", exc)
    log_message("Run complete.")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()