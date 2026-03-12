#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

PRESETS = {
    "baseline-adamw": {
        "launcher": "volcano_owt10b_baseline_adamw.sh",
        "description": "Keller-style 10B baseline that calls train.py with the baseline config.",
        "entrypoint": "train.py",
        "config": "configs/train_gpt2/openwebtext_tokens10b_baseline.yaml",
        "project_name": "nanugpt-owt10k",
        "run_name": "owt-10b-baseline",
        "run_description": "Baseline: Karpathy's model with WSD sched and 3X LR, 10.7B tokens",
        "env": {
            "JOB_NAME": "gpt-std",
            "TRANSFER_VARS": "DATA_ROOT WANDB_API_KEY WANDB_HOST",
            "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING": "1",
            "TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS": "1",
        },
    },
    "karpathy-classic": {
        "launcher": "volcano_karpathy_classic_owt10b.sh",
        "description": "Karpathy classic OpenWebText 10B-token run.",
        "entrypoint": "train.py",
        "config": "configs/train_gpt2/openwebtext_tokens10b_karpathy_classic.yaml",
        "project_name": "nanugpt-owt10k",
        "run_name": "owt-10b-karpathy-classic",
        "run_description": "Baseline: Karpathy classic 10.666B tokens",
        "env": {
            "JOB_NAME": "gpt-std",
            "TRANSFER_VARS": "DATA_ROOT WANDB_API_KEY WANDB_HOST",
            "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING": "1",
            "TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS": "1",
        },
    },
    "karpathy-llmc": {
        "launcher": "volcano_karpathy_llmc_owt10b.sh",
        "description": "Karpathy llm.c OpenWebText 10B-token run.",
        "entrypoint": "train.py",
        "config": "configs/train_gpt2/openwebtext_tokens10b_karpathy_llmc.yaml",
        "project_name": "nanugpt-owt10k",
        "run_name": "owt-10b-karpathy-llmc",
        "run_description": "Baseline: Karpathy llm.c 10.666B tokens",
        "env": {
            "JOB_NAME": "gpt-std",
            "TRANSFER_VARS": "DATA_ROOT WANDB_API_KEY WANDB_HOST",
            "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING": "1",
            "TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS": "1",
        },
    },
    "baseline-muon": {
        "launcher": "volcano_owt10b_baseline_muon.sh",
        "description": "Keller muon variant using the alternate training entrypoint.",
        "entrypoint": "scripts/alt_training/keller_train_gpt2_muon.py",
        "config": None,
        "project_name": None,
        "run_name": None,
        "run_description": None,
        "env": {
            "JOB_NAME": "gpt-std",
            "TRANSFER_VARS": "DATA_ROOT WANDB_API_KEY WANDB_HOST",
            "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING": "0",
            "TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS": "0",
        },
    },
}

PREFERRED_ENV_ORDER = [
    "JOB_NAME",
    "NODES",
    "GPUS_PER_NODE",
    "NPROC_PER_NODE",
    "DATA_ROOT",
    "OUT_DIR",
    "TRANSFER_VARS",
    "VOLCANO_NAMESPACE",
    "VOLCANO_DATA_PVC_NAME",
    "CONTAINER_IMAGE_PATH",
    "ENV_SETUP_SCRIPT",
    "USE_TORCHRUN",
    "INSTALL_PACKAGE",
    "UPDATE_PYTHONPATH",
    "SOURCE_DIR",
    "LOCAL_OUT_DIR",
    "CPU_REQUESTS",
    "MEMORY_REQUESTS",
    "RDMA_REQUESTS",
    "MEMORY_SIZE_LIMIT",
    "PRIORITY",
    "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING",
    "TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS",
]


def parse_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description="Prepare or submit nanugpt Volcano jobs through scripts/volcano/vsubmit.sh.",
    )
    parser.add_argument("--repo-root", help="Path to the nanugpt repo root.")
    parser.add_argument("--list-presets", action="store_true", help="List supported repo presets and exit.")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Preset mapped from a repo launcher.")
    parser.add_argument("--entrypoint", help="Training entrypoint relative to the repo root.")
    parser.add_argument("--config", help="Config path relative to the repo root.")
    parser.add_argument("--job-name", help="JOB_NAME passed to vsubmit.sh.")
    parser.add_argument("--nodes", type=int, help="NODES passed to vsubmit.sh.")
    parser.add_argument("--gpus-per-node", type=int, help="GPUS_PER_NODE passed to vsubmit.sh.")
    parser.add_argument("--nproc-per-node", type=int, help="NPROC_PER_NODE passed to vsubmit.sh.")
    parser.add_argument("--project-name", help="Value for --general.project_name.")
    parser.add_argument("--run-name", help="Value for --general.run_name.")
    parser.add_argument("--run-description", help="Value for --general.run_description.")
    parser.add_argument("--data-root", help="DATA_ROOT exported before submission.")
    parser.add_argument("--out-dir", help="OUT_DIR exported before submission.")
    parser.add_argument("--source-dir", help="SOURCE_DIR exported before submission.")
    parser.add_argument("--local-out-dir", help="LOCAL_OUT_DIR exported before submission.")
    parser.add_argument("--transfer-vars", help="TRANSFER_VARS value.")
    parser.add_argument(
        "--add-transfer-var",
        action="append",
        default=[],
        help="Additional variable name to append to TRANSFER_VARS.",
    )
    parser.add_argument("--container-image", help="CONTAINER_IMAGE_PATH exported before submission.")
    parser.add_argument("--env-setup-script", help="ENV_SETUP_SCRIPT exported before submission.")
    parser.add_argument("--volcano-namespace", help="VOLCANO_NAMESPACE exported before submission.")
    parser.add_argument("--volcano-data-pvc-name", help="VOLCANO_DATA_PVC_NAME exported before submission.")
    parser.add_argument("--priority", help="PriorityClassName value used by the job template.")
    parser.add_argument("--cpu-requests", help="CPU_REQUESTS exported before submission.")
    parser.add_argument("--memory-requests", help="MEMORY_REQUESTS exported before submission.")
    parser.add_argument("--rdma-requests", help="RDMA_REQUESTS exported before submission.")
    parser.add_argument("--memory-size-limit", help="MEMORY_SIZE_LIMIT exported before submission.")
    parser.add_argument("--use-torchrun", choices=["0", "1"], help="USE_TORCHRUN override.")
    parser.add_argument("--install-package", choices=["0", "1"], help="INSTALL_PACKAGE override.")
    parser.add_argument("--update-pythonpath", choices=["0", "1"], help="UPDATE_PYTHONPATH override.")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra environment assignment in KEY=VALUE format.",
    )
    parser.add_argument("--json", action="store_true", help="Emit dry-run details as JSON.")
    parser.add_argument("--submit", action="store_true", help="Execute the rendered vsubmit.sh command.")
    parser.add_argument(
        "training_args",
        nargs=argparse.REMAINDER,
        help="Extra args appended to the training command. Put them after --.",
    )
    args = parser.parse_args()
    if args.training_args and args.training_args[0] == "--":
        args.training_args = args.training_args[1:]
    return args, parser


def list_presets() -> int:
    for name in sorted(PRESETS):
        preset = PRESETS[name]
        config = preset["config"] or "<none>"
        print(f"{name}: {preset['description']}")
        print(f"  launcher: {preset['launcher']}")
        print(f"  entrypoint: {preset['entrypoint']}")
        print(f"  config: {config}")
    return 0


def find_repo_root(start: Path) -> Path | None:
    for candidate in [start, *start.parents]:
        if (candidate / "scripts/volcano/vsubmit.sh").is_file() and (candidate / "train.py").is_file():
            return candidate.resolve()
    return None


def resolve_repo_root(repo_root_arg: str | None) -> Path:
    if repo_root_arg:
        repo_root = Path(repo_root_arg).expanduser().resolve()
        if not (repo_root / "scripts/volcano/vsubmit.sh").is_file():
            raise SystemExit(f"repo root does not contain scripts/volcano/vsubmit.sh: {repo_root}")
        return repo_root
    repo_root = find_repo_root(Path.cwd().resolve())
    if repo_root is None:
        raise SystemExit(
            "could not locate the nanugpt repo root from the current directory; run inside the repo or pass --repo-root"
        )
    return repo_root


def parse_env_assignments(items: list[str]) -> dict[str, str]:
    env = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"invalid --env value '{item}'; expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"invalid --env value '{item}'; key is empty")
        env[key] = value
    return env


def merge_transfer_vars(existing: str | None, additions: list[str]) -> str | None:
    values = []
    seen = set()
    for item in (existing or "").split():
        if item not in seen:
            values.append(item)
            seen.add(item)
    for item in additions:
        if item not in seen:
            values.append(item)
            seen.add(item)
    return " ".join(values) if values else None


def resolve_file(repo_root: Path, raw_path: str | None, label: str, required: bool = False) -> str | None:
    if not raw_path:
        if required:
            raise SystemExit(f"{label} is required")
        return None
    original = Path(raw_path).expanduser()
    candidate = original
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    if not candidate.exists():
        raise SystemExit(f"{label} does not exist: {raw_path}")
    try:
        return str(candidate.resolve().relative_to(repo_root))
    except ValueError:
        return str(candidate.resolve())


def build_start_command(args: argparse.Namespace, repo_root: Path) -> list[str]:
    preset = PRESETS.get(args.preset, {})
    entrypoint = args.entrypoint or preset.get("entrypoint")
    entrypoint = resolve_file(repo_root, entrypoint, "entrypoint", required=True)
    config = args.config if args.config is not None else preset.get("config")
    if config:
        config = resolve_file(repo_root, config, "config")

    project_name = args.project_name if args.project_name is not None else preset.get("project_name")
    run_name = args.run_name if args.run_name is not None else preset.get("run_name")
    run_description = (
        args.run_description if args.run_description is not None else preset.get("run_description")
    )

    argv = [entrypoint]
    if config:
        argv.append(config)
    if project_name:
        argv.extend(["--general.project_name", project_name])
    if run_name:
        argv.extend(["--general.run_name", run_name])
    if run_description:
        argv.extend(["--general.run_description", run_description])
    argv.extend(args.training_args)
    return argv


def preferred_env_items(env_updates: dict[str, str]) -> list[tuple[str, str]]:
    order = {key: index for index, key in enumerate(PREFERRED_ENV_ORDER)}
    return sorted(env_updates.items(), key=lambda item: (order.get(item[0], len(order)), item[0]))


def render_shell_command(env_updates: dict[str, str], start_argv: list[str]) -> str:
    env_part = " ".join(f"{key}={shlex.quote(value)}" for key, value in preferred_env_items(env_updates))
    argv_part = " ".join(shlex.quote(token) for token in ["bash", "scripts/volcano/vsubmit.sh", *start_argv])
    return f"{env_part} {argv_part}".strip()


def build_env_updates(args: argparse.Namespace) -> dict[str, str]:
    preset = PRESETS.get(args.preset, {})
    env_updates = dict(preset.get("env", {}))

    structured_updates = {
        "JOB_NAME": args.job_name,
        "NODES": str(args.nodes) if args.nodes is not None else None,
        "GPUS_PER_NODE": str(args.gpus_per_node) if args.gpus_per_node is not None else None,
        "NPROC_PER_NODE": str(args.nproc_per_node) if args.nproc_per_node is not None else None,
        "DATA_ROOT": args.data_root,
        "OUT_DIR": args.out_dir,
        "SOURCE_DIR": args.source_dir,
        "LOCAL_OUT_DIR": args.local_out_dir,
        "TRANSFER_VARS": args.transfer_vars,
        "CONTAINER_IMAGE_PATH": args.container_image,
        "ENV_SETUP_SCRIPT": args.env_setup_script,
        "VOLCANO_NAMESPACE": args.volcano_namespace,
        "VOLCANO_DATA_PVC_NAME": args.volcano_data_pvc_name,
        "PRIORITY": args.priority,
        "CPU_REQUESTS": args.cpu_requests,
        "MEMORY_REQUESTS": args.memory_requests,
        "RDMA_REQUESTS": args.rdma_requests,
        "MEMORY_SIZE_LIMIT": args.memory_size_limit,
        "USE_TORCHRUN": args.use_torchrun,
        "INSTALL_PACKAGE": args.install_package,
        "UPDATE_PYTHONPATH": args.update_pythonpath,
    }
    for key, value in structured_updates.items():
        if value is not None:
            env_updates[key] = value

    merged_transfer_vars = merge_transfer_vars(env_updates.get("TRANSFER_VARS"), args.add_transfer_var)
    if merged_transfer_vars is not None:
        env_updates["TRANSFER_VARS"] = merged_transfer_vars

    env_updates.update(parse_env_assignments(args.env))
    return env_updates


def validate_submit_prereqs(env_updates: dict[str, str], effective_env: dict[str, str]) -> list[str]:
    warnings = []
    transfer_vars = env_updates.get("TRANSFER_VARS", "")
    for key in transfer_vars.split():
        if not effective_env.get(key):
            warnings.append(
                f"transfer var {key} is not set in the current environment; it will not be exported into the container"
            )
    for key in ("VOLCANO_NAMESPACE", "VOLCANO_DATA_PVC_NAME"):
        if not effective_env.get(key):
            warnings.append(f"{key} is not set")
    return warnings


def main() -> int:
    args, parser = parse_args()

    if args.list_presets:
        return list_presets()

    if args.json and args.submit:
        raise SystemExit("--json is only supported for dry-run output")

    if not args.preset and not args.entrypoint:
        parser.error("provide either --preset or --entrypoint")

    repo_root = resolve_repo_root(args.repo_root)
    start_argv = build_start_command(args, repo_root)
    env_updates = build_env_updates(args)

    if "JOB_NAME" not in env_updates:
        raise SystemExit("JOB_NAME is required; pass --job-name or set it through --env JOB_NAME=...")

    effective_env = os.environ.copy()
    effective_env.update(env_updates)
    warnings = validate_submit_prereqs(env_updates, effective_env)

    if args.submit:
        if shutil.which("kubectl") is None:
            raise SystemExit("kubectl is not available in PATH")
        missing = [key for key in ("VOLCANO_NAMESPACE", "VOLCANO_DATA_PVC_NAME") if not effective_env.get(key)]
        if missing:
            raise SystemExit("cannot submit; missing required cluster variables: " + ", ".join(missing))

    command = render_shell_command(env_updates, start_argv)

    summary = {
        "mode": "submit" if args.submit else "dry-run",
        "repo_root": str(repo_root),
        "preset": args.preset,
        "start_command": start_argv,
        "env_updates": dict(preferred_env_items(env_updates)),
        "warnings": warnings,
        "command": command,
    }

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=False))
        return 0

    print(f"Mode: {summary['mode']}")
    print(f"Repo root: {summary['repo_root']}")
    print(f"Preset: {summary['preset'] or '<custom>'}")
    print(f"Start command: {' '.join(shlex.quote(token) for token in start_argv)}")
    print("Rendered vsubmit command:")
    print(command)
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")

    if not args.submit:
        return 0

    print("Submitting with scripts/volcano/vsubmit.sh ...")
    result = subprocess.run(
        ["bash", "scripts/volcano/vsubmit.sh", *start_argv],
        cwd=repo_root,
        env=effective_env,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
