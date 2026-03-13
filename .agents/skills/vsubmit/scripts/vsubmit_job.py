#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

TEMPLATE_ENV_KEYS = {
    "JOB_NAME",
    "PROJECT_NAME",
    "NODES",
    "GPUS_PER_NODE",
    "NPROC_PER_NODE",
    "DATA_ROOT",
    "OUT_DIR",
    "SOURCE_DIR",
    "LOCAL_OUT_DIR",
    "TRANSFER_VARS",
    "CONTAINER_IMAGE_PATH",
    "ENV_SETUP_SCRIPT",
    "USE_TORCHRUN",
    "INSTALL_PACKAGE",
    "UPDATE_PYTHONPATH",
    "CPU_REQUESTS",
    "MEMORY_REQUESTS",
    "RDMA_REQUESTS",
    "MEMORY_SIZE_LIMIT",
    "PRIORITY",
    "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING",
    "TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS",
}

PREFERRED_ENV_ORDER = [
    "JOB_NAME",
    "PROJECT_NAME",
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

CRITICAL_RENDER_KEYS = ("JOB_NAME", "PROJECT_NAME")
CRITICAL_SUBMIT_KEYS = ("JOB_NAME", "PROJECT_NAME", "VOLCANO_NAMESPACE", "VOLCANO_DATA_PVC_NAME")
OBSERVED_ENV_KEYS = tuple(dict.fromkeys([*PREFERRED_ENV_ORDER, *CRITICAL_SUBMIT_KEYS]))
VAR_REF_RE = re.compile(r"\$(\w+)|\$\{([^}]+)\}")


@dataclass
class TemplateSpec:
    path: str
    context_vars: dict[str, str]
    env_defaults: dict[str, str]
    start_command: list[str]


def parse_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description="Prepare or submit Volcano jobs through scripts/volcano/vsubmit.sh using a launcher template.",
    )
    parser.add_argument("--repo-root", help="Path to the repo root that contains scripts/volcano/vsubmit.sh.")
    parser.add_argument(
        "--template",
        help="Launcher shell script used to infer defaults. Defaults to volcano_owt10b_baseline_adamw.sh if present.",
    )
    parser.add_argument(
        "--start-command",
        help="Full shell command to run inside the container. If omitted, the helper uses the template command.",
    )
    parser.add_argument("--job-name", help="JOB_NAME passed to vsubmit.sh.")
    parser.add_argument("--workstream", help="PROJECT_NAME used by vsubmit.sh and volcano_job.yaml.")
    parser.add_argument("--nodes", type=int, help="NODES passed to vsubmit.sh.")
    parser.add_argument("--gpus-per-node", type=int, help="GPUS_PER_NODE passed to vsubmit.sh.")
    parser.add_argument("--nproc-per-node", type=int, help="NPROC_PER_NODE passed to vsubmit.sh.")
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
    parser.add_argument(
        "--append-arg",
        action="append",
        default=[],
        help="Extra command argument appended after the inferred or explicit start command.",
    )
    parser.add_argument("--json", action="store_true", help="Emit details as JSON.")
    parser.add_argument("--submit", action="store_true", help="Execute the rendered vsubmit.sh command.")
    args = parser.parse_args()
    return args, parser


def find_repo_root(start: Path) -> Path | None:
    for candidate in [start, *start.parents]:
        if (candidate / "scripts/volcano/vsubmit.sh").is_file():
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
            "could not locate a repo root containing scripts/volcano/vsubmit.sh; run inside the repo or pass --repo-root"
        )
    return repo_root


def resolve_template_path(repo_root: Path, template_arg: str | None) -> Path | None:
    default_template = repo_root / "volcano_owt10b_baseline_adamw.sh"
    raw = template_arg
    if raw is None and default_template.is_file():
        raw = str(default_template)
    if raw is None:
        return None
    template_path = Path(raw).expanduser()
    if not template_path.is_absolute():
        template_path = repo_root / template_path
    if not template_path.is_file():
        raise SystemExit(f"template does not exist: {raw}")
    return template_path.resolve()


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


def shell_unquote(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def join_logical_lines(text: str) -> list[str]:
    logical_lines = []
    current = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if current:
            current = f"{current} {line}"
        else:
            current = line
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        logical_lines.append(current)
        current = ""
    if current:
        logical_lines.append(current)
    return logical_lines


def expand_vars(text: str, variables: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        return variables.get(name, match.group(0))

    return VAR_REF_RE.sub(replace, text)


def parse_template(template_path: Path) -> TemplateSpec:
    logical_lines = join_logical_lines(template_path.read_text())
    context_vars: dict[str, str] = {}
    invocation_tokens: list[str] | None = None

    export_re = re.compile(r"^export\s+([A-Za-z_][A-Za-z0-9_]*)=(.+)$")
    assign_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.+)$")

    for line in logical_lines:
        export_match = export_re.match(line)
        if export_match:
            key, value = export_match.groups()
            context_vars[key] = shell_unquote(value)
            continue
        if "vsubmit.sh" in line:
            invocation_tokens = shlex.split(line)

    if invocation_tokens is None:
        raise SystemExit(f"template does not contain a vsubmit.sh invocation: {template_path}")

    prefix_assignments: dict[str, str] = {}
    start_index = None
    for index, token in enumerate(invocation_tokens):
        if token == "bash":
            start_index = index
            break
        assign_match = assign_re.match(token)
        if assign_match:
            key, value = assign_match.groups()
            prefix_assignments[key] = shell_unquote(value)

    if start_index is None:
        raise SystemExit(f"template invocation does not call bash ... vsubmit.sh: {template_path}")

    script_index = None
    for index in range(start_index + 1, len(invocation_tokens)):
        if "vsubmit.sh" in invocation_tokens[index]:
            script_index = index
            break
    if script_index is None:
        raise SystemExit(f"template invocation does not include the vsubmit.sh path: {template_path}")

    context_vars = {**context_vars, **prefix_assignments}
    start_command = [expand_vars(token, context_vars) for token in invocation_tokens[script_index + 1 :]]
    env_defaults = {
        key: expand_vars(value, context_vars)
        for key, value in context_vars.items()
        if key in TEMPLATE_ENV_KEYS
    }

    return TemplateSpec(
        path=str(template_path),
        context_vars=context_vars,
        env_defaults=env_defaults,
        start_command=start_command,
    )


def preferred_env_items(env_updates: dict[str, str]) -> list[tuple[str, str]]:
    order = {key: index for index, key in enumerate(PREFERRED_ENV_ORDER)}
    return sorted(env_updates.items(), key=lambda item: (order.get(item[0], len(order)), item[0]))


def render_shell_command(env_updates: dict[str, str], start_argv: list[str]) -> str:
    env_part = " ".join(f"{key}={shlex.quote(value)}" for key, value in preferred_env_items(env_updates))
    argv_part = " ".join(shlex.quote(token) for token in ["bash", "scripts/volcano/vsubmit.sh", *start_argv])
    return f"{env_part} {argv_part}".strip()


def build_env_updates(args: argparse.Namespace, template: TemplateSpec | None) -> dict[str, str]:
    env_updates = dict(template.env_defaults if template else {})
    structured_updates = {
        "JOB_NAME": args.job_name,
        "PROJECT_NAME": args.workstream,
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


def build_start_command(args: argparse.Namespace, template: TemplateSpec | None) -> list[str]:
    if args.start_command:
        start_argv = shlex.split(args.start_command)
    elif template:
        start_argv = list(template.start_command)
    else:
        raise SystemExit("provide --start-command or use a template that defines one")

    start_argv.extend(args.append_arg)
    if not start_argv:
        raise SystemExit("start command is empty")
    return start_argv


def find_missing_values(effective_env: dict[str, str], required_keys: tuple[str, ...]) -> list[str]:
    return [key for key in required_keys if not effective_env.get(key)]


def collect_inherited_env(env_updates: dict[str, str], effective_env: dict[str, str]) -> dict[str, str]:
    inherited = {}
    for key in OBSERVED_ENV_KEYS:
        if key in env_updates:
            continue
        value = effective_env.get(key)
        if value:
            inherited[key] = value
    return inherited


def validate_submit_prereqs(env_updates: dict[str, str], effective_env: dict[str, str]) -> list[str]:
    warnings = []
    transfer_vars = env_updates.get("TRANSFER_VARS", "")
    for key in transfer_vars.split():
        if not effective_env.get(key):
            warnings.append(
                f"transfer var {key} is not set in the current environment; it will not be exported into the container"
            )
    return warnings


def main() -> int:
    args, _parser = parse_args()

    if args.json and args.submit:
        raise SystemExit("--json is only supported for dry-run output")

    repo_root = resolve_repo_root(args.repo_root)
    template_path = resolve_template_path(repo_root, args.template)
    template = parse_template(template_path) if template_path else None
    start_argv = build_start_command(args, template)
    env_updates = build_env_updates(args, template)

    effective_env = os.environ.copy()
    effective_env.update(env_updates)
    render_missing = find_missing_values(effective_env, CRITICAL_RENDER_KEYS)
    submit_missing = find_missing_values(effective_env, CRITICAL_SUBMIT_KEYS)
    inherited_env = collect_inherited_env(env_updates, effective_env)
    warnings = validate_submit_prereqs(env_updates, effective_env)

    if args.submit:
        if shutil.which("kubectl") is None:
            raise SystemExit("kubectl is not available in PATH")
        if submit_missing:
            raise SystemExit("cannot submit; missing required values: " + ", ".join(submit_missing))

    command = render_shell_command(env_updates, start_argv)
    summary = {
        "mode": "submit" if args.submit else "dry-run",
        "repo_root": str(repo_root),
        "template": template.path if template else None,
        "template_env": dict(preferred_env_items(template.env_defaults)) if template else {},
        "template_start_command": template.start_command if template else [],
        "start_command": start_argv,
        "env_updates": dict(preferred_env_items(env_updates)),
        "inherited_env": dict(preferred_env_items(inherited_env)),
        "missing_for_render": render_missing,
        "missing_for_submit": submit_missing,
        "warnings": warnings,
        "command": command,
    }

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=False))
        return 0

    print(f"Mode: {summary['mode']}")
    print(f"Repo root: {summary['repo_root']}")
    print(f"Template: {summary['template'] or '<none>'}")
    if template:
        print("Template start command:")
        print(" ".join(shlex.quote(token) for token in summary["template_start_command"]))
    print("Start command:")
    print(" ".join(shlex.quote(token) for token in summary["start_command"]))
    if summary["env_updates"]:
        print("Environment:")
        for key, value in preferred_env_items(summary["env_updates"]):
            print(f"- {key}={value}")
    if summary["inherited_env"]:
        print("Inherited environment from the current shell:")
        for key, value in preferred_env_items(summary["inherited_env"]):
            print(f"- {key}={value}")
    print("Rendered vsubmit command:")
    print(command)
    if summary["missing_for_render"]:
        print("Missing values that would cause vsubmit.sh to prompt or fail:")
        for key in summary["missing_for_render"]:
            print(f"- {key}")
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
