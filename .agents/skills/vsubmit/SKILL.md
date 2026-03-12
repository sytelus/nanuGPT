---
name: vsubmit
description: Build, explain, dry-run, and submit nanugpt training jobs to this repo's Kubernetes Volcano cluster via scripts/volcano/vsubmit.sh. Use when the user wants to launch or modify a Volcano job from natural language, reuse one of the repo's Volcano launcher presets, generate the exact submit command, or troubleshoot the required Volcano submission variables for nanugpt.
---

# Vsubmit

Use this skill only for the `nanugpt` Volcano workflow in the current repo.

## Workflow

1. Distinguish between `explain`, `prepare`, and `submit`.
2. Prefer dry-run output unless the user explicitly asks to submit a job.
3. Read [references/volcano-workflow.md](references/volcano-workflow.md) when you need preset names, required environment variables, or repo-specific defaults.
4. Prefer the helper script for command construction:

If the skill is repo-local, the script path from the repo root is `.agents/skills/vsubmit/scripts/vsubmit_job.py`. If the skill is globally installed, use the same script under `~/.codex/skills/vsubmit/scripts/`.

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py --list-presets
python .agents/skills/vsubmit/scripts/vsubmit_job.py --preset baseline-adamw
python .agents/skills/vsubmit/scripts/vsubmit_job.py --preset baseline-adamw --job-name my-exp --nodes 2 --submit
```

5. Show the exact rendered `vsubmit.sh` command before execution. Briefly explain only non-default choices.

## Command Building

- Use a preset when the user's request matches one of the repo launchers: `baseline-adamw`, `karpathy-classic`, `karpathy-llmc`, or `baseline-muon`.
- For custom runs, pass `--entrypoint` and optionally `--config`.
- Put extra training overrides after `--` so they are forwarded verbatim to the training command.
- Use structured flags for common submission controls: `--job-name`, `--nodes`, `--gpus-per-node`, `--nproc-per-node`, `--project-name`, `--run-name`, `--run-description`, `--data-root`, `--out-dir`, `--container-image`, `--volcano-namespace`, `--volcano-data-pvc-name`.
- Use `--env KEY=VALUE` for other `vsubmit.sh` environment variables.
- Use `--add-transfer-var NAME` when a new local environment variable must be exported into the cluster container.

## Safety

- Do not silently invent config paths, preset names, or cluster settings.
- Do not submit if the user only asked for a command, example, or review.
- When `VOLCANO_NAMESPACE` or `VOLCANO_DATA_PVC_NAME` is missing, surface that clearly before any submit attempt.
- Treat helper warnings about unset transfer variables as important; mention them in the response.
- Run submission commands from the repo root or pass `--repo-root` explicitly if Codex is not already there.

## Examples

Dry run the baseline launcher:

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py --preset baseline-adamw
```

Dry run with overrides:

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py \
  --preset karpathy-llmc \
  --job-name gpt-llmc-2n \
  --nodes 2 \
  --project-name nanugpt-owt10k \
  --run-name owt-10b-karpathy-2n \
  --run-description "Karpathy llm.c preset on 2 nodes" \
  -- --optimizer.lr 0.0006
```

Submit a custom config:

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py \
  --entrypoint train.py \
  --config configs/train_gpt2/openwebtext_tokens10b_keller_adamw.yaml \
  --job-name keller-adamw \
  --nodes 1 \
  --volcano-namespace my-namespace \
  --volcano-data-pvc-name my-pvc \
  --submit
```
