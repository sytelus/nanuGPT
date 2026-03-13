---
name: vsubmit
description: Build, review, dry-run, and submit Kubernetes Volcano jobs through scripts/volcano/vsubmit.sh. Use when the user wants to turn a natural-language job request into the exact submission command, infer defaults from launcher scripts such as volcano_owt10b_baseline_adamw.sh, adjust env vars or training overrides, or submit only after an explicit confirmation step.
---

# Vsubmit

Use this skill whenever the repo contains `scripts/volcano/vsubmit.sh` and the user wants a Volcano command, a dry run, or a real submission.

## Workflow

1. Determine the intent: explain, prepare, dry-run, or submit.
2. Start from a launcher template instead of hard-coded presets.
   In this repo, default to `volcano_owt10b_baseline_adamw.sh` unless the user names a different launcher or clearly needs a different command.
3. Infer as much as possible from the prompt and the template:
   - `START_COMMAND`
   - `JOB_NAME`
   - `PROJECT_NAME` workstream label for `vsubmit.sh`
   - `OUT_DIR`, `DATA_ROOT`, `TRANSFER_VARS`, and other submission env vars
   - run metadata embedded in the training command, such as `--general.run_name` and `--general.run_description`
4. For `nanugpt`, the common command shape is `train.py <config> <override args>`, but do not assume that shape when the prompt or launcher points to a different entrypoint.
5. If any critical value is still unknown, ask only for those values and stop.
   Critical values:
   - `JOB_NAME`
   - `PROJECT_NAME`
   - `VOLCANO_NAMESPACE` for a real submit
   - `VOLCANO_DATA_PVC_NAME` for a real submit
   - any command argument the target program cannot run without
6. Before any real submit, present a proposal that includes:
   - inferred values and the important overrides
   - final `START_COMMAND`
   - final environment variables
   - exact rendered `vsubmit.sh` shell command
7. Ask for confirmation and allow changes.
   Never submit in the same turn that first introduces the command.
8. Submit only after the user explicitly confirms the final command.

## Helper Script

Prefer the helper script for template parsing and command rendering:

If the skill is repo-local, use `.agents/skills/vsubmit/scripts/vsubmit_job.py`.
If the skill is globally installed, use the same relative path under `~/.codex/skills/vsubmit/`.

Use `--json` during planning so you can inspect inferred defaults, missing critical values, warnings, and the exact rendered shell command.

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py --json
```

Use `--template` when the user names a different launcher:

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py \
  --template volcano_karpathy_llmc_owt10b.sh \
  --json
```

Use `--start-command` when the final command should differ from the template command:

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py \
  --start-command "train.py configs/train_gpt2/openwebtext_tokens10b_karpathy_llmc.yaml --general.project_name nanugpt-owt10k --general.run_name owt-10b-karpathy-llmc --general.run_description 'Karpathy llm.c run'" \
  --job-name gpt-std \
  --workstream rlscaling \
  --json
```

Append extra command arguments without rebuilding the whole command:

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py \
  --json \
  --append-arg=--optimizer.lr \
  --append-arg 0.0006
```

Use the `--append-arg=...` form when the appended token itself starts with `-`.

After the user confirms, rerun the same command with `--submit`.

## Inference Rules

- Prefer values found in the user prompt over template defaults.
- Prefer template defaults over guessing.
- Infer run names and descriptions from the config name and the user's stated goal when the prompt does not specify them.
- Keep `PROJECT_NAME` separate from training flags such as `--general.project_name`.
- If the template exports `TRANSFER_VARS`, preserve it and add new entries with `--add-transfer-var` instead of replacing it unless the user clearly wants a replacement.
- Mention transfer vars that are not set in the local environment before any submit.

## Safety

- Do not silently invent launcher names, config paths, namespaces, or PVC names.
- Do not submit if the user asked only for a command, explanation, or review.
- Do not rely on `vsubmit.sh` interactive prompts for missing `JOB_NAME` or `PROJECT_NAME`; ask the user first.
- For actual submissions, run from the repo root or pass `--repo-root` explicitly.
- If the user wants to change a proposed value after the dry run, regenerate the command and show the full updated command again before submitting.
