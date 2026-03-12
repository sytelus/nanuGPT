# HOWTO: Codex `vsubmit` Skill for nanugpt

This repo now includes a Codex skill at `.agents/skills/vsubmit` that turns natural-language job requests into the exact `scripts/volcano/vsubmit.sh` command used by `nanugpt` on a Kubernetes Volcano cluster.

It is intentionally narrow: it only covers this repo's Volcano workflow. That is a feature, not a limitation. OpenAI's current skills guidance recommends repo-local skills for project-specific workflows, concise trigger descriptions, and moving repeatable deterministic work into scripts instead of long prompt text. This skill follows that model.

## What the Skill Does

The skill helps Codex:

- recognize requests such as "submit the baseline AdamW run on 2 nodes"
- map those requests to this repo's launcher presets or a custom `train.py` + config command
- render the exact `vsubmit.sh` invocation before execution
- warn about missing Volcano prerequisites such as `VOLCANO_NAMESPACE` and `VOLCANO_DATA_PVC_NAME`
- submit the job only when the prompt clearly asks for a real submission

The helper script that does the deterministic rendering lives at:

`./.agents/skills/vsubmit/scripts/vsubmit_job.py`

## Why This Skill Is Written This Way

These design choices come directly from OpenAI's current skills guidance:

- Keep the skill focused on one workflow. This one only handles `nanugpt` Volcano submissions.
- Put the trigger conditions in `SKILL.md` frontmatter `description`, because that is what Codex uses to decide when a skill applies.
- Keep `SKILL.md` concise and move repo-specific details into a reference file.
- Use a script for command rendering because shell quoting and environment assembly are deterministic tasks that should not be rebuilt from scratch on every prompt.
- Prefer repo-local installation for repo-local behavior. OpenAI's Codex docs describe project skills under `.agents/skills/`.

## Files Added

- `.agents/skills/vsubmit/SKILL.md`
- `.agents/skills/vsubmit/agents/openai.yaml`
- `.agents/skills/vsubmit/references/volcano-workflow.md`
- `.agents/skills/vsubmit/scripts/vsubmit_job.py`

## Prerequisites

Before a real submit, make sure the shell environment used by Codex has:

- `kubectl` installed and pointed at the right cluster
- `VOLCANO_NAMESPACE`
- `VOLCANO_DATA_PVC_NAME`

Common optional variables:

- `DATA_ROOT`
- `OUT_DIR`
- `WANDB_API_KEY`
- `WANDB_HOST`

The helper warns when transfer variables are missing. `vsubmit.sh` can still render a command without them, but an actual job may not behave as intended.

## Installation Options

### Option 1: Repo-local, recommended

If you are running Codex inside this repo, no separate installation is needed beyond having the skill folder in place:

```text
.agents/skills/vsubmit/
```

OpenAI's Codex skills docs describe project skills in `.agents/skills/`. Restart Codex after adding the skill if the current session does not pick it up automatically.

### Option 2: Global manual install

If you want the skill available across sessions, copy it into your personal Codex skills directory:

```bash
mkdir -p ~/.codex/skills
cp -R .agents/skills/vsubmit ~/.codex/skills/vsubmit
```

Restart Codex after copying it.

This is useful if you work in `nanugpt` often and want `$vsubmit` available even when the skill is not only repo-local.

### Option 3: Install from GitHub with the skill installer

This works after the new skill directory has been pushed to GitHub:

```bash
~/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo sytelus/nanugpt \
  --path .agents/skills/vsubmit
```

That installs into `~/.codex/skills/vsubmit`. Restart Codex afterward.

If the repo changes are still only local, use Option 1 or Option 2 for now.

## How Codex Should Use the Skill

The skill is designed for three modes:

- `explain`: explain what command would be used and why
- `dry-run`: render the final `vsubmit.sh` command without running it
- `submit`: run the command for real

Recommended behavior:

- default to `dry-run` if the user asks for help, an example, or a command
- only `submit` when the prompt clearly asks to launch the job
- show the rendered command before execution

## Supported Presets

The helper script mirrors these repo launchers:

- `baseline-adamw` -> `volcano_owt10b_baseline_adamw.sh`
- `karpathy-classic` -> `volcano_karpathy_classic_owt10b.sh`
- `karpathy-llmc` -> `volcano_karpathy_llmc_owt10b.sh`
- `baseline-muon` -> `volcano_owt10b_baseline_muon.sh`

To list them from the command line:

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py --list-presets
```

## Direct Helper Script Usage

### 1. Dry-run the default baseline preset

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py --preset baseline-adamw
```

### 2. Dry-run with job and node overrides

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py \
  --preset karpathy-llmc \
  --job-name gpt-llmc-2n \
  --nodes 2 \
  --run-name owt-10b-karpathy-2n \
  --run-description "Karpathy llm.c run on 2 nodes"
```

### 3. Forward training overrides to `train.py`

Anything after `--` is appended to the training command:

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py \
  --preset baseline-adamw \
  --job-name baseline-lr-test \
  -- --optimizer.learning_rate 0.0006 --training.max_steps 2000
```

### 4. Build a completely custom submission

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py \
  --entrypoint train.py \
  --config configs/train_gpt2/openwebtext_tokens10b_keller_adamw.yaml \
  --job-name keller-adamw-custom \
  --project-name nanugpt-owt10k \
  --run-name owt-10b-keller-adamw \
  --run-description "Custom keller adamw run"
```

### 5. Submit for real

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py \
  --preset baseline-adamw \
  --job-name baseline-real \
  --volcano-namespace my-namespace \
  --volcano-data-pvc-name my-pvc \
  --submit
```

The script prints warnings first, then executes `bash scripts/volcano/vsubmit.sh ...` from the repo root.

## Prompt Examples for Codex

These are the kinds of prompts the skill is built to simplify:

```text
Use $vsubmit to dry-run the baseline AdamW Volcano job with job name gpt-std-test.
```

```text
Use $vsubmit to prepare a 2-node Karpathy llm.c submit command and append --optimizer.learning_rate 0.0006.
```

```text
Use $vsubmit to submit train.py with configs/train_gpt2/openwebtext_tokens10b_keller_adamw.yaml on Volcano in namespace ml, pvc train-data, job name keller-adamw-01.
```

```text
Use $vsubmit to explain which env vars are missing for a real submit in this shell.
```

The better the prompt, the better the result. Include:

- whether you want `dry-run` or `submit`
- preset name or entrypoint + config
- job name
- node count and other resource overrides
- cluster namespace and PVC if they are not already in the environment
- any extra training flags

## Prompting Best Practices

For this skill, effective prompts are explicit about intent and constraints. Good prompts answer these questions:

- Do you want a command, an explanation, or an actual submission?
- Are you using one of the known presets or a custom config?
- What should `JOB_NAME` be?
- Do you need non-default nodes, GPUs, image, namespace, or PVC?
- Are there extra training overrides after the base config?

Avoid vague prompts like:

```text
run the job
```

Prefer prompts like:

```text
Use $vsubmit to dry-run the baseline-adamw preset as job gpt-baseline-test on 2 nodes in namespace ml with pvc train-data.
```

## Troubleshooting

### Skill does not appear in Codex

- Confirm the folder exists at `.agents/skills/vsubmit` for repo-local use, or `~/.codex/skills/vsubmit` for global use.
- Restart Codex.

### The helper says it cannot find the repo root

Run Codex from inside the `nanugpt` repo or pass:

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py --repo-root /path/to/nanugpt ...
```

### The helper warns that `VOLCANO_NAMESPACE` or `VOLCANO_DATA_PVC_NAME` is missing

Set them in the shell or pass them as flags:

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py \
  --preset baseline-adamw \
  --job-name test \
  --volcano-namespace my-namespace \
  --volcano-data-pvc-name my-pvc
```

### The helper warns about missing transfer vars

That means a variable listed in `TRANSFER_VARS` is not set in the current shell. Either export it first or remove it from the transfer list.

## Official References

These informed both the skill design and the usage guidance:

- OpenAI Developers: https://developers.openai.com/codex/ide/skills
- OpenAI Developers: https://developers.openai.com/codex/prompting-guide
- Local system guidance used by Codex in this environment: `/home/shitals/.codex/skills/.system/skill-creator/SKILL.md`
