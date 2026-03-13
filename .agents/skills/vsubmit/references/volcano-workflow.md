# Volcano Workflow Reference

## What `vsubmit.sh` Needs

`scripts/volcano/vsubmit.sh` is the generic primitive. It needs:

- a command to run inside the container
- `JOB_NAME`
- `PROJECT_NAME`

For a real cluster submission it also needs:

- `VOLCANO_NAMESPACE`
- `VOLCANO_DATA_PVC_NAME`

Common optional variables that are often worth inferring from a launcher template:

- `DATA_ROOT`
- `OUT_DIR`
- `TRANSFER_VARS`
- `CONTAINER_IMAGE_PATH`
- `ENV_SETUP_SCRIPT`
- `NODES`
- `GPUS_PER_NODE`
- `NPROC_PER_NODE`

## Important Defaults From `scripts/volcano/vsubmit.sh`

- `NODES=1`
- `GPUS_PER_NODE=8`
- `NPROC_PER_NODE=GPUS_PER_NODE`
- `INSTALL_PACKAGE=1`
- `UPDATE_PYTHONPATH=0`
- `USE_TORCHRUN=1`
- `CONTAINER_IMAGE_PATH=sytelus/gpu-devbox:latest`
- `OUT_DIR=/data/runs/${USER_ALIAS}`
- `SOURCE_DIR=.`
- `LOCAL_OUT_DIR=/tmp/volcano_jobs`
- `TRANSFER_VARS=""` unless the caller sets it

`vsubmit.sh` will prompt interactively if `PROJECT_NAME` is missing. The skill should avoid that by asking the user before execution.

## Template-First Pattern

Use a launcher script as a template for defaults. In this repo, the normal default template is:

- `volcano_owt10b_baseline_adamw.sh`

That template usually provides:

- `JOB_NAME`
- `PROJECT_NAME`
- `OUT_DIR`
- `DATA_ROOT`
- `TRANSFER_VARS`
- `NODES`
- a default `START_COMMAND`

The helper script can parse those values and render the final command without assuming one of a few fixed presets.

## Helper Script Usage

Render inferred defaults from the default template:

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py --json
```

Switch to another launcher template:

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py \
  --template volcano_karpathy_classic_owt10b.sh \
  --json
```

Override the command while keeping inferred env defaults:

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py \
  --template volcano_owt10b_baseline_adamw.sh \
  --start-command "train.py configs/train_gpt2/openwebtext_tokens10b_baseline.yaml --general.project_name nanugpt-owt10k --general.run_name my-run --general.run_description 'custom baseline run'" \
  --job-name gpt-std \
  --workstream rlscaling \
  --json
```

Append additional training arguments:

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py \
  --json \
  --append-arg=--optimizer.lr \
  --append-arg 0.0006
```

Submit only after confirmation:

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py \
  --template volcano_owt10b_baseline_adamw.sh \
  --start-command "train.py configs/train_gpt2/openwebtext_tokens10b_baseline.yaml --general.project_name nanugpt-owt10k --general.run_name approved-run --general.run_description 'approved baseline run'" \
  --job-name gpt-std \
  --workstream rlscaling \
  --volcano-namespace my-namespace \
  --volcano-data-pvc-name my-pvc \
  --submit
```

## Prompt Fields To Extract

When translating a prompt into a submission proposal, extract:

- intent: explain, prepare, dry-run, or submit
- launcher template, if the user names one
- start command or command shape
- workstream label for `PROJECT_NAME`
- job name
- cluster placement: namespace and PVC
- output and data paths
- extra environment variables
- extra command arguments or config overrides

If the prompt omits a real-submit requirement such as namespace or PVC, stop and ask for it instead of guessing.
