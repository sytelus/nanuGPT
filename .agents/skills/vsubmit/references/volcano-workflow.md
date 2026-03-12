# Volcano Workflow Reference

## Required Cluster Environment

`scripts/volcano/vsubmit.sh` requires a working `kubectl` context and these cluster variables before an actual submission:

- `VOLCANO_NAMESPACE`
- `VOLCANO_DATA_PVC_NAME`

Common optional variables:

- `DATA_ROOT`
- `OUT_DIR`
- `WANDB_API_KEY`
- `WANDB_HOST`
- `CONTAINER_IMAGE_PATH`
- `ENV_SETUP_SCRIPT`

`JOB_NAME` is mandatory for every run, but the helper script can set it directly.

## Default Behavior in `vsubmit.sh`

Important defaults from `scripts/volcano/vsubmit.sh`:

- `NODES=1`
- `GPUS_PER_NODE=8`
- `NPROC_PER_NODE=GPUS_PER_NODE`
- `INSTALL_PACKAGE=1`
- `UPDATE_PYTHONPATH=0`
- `USE_TORCHRUN=1`
- `CONTAINER_IMAGE_PATH=sytelus/gpu-devbox:latest`
- `TRANSFER_VARS=""` unless set by the caller

The script copies the working tree to PVC storage, renders `scripts/volcano/volcano_job.yaml`, creates a Volcano job, waits for pods, streams logs, and then reports pod termination status.

## Repo Presets

These presets mirror the launcher scripts in the repo root:

| Preset | Launcher | Entrypoint | Config | Notes |
|:--|:--|:--|:--|:--|
| `baseline-adamw` | `volcano_owt10b_baseline_adamw.sh` | `train.py` | `configs/train_gpt2/openwebtext_tokens10b_baseline.yaml` | Keller-style baseline with run metadata passed through CLI |
| `karpathy-classic` | `volcano_karpathy_classic_owt10b.sh` | `train.py` | `configs/train_gpt2/openwebtext_tokens10b_karpathy_classic.yaml` | Karpathy classic 10B-token run |
| `karpathy-llmc` | `volcano_karpathy_llmc_owt10b.sh` | `train.py` | `configs/train_gpt2/openwebtext_tokens10b_karpathy_llmc.yaml` | Karpathy llm.c variant |
| `baseline-muon` | `volcano_owt10b_baseline_muon.sh` | `scripts/alt_training/keller_train_gpt2_muon.py` | none | Alternate training entrypoint, no config argument by default |

All presets also set:

- `TRANSFER_VARS="DATA_ROOT WANDB_API_KEY WANDB_HOST"`
- `JOB_NAME=gpt-std` by default

Preset-specific tuning flags:

- `baseline-adamw`, `karpathy-classic`, `karpathy-llmc`
  - `TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1`
  - `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1`
- `baseline-muon`
  - `TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=0`
  - `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=0`

## Helper Script Usage

If the skill is globally installed instead of repo-local, replace `.agents/skills/vsubmit/` with `~/.codex/skills/vsubmit/` in the examples below.

List presets:

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py --list-presets
```

Prepare a preset command:

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py --preset baseline-adamw --job-name my-run
```

Forward extra training flags after `--`:

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py \
  --preset baseline-adamw \
  --job-name my-run \
  -- --optimizer.lr 0.0006 --training.max_steps 2000
```

Prepare a manual command:

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py \
  --entrypoint train.py \
  --config configs/train_gpt2/tinyshakespeare.yaml \
  --job-name tiny-test
```

Submit for real:

```bash
python .agents/skills/vsubmit/scripts/vsubmit_job.py \
  --preset baseline-adamw \
  --job-name my-run \
  --volcano-namespace my-ns \
  --volcano-data-pvc-name my-pvc \
  --submit
```

## Prompt Inputs That Matter

When translating a prompt into a submission, extract these fields if present:

- intent: `explain`, `dry-run`, or `submit`
- preset or explicit `entrypoint` + `config`
- `job_name`
- `nodes`, `gpus_per_node`, `nproc_per_node`
- `project_name`, `run_name`, `run_description`
- cluster placement: namespace, PVC, image, priority
- data paths and secrets to transfer
- trailing config overrides after `--`

If the user omits one of the cluster variables for a real submit, stop and point out what is missing.
