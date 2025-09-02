#! /bin/bash
set -eu -o pipefail # fail if any command failes, log all commands, -o xtrace

# register code direcoty as python package and do the torchrun

REQUIRED_VARS=("GPUS_PER_NODE" "REMOTE_JOB_OUT_DIR" "NPROC_PER_NODE" \
                "NODES" "RANK" "MASTER_ADDR" "MASTER_PORT" \
                 "START_COMMAND" "INSTALL_PACKAGE" "UPDATE_PYTHONPATH")

### ---------- Check required environment variables
for var in "${REQUIRED_VARS[@]}"; do
    if [[ -z "${!var:-}" ]]; then echo "Error: Required environment variable '$var' is not set." >&2; exit 1; fi
done
### ---------- End check required environment variables

# if file job_env_setup.sh exist then source it
if [ -f "${REMOTE_JOB_OUT_DIR}/job_env_setup.sh" ]; then
    source "${REMOTE_JOB_OUT_DIR}/job_env_setup.sh"
fi

cd ${REMOTE_JOB_OUT_DIR}/source_dir

# package installation if requested
if [ "${INSTALL_PACKAGE}" = "1" ]; then
    LOCK=/tmp/pip-install.lock
    STAMP=/tmp/pip-install.done

    # Everyone blocks on the lock. Only the first one actually installs; others skip.
    flock -x "$LOCK" bash -eu -c '
    if [[ ! -f "'"$STAMP"'" ]]; then
        echo "[info] installing pcakge..."
        pip install -e .
        # Create the stamp only if install succeeded
        : > "'"$STAMP"'"
        echo "[info] install complete; stamp written."
    else
        echo "[info] already installed by another process; skipping."
    fi
    '
else
    # add the package to PYTHONPATH so it acts like installed package
    if [ "${UPDATE_PYTHONPATH}" = "1" ]; then
        export PYTHONPATH="${TARGET_SOURCE_DIR}:${PYTHONPATH:-}"
    fi
fi

# build the torchrun command
# RANK variable is set by Pytorch plugin and its actually one per node (i.e. node index), as opposed to global rank of worker which torchrun will reset to
TORCH_RUN_ARGS="--nproc_per_node=${NPROC_PER_NODE} --nnodes=${NODES} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}"
eval "OUT_DIR=\"${JOB_OUT_DIR}\" torchrun ${TORCH_RUN_ARGS} ${START_COMMAND}"
