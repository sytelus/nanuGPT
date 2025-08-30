#! /bin/bash
set -eu -o pipefail # fail if any command failes, log all commands, -o xtrace

REQUIRED_VARS=("GPUS_PER_NODE" "CONTAINER_IMAGE_PATH" "JOB_OUT_DIR" "TARGET_SOURCE_DIR" "VOLCANO_SCRIPT_DIR" \
                 "START_COMMAND" "INSTALL_PACKAGE" "UPDATE_PYTHONPATH")
JOB_ENV_SETUP_SCRIPT=${JOB_ENV_SETUP_SCRIPT:-}

### ---------- Check required environment variables
for var in "${REQUIRED_VARS[@]}"; do
    [ -z "${!var}" ] && { echo "Error: Required environment variable '$var' is not set." >&2; exit 1; }
done
### ---------- End check required environment variables

# setup cluster specific environment variables, these will be inherited by the container
if [ ! -z "${JOB_ENV_SETUP_SCRIPT}" ]; then
    source "${JOB_ENV_SETUP_SCRIPT}"
fi

cd "${TARGET_SOURCE_DIR}"

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
TORCH_RUN_ARGS="--nproc_per_node=${NPROC_PER_NODE} --nnodes=${NNODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}
eval "OUT_DIR=\"${JOB_OUT_DIR}\" torchrun ${TORCH_RUN_ARGS} ${START_COMMAND}"
