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

# Reconstruct original argv from START_COMMAND (shell-escaped tokens) and exec torchrun
# below replace the script’s positional parameters ($1, $2, …).
eval "set -- ${START_COMMAND}"

USE_TORCHRUN="${USE_TORCHRUN:-1}"
case "${USE_TORCHRUN}" in
    1|true|TRUE|True) USE_TORCHRUN=1 ;;
    0|false|FALSE|False) USE_TORCHRUN=0 ;;
    *) echo "Error: USE_TORCHRUN must be 0/1/true/false (got '${USE_TORCHRUN}')." >&2; exit 1 ;;
esac

if [[ "${USE_TORCHRUN}" == "1" ]]; then
    # build the torchrun command
    # RANK variable is set by Pytorch plugin and its index of the node (this is not same as GLOBAL_RANK which is index of the process and set later by torchrun)
    TORCH_RUN_ARGS="--nproc_per_node=${NPROC_PER_NODE} --nnodes=${NODES} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}"
    OUT_DIR="${JOB_OUT_DIR}" exec torchrun ${TORCH_RUN_ARGS} "$@"
fi

if [[ "${NODES}" != "1" ]]; then
    echo "Running '${START_COMMAND}' once per pod/node..." >&2
fi
OUT_DIR="${JOB_OUT_DIR}" exec "$@"
