
# run node_stat.py from this script locations
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1; pwd)
python "${SCRIPT_DIR}/node_stat.py" "$@"