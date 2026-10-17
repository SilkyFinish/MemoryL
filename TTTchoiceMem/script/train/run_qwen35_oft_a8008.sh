#!/bin/bash
# Historical filename retained for callers; defaults now follow local stage 1.
# By default this freezes a plan only. --launch is an explicit training request.
set -euo pipefail

project_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python="${TRAINING_PYTHON:-python3}"
launch=false
configuration_arguments=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --launch)
            launch=true
            shift
            ;;
        --python)
            if [ "$#" -lt 2 ] || [[ "$2" = --* ]]; then
                printf '%s\n' 'Error: --python requires an explicit interpreter.' >&2
                exit 2
            fi
            python="$2"
            configuration_arguments+=(--python "$2")
            shift 2
            ;;
        *)
            configuration_arguments+=("$1")
            shift
            ;;
    esac
done

if [ "$launch" = false ]; then
    exec "$python" -u "$project_root/configure_training_job.py" "${configuration_arguments[@]}"
fi
for argument in "${configuration_arguments[@]}"; do
    if [ "$argument" = --dry-run ]; then
        printf '%s\n' 'Error: --launch and --dry-run are mutually exclusive.' >&2
        exit 2
    fi
done
plan_path="$("$python" "$project_root/configure_training_job.py" "${configuration_arguments[@]}")"
exec "$python" -u "$project_root/run_training_job.py" --plan "$plan_path"
