#!/usr/bin/env bash
# Driver for the pseudo-image CNN job (conventions §8): one script lists, sizes and
# runs the work. Every variable below can be overridden from the environment.
#
#   src/utils/data_analysis_02_pseudo_image_cnn_job.sh shards   # list the shards
#   src/utils/data_analysis_02_pseudo_image_cnn_job.sh probe    # time one shard
#   src/utils/data_analysis_02_pseudo_image_cnn_job.sh local    # run all, then merge
#   SHARD=7 src/utils/data_analysis_02_pseudo_image_cnn_job.sh local   # one array task
#   src/utils/data_analysis_02_pseudo_image_cnn_job.sh merge
set -euo pipefail

PROFILE="${PROFILE:-smoke}"
PYTHON="${PYTHON:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB="$ROOT/src/utils/data_analysis_02_pseudo_image_cnn_job.py"
LOG="$ROOT/logs/02_pseudo_image_cnn_${PROFILE}_run.log"

command="${1:?usage: $0 shards|probe|local|merge}"
args=(--profile "$PROFILE")
[[ -n "${SHARD:-}" ]] && args+=(--shard "$SHARD")
[[ -n "${THREADS:-}" ]] && args+=(--threads "$THREADS")
[[ -n "${FORCE:-}" ]] && args+=(--force)

mkdir -p "$ROOT/logs"
"$PYTHON" "$JOB" "$command" "${args[@]}" 2>&1 | tee -a "$LOG"
