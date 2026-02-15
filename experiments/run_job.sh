#!/bin/bash
# run_single_job.sh
# Wrapper script for running a single training job with HTCondor

set -e

MULTIRUN_NAME=$1    # typically datetime
PROCESS_ID=$2       # Job ID

# Specify config & hydra output
SWEEP_DIR="sweeps/${MULTIRUN_NAME}"
CONFIG_FILE="${SWEEP_DIR}/configs/config_${PROCESS_ID}.txt"
OUTPUT_DIR="${SWEEP_DIR}/outputs/"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found"
    exit 1
fi

# Read Hydra overrides from config file
HYDRA_OVERRIDES=$(cat "$CONFIG_FILE")

echo "=========================================="
echo "HTCondor Job Information"
echo "=========================================="
echo "Process ID: $PROCESS_ID"
echo "Hostname: $(hostname)"
echo "Working Directory: $(pwd)"
echo "Date: $(date)"
echo "=========================================="
echo "Hyperparameters:"
echo "$HYDRA_OVERRIDES"
echo "=========================================="

# Run the training script with Hydra overrides
# shellcheck disable=SC2086
python train.py \
    --multirun \
    hydra.sweep.dir="${OUTPUT_DIR}" \
    hydra.sweep.subdir="${PROCESS_ID}" \
    hydra.job.num="${PROCESS_ID}" \
    $HYDRA_OVERRIDES

echo "=========================================="
echo "Job completed successfully"
echo "=========================================="
