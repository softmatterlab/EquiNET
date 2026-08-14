#!/usr/bin/env bash
#SBATCH --account=YOUR_PROJECT_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=1-00:00:00
#SBATCH --job-name=hairpin_sim
#SBATCH --array=0-99
#SBATCH --output=logs/hairpin_%A_%a.out
#SBATCH --error=logs/hairpin_%A_%a.err
#SBATCH --cpus-per-task=8

set -uo pipefail

mkdir -p logs

# Configure these paths for your system before submitting the job.
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/output}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/path/to/hairpin.sif}"

BASE_OUTPUT_DIR="${OUTPUT_ROOT}/hairpin_100000_${SLURM_ARRAY_JOB_ID}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}"

mkdir -p "${OUTPUT_DIR}"

N_TOTAL=1000
N_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
CHUNK_SIZE=10

# Give every SLURM array task a distinct reproducible seed.
BASE_SEED=$((123456 + SLURM_ARRAY_TASK_ID))

echo "Array job: ${SLURM_ARRAY_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Base seed: ${BASE_SEED}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Container image: ${CONTAINER_IMAGE}"

apptainer exec \
    --nv \
    --bind "${PWD}:/workspace" \
    --bind "${OUTPUT_ROOT}:${OUTPUT_ROOT}" \
    "${CONTAINER_IMAGE}" \
    python /workspace/hairpin_simulation.py \
        "${OUTPUT_DIR}" \
        "${N_TOTAL}" \
        "${N_WORKERS}" \
        "${CHUNK_SIZE}" \
        "${BASE_SEED}"

status=$?

if [[ ${status} -eq 0 ]]; then
    echo "SUCCESS: Batch ${SLURM_ARRAY_TASK_ID} completed at $(date)"
else
    echo "ERROR: Batch ${SLURM_ARRAY_TASK_ID} failed at $(date)" >&2
fi

exit "${status}"