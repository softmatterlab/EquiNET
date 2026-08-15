```bash
#!/usr/bin/env bash

#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=0-40:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=hairpin_ml
#SBATCH --output=logs/hairpin_ml_%A_%a.out
#SBATCH --error=logs/hairpin_ml_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --array=1-36%4

set -u

mkdir -p logs output

###############################################################################
# User configuration
###############################################################################

CONTAINER_IMAGE="${CONTAINER_IMAGE:-hairpin.sif}"
DATA_ROOT="${DATA_ROOT:-/path/to/hairpin/output}"

DATA_DIRS=(
    "${DATA_ROOT}/dataset_1/final"
    "${DATA_ROOT}/dataset_2/final"
    "${DATA_ROOT}/dataset_3/final"
    "${DATA_ROOT}/dataset_4/final"
)

LABELS=(
    "dataset_1"
    "dataset_2"
    "dataset_3"
    "dataset_4"
)

DATA_SIZES=(100 1000 5000)
N_REPEATS=3

###############################################################################
# Map each SLURM array task to a dataset, data size, and repeat
###############################################################################

N_PARAMS=${#DATA_DIRS[@]}
N_SIZES=${#DATA_SIZES[@]}
TASKS_PER_PARAM=$((N_SIZES * N_REPEATS))
EXPECTED_TASKS=$((N_PARAMS * TASKS_PER_PARAM))

TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))

if (( TASK_ID < 0 || TASK_ID >= EXPECTED_TASKS )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is out of range." >&2
    echo "Expected array range: 1-${EXPECTED_TASKS}" >&2
    exit 1
fi

if (( ${#LABELS[@]} != N_PARAMS )); then
    echo "ERROR: DATA_DIRS and LABELS must have the same number of entries." >&2
    exit 1
fi

PARAM_IDX=$((TASK_ID / TASKS_PER_PARAM))
WITHIN_PARAM=$((TASK_ID % TASKS_PER_PARAM))
SIZE_IDX=$((WITHIN_PARAM / N_REPEATS))
REPEAT_IDX=$((WITHIN_PARAM % N_REPEATS + 1))

DATA_DIR="${DATA_DIRS[$PARAM_IDX]}"
LABEL="${LABELS[$PARAM_IDX]}"
N_DATA="${DATA_SIZES[$SIZE_IDX]}"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: Input directory does not exist: ${DATA_DIR}" >&2
    exit 1
fi

if [[ ! -f "${CONTAINER_IMAGE}" ]]; then
    echo "ERROR: Container image does not exist: ${CONTAINER_IMAGE}" >&2
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${PWD}/output/${LABEL}/N_${N_DATA}/run_${REPEAT_IDX}_${TIMESTAMP}"

mkdir -p "${OUTPUT_DIR}"

echo "SLURM job ID     : ${SLURM_JOB_ID:-unknown}"
echo "Array task       : ${SLURM_ARRAY_TASK_ID}"
echo "Dataset          : ${LABEL}"
echo "Dataset index    : ${PARAM_IDX}"
echo "Training samples : ${N_DATA}"
echo "Repeat           : ${REPEAT_IDX}"
echo "Input directory  : ${DATA_DIR}"
echo "Output directory : ${OUTPUT_DIR}"
echo "Container image  : ${CONTAINER_IMAGE}"

apptainer exec \
    --nv \
    --bind "${PWD}:/workspace" \
    --bind "${DATA_ROOT}:${DATA_ROOT}" \
    "${CONTAINER_IMAGE}" \
    python /workspace/egnn.py \ #egnn_cg.py
        "${DATA_DIR}" \
        "${OUTPUT_DIR}" \
        "${N_DATA}"

status=$?

if [[ ${status} -eq 0 ]]; then
    echo "SUCCESS: Job completed at $(date)"
else
    echo "ERROR: Job failed with status ${status} at $(date)" >&2
fi

exit "${status}"
```
