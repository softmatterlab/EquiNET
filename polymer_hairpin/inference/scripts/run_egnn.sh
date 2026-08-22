#!/usr/bin/env bash

#SBATCH -A <PROJECT_ACCOUNT>
#SBATCH -p alvis
#SBATCH -t 0-08:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --job-name=neep_ml
#SBATCH --output=logs/neep_ml_%A_%a.out
#SBATCH --error=logs/neep_ml_%A_%a.err
#SBATCH --array=1-144%4

set -euo pipefail

# ================================================================
# Usage
# ================================================================
#
# Before submitting:
#
#   mkdir -p logs output
#   sbatch submit_neep_ml.sh
#
# The array runs:
#
#   4 parameter sets
# x 6 dataset sizes
# x 6 independent train/test splits
# = 144 jobs
#
# At most 4 array jobs run simultaneously because of:
#
#   #SBATCH --array=1-144%4
#
# Each Python run receives:
#
#   DATA_DIR OUTPUT_DIR N_DATA SEED
#
# where the Python script constructs:
#
#   N_train = N_DATA
#   N_test  = N_DATA
#
# from disjoint randomly sampled trajectory indices.
# ================================================================

cd "${SLURM_SUBMIT_DIR}"

mkdir -p output

# ================================================================
# DATASETS
# ================================================================
#
# Replace these paths with the locations of your trajectory datasets.
# Each directory must contain:
#
#   traj_fwd.npy
#   traj_rev.npy
#

DATA_DIRS=(
    "/path/to/eps12/final"
    "/path/to/eps9/final"
    "/path/to/eps6/final"
    "/path/to/eps3/final"
)

LABELS=(
    "eps12_out"
    "eps9_out"
    "eps6_out"
    "eps3_out"
)

# Number of trajectories used independently for training and testing.
#
# For example, N_DATA=10000 means:
#
#   10000 training trajectories
#   10000 held-out testing trajectories
#
# so at least 20000 trajectories must be available.

DATA_SIZES=(
    50
    10000
    20000
    30000
    40000
    50000
)

# Number of independent random train/test splits for each dataset size.
N_REPEATS=6

# ================================================================
# ARRAY MAPPING
# ================================================================

N_PARAMS=${#DATA_DIRS[@]}
N_LABELS=${#LABELS[@]}
N_SIZES=${#DATA_SIZES[@]}

TASKS_PER_PARAM=$((N_SIZES * N_REPEATS))
EXPECTED_TASKS=$((N_PARAMS * TASKS_PER_PARAM))

if (( N_PARAMS != N_LABELS )); then
    echo "ERROR: DATA_DIRS and LABELS must have the same length." >&2
    exit 1
fi

if (( SLURM_ARRAY_TASK_ID < 1 || SLURM_ARRAY_TASK_ID > EXPECTED_TASKS )); then
    echo "ERROR: Invalid array task ID ${SLURM_ARRAY_TASK_ID}." >&2
    echo "Expected IDs: 1-${EXPECTED_TASKS}" >&2
    exit 1
fi

# Convert the 1-based Slurm array ID to a zero-based task index.
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))

# Identify which parameter set this task belongs to.
PARAM_IDX=$((TASK_ID / TASKS_PER_PARAM))

# Position within this parameter set.
WITHIN_PARAM=$((TASK_ID % TASKS_PER_PARAM))

# Dataset-size index.
SIZE_IDX=$((WITHIN_PARAM / N_REPEATS))

# Repeat number: 1,...,N_REPEATS.
REPEAT_IDX=$((WITHIN_PARAM % N_REPEATS + 1))

DATA_DIR="${DATA_DIRS[$PARAM_IDX]}"
LABEL="${LABELS[$PARAM_IDX]}"
N_DATA="${DATA_SIZES[$SIZE_IDX]}"

# ================================================================
# INPUT VALIDATION
# ================================================================

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: Data directory does not exist:"
    echo "  ${DATA_DIR}" >&2
    exit 1
fi

for required_file in traj_fwd.npy traj_rev.npy; do
    if [[ ! -f "${DATA_DIR}/${required_file}" ]]; then
        echo "ERROR: Missing required file:"
        echo "  ${DATA_DIR}/${required_file}" >&2
        exit 1
    fi
done

# ================================================================
# RANDOM SEED
# ================================================================
#
# Every array task gets a different reproducible seed within this
# submission.
#
# Thus the six repeats for a given N_DATA use six independently drawn
# train/test splits.
#
# Note:
# The splits are disjoint within each individual run, but trajectories
# may overlap between different repeats.

SEED=$((SLURM_ARRAY_JOB_ID * 1000 + SLURM_ARRAY_TASK_ID))

# ================================================================
# OUTPUT DIRECTORY
# ================================================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

OUTPUT_DIR="output/${LABEL}/N_${N_DATA}/run_${REPEAT_IDX}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TIMESTAMP}"

mkdir -p "${OUTPUT_DIR}"

# ================================================================
# JOB INFORMATION
# ================================================================

echo "============================================================"
echo "NEEP / EGNN run"
echo "============================================================"
echo "Array job ID:       ${SLURM_ARRAY_JOB_ID}"
echo "Array task ID:      ${SLURM_ARRAY_TASK_ID}"
echo "Parameter index:    ${PARAM_IDX}"
echo "Parameter label:    ${LABEL}"
echo "Dataset-size index: ${SIZE_IDX}"
echo "Repeat index:       ${REPEAT_IDX}/${N_REPEATS}"
echo "N_train:            ${N_DATA}"
echo "N_test:             ${N_DATA}"
echo "Seed:               ${SEED}"
echo "Data directory:     ${DATA_DIR}"
echo "Output directory:   ${OUTPUT_DIR}"
echo "Node:               $(hostname)"
echo "Started:            $(date)"
echo "============================================================"

# ================================================================
# RUN
# ================================================================

apptainer exec \
    --nv \
    --bind "${SLURM_SUBMIT_DIR}:/workspace" \
    --bind "/mimer:/mimer" \
    --pwd /workspace \
    neep.sif \
    python /workspace/egnn.py \     #egnn_cg.py
    "${DATA_DIR}" \
    "/workspace/${OUTPUT_DIR}" \
    "${N_DATA}" \
    "${SEED}"

echo "============================================================"
echo "SUCCESS"
echo "Finished: $(date)"
echo "Results:  ${OUTPUT_DIR}"
echo "============================================================"