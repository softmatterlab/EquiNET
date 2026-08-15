#!/usr/bin/env bash

#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=0-02:00:00
#SBATCH --job-name=hairpin_merge
#SBATCH --output=logs/hairpin_merge_%j.out
#SBATCH --error=logs/hairpin_merge_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

set -euo pipefail

mkdir -p logs

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch $0 /path/to/hairpin/output_directory" >&2
    exit 1
fi

BASE_OUTPUT_DIR=$(realpath "$1")
FINAL_DIR="${BASE_OUTPUT_DIR}/final"

if [[ ! -d "${BASE_OUTPUT_DIR}" ]]; then
    echo "ERROR: Base output directory does not exist: ${BASE_OUTPUT_DIR}" >&2
    exit 1
fi

mkdir -p "${FINAL_DIR}"

export BASE_OUTPUT_DIR
export FINAL_DIR

python <<'PYTHON'
import os
from pathlib import Path

import numpy as np


base_dir = Path(os.environ["BASE_OUTPUT_DIR"])
final_dir = Path(os.environ["FINAL_DIR"])

batch_dirs = sorted(
    path
    for path in base_dir.iterdir()
    if path.is_dir() and path.name.startswith("batch_")
)

if not batch_dirs:
    raise RuntimeError(f"No batch directories found in {base_dir}")

required_files = ("traj_fwd.npy", "traj_rev.npy")

valid_batch_dirs = []
for batch_dir in batch_dirs:
    missing = [
        filename
        for filename in required_files
        if not (batch_dir / filename).is_file()
    ]

    if missing:
        print(
            f"WARNING: Skipping {batch_dir}; missing files: "
            f"{', '.join(missing)}"
        )
        continue

    valid_batch_dirs.append(batch_dir)

if not valid_batch_dirs:
    raise RuntimeError("No complete batch directories were found.")

sample_fwd = np.load(
    valid_batch_dirs[0] / "traj_fwd.npy",
    mmap_mode="r",
)
sample_rev = np.load(
    valid_batch_dirs[0] / "traj_rev.npy",
    mmap_mode="r",
)

if sample_fwd.shape != sample_rev.shape:
    raise ValueError(
        f"Forward and reverse arrays have different shapes in "
        f"{valid_batch_dirs[0]}: {sample_fwd.shape} and {sample_rev.shape}"
    )

dtype = sample_fwd.dtype
sample_shape = sample_fwd.shape[1:]

batch_sizes = []
for batch_dir in valid_batch_dirs:
    fwd = np.load(batch_dir / "traj_fwd.npy", mmap_mode="r")
    rev = np.load(batch_dir / "traj_rev.npy", mmap_mode="r")

    if fwd.shape != rev.shape:
        raise ValueError(
            f"Forward and reverse arrays have different shapes in "
            f"{batch_dir}: {fwd.shape} and {rev.shape}"
        )

    if fwd.shape[1:] != sample_shape:
        raise ValueError(
            f"Inconsistent trajectory shape in {batch_dir}: "
            f"expected {sample_shape}, found {fwd.shape[1:]}"
        )

    if fwd.dtype != dtype or rev.dtype != dtype:
        raise ValueError(
            f"Inconsistent dtype in {batch_dir}: "
            f"expected {dtype}, found {fwd.dtype} and {rev.dtype}"
        )

    batch_sizes.append(fwd.shape[0])

n_total = sum(batch_sizes)
final_shape = (n_total,) + sample_shape

print(f"Merging {len(valid_batch_dirs)} batches")
print(f"Total trajectories: {n_total}")
print(f"Final array shape: {final_shape}")
print(f"Output directory: {final_dir}")

fwd_path = final_dir / "traj_fwd.npy"
rev_path = final_dir / "traj_rev.npy"

fwd_out = np.lib.format.open_memmap(
    fwd_path,
    mode="w+",
    dtype=dtype,
    shape=final_shape,
)

rev_out = np.lib.format.open_memmap(
    rev_path,
    mode="w+",
    dtype=dtype,
    shape=final_shape,
)

offset = 0

for batch_dir, batch_size in zip(valid_batch_dirs, batch_sizes):
    fwd = np.load(batch_dir / "traj_fwd.npy", mmap_mode="r")
    rev = np.load(batch_dir / "traj_rev.npy", mmap_mode="r")

    end = offset + batch_size

    fwd_out[offset:end] = fwd
    rev_out[offset:end] = rev

    print(
        f"Merged {batch_dir.name}: "
        f"indices {offset}:{end}"
    )

    offset = end

fwd_out.flush()
rev_out.flush()

print(f"Created {fwd_path}")
print(f"Created {rev_path}")
print("Merge completed successfully.")
