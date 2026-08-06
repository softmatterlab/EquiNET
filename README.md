# Inferring Equilibrium Free Energy Differences from Nonequilibrium Fluctuations in High-Dissipation Regimes

This repository contains the simulation, inference, and analysis code accompanying the paper:

> **Inferring Equilibrium Free Energy Differences from Nonequilibrium Fluctuations in High-Dissipation Regimes**

The workflow estimates equilibrium free-energy differences from typical nonequilibrium trajectories by inferring entropy production through a variational short-time thermodynamic uncertainty relation. The repository contains a one-dimensional double-well demonstration and a three-dimensional polymer-hairpin application using time-conditioned E(3)-equivariant graph neural networks (EGNNs).

## Repository contents

```text
nonequilibrium-free-energy-inference/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── double_well/
│   ├── double_well_demo.ipynb
│   ├── simulation.py
│   ├── work_entropy_plot.py
│   └── protocol_plot.py
│
├── polymer_hairpin/
│   ├── hairpin_parallel_x_trap.py
│   ├── hairpin_work_array.py
│   ├── egnn_fast_multisize_split.py
│   ├── egnn_cg.py
│   ├── merge_hairpin.sh
│   ├── submit_hairpin_array.sh
│   ├── submit_neep_ml.sh
│   └── submit_neep_cg.sh
│
├── analysis/
│   ├── convergence_analysis.py
│   ├── duplicate_check.py
│   └── make_paper_figures.py
│
├── figures/
│   └── README.md
│
└── data/
    └── README.md
```

Large trajectory files, model checkpoints, cluster logs, and generated figures are not tracked by Git.

## Installation

Create a Python environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

A minimal `requirements.txt` is:

```text
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
seaborn>=0.12
torch>=2.0
tqdm>=4.65
jupyter>=1.0
ipykernel>=6.25
nbformat>=5.9
```

The cluster jobs in this repository use an Apptainer image named `neep.sif`. The image is not included in the repository.

## 1. Double-well example

The one-dimensional example simulates overdamped Langevin dynamics in a time-dependent asymmetric double-well potential. The protocol contains:

1. an initial harmonic equilibration plateau,
2. a smooth finite-time transformation through the double-well potential,
3. a final harmonic equilibration plateau.

The notebook and scripts:

- define the static and time-dependent potentials,
- generate forward and reverse trajectory ensembles,
- estimate the instantaneous entropy-production rate,
- compute protocol work using a midpoint, Stratonovich-consistent discretization,
- compare work, entropy production, and free-energy estimates,
- generate the figures used in the paper.

Run the notebook with:

```bash
jupyter notebook double_well/double_well_demo.ipynb
```

## 2. Polymer-hairpin trajectory generation

The polymer hairpin is represented by 13 beads in three dimensions. The force field contains:

- harmonic nearest-neighbor bond forces,
- a bending force based on the discrete second difference of the bead coordinates,
- attractive Lennard-Jones interactions for the native contacts,
- short-range repulsive Lennard-Jones interactions for nonbonded, non-native pairs,
- a stiff harmonic anchor on the first bead,
- a moving harmonic trap coupled to the last bead.

The forward protocol holds the trap fixed during the initial plateau, translates it in the x direction during the pulling stage, and holds it fixed again during the final plateau. The reverse protocol is obtained by reversing the trap sequence.

### SLURM array workflow

The trajectory generator is designed to run as a SLURM array. Each array task writes an independent batch to

```text
hairpin_100000_<array-job-id>/batch_<array-task-id>/
```

A typical batch contains:

```text
traj_fwd.npy
traj_rev.npy
trap_fwd.npy
trap_rev.npy
```

Submit the array with:

```bash
mkdir -p logs
sbatch polymer_hairpin/submit_hairpin_array.sh
```

The default setup uses 100 array tasks with 1000 trajectories per task, giving 100,000 forward and 100,000 reverse trajectories.

### Random seeds

Each SLURM array task must use a distinct random seed. Reusing the same base seed in every array task produces repeated trajectory blocks.

For the batch generator, a safe pattern is:

```python
TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
JOB_ID = int(os.environ.get("SLURM_ARRAY_JOB_ID", "0"))
BASE_SEED = int(os.environ.get("BASE_SEED", "123456"))

seed_sequence = np.random.SeedSequence([
    BASE_SEED,
    JOB_ID,
    TASK_ID,
])

rng = np.random.default_rng(seed_sequence)
```

For scripts that use the legacy `np.random.seed`, keep the integer seed in the interval `0 <= seed < 2**32`.

## 3. Merging trajectory batches

After all array tasks finish successfully, merge the batch files with a dependent SLURM job:

```bash
sbatch \
  --dependency=afterok:<ARRAY_JOB_ID> \
  polymer_hairpin/merge_hairpin.sh \
  /mimer/NOBACKUP/groups/naiss2026-4-244/sreekm/output/hairpin_100000_<ARRAY_JOB_ID>
```

For example:

```bash
sbatch \
  --dependency=afterok:10082462 \
  polymer_hairpin/merge_hairpin.sh \
  /mimer/NOBACKUP/groups/naiss2026-4-244/sreekm/output/hairpin_100000_10082462
```

The merged files are written to:

```text
hairpin_100000_<ARRAY_JOB_ID>/final/traj_fwd.npy
hairpin_100000_<ARRAY_JOB_ID>/final/traj_rev.npy
```

After verifying the merged arrays, the intermediate batch directories can be removed from inside the run directory with:

```bash
rm -rf batch_*
```

## 4. Duplicate-trajectory check

Use the following script to detect exact duplicate trajectories:

```python
import hashlib
from collections import defaultdict
import numpy as np

traj = np.load("traj_fwd.npy", mmap_mode="r")

groups = defaultdict(list)

for i in range(traj.shape[0]):
    digest = hashlib.sha256(traj[i].tobytes()).hexdigest()
    groups[digest].append(i)

duplicate_groups = [
    indices for indices in groups.values()
    if len(indices) > 1
]

print("Total trajectories:", traj.shape[0])
print("Unique trajectories:", len(groups))
print("Duplicate groups:", len(duplicate_groups))

for indices in sorted(duplicate_groups, key=len, reverse=True)[:10]:
    print(
        f"Occurrences: {len(indices)}; "
        f"first indices: {indices[:20]}"
    )
```

Preallocated memmap files can also contain unwritten all-zero trajectories. Check them separately:

```python
all_zero = np.all(traj == 0, axis=tuple(range(1, traj.ndim)))
print("Number of all-zero trajectories:", int(all_zero.sum()))
```

## 5. Full-coordinate EGNN inference

The time-conditioned EGNN predicts a bead-wise coefficient vector field from nonequilibrium trajectories.

The molecular configuration is represented as a graph:

- nodes correspond to beads,
- directed edges encode nearest-neighbor backbone interactions,
- next-nearest-neighbor interactions,
- native-contact interactions.

Each node receives a learned bead embedding and a global time embedding. The time embedding is formed from Gaussian basis functions with trainable centers and widths. Edge messages are computed from the sender and receiver node features together with the squared interbead distance. Coordinate updates are built from relative coordinates multiplied by learned scalar weights, preserving E(3) equivariance. Node features are updated from summed incoming messages, and the final node embeddings are mapped to the bead-wise coefficient field.

The full-coordinate jobs can be submitted with:

```bash
mkdir -p logs output
sbatch polymer_hairpin/submit_neep_ml.sh
```

A typical array setup uses:

- four hairpin parameter sets,
- four training-set sizes,
- six independent repeats,
- 96 total array tasks.

## 6. Coarse-grained EGNN inference

The coarse-grained model uses only the x coordinates of selected beads, including beads participating in native contacts and the terminal beads. The graph is reconstructed by retaining physically relevant edges whose endpoints are present in the selected bead set.

Submit the coarse-grained jobs with:

```bash
mkdir -p logs output
sbatch polymer_hairpin/submit_neep_cg.sh
```

The default setup uses four parameter sets and six repeats, giving 24 array tasks.

### Alvis GPU resource requirement

On Alvis, one A40 GPU is associated with 16 CPU cores. Request:

```bash
#SBATCH --gpus-per-node=A40:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
```

Requests such as `--cpus-per-task=2` are rejected because the CPU count does not match the GPU allocation policy.

## 7. Reproducible model training

Each training script accepts a seed as a command-line argument:

```python
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
```

Set all relevant random-number generators before constructing the networks:

```python
import os
import random
import numpy as np
import torch

os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed % (2**32))
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
```

Export `PYTHONHASHSEED` in the SLURM script before starting Python:

```bash
export PYTHONHASHSEED="${SEED}"
```

The forward and reverse networks are constructed sequentially from the same seeded PyTorch stream, so their initializations are reproducible but not identical.

## 8. Training subsets and repeated runs

If repeated runs are intended to use the same training trajectories but different network initializations and minibatches, select:

```python
traj_fwd_train = traj_fwd_all[:N_train]
traj_rev_train = traj_rev_all[:N_train]
```

If each repeat should also use a different random subset, use:

```python
rng = np.random.default_rng(seed)
train_indices = rng.choice(N_all, size=N_train, replace=False)

traj_fwd_train = traj_fwd_all[train_indices]
traj_rev_train = traj_rev_all[train_indices]
```

Save the indices and seed with the model outputs:

```python
np.save(os.path.join(output_dir, "train_indices.npy"), train_indices)
np.save(os.path.join(output_dir, "seed.npy"), np.array([seed], dtype=np.int64))
```

## 9. Outputs

A typical EGNN run produces:

```text
sigma_fwd.npy
sigma_rev.npy
loss_fwd.npy
loss_rev.npy
time_arr.npy
dt_inf.npy
selected_bead_ids.npy
seed.npy
net_fwd_x_native_egnn.pt
net_rev_x_native_egnn.pt
gaussian_params_x_native_egnn.npz
epr_fwd_rev_x_native_egnn.png
```

The cumulative entropy production is computed as:

```python
total_fwd = float(np.sum(sigma_fwd) * dt_inf)
total_rev = float(np.sum(sigma_rev) * dt_inf)
```

## 10. Data availability

The raw and merged trajectory files are large and are not included in this repository. The `data/README.md` file should describe:

- the expected file names and shapes,
- the storage location used for the published analysis,
- instructions for obtaining archived data,
- checksums if the data are deposited externally.

## 11. Suggested `.gitignore`

```gitignore
__pycache__/
*.pyc
.ipynb_checkpoints/

logs/
output/
results/
checkpoints/

*.out
*.err
slurm-*.out

*.npy
*.npz
*.pt
*.pth
*.sif

figures/generated/
```

## 12. Reproducing the paper workflow

A typical end-to-end workflow is:

1. Generate independent forward and reverse hairpin trajectory batches.
2. Verify that batches are not exact duplicates and contain no unwritten zero trajectories.
3. Merge the completed batches after the SLURM array finishes successfully.
4. Train the full-coordinate EGNN for each parameter set, training size, and repeat.
5. Train the coarse-grained EGNN for each parameter set and repeat.
6. Evaluate the inferred entropy-production rates over the full trajectory ensembles.
7. Integrate the entropy-production rates and combine them with the measured work to estimate the equilibrium free-energy differences.
8. Run the analysis scripts to reproduce the convergence plots and paper figures.

## Code availability statement

The code used to generate the simulation data, infer entropy production, estimate equilibrium free-energy differences, and reproduce the figures presented in this work is available in this repository. A permanent archival DOI can be added after creating a tagged GitHub release and linking the repository to Zenodo.

## Citation

A `CITATION.cff` file should be added when the paper metadata and author list are finalized. Until then, please cite the accompanying paper:

> *Inferring Equilibrium Free Energy Differences from Nonequilibrium Fluctuations in High-Dissipation Regimes.*

## License

Add the license selected for the project, for example MIT or BSD-3-Clause.
