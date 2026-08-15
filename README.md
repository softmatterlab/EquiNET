# EquiNET

This repository contains the simulation, inference, and analysis code accompanying the paper:

> **Inferring Equilibrium Free Energy Differences from Nonequilibrium Fluctuations in High-Dissipation Regimes**

The workflow estimates equilibrium free-energy differences from typical nonequilibrium trajectories by inferring entropy production using a variational inference scheme based on the short-time thermodynamic uncertainty relation.

The repository contains two applications:

* a one-dimensional double-well demonstration;
* a three-dimensional polymer-hairpin model analyzed using time-conditioned E(3)-equivariant graph neural networks (EGNNs).

The repository includes simulation and inference code, scripts used to generate the datasets, processed data required to reproduce the figures, and Apptainer definition files specifying the computational environments used for the simulations and neural-network training.

The full polymer-hairpin datasets are computationally intensive to generate and were produced using high-performance computing resources. The simulations are divided into independent batches that can be run in parallel on a compute cluster. The largest dataset used in the paper contains (10^5) trajectories, but substantially smaller datasets already provide good inference results, with approximately 500--1000 trajectories sufficient in many of the cases considered here.

A self-contained demonstration notebook is also provided for the polymer-hairpin example. To make the complete workflow practical to run interactively, this notebook uses a coarser integration timestep than the production simulations used for the paper. The distinction between the demonstration and production settings is described below.

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/sreekmnoneq/EquiNET.git
cd EquiNET
```

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

For the production calculations, the provided Apptainer definition files can be used to reproduce the computational environments used for the simulations and neural-network training.

For example, the polymer-hairpin simulation container can be built with:

```bash
apptainer build hairpin.sif polymer_hairpin/simulation/cluster/hairpin.def
```

Depending on the HPC system, Apptainer images may need to be built outside the cluster or using a remote builder.

## Repository structure

```text
.
├── LICENSE
├── README.md
├── double_well
│   ├── double_well_inference.ipynb
│   ├── protocol_plot.pdf
│   └── work_entropy.pdf
│
├── polymer_hairpin
│   ├── hairpin_demo.ipynb
│   │
│   ├── inference
│   │   ├── cluster
│   │   │   └── hairpin.def
│   │   ├── scripts
│   │   │   └── run_egnn.sh
│   │   └── src
│   │       ├── egnn.py
│   │       └── egnn_cg.py
│   │
│   ├── plotting
│   │   ├── data
│   │   │   ├── Fig_3.npz
│   │   │   └── Fig_4.npz
│   │   └── script
│   │       ├── Fig_3.py
│   │       └── Fig_4.py
│   │
│   └── simulation
│       ├── cluster
│       │   └── hairpin.def
│       ├── scripts
│       │   ├── merge_hairpin.sh
│       │   └── submit_hairpin_array.sh
│       └── src
│           └── hairpin_simulation.py
│
└── requirements.txt
```

Generated figures and other output files are omitted from the structure above for clarity.

## Double-well example

The one-dimensional double-well example is contained in:

```text
double_well/
```

The complete inference workflow is implemented in:

```text
double_well/double_well_inference.ipynb
```

To run the notebook:

```bash
jupyter notebook double_well/double_well_inference.ipynb
```

The notebook performs the inference and reproduces the corresponding double-well results.

## Polymer-hairpin workflow

The polymer-hairpin calculation consists of three main stages:

1. generating nonequilibrium trajectories;
2. performing EquiNET inference;
3. reproducing the figures from the processed data.

The corresponding directories are:

```text
polymer_hairpin/
├── simulation/
├── inference/
└── plotting/
```

A self-contained demonstration notebook is additionally provided in:

```text
polymer_hairpin/hairpin_demo.ipynb
```

## Demonstration notebook

`polymer_hairpin/hairpin_demo.ipynb` provides a self-contained version of the polymer-hairpin workflow that can be run in Google Colab or locally in Jupyter.

The notebook includes:

* trajectory generation;
* work-distribution sampling;
* full-coordinate EquiNET inference;
* coarse-grained EquiNET inference;
* analysis and visualization routines.

To reduce the computational cost of running the complete workflow interactively, the demonstration notebook uses a coarser integration timestep,

```text
dt = 1e-4
```

instead of the finer timestep used for the production calculations reported in the paper,

```text
dt = 1e-5
```

The number of integration steps and the storage strides in the demonstration notebook are reduced by the corresponding factor of 10. Consequently, the physical durations of the protocol stages and the time interval between stored configurations are kept unchanged.

For the trajectory simulations, the corresponding settings are:

| Parameter                          | Demonstration notebook | Paper calculations |
| ---------------------------------- | ---------------------: | -----------------: |
| Integration timestep               |                 `1e-4` |             `1e-5` |
| Initial equilibration              |         `17,000` steps |    `170,000` steps |
| Pulling stage                      |         `60,000` steps |    `600,000` steps |
| Final equilibration                |         `15,000` steps |    `150,000` steps |
| Trajectory storage stride          |            `500` steps |       `5000` steps |
| Time between stored configurations |                 `0.05` |             `0.05` |

For example,

```text
500 × 1e-4 = 5000 × 1e-5 = 0.05
```

so the temporal resolution of the trajectories supplied to EquiNET is the same in the demonstration and production settings.

The demonstration notebook is intended as an accessible example of the complete simulation-to-inference workflow. **All numerical results reported in the paper were generated using the finer production timestep `dt = 1e-5` and the corresponding production step counts and storage strides.**

The production scripts in `polymer_hairpin/simulation/` and `polymer_hairpin/inference/` retain the numerical settings used for the paper.

## 1. Generating polymer-hairpin trajectories

The production simulation code is located in:

```text
polymer_hairpin/simulation/src/hairpin_simulation.py
```

The polymer hairpin consists of 13 beads evolving in three dimensions.

For the production simulations used in the paper, the integration timestep is:

```text
dt = 1e-5
```

A trajectory contains approximately (9\times10^5) integration steps. To reduce storage requirements, configurations are not written at every integration step; instead, the system is stored every 5000 simulation steps.

Thus, the integration timestep and the stored trajectory resolution are different: the dynamics are evolved using all integration steps, while only periodically sampled configurations are saved for subsequent inference.

The self-contained demonstration notebook uses the computationally lighter discretization described above and should therefore not be confused with the finer production settings used to generate the published results.

### Number of trajectories

The largest datasets used in the paper contain **100,000 independent trajectories**. This represents the maximum dataset size used in the calculations rather than a minimum requirement for obtaining useful results.

In practice, good inference results are already obtained with substantially fewer trajectories. For many of the cases considered in the paper, approximately **500--1000 trajectories** are sufficient to obtain accurate estimates.

The larger datasets are used primarily to characterize convergence and the behavior of the inference method as the amount of available data is increased.

### Running a reduced simulation

For a quick demonstration of the complete workflow, use:

```text
polymer_hairpin/hairpin_demo.ipynb
```

The notebook uses a reduced computational setup and is intended to make the full pipeline easier to run interactively.

Alternatively, the production simulation script can be run directly:

```bash
python polymer_hairpin/simulation/src/hairpin_simulation.py
```

The exact simulation parameters are defined in the simulation script.

Datasets containing several hundred to approximately 1000 trajectories are much less computationally demanding than the largest production runs and are already representative of the regime in which EquiNET performs well.

### Running production simulations on a cluster

For larger datasets, the trajectories are generated in independent batches of **1000 trajectories**.

The largest dataset is obtained using:

```text
100 batches × 1000 trajectories = 100000 trajectories
```

Since the individual trajectories and batches are independent, the calculation is embarrassingly parallel and can be distributed efficiently across compute nodes.

The cluster submission script is:

```text
polymer_hairpin/simulation/scripts/submit_hairpin_array.sh
```

For a SLURM-based system, the job array can typically be submitted with:

```bash
sbatch polymer_hairpin/simulation/scripts/submit_hairpin_array.sh
```

The supplied submission script reflects the HPC environment used for the calculations in the paper and may need to be adapted for another system.

Users should check:

* partition or queue names;
* requested CPUs and memory;
* wall-time limits;
* job-array ranges;
* Apptainer image paths;
* input and output directories;
* filesystem paths specific to the cluster.

The number of submitted batches can be reduced depending on the desired dataset size. For example, a single batch produces 1000 trajectories and is already sufficient for testing the complete inference pipeline.

### Merging simulation batches

After all requested simulation batches have completed, the individual output files can be combined using:

```text
polymer_hairpin/simulation/scripts/merge_hairpin.sh
```

Run:

```bash
bash polymer_hairpin/simulation/scripts/merge_hairpin.sh
```

The resulting merged dataset is then used as input for the EquiNET inference.

## 2. Running EquiNET inference

The EquiNET inference code for the polymer hairpin is located in:

```text
polymer_hairpin/inference/src/
```

The main scripts are:

```text
egnn.py
egnn_cg.py
```

`egnn.py` performs the inference using the full set of observed coordinates considered in the model.

`egnn_cg.py` performs the inference using the reduced, partially observed representation considered in the paper.

The cluster submission script is:

```text
polymer_hairpin/inference/scripts/run_egnn.sh
```

For a SLURM-based system, the inference job can typically be submitted with:

```bash
sbatch polymer_hairpin/inference/scripts/run_egnn.sh
```

Before running the inference, users should check the paths to:

* the merged simulation dataset;
* the output directory;
* the Apptainer image;
* any model or training parameters specified in the submission script.

Full-scale inference can be performed on the largest datasets, but this is not required to test the method. A dataset containing approximately **500--1000 trajectories** already provides a useful reduced-scale example and, for many of the cases studied here, gives results close to those obtained with substantially larger datasets.

## 3. Reproducing the figures

The processed data used to produce the polymer-hairpin figures are provided in:

```text
polymer_hairpin/plotting/data/
```

The supplied processed data include:

```text
polymer_hairpin/plotting/data/Fig_3.npz
polymer_hairpin/plotting/data/Fig_4.npz
```

The corresponding plotting scripts are located in:

```text
polymer_hairpin/plotting/script/
```

and are:

```text
polymer_hairpin/plotting/script/Fig_3.py
polymer_hairpin/plotting/script/Fig_4.py
```

For example:

```bash
python polymer_hairpin/plotting/script/Fig_3.py
python polymer_hairpin/plotting/script/Fig_4.py
```

The plotting scripts operate directly on the supplied processed data. Therefore, reproducing the published figures does **not** require rerunning the full simulations or neural-network training.

## Recommended workflow

For users interested primarily in testing the method, we recommend starting with:

```text
polymer_hairpin/hairpin_demo.ipynb
```

This provides the simplest way to run the complete simulation and inference pipeline with reduced computational cost.

For users who want to reproduce the production workflow more closely:

1. Clone the repository and install the dependencies.
2. Generate a reduced dataset using the production simulation code, for example 500--1000 trajectories.
3. Verify that the expected simulation output is produced.
4. If multiple batches are used, combine them with `merge_hairpin.sh`.
5. Run EquiNET on the resulting dataset.
6. Verify that the inference output is generated correctly.
7. Use the supplied processed data to reproduce the figures.
8. Increase the number of trajectories only if a larger-scale convergence study is desired.

There is therefore no need to generate the full (10^5)-trajectory dataset simply to test or use the method.

## Computational requirements

The production polymer-hairpin simulations reported in the paper use:

```text
dt = 1e-5
```

with approximately **900,000 integration steps per trajectory** and configurations stored every **5000 steps**.

The maximum dataset used in the paper contains:

* 100 independent batches;
* 1000 trajectories per batch;
* 100,000 trajectories in total.

These full-scale calculations are best suited to HPC resources. However, this maximum dataset size should not be interpreted as a requirement for obtaining useful results. In many cases, good estimates are already obtained using approximately **500--1000 trajectories**.

The demonstration notebook uses `dt = 1e-4` together with proportionally reduced step counts and storage strides to make the workflow substantially faster while preserving the protocol durations and stored-data time resolution.

Users interested only in reproducing the published figures do not need access to a cluster, since the processed plotting data are included in the repository.

## Adapting the cluster scripts

The supplied cluster scripts correspond to the computing environment used for the calculations in the paper. They will generally require some modification before being used on another HPC system.

Users should check:

* scheduler directives;
* partition or queue names;
* CPU and GPU requests;
* memory requirements;
* wall-time limits;
* job-array settings;
* Apptainer image locations;
* input and output paths;
* Python, CUDA, or other environment settings.

The scientific code itself is not tied to a particular scheduler. In most cases, only the cluster submission scripts and filesystem paths need to be adapted.

## License

Released under the MIT License.
