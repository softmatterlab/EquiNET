# EquiNET

This repository contains the simulation, inference, and analysis code accompanying the paper:

> **Inferring Equilibrium Free Energy Differences from Nonequilibrium Fluctuations in High-Dissipation Regimes**

The method is named EquiNET and it estimates equilibrium free-energy differences from nonequilibrium trajectories by inferring entropy production.

The repository contains two applications:

* a one-dimensional double-well demonstration;
* a three-dimensional polymer-hairpin model analyzed using a time-conditioned E(3)-equivariant graph neural networks (EGNNs).

A self-contained demonstration notebook is provided for the double well demonstration

The repository for the polymer hairpin model includes simulation and inference code and scripts used to generate the datasets, and Apptainer definition files specifying the computational environments used for the simulations and neural-network training.

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

## Demonstration notebook for the polymer hairpin model

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

The number of integration steps and the storage strides in the demonstration notebook are reduced by the corresponding factor of 10. The production scripts in `polymer_hairpin/simulation/` and `polymer_hairpin/inference/` retain the numerical settings used for the paper.

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

### Running production simulations on a cluster

For larger datasets, the trajectories are generated in independent batches of **1000 trajectories**.

The largest dataset is obtained using:

```text
100 batches × 1000 trajectories = 100000 trajectories
```

Since the individual trajectories and batches are independent, the calculation can be distributed efficiently across compute nodes.

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

Run the script and provide the path to the folder containing the batch files:

```bash
bash polymer_hairpin/simulation/scripts/merge_hairpin.sh /path/to/batch_folder
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
7. Use the supplied script in the jupyter notebook to reproduce the figures.

## License

Released under the MIT License.
