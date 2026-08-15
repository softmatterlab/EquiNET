# EquiNET

This repository contains the simulation, inference, and analysis code accompanying the paper:

> **Inferring Equilibrium Free Energy Differences from Nonequilibrium Fluctuations in High-Dissipation Regimes**

The workflow estimates equilibrium free-energy differences from typical nonequilibrium trajectories by inferring entropy production using a variational inference scheme based on the short-time thermodynamic uncertainty relation.

The repository contains two applications:

* a one-dimensional double-well demonstration;
* a three-dimensional polymer-hairpin model analyzed using time-conditioned E(3)-equivariant graph neural networks (EGNNs).

The repository includes the simulation and inference code, scripts used to generate the datasets, processed data required to reproduce the figures, and Apptainer definition files specifying the computational environments used for the simulations and neural-network training.

The full polymer-hairpin datasets are computationally intensive to generate and were produced using high-performance computing resources. The simulations are divided into independent batches that can be run in parallel on a compute cluster. Importantly, the largest dataset contains (10^5) trajectories, but substantially smaller datasets already provide good inference results, with approximately 500--1000 trajectories sufficient in many of the cases considered here.

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
apptainer build hairpin.sif PolymerHairpin/Simulation/cluster/hairpin.def
```

Depending on the HPC system, Apptainer images may need to be built outside the cluster or using a remote builder.

## Repository structure

```text
.
├── DoubleWell
│   └── DoubleWell_Inference.ipynb
│
├── PolymerHairpin
│   ├── Inference
│   │   ├── cluster
│   │   │   └── hairpin.def
│   │   ├── scripts
│   │   │   └── run_egnn.sh
│   │   └── src
│   │       ├── egnn.py
│   │       └── egnn_cg.py
│   │
│   ├── Plotting
│   │   ├── Data
│   │   └── Plotting_Script
│   │
│   └── Simulation
│       ├── cluster
│       │   └── hairpin.def
│       ├── scripts
│       │   ├── merge_hairpin.sh
│       │   └── submit_hairpin_array.sh
│       └── src
│           └── hairpin_simulation.py
│
├── LICENSE
├── README.md
└── requirements.txt
```

Generated figures and other output files are omitted from the structure above for clarity.

## Double-well example

The one-dimensional double-well example is contained in:

```text
DoubleWell/
```

The complete inference workflow is implemented in:

```text
DoubleWell/DoubleWell_Inference.ipynb
```

To run the notebook:

```bash
jupyter notebook DoubleWell/DoubleWell_Inference.ipynb
```

The notebook performs the inference and reproduces the corresponding double-well results.

## Polymer-hairpin workflow

The polymer-hairpin calculation consists of three main stages:

1. generating nonequilibrium trajectories;
2. performing EquiNET inference;
3. reproducing the figures from the processed data.

The corresponding directories are:

```text
PolymerHairpin/
├── Simulation/
├── Inference/
└── Plotting/
```

## 1. Generating polymer-hairpin trajectories

The simulation code is located in:

```text
PolymerHairpin/Simulation/src/hairpin_simulation.py
```

The polymer hairpin consists of 13 beads evolving in three dimensions. A trajectory contains approximately (9\times10^5) integration steps. To reduce storage requirements, configurations are not written at every integration step; instead, the system is stored every 5000 simulation steps.

Thus, the simulation time resolution and the stored trajectory resolution are different: the underlying dynamics are integrated using all simulation steps, while only periodically sampled configurations are saved for subsequent inference.

### Number of trajectories

The largest datasets used in the paper contain **100,000 independent trajectories**. This represents the maximum dataset size used in our calculations rather than a minimum requirement for obtaining useful results.

In practice, good inference results are already obtained with substantially fewer trajectories. For many of the cases considered in the paper, approximately **500--1000 trajectories** are sufficient to obtain accurate estimates.

The larger datasets are used primarily to characterize convergence and the behavior of the inference method as the amount of available data is increased.

### Running a reduced simulation

For testing the code locally, a reduced number of trajectories can be used. In particular, datasets containing several hundred to approximately 1000 trajectories are much less computationally demanding than the largest production runs and are already representative of the regime in which EquiNET performs well.

The simulation script can be run with:

```bash
python PolymerHairpin/Simulation/src/hairpin_simulation.py
```

The exact simulation parameters are defined in the simulation script.

### Running production simulations on a cluster

For larger datasets, the trajectories are generated in independent batches of **1000 trajectories**.

The largest dataset is obtained using:

```text
100 batches × 1000 trajectories = 100000 trajectories
```

Since the individual trajectories and batches are independent, the calculation is embarrassingly parallel and can be distributed efficiently across compute nodes.

The cluster submission script is:

```text
PolymerHairpin/Simulation/scripts/submit_hairpin_array.sh
```

For a SLURM-based system, the job array can typically be submitted with:

```bash
sbatch PolymerHairpin/Simulation/scripts/submit_hairpin_array.sh
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
PolymerHairpin/Simulation/scripts/merge_hairpin.sh
```

Run:

```bash
bash PolymerHairpin/Simulation/scripts/merge_hairpin.sh
```

The resulting merged dataset is then used as input for the EquiNET inference.

## 2. Running EquiNET inference

The EquiNET inference code for the polymer hairpin is located in:

```text
PolymerHairpin/Inference/src/
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
PolymerHairpin/Inference/scripts/run_egnn.sh
```

For a SLURM-based system, the inference job can typically be submitted with:

```bash
sbatch PolymerHairpin/Inference/scripts/run_egnn.sh
```

Before running the inference, users should check the paths to:

* the merged simulation dataset;
* the output directory;
* the Apptainer image;
* any model or training parameters specified in the submission script.

Full-scale inference can be performed on the largest datasets, but this is not required to test the method. A dataset containing approximately 500--1000 trajectories already provides a useful reduced-scale example and, for many of the cases studied here, gives results close to those obtained with substantially larger datasets.

## 3. Reproducing the figures

The processed data used to produce the polymer-hairpin figures are provided in:

```text
PolymerHairpin/Plotting/Data/
```

The corresponding plotting scripts are located in:

```text
PolymerHairpin/Plotting/Plotting_Script/
```

The plotting scripts operate directly on the supplied processed data. Therefore, reproducing the published figures does **not** require rerunning the full simulations or neural-network training.

## Recommended workflow

For users who want to test the complete pipeline, we recommend:

1. Clone the repository and install the dependencies.
2. Generate a reduced dataset, for example 500--1000 trajectories.
3. Verify that the expected simulation output is produced.
4. If multiple batches are used, combine them with `merge_hairpin.sh`.
5. Run EquiNET on the resulting dataset.
6. Verify that the inference output is generated correctly.
7. Use the supplied processed data to reproduce the figures.
8. Increase the number of trajectories only if a larger-scale convergence study is desired.

There is therefore no need to generate the full (10^5)-trajectory dataset simply to test or use the method.

## Computational requirements

Each polymer-hairpin trajectory consists of approximately **900,000 integration steps**, with the system configuration stored every **5000 steps**.

The maximum dataset used in the paper contains:

* 100 independent batches;
* 1000 trajectories per batch;
* 100,000 trajectories in total.

These full-scale calculations are best suited to HPC resources. However, this maximum dataset size should not be interpreted as a requirement for obtaining useful results. In many cases, good estimates are already obtained using approximately **500--1000 trajectories**.

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
