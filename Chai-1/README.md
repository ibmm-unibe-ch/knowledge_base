# 🧪 Chai-1: Protein and Molecular Modeling Framework

Chai-1 is a deep learning framework for molecular modeling, protein embeddings, and protein–ligand simulations. This repository provides utilities for downloading pretrained models, handling conformers, and running ESM-based embeddings.

---

## ⚙️ 1. Installation

It is recommended to install Chai-1 inside a dedicated conda environment to avoid dependency conflicts.

### 1.1 Create and Activate Conda Environment

```bash
# Create a new conda environment
conda create -n chai_env python=3.10 -y

# Activate the environment
conda activate chai_env
```

### 1.2 Install Chai-1 and Dependencies
```bash
# Install Chai-1 directly from GitHub
pip install git+https://github.com/chaidiscovery/chai-lab.git

# Install supporting packages
pip install ipython transformers tqdm
```

These packages provide:

- chai-lab → main framework (model loading, utilities, chemistry modules)
- ipython → enhanced interactive Python shell
- tqdm → print progress bar
- transformers → required for ESM (protein language model) support

### 1.3 Download Model Weights
Before running Chai-1, you must download the pretrained models and cached resources. Use the helper script:

```bash
python download_chai.py
```

This script will automatically:

- Download pretrained Chai-1 model weights (*.pt files)
- Cache RDKit conformers
- Fetch the ESM-2 tokenizer and model from Hugging Face

### 1.4 Verify Installation
```bash
python -c "import chai_lab; print('Chai-1 successfully installed!')"
```
Expected output:
```
Chai-1 successfully installed!
```

## ⚙️ 2. Usage

### 2.1 Example Input File

Your input CSV should have the following columns:
```
Protein_ID,Sequence,Ligand
prot1,MVLSPADKTNVKAA...,LIG
prot2,AVLIPFSTCWYQN...,None
```
Ligand can be the ligand SMILES representation. Use None if no ligand is present. Example of input in `example.csv`.

### 2.2 How to run

```bash
conda activate chai_env

python run_inference.py \
    --input_csv example.csv \
    --output_dir results_chai \
    --summary_csv results_chai/summary.csv \
    --batch_size 4 \
    --gpus 0 1 2 \
    --num_trunk_recycles 3 \
    --num_diffn_timesteps 200 \
    --use_esm_embeddings True \
    --seed 42
```

#### Explanation of Inputs

| Argument | Description | Example / Notes |
|----------|-------------|----------------|
| `--input_csv` | Path to your input CSV file containing sequences to fold. The CSV must have the columns: `Protein_ID`, `Sequence`, `Ligand`. If no ligand is present, use `None`. | `sequences.csv` |
| `--output_dir` | Directory where all Chai-1 outputs will be saved. Each protein gets its own subfolder containing all predicted structures. | `results_chai` |
| `--summary_csv` | CSV file where a summary of predictions (e.g., pLDDT scores, confidence metrics) will be stored. | `results_chai/summary.csv` |
| `--batch_size` | Number of sequences to process per GPU at a time. Useful to balance memory and throughput. | `4` |
| `--gpus` | List of GPU device indices to use. If not provided, the script will use all available GPUs. | `0 1 2` |
| `--num_trunk_recycles` | Number of trunk recycles in Chai-1 for structural refinement. Higher values may improve accuracy but increase runtime. | `3` |
| `--num_diffn_timesteps` | Number of diffusion timesteps in Chai-1. Determines how thoroughly the model explores conformations. | `200` |
| `--use_esm_embeddings` | Boolean flag to use ESM embeddings as input features. Helps improve folding accuracy. | `True` |
| `--seed` | Random seed for reproducibility of results. | `42` |

This command will:

- Load your sequences from sequences.csv.
- Use GPUs 0, 1, 2 in parallel, with a batch size of 4 sequences per GPU.
- Save all candidate structures for each sequence in results_chai/.
- Produce a summary.csv with per-sequence metrics for downstream analysis.

### 2.3 Running chai-1 on a cluster using slurm

The `chai1.slurm` script allows you to run Chai-1 on a GPU cluster using SLURM. Below is an example and explanation of the SLURM flags and inputs:

#### Example `chai1.slurm` Script

```bash
#!/bin/bash
#SBATCH --job-name="chai_test"
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:h100:2
#SBATCH --mem-per-gpu=90G

module load CUDA
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate chai_env

python run_inference.py \
    --input_csv example.csv \
    --output_dir results_chai \
    --summary_csv results_chai/summary.csv \
    --batch_size 2
```

| Flag                                 | Description                                                                                                                                                            |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--job-name="chai_test"` | Name of the job displayed in the scheduler.                                                                                                                            |
| `--time=01:00:00`        | Maximum runtime for the job (HH:MM:SS).                                                                                                                                |
| `--partition=gpu` | Partition (queue) to submit the job. Use `gpu` or `gpu-invest` for standard GPU jobs.                                                                                  |
| `--qos=job_gpu_preemptable`   | Quality of Service. If partition=`gpu-invest`, use `job_gpu_preemptable` for preemptable jobs. If partition=`gpu`, use `job_gpu`.                                 |
| `--gres=gpu:h100:2`      | Requests 2 GPUs of type `h100`. Adjust based on GPU type and availability.                                                                                             |
| `--mem-per-gpu=90G`      | Allocates 90 GB RAM per GPU. Adjust depending on dataset size and GPU memory.                                                                                          |


#### Submitting the job
```bash
sbatch chai1.slurm
```

