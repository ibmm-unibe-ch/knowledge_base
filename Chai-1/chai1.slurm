#!/bin/bash
#SBATCH --job-name="test"
#SBATCH --time=00:10:00
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
