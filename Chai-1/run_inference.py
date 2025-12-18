#!/usr/bin/env python3
import os
import sys
import torch
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Process, Queue
from tqdm import tqdm
from chai_lab.chai1 import run_inference
import contextlib
import datetime

# --- Utility print with timestamps ---
print_ = print
def print(*args, **kwargs):
    ts = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print_(ts, *args, **kwargs)

# --- Suppress stdout/stderr (to silence Chai internal logs) ---
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err

# --- Argument parser ---
parser = ArgumentParser(description="Run Chai-1 on multiple sequences with optional ligands.")
parser.add_argument("--input_csv", required=True, type=str,
                    help="CSV file with columns: Protein_ID, Sequence, Ligand(optional)")
parser.add_argument("--output_dir", required=True, type=str,
                    help="Directory where predictions and models will be stored")
parser.add_argument("--summary_csv", required=True, type=str,
                    help="Path to write summary CSV of predictions")
parser.add_argument("--batch_size", type=int, default=4,
                    help="Number of sequences per GPU batch")
parser.add_argument("--recycles", type=int, default=3,
                    help="Number of trunk recycles for Chai-1 inference")
parser.add_argument("--timesteps", type=int, default=200,
                    help="Number of diffusion timesteps")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
args = parser.parse_args()

# --- Input handling ---
df = pd.read_csv(args.input_csv, keep_default_na=False, na_filter=False)
required_cols = {"Protein_ID", "Sequence"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Input CSV must contain columns: {required_cols}")

# Add Ligand column if missing
if "Ligand" not in df.columns:
    df["Ligand"] = ""
else:
    df["Ligand"] = df["Ligand"].fillna("").astype(str)

print(f"✅ Loaded {len(df)} sequences from {args.input_csv}")

# --- Create output directories ---
output_root = Path(args.output_dir)
output_root.mkdir(parents=True, exist_ok=True)
tmp_in = output_root / "tmp_in"
tmp_out = output_root / "tmp_out"
tmp_in.mkdir(parents=True, exist_ok=True)
tmp_out.mkdir(parents=True, exist_ok=True)

# --- Core function to run Chai inference on a subset of data ---
def run_chai_batch(sub_df, gpu_id):
    results = []
    for _, row in sub_df.iterrows():
        pid, seq, lig = row["Protein_ID"], row["Sequence"], row["Ligand"]

        fasta_file = tmp_in / f"{pid}.fasta"
        with open(fasta_file, "w") as f:
            if lig and lig.strip():
                f.write(f">protein|name={pid}\n{seq}\n>ligand|name=LIG2\n{lig}\n")
            else:
                f.write(f">protein|name={pid}\n{seq}\n")

        out_dir = tmp_out / pid
        out_dir.mkdir(parents=True, exist_ok=True)

        with suppress_output():
            candidates = run_inference(
                fasta_file=fasta_file,
                output_dir=out_dir,
                num_trunk_recycles=args.recycles,
                num_diffn_timesteps=args.timesteps,
                seed=args.seed,
                device=f"cuda:{gpu_id}",
                use_esm_embeddings=True,
            )

        # Extract all model data
        for i, rd in enumerate(candidates.ranking_data):
            results.append({
                "Protein_ID": pid,
                "Model_Index": i,
                "Aggregate_Score": rd.aggregate_score.item(),
                "Complex_pLDDT": float(rd.plddt_scores.complex_plddt),
                "Protein_pLDDT": float(rd.plddt_scores.per_chain_plddt[0][0]),
                "Ligand_pLDDT": float(rd.plddt_scores.per_chain_plddt[0][1]) if len(rd.plddt_scores.per_chain_plddt[0]) > 1 else None,
                "Complex_pTM": float(rd.ptm_scores.complex_ptm),
                "Protein_pTM": float(rd.ptm_scores.per_chain_ptm[0][0]),
                "Ligand_pTM": float(rd.ptm_scores.per_chain_ptm[0][1]) if len(rd.ptm_scores.per_chain_ptm[0]) > 1 else None,
                "CIF_Path": str(candidates.cif_paths[i]),
                "Output_Folder": str(out_dir)
            })

    return results

# --- Multiprocessing setup ---
def gpu_worker(gpu_id, queue_in, queue_out):
    while True:
        task = queue_in.get()
        if task is None:
            break
        idx, sub_df = task
        res = run_chai_batch(sub_df, gpu_id)
        queue_out.put(res)

available_gpus = list(range(torch.cuda.device_count()))
if not available_gpus:
    raise RuntimeError("No GPUs detected! Please ensure CUDA is available.")

print(f"🧠 Detected GPUs: {available_gpus}")

batch_size = args.batch_size
chunks = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]
queue_in, queue_out = Queue(), Queue()

for idx, chunk in enumerate(chunks):
    queue_in.put((idx, chunk))
for _ in available_gpus:
    queue_in.put(None)

# --- Start GPU workers ---
workers = []
for gpu_id in available_gpus:
    p = Process(target=gpu_worker, args=(gpu_id, queue_in, queue_out))
    p.start()
    workers.append(p)

print(f"🚀 Starting Chai inference on {len(chunks)} batches...")
all_results = []
for _ in tqdm(range(len(chunks)), total=len(chunks), desc="Running batches"):
    res = queue_out.get()
    if res:
        all_results.extend(res)

for p in workers:
    p.join()

# --- Save summary ---
summary_df = pd.DataFrame(all_results)
summary_df.to_csv(args.summary_csv, index=False)
print(f"✅ All predictions complete! Summary written to: {args.summary_csv}")
print(f"🧩 Raw outputs stored in: {args.output_dir}")
