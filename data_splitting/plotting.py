import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from collections import Counter


def plot_counts(df: pd.DataFrame, parent_dir: Path):
    sizes = []
    lengths = []
    protein_ids = []
    for i in tqdm(df.columns, total=len(df.columns)):
        sizes.append(len(df[i].dropna()))
        for j in df[i].dropna():
            id_, range_ = j.split("/")
            start, end = range_.split("-")
            lengths.append(int(end) - int(start) + 1)
            protein_ids.append(id_)

    print(f"Number of clusters {len(df.columns)}")
    print(f"Number of proteins {len(lengths)}")
    print(f"Number of unique proteins {len(np.unique(protein_ids))}")

    plt.hist(lengths, bins=100)
    plt.xlabel("Length (residues)")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(parent_dir / "length_plot.svg", transparent=True)
    plt.gca()

    plt.hist(sizes, bins=100)
    plt.xlabel("cluster size")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(parent_dir / "cluster_plot.svg", transparent=True)
    plt.gca()

    c = Counter(Counter(protein_ids).values())
    plt.bar(c.keys(), c.values())
    plt.yscale("log")
    plt.xlabel("domains per protein")  # TODO not sure?
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.savefig(parent_dir / "domains_plot.svg", transparent=True)
    plt.gca()


def create_histogram(identities, binwidth, full_pident, diversity_path: Path):
    file_names = []
    for ident in identities:
        file_names += [f"{diversity_path}/DataBase_{ident}/db_occ.dat"]

    _, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    for file_name, pident in zip(file_names, identities):
        data = pd.read_csv(file_name, header=None)
        bins_ = range(min(data[0]), max(data[0]) + binwidth, binwidth)
        ax0.hist(data[0], bins=bins_, histtype="bar", label=str(pident), alpha=0.4)
    ax0.legend(prop={"size": 10})
    ax0.set_xlabel("Cluster size", fontsize=12)
    ax0.set_ylabel("Counter", fontsize=12)
    ax0.xaxis.set_tick_params(labelsize=10)
    ax0.yaxis.set_tick_params(labelsize=10)
    ax0.set_yscale("log")
    ax0.set_title(f"Distribution of cluster sizes")
    y = []
    for file_name, pident in zip(file_names, full_pident):
        data = pd.read_csv(file_name, header=None)
        y += [len(data)]
    x = full_pident
    y = np.array(y) / sum(data[0]) * 100
    ax1.xaxis.set_ticks(x)
    ax1.plot(x, y, "*-", color="gray")
    ax1.set_xlabel(
        "Minimum Percentage of sequence\nidentity in each cluster (%)", fontsize=12
    )
    ax1.set_ylabel("*1 - Quantization (%)", fontsize=12)
    ax1.xaxis.set_tick_params(labelsize=10)
    ax1.yaxis.set_tick_params(labelsize=10)
    ax1.set_title(f"# of Clusters vs Minimum Percentage of Identity")
    plt.subplots_adjust(wspace=0.65)
    plt.tight_layout()
    plt.savefig(diversity_path / "DatasetDiversity.svg", transparent=True)
    plt.gca()


def plot_final_clusters(clusters: pd.DataFrame, output_path: Path):
    counter = (
        clusters.groupby(by=["graph_id"])
        .count()
        .reset_index()
        .sort_values(by=["node_name"], ascending=False)
    )
    plt.hist(counter.node_name.values, bins=100)
    plt.yscale("log")
    plt.ylabel("Counts")
    plt.xlabel("Cluster sizes")
    plt.tight_layout()
    plt.savefig(output_path / "Final_cluster_sizes.svg", transparent=True)
    plt.gca()


def plot_final_lengths(
    training_sequences: pd.DataFrame,
    validation_sequences: pd.DataFrame,
    test_sequences: pd.DataFrame,
    output_path: Path,
):
    _, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(training_sequences.length.values, bins=100, label="training")
    axes[1].hist(validation_sequences.length.values, bins=100, label="validation")
    axes[2].hist(test_sequences.length.values, bins=100, label="test")
    for ax in axes:
        ax.set_xlabel("Lengths (residues)")
        ax.set_ylabel("Counts")
        ax.set_yscale("log")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / "Final_lengths.svg", transparent=True)
    plt.gca()


def plot_final_identities(
    test_train: pd.DataFrame,
    val_train: pd.DataFrame,
    test_val: pd.DataFrame,
    output_path: Path,
):
    _, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(test_train.pident.values, bins=100, label="test - training")
    axes[1].hist(val_train.pident.values, bins=100, label="validation - training")
    axes[2].hist(test_val.pident.values, bins=100, label="test - validatikon")
    for ax in axes:
        ax.set_xlabel("Sequence identity (%)")
        ax.set_ylabel("Counts")
        ax.set_yscale("log")
        ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / "Final_identities.svg", transparent=True)
    plt.gca()
