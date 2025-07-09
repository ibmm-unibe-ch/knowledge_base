import pandas as pd
from tqdm import tqdm
import os
import shutil
import networkx as nx
import multiprocessing as mp
from argparse import ArgumentParser
from pathlib import Path
from plotting import (
    create_histogram,
    plot_final_clusters,
    plot_final_identities,
    plot_final_lengths,
)
from utils import read_data, save_split, find_last_index, pandas_to_fasta
from constants import MMSEQS_PATH, BASH_SCRIPTS_PATH
from typing import List, Tuple


def make_diversitycheck(diversity_path: Path, fasta_path: Path, own_pident:int):  # sequences.fasta
    if not diversity_path.is_dir():
        diversity_path.mkdir(parents=True, exist_ok=True)
        os.system(
            f"bash {BASH_SCRIPTS_PATH}/DiversityCheck.sh {fasta_path} {MMSEQS_PATH} {diversity_path} {own_pident}"
        )
    pident = sorted(set(range(20, 100, 10)) | {own_pident})
    print(f"Pident Ranges for the histogram: {pident}")
    create_histogram(pident, 50, pident, diversity_path)


def clustering_sequences(
    seqs: pd.DataFrame, pident: int, output_path: Path
) -> pd.DataFrame:
    # Graph based clustering of input sequences
    nodes = seqs.id.unique()  # Get all sequence ids to be added to the graph
    splitting_path = output_path / "Splitting"
    splitting_path.mkdir(parents=True, exist_ok=True)
    pandas_to_fasta(seqs, splitting_path / "dataset.fasta")
    # Pair-wise search to get the pident between all sequences in the dataset
    os.system(
        f"bash {BASH_SCRIPTS_PATH}/cluster.sh {splitting_path}/dataset.fasta {splitting_path}/DataBase {pident/100} {MMSEQS_PATH} {splitting_path}/pident_{pident}_cluster.tsv"
    )
    aln = pd.read_csv(
        f"{splitting_path}/pident_{pident}_cluster.tsv",
        header=None,
        sep="\t",
        names=[
            "query",
            "target",
            "pident",
            "aln_length",
            "mismatches",
            "gaps",
            "qstart",
            "qend",
            "tstart",
            "tend",
            "evalue",
            "bit_score",
            "qcov",
        ],
    )
    aln = aln[aln["query"] != aln.target]  # remove the identity matches
    filtered_aln = aln[
        aln["pident"] > pident
    ]  # filter out rows with lower pident than the given

    # Graph creation
    G = nx.Graph()  # initialize the graph
    G.add_nodes_from(nodes)  # add all nodes to the graph
    # Add edges to the graph
    G = add_edges_to_graph(filtered_aln, G)

    # Find all connected components (subgraphs)
    connected_components = list(nx.connected_components(G))
    subgraphs = []
    for c in tqdm(
        nx.connected_components(G),
        total=len(connected_components),
        desc="Splitting to subgraphs",
    ):
        subgraphs.append(G.subgraph(c).copy())
    data = []
    # Iterate over subgraphs and collect node information
    for i, sg in tqdm(
        enumerate(subgraphs),
        total=len(subgraphs),
        desc="Iterating through the subgraphs",
    ):
        nodes = list(sg.nodes())
        for node in nodes:
            data.append({"graph_id": i, "node_name": node})
    return pd.DataFrame(data)


def add_edges_to_graph(
    filtered_aln: pd.DataFrame, G: nx.Graph, batch_size: int = 1000
) -> nx.Graph:
    # Divide filtered_aln into batches
    row_batches = [
        filtered_aln.iloc[i : i + batch_size]
        for i in range(0, len(filtered_aln), batch_size)
    ]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_batch, row_batches),
                total=len(row_batches),
                desc="Adding edges to graph",
            )
        )

    # Flatten the list of results
    edges = [edge for batch in results for edge in batch]

    # Add edges to the graph in the main process
    for edge in edges:
        G.add_edge(*edge)
    return G


# Define the function to process each batch of rows
def process_batch(row_batch: pd.DataFrame) -> List:
    # Ensure row_batch is a DataFrame
    if not isinstance(row_batch, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(row_batch)}")
    edges = [(row["query"], row["target"]) for _, row in row_batch.iterrows()]
    return edges


def make_sets(
    seqs: pd.DataFrame, clusters: pd.DataFrame, training_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counter = (
        clusters.groupby(by=["graph_id"])
        .count()
        .reset_index()
        .sort_values(by=["node_name"], ascending=False)
    )
    training_size = training_size * len(clusters)
    training_graph_ids = counter.head(
        find_last_index(counter.node_name.values, training_size)
    ).graph_id.values
    training_sequences = seqs[
        seqs.id.isin(clusters.node_name[clusters.graph_id.isin(training_graph_ids)])
    ]
    print(f"Size of the training set is: ", len(training_sequences))

    aln = clusters[~clusters.node_name.isin(training_sequences.id)]
    counter = (
        aln.groupby(by=["graph_id"])
        .count()
        .reset_index()
        .sort_values(by=["node_name"], ascending=False)
    )
    validation_size = 0.5 * len(aln)
    validation_graph_ids = counter.head(
        find_last_index(counter.node_name.values, validation_size)
    ).graph_id.values
    validation_sequences = seqs[
        seqs.id.isin(aln.node_name[aln.graph_id.isin(validation_graph_ids)])
    ]
    print(f"Size of the validation set is: ", len(validation_sequences))

    test_sequences = seqs[
        ~(seqs.id.isin(training_sequences.id))
        & ~(seqs.id.isin(validation_sequences.id))
    ]
    print(f"Size of the test set is: ", len(test_sequences))
    return training_sequences, validation_sequences, test_sequences


def make_sanity_check(
    training_sequences: pd.DataFrame,
    validation_sequences: pd.DataFrame,
    test_sequences: pd.DataFrame,
    output_path: Path,
    splitting_pident: int,
):
    # Splits sanity check
    print("\n\n--INFO: Performing splits sanity check...")
    sanity_check_path = output_path / "Splitting" / "SanityCheck"
    sanity_check_path.mkdir(parents=True, exist_ok=True)
    # Save the splits into fasta files
    pandas_to_fasta(training_sequences, sanity_check_path / "train.fasta")
    pandas_to_fasta(validation_sequences, sanity_check_path / "val.fasta")
    pandas_to_fasta(test_sequences, sanity_check_path / "test.fasta")
    # Run sanity check
    os.system(
        f"bash {BASH_SCRIPTS_PATH}/splits_sanity_check.sh {sanity_check_path}/train.fasta {sanity_check_path}/test.fasta {sanity_check_path}/val.fasta {sanity_check_path} {MMSEQS_PATH} {splitting_pident}"
    )


def exclude_similar_seqs(
    seqs: pd.DataFrame, output_path: Path, exclude_pident: int,
) -> pd.DataFrame:  # Exclude very similar sequences
    print(
        f"\n\n--INFO: Excluding sequences more similar than {exclude_pident}% identity..."
    )
    aln = pd.read_csv(
        output_path / f"DataBase_{exclude_pident}/db_clu.tsv",
        header=None,
        sep="\t",
        names=["centroid", "element"],
    )
    print("Total number of sequences: ", len(aln))
    seqs = seqs[seqs.id.isin(aln.centroid)]
    print(
        f"Total number of sequences after removing sequences with over {exclude_pident}% identity: ",
        len(seqs),
    )
    return seqs


def main(args):
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    seqs = read_data(input_path)
    # plot_counts(df, output_path/"counts")
    pandas_to_fasta(seqs, output_path / "input.fasta")
    make_diversitycheck(output_path / "DiversityCheck", output_path / "input.fasta", int(args.exclude_pident))
    # Load sequences
    if (output_path / "Splitting").exists():
        shutil.rmtree((output_path / "Splitting"), ignore_errors=True)
    seqs["length"] = seqs["sequence"].str.len()
    seqs = exclude_similar_seqs(seqs, output_path / "DiversityCheck", exclude_pident=args.exclude_pident)
    print(
        f"\n\n--INFO: Excluding sequences shorter than {args.minimum_length} amino acids..."
    )
    seqs = seqs[seqs["length"] > args.minimum_length]
    print(
        f"Total number of sequences after removing sequences shorter than {args.minimum_length} amino acids: ",
        len(seqs),
    )
    # Cluster sequences at identity, default 30
    print(f"\n\n--INFO: Clustering sequences at {args.splitting_pident}% identity...")
    clusters = clustering_sequences(seqs, args.splitting_pident, output_path)
    # Split into training, validation, and test sets
    print("\n\n--INFO: Splitting into training, validation, and test sets...")
    training_sequences, validation_sequences, test_sequences = make_sets(
        seqs, clusters, args.training_size/100
    )
    make_sanity_check(
        training_sequences,
        validation_sequences,
        test_sequences,
        output_path,
        args.splitting_pident,
    )
    sanity_check_path = output_path / "Splitting" / "SanityCheck"
    test_train = pd.read_csv(
        sanity_check_path / "Test_Train.aln",
        header=None,
        sep="\t",
        names=[
            "query",
            "target",
            "pident",
            "aln_length",
            "mismatches",
            "gaps",
            "qstart",
            "qend",
            "tstart",
            "tend",
            "evalue",
            "bit_score",
            "qcov",
        ],
    )
    val_train = pd.read_csv(
        sanity_check_path / "Train_Val.aln",
        header=None,
        sep="\t",
        names=[
            "query",
            "target",
            "pident",
            "aln_length",
            "mismatches",
            "gaps",
            "qstart",
            "qend",
            "tstart",
            "tend",
            "evalue",
            "bit_score",
            "qcov",
        ],
    )
    test_val = pd.read_csv(
        sanity_check_path / "Test_Val.aln",
        header=None,
        sep="\t",
        names=[
            "query",
            "target",
            "pident",
            "aln_length",
            "mismatches",
            "gaps",
            "qstart",
            "qend",
            "tstart",
            "tend",
            "evalue",
            "bit_score",
            "qcov",
        ],
    )

    output_path_output = output_path / "Output"
    output_path_output.mkdir(parents=True, exist_ok=True)
    plot_final_clusters(clusters, output_path_output)
    plot_final_lengths(
        training_sequences, validation_sequences, test_sequences, output_path_output
    )
    plot_final_identities(test_train, val_train, test_val, output_path_output)
    save_split(training_sequences, output_path_output / "training.fasta")
    save_split(validation_sequences, output_path_output / "validation.fasta")
    save_split(test_sequences, output_path_output / "test.fasta")


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "--input_path",
        type=str,
        help="Path to input file with just sequences (and index)",
    )
    argparser.add_argument("--output_path", type=str, help="Ouput directory")
    argparser.add_argument(
        "--exclude_pident", type=int,help="Exclusion identity percentage ([20, 30, 40, 50, 60, 70, 80, 90])"
    #choices=[20, 30, 40, 50, 60, 70, 80, 90], 
    )
    argparser.add_argument(
        "--minimum_length", type=int, help="Minimum length of sequences"
    )
    argparser.add_argument(
        "--splitting_pident",
        type=int,
        help="Splitting identity percentage (in integers)",
    )
    argparser.add_argument(
        "--training_size",
        type=int,
        help="Size of training set in percentages (in integers)",
    )
    args = argparser.parse_args()
    main(args)
