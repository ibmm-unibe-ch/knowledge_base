import pandas as pd
from pathlib import Path
from typing import List
from Bio import SeqIO

def read_data(data_name: Path) -> pd.DataFrame:
    # Read data
    print(f"Read {data_name}")
    if data_name.name.endswith("xlsx"):
        # Read clusters excel file
        print("Read as excel")
        df = pd.read_excel(data_name, skiprows=1, header=None, names=["id", "sequence"])
    elif data_name.name.endswith("tsv"):
        print("Read as tsv")
        df = pd.read_csv(data_name, "\t", header=None, names=["id", "sequence"])
    elif data_name.name.endswith(".fasta"):
        records = []
        for record in SeqIO.parse(data_name, "fasta"):
            records.append({'id': record.id, 'sequence': str(record.seq)})
        df = pd.DataFrame(records)
    else:
        print("Read as csv")
        df = pd.read_csv(data_name, header=None, sep=",", names=["id", "sequence"])
    return df


def save_split(df: pd.DataFrame, filename: Path):
    # Save pandas to fasta file
    print(f"Save split to {filename}")
    with open(filename, "w") as f:
        for _, i in df.iterrows():
            f.write(f">{i.id}\n{i.sequence}\n")
        f.close()


def find_last_index(arr: List, x: float) -> List:
    running_sum = 0
    for i in range(len(arr)):
        running_sum += arr[i]
        if running_sum >= x:
            return i
    return len(arr) - 1


def pandas_to_fasta(df: pd.DataFrame, filename: Path):
    # Save pandas to fasta file
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        for _, i in df.iterrows():
            f.write(f">{i.id}\n{i.sequence}\n")
        f.close()
