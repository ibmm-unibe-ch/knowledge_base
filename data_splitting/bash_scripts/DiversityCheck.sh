#!/bin/sh

################################## Inputs ##################################
# $1 --> Input dataset in FASTA format
# $2 --> Folder containing MMseqs2 binary
# $3 --> Output folder
# $4 --> Own pident
############################################################################

# Save current working directory
cwd=$(pwd)

# Create working database directory and copy input FASTA
mkdir -p "$3/DataBase"
cp "$1" "$3/DataBase/"

# Extract filename from path
file=$(basename "$1")

echo "##### Step 1: Create MMseqs2 Database from FASTA #####"

cd "$3/DataBase" || exit 1
"$2" createdb "$file" db > out.out
cd "$cwd" || exit 1

# Run clustering with varying percent identities
for pident in "$4" 20 30 40 50 60 70 80 90; do
    echo "##### Step 2: Clustering sequences with minimum identity: ${pident}% #####"

    out_dir="$3/DataBase_${pident}"
    mkdir -p "$out_dir"

    "$2" cluster "$3/DataBase/db" "$out_dir/db_clu" tmp \
        --min-seq-id "0.$pident" > "$out_dir/out.out"

    "$2" createtsv "$3/DataBase/db" "$3/DataBase/db" "$out_dir/db_clu" "$out_dir/db_clu.tsv" \
        > "$out_dir/out_.out"

    # Count unique representatives in the clustering result
    awk -F '\t' '{print $1}' "$out_dir/db_clu.tsv" | sort | uniq -c | awk '{print $1}' > "$out_dir/db_occ.dat"
done

echo "##### All clustering tasks completed. #####"
