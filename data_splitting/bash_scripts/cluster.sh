#!/bin/sh

################################## Inputs ##################################
# $1 --> Input dataset in FASTA format
# $2 --> Output folder
# $3 --> Minimum sequence identity for search
# $4 --> Path to MMseqs2 binary
# $5 --> Final output path for alignment results
############################################################################

echo "##### Cluster Sequences #####"

# Create output directory if it doesn't exist
mkdir -p "$2"

echo "[Step 1] Creating database from input sequences..."
"$4" createdb "$1" "$2/db" > "$2/dbcreatedb.log"

echo "[Step 2] Creating index file for the database..."
"$4" createindex "$2/db" tmp > "$2/ind.log"

echo "[Step 3] Running MMseqs2 search for sequence clustering..."
"$4" search "$2/db" "$2/db" "$2/resultdb" tmp \
  --min-seq-id "$3" \
  -s 7.0 \
  -c 0.0 \
  --cov-mode 0 \
  --alignment-mode 3 > "$2/search.log"

echo "[Step 4] Converting results to human-readable alignment format..."

# Output fields:
# query, target, percent identity, alignment length, mismatches, gap opens,
# query start/end, target start/end, E-value, bit score, query coverage
"$4" convertalis "$2/db" "$2/db" "$2/resultdb" "$2/Alignment_Results" \
  --format-output "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qcov" \
  > "$2/convertalis.log"

# Move final result to the desired output location
mv "$2/Alignment_Results" "$5"

echo "[Done] Alignment results saved to $5"