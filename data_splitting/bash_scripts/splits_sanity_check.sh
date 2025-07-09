#!/bin/sh

################################### Inputs ###################################
# $1 --> Training dataset (FASTA)
# $2 --> Test dataset (FASTA)
# $3 --> Validation dataset (FASTA)
# $4 --> Output folder
# $5 --> MMseqs2 binary folder
# $6 --> Specified splitting pident
##############################################################################

echo "##### CHECKING OVERLAP BETWEEN DATASETS #####"

# Ensure output folder exists
mkdir -p "$4"

# Set MMseqs2 binary
MMSEQS_BINARY="$5"
SPLITTING_PIDENT="$6"

# Utility: Create MMseqs2 database
create_database() {
    local dataset="$1"
    local output_dir="$2"
    mkdir -p "$output_dir"
    "$MMSEQS_BINARY" createdb "$dataset" "$output_dir/db" > "$output_dir/dbcreatedb.log"
    "$MMSEQS_BINARY" createindex "$output_dir/db" tmp > "$output_dir/ind.log"
    echo "$output_dir/db"
}

# Utility: Search one DB against another
search_database() {
    local result_db="$1"
    local output_file="$2"
    local query_db="$3"
    local target_db="$4"

    echo "===================================================================================="
    echo "[INFO] Running MMseqs2 search between:"
    echo "       Query : $query_db"
    echo "       Target: $target_db"
    echo "===================================================================================="

    "$MMSEQS_BINARY" search "$query_db" "$target_db" "$result_db" tmp \
        -s 7.0 -c 0.0 --cov-mode 0 --alignment-mode 3 > "$result_db.search.log"
    mkdir -p "${4}_dir"
    "$MMSEQS_BINARY" convertalis "$query_db" "$target_db" "$result_db" "${4}_dir/Alignment_results" \
        --format-output "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qcov" \
        > "$result_db.convertalis.log"

    echo "[RESULT] Sequences with identity > 30%: $(awk -F'\t' '$3 > 30 { print $2 }' "${4}_dir/Alignment_results" | sort -u | wc -l)"
    echo "[RESULT] Sequences with identity > 50%: $(awk -F'\t' '$3 > 50 { print $2 }' "${4}_dir/Alignment_results" | sort -u | wc -l)"
    echo "[RESULT] Sequences with identity > $SPLITTING_PIDENT%: $(awk -v pident=$SPLITTING_PIDENT -F'\t' '$3 > pident { print $2 }' "${4}_dir/Alignment_results" | sort -u | wc -l)"
    mv "${4}_dir/Alignment_results" "$output_file"
}

# Create databases
train_db=$(create_database "$1" "$4/train")
test_db=$(create_database "$2" "$4/test")
val_db=$(create_database "$3" "$4/val")

# Perform pairwise overlap checks
search_database "$4/TestValdb"   "$4/Test_Val.aln"   "$test_db"  "$val_db"
search_database "$4/ValTraindb"  "$4/Train_Val.aln"  "$train_db" "$val_db"
search_database "$4/TestTraindb" "$4/Test_Train.aln" "$test_db"  "$train_db"

echo "##### All comparisons complete. #####"
