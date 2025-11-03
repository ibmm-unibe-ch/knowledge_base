"""
ESM3 Embedding Generator

This script generates protein sequence embeddings using the ESM3 model.
It can process individual sequences or batch process FASTA files.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import glob
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import torch


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the ESM3 embedding generator.
    
    Returns:
        argparse.Namespace: Parsed command line arguments with attributes:
            - sequence: Optional input protein sequence string
            - path: Optional path to FASTA file or directory
            - depth: Directory search depth for FASTA files
            - output: Optional output file or directory path
            - char_embedding: use sentence-level embeddings or character-level embeddings
    """
    parser = argparse.ArgumentParser(description="Generate ESM3 embeddings for protein sequences.")
    parser.add_argument(
        "--sequence", 
        type=str,
        help="A protein sequence to process. If provided, --path and --depth are ignored."
    )
    parser.add_argument(
        "--path", 
        type=str,
        help="Path to FASTA file or directory (used only if --sequence is not provided)."
    )
    parser.add_argument(
        "--depth", 
        type=int,
        help="Directory search depth for FASTA files (used only if --sequence is not provided)."
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="Output file or directory path for saving embeddings."
    )
    parser.add_argument(
        "--char_embedding",
        action="store_true",
        help="If set, return character-level embeddings instead of mean pooled embeddings."
    )
    
    args = parser.parse_args()
    
    # Enforce conditional argument logic
    if (args.sequence is None) and (args.path is None):
        parser.error("When --sequence is not provided, both --path and --depth are required.")
    
    return args


def read_fasta(fasta_paths: List[Path], sequence_amount: Optional[int] = 1) -> Dict[str, str]:
    """
    Read FASTA files and extract protein sequences.
    
    Args:
        fasta_paths: List of paths to FASTA files to read
        sequence_amount: Maximum number of sequences to read from each file
        
    Returns:
        Dictionary mapping sequence IDs to sequence strings. Sequence IDs are
        constructed as "filename___header" to ensure uniqueness.
        
    Raises:
        FileNotFoundError: If any FASTA file cannot be found
        IOError: If there are issues reading the FASTA files
    """
    sequences: Dict[str, str] = {}
    
    for fasta_path in fasta_paths:
        count = sequence_amount
        with open(fasta_path, 'r') as f:
            seq_id: Optional[str] = None
            seq_lines: List[str] = []
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith(">"):
                    # Save previous sequence if exists
                    if seq_id and count > 0:
                        sequences[seq_id] = "".join(seq_lines)
                        count -= 1
                    
                    # Start new sequence - take first word after '>' and prefix with filename
                    seq_id = f"{fasta_path}___{line[1:].split()[0]}"
                    seq_lines = []
                else:
                    seq_lines.append(line)
            
            # Don't forget the last sequence in the file
            if seq_id and count > 0:
                sequences[seq_id] = "".join(seq_lines)
                
    return sequences


def find_fasta_files(base_path: str, depth: int) -> List[Path]:
    """
    Find all .fasta files under base_path up to a given directory depth.
    
    Args:
        base_path: The base directory to search for FASTA files
        depth: Maximum subdirectory depth to search (0 = only base directory)
        
    Returns:
        List of Path objects for all matching .fasta files
        
    Raises:
        ValueError: If base_path doesn't exist or depth is negative
    """
    base = Path(base_path).resolve()
    
    if not base.exists():
        raise ValueError(f"Base path {base_path} does not exist")
    
    if depth < 0:
        raise ValueError("Depth cannot be negative")
    
    if depth == 0:
        pattern = str(base / "*.fasta")
    else:
        # Build pattern for recursive search up to specified depth
        pattern = str(base / ("*/" * depth) / "*.fasta")
    
    return [Path(p) for p in glob.glob(pattern, recursive=False)]


def get_sequences(args: argparse.Namespace) -> Tuple[List[str], List[Path]]:
    """
    Get sequences and output paths based on command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Tuple containing:
            - List of sequence strings to process
            - List of output file paths for saving embeddings
            
    Raises:
        ValueError: If no valid sequences are found
    """
    if args.sequence:
        # Single sequence mode
        sequences = [args.sequence]
        
        if args.output is None:
            output_paths = [Path("ESM3_embedding.pt")]
        else:
            output_path = Path(args.output)
            if output_path.suffix:  # Has file extension
                output_paths = [output_path]
            else:
                output_paths = [output_path / "ESM3_embedding.pt"]
    else:
        # FASTA file mode
        if str(args.path).endswith(".fasta"):
            fasta_paths = [Path(args.path)]
        else:
            fasta_paths = find_fasta_files(args.path, args.depth)
        
        if not fasta_paths:
            raise ValueError(f"No FASTA files found at path: {args.path}")
        
        sequence_dict = read_fasta(fasta_paths)
        sequences = list(sequence_dict.values())
        output_paths = [
            Path(str(path).replace(".fasta", "_ESM3_embedding.pt")) 
            for path in fasta_paths
        ]
    
    return sequences, output_paths


def embed_sequences(sequences: List[str], char_embedding: bool = True) -> List[torch.Tensor]:
    """
    Generate ESM3 embeddings for a list of protein sequences.
    
    Args:
        sequences: List of protein sequence strings to embed
        char_embedding: If True, return character-level embeddings;
                       if False, return mean-pooled sequence embeddings
                       
    Returns:
        List of embedding tensors for each input sequence
        
    Raises:
        RuntimeError: If CUDA is requested but not available
        ValueError: If sequences are invalid or empty
    """
    if not sequences:
        raise ValueError("No sequences provided for embedding")
    
    # Initialize ESM3 model
    client = ESMC.from_pretrained("esmc_300m").to("cuda")  # or "cpu"
    embeddings: List[torch.Tensor] = []
    
    for sequence in sequences:
        if not sequence or not isinstance(sequence, str):
            raise ValueError(f"Invalid sequence: {sequence}")
            
        protein = ESMProtein(sequence=sequence)
        protein_tensor = client.encode(protein)
        
        # Get logits and embeddings
        logits_output = client.logits(
            protein_tensor, 
            LogitsConfig(sequence=True, return_embeddings=True)
        )
        
        # logits_output.embeddings shape: [1, length+2, 960]
        # (batch=1, seq_length=sequence_length+beginining+cls token, embedding_dim=960)
        if char_embedding:
            embeddings.append(logits_output.embeddings[0])
        else:
            # Use mean pooling as recommended by ESM authors
            # See: https://github.com/evolutionaryscale/esm/issues/116
            #      https://github.com/evolutionaryscale/esm/issues/162
            embeddings.append(torch.mean(logits_output.embeddings[0], dim=0))
    
    return embeddings


def save_embedding(embedding: torch.Tensor, path: Union[str, Path]) -> None:
    """
    Save a PyTorch embedding tensor to disk.
    
    Args:
        embedding: PyTorch tensor containing the embedding to save
        path: File path where the embedding will be saved
        
    Raises:
        IOError: If the file cannot be written
        PermissionError: If lacking permissions to write to the path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    torch.save(embedding, path)


def write_embeddings(embeddings: List[torch.Tensor], paths: List[Path]) -> None:
    """
    Save multiple embeddings to their respective file paths.
    
    Args:
        embeddings: List of embedding tensors to save
        paths: List of file paths for saving each embedding
        
    Raises:
        ValueError: If the number of embeddings and paths don't match
    """
    if len(embeddings) != len(paths):
        raise ValueError(f"Mismatch between embeddings count ({len(embeddings)}) and paths count ({len(paths)})")
    
    for path, embedding in zip(paths, embeddings):
        save_embedding(embedding, path)


def main() -> None:
    """
    Main function to orchestrate ESM3 embedding generation.
    
    Handles argument parsing, sequence loading, embedding generation,
    and result saving.
    """
    args = parse_args()
    
    try:
        sequences, output_paths = get_sequences(args)
        print(f"Processing {len(sequences)} sequence(s)...")
        
        embeddings = embed_sequences(sequences, args.char_embedding)
        print(f"Generated {len(embeddings)} embedding(s)")
        
        write_embeddings(embeddings, output_paths)
        print(f"Saved embeddings to: {[str(p) for p in output_paths]}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
