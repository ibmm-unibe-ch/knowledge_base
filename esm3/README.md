# ESM3 Embeddings
Some help in running [ESM3](https://github.com/evolutionaryscale/esm) on our servers.
## Installation on Skywalker / Vader
Create a new environment called _esm3_ with Python 3.12
```
micromamba create --name esm3 -c conda-forge python=3.12 
```
Otherwise, to use ESM3 in a pre-existing environment, you may need to update your Python version with
```
micromamba install update python
```
Install a [Pytorch](https://pytorch.org/) version compatible with our CUDA version and esm
```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 
pip install esm
```
Optionally, install [flash-attention](https://github.com/Dao-AILab/flash-attention) to be a bit faster and use less memory.
```
pip install flash-attn --no-build-isolation
```
## Predict embeddings
ESM3C is suggested to make better embeddings, if you want to predict a 3D-structure, then you may change the model to ESM3 without suffix.
There seems to be no mode for batches, that is why we do every protein after another, but you can save some time by loading the model only once.

The script that might help you is provided in `embed.py`

```
Parse command line arguments for the ESM3 embedding generator.
Returns:
   argparse.Namespace: Parsed command line arguments with attributes:
   - sequence: Optional input protein sequence string
   - path: Optional path to FASTA file or directory
   - depth: Directory search depth for FASTA files
   - output: Optional output file or directory path
   - char_embedding: use sentence-level embeddings or character-level embeddings
```
