# Data splitting script
Script for [splitting data](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets) by first creating a seq identity graph before splitting this graph into a train and a test set.

## Usage
You need to install [MMseqs2](https://github.com/soedinglab/MMseqs2/) and give the path to it in `constants.py`. (You can get the path easily by typing `which mmseqs`)

The main command script is `data_splitting.py` and check the code or the help from the script to use it like so in the end:

`python data_splitting.py --input_path PATH_TO_YOUR_INPUT/file.csv --output_path PATH_TO_YOUR_OUTPUT_DIR --exclude_pident 90 --minimum_length 20 --splitting_pident 30 --training_size 80`

The used libraries are pretty minimal, but we provide a [mamba](https://mamba.readthedocs.io/en/latest/index.html) environment to load the libraries in `env.yml`.

You can see a potential output in file `example_output.txt`
### Authors
The idea and initial code came from Symela Lazaridi and was refactored by Jannik Gut.
If you have question, address them to symela.lazaridi@unibe.ch or jannik.gut@unibe.ch.
