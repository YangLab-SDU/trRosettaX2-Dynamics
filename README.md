# trRosettaX2-Dynamics: Predicting Alternative Protein Conformations through a Novel Output-Driven Approach



Installation
----
### Step 1. Clone the repository

```bash
git clone https://github.com/YangLabSDU/trX2-Mult.git
cd trRosettaX2-Dynamics
```
### Step 2. Environment installation

A new environment for the trX2-Mult can be created and activated with conda 

```
# create conda environment for trX2-Mult
conda env create -f environment.yml

# acitvate the installed environment
conda activate trX2dy
```
----



Usage
----
### Running Inference
The main script inference.py performs structure prediction. It supports both single-sample and batch modes.
* **Single-sample mode:** Predict structure for one sequence.
	* **example**
  ```bash
  python run_inference.py \
    --fasta ./example/seq.fasta \
    --msa ./data/seq.a3m \
    --name seq \
    --save_dir ./example/output \
    [--init_num 20] [--Nmax 300] [--angle/--no-angle] [--mult_two_models/--no-mult_two_models]
  ```

  * `--fasta`: Path to the FASTA file.
  * `--msa`: Path to the MSA (.a3m) file.
  * `--name`: Base name (identifier) for this sample.
  * `--save_dir`: Directory where results will be saved.
  * Optional:

    * `--init_num`: Initial number of structures to generate (default: 10).
    * `--Nmax`: Maximum number of iterations (default: 300).
    * `--angle`/`--no-angle`: Include (or exclude) orientation angles (default: included).
    * `--mult_two_models`/`--no-mult_two_models`: Use two trX2 variant (NMR and X-ray)  or only trRosettaX2 (NMR) (default: used).

* **Batch mode:** Predict structures for multiple sequences.

	* **example**
  
  ```bash
  python run_inference.py \
    --fasta_dir ./example \
    --msa_dir ./example \
    --name_lst ./example/name_lst \
    --save_dir ./example/output \
  [--init_num 10] [--Nmax 300] [--angle/--no-angle] [--mult_two_models/--no-mult_two_models]
  ```
  
  * `--fasta_dir`: Directory containing FASTA files (`{name}.fasta`).
  * `--msa_dir`: Directory containing MSA files (`{name}.a3m`).
  * `--name_lst`: Text file with one sample name per line (without extensions).
  * `--save_dir`: Directory where all results will be saved (each sample in its own subfolder).

Each sample named `name` will have results in `save_dir/name/`.

### Evaluation (Optional)

Use `evaluate.py` to compare predicted structures against native structures:

```bash
python evaluate.py \
  --native_dir /path/to/native_pdbs \
  --pred_dir /path/to/predicted_pdbs \
  [--output /path/to/summary.txt] \
  [--align]
```

* `--native_dir`: Directory of native (reference) PDB files.
* `--pred_dir`: Directory of predicted PDB files.
* `--output`: (Optional) Output summary file or directory. If ending in `.txt`, a file is created; otherwise a directory is used (summary.txt inside). Default is `pred_dir/summary.txt`.
* `--align`: Use sequence alignment in TM-score (`-seq` option).

The script prints summary statistics (min RMSD, max TM-score, etc.) and saves detailed results to the specified output.

### Clustering (Optional)

Use `cluster.py` to cluster predicted structures:

```bash
python cluster.py \
  --pdb_dir /path/to/predicted_pdbs \
  [--mode glocon] \
  [--output_dir /path/to/clusters] \
  [--n_clusters 10] \
  [--n_files 5]
```

* `--pdb_dir`: Directory containing PDB models to cluster.
* `--mode`: Clustering mode (`glocon`, `tmscore`, or `rmsd`; default: `glocon`).
* `--output_dir`: Directory to save clustering results (default: `<pdb_dir>/clusters_result`).
* `--n_clusters`: Number of clusters for KMeans (default: 10).
* `--n_files`: Number of top files to keep per cluster (default: 5).

The script copies the top `n_files` structures from each cluster into the output directory.

---

**Note:** 

* Ensure all paths and file names are correctly specified. Consult the inline help (`-h` or `--help`) for each script for detailed usage.




