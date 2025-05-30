#!/usr/bin/env python
"""Structure clustering script to group similar PDB models."""
import os
import argparse
from utils_trX2dy.utils import save_cluster_result

def main():
    parser = argparse.ArgumentParser(
        description="Cluster predicted structures based on GloCon or TM-score/RMSD."
    )
    parser.add_argument("--pdb_dir", "-d", required=True, type=str,
                        help="Directory containing PDB files to cluster.")
    parser.add_argument("--mode", "-m", choices=["glocon", "tmscore", "rmsd"], default="glocon",
                        help="Clustering mode: 'glocon', 'tmscore', or 'rmsd'.")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="Directory to save clustering results (default: pdb_dir/clusters_result).")
    parser.add_argument("--n_clusters", type=int, default=10,
                        help="Number of clusters for KMeans (default: 10).")
    parser.add_argument("--n_files", type=int, default=5,
                        help="Number of files to keep per cluster (default: 5).")
    args = parser.parse_args()

    pdb_dir = args.pdb_dir
    mode = args.mode
    n_clusters = args.n_clusters
    n_files = args.n_files
    output_dir = args.output_dir or os.path.join(pdb_dir, "clusters_result")
    os.makedirs(output_dir, exist_ok=True)

    result = save_cluster_result(pdb_dir,
                                n_clusters=n_clusters,
                                n_files=n_files,
                                output_dir=output_dir,
                                mode=mode)
    if result == "no_cluster":
        print("Clustering failed or not possible.")
    else:
        print(f"Clustering completed. Results saved in {output_dir}.")

if __name__ == "__main__":
    main()
