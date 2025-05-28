#!/usr/bin/env python
"""Inference script for protein structure prediction using trRosettaX2."""
import os
import argparse
import shutil
import numpy as np

from utils.utils import (
    folding_with_pred_pdb,
    calculate_reliability_score,
    get_npz_from_pred_pdb,
    pred_2d_geometry,
)

def generate_npz_and_pdb(
    pdb_name,
    processed_npz_dir,
    pred_pdb_dir,
    initial_npz,
    fasta,
    N=10,
    Nmax=500,
    begin_num=0,
    sigma=1.0,
    tta_opt="-m 2 -r no-idp --orient ",
    angle=True,
):
    """
    Generate multiple structures iteratively using predicted 2D geometry.

    Args:
        pdb_name (str): Base name for output PDB files.
        processed_npz_dir (str): Directory to store intermediate NPZ files.
        pred_pdb_dir (str): Directory to store predicted PDB structures.
        initial_npz (str): Path to initial predicted NPZ file.
        fasta (str): Path to FASTA file for the sequence.
        N (int): Number of initial structures to generate (default: 10).
        Nmax (int): Maximum number of iterations (default: 500).
        begin_num (int): Starting index for structure numbering (default: 0).
        sigma (float): Sigma parameter for gaussian smoothing (default: 1.0).
        tta_opt (str): Options for Energy Minimization.
        angle (bool): Whether to include orientation angles (default: True).

    Returns:
        int: Total number of structures generated (index of last structure).
    """
    print("Start generating the initial structures")
    # Generate N initial structures using trRosetta��
    folding_with_pred_pdb(
        base_npz=f'"{initial_npz}"',
        base_fasta=f'"{fasta}"',
        base_out=f'"{pred_pdb_dir}"',
        out_name="initial",
        options=tta_opt,
        repeat=N,
    )
    print("Done generating initial structures")

    # Collect initial PDB file names
    initial_pdbs = [os.path.join(pred_pdb_dir, f"initial{i}.pdb") for i in range(N)]
    # Select best initial structure by highest Ramachandran (reliability) score
    best_score = -np.inf
    best_pdb = initial_pdbs[0]
    for pdb in initial_pdbs:
        score = calculate_reliability_score(pdb)
        if score > best_score:
            best_score = score
            best_pdb = pdb

    # Load old distance distribution from the initial NPZ
    old_tmp_dist = np.load(initial_npz)["dist"]
    # Process the best initial structure to update geometry
    if angle:
        processed = get_npz_from_pred_pdb(initial_npz, best_pdb, simga=sigma, angle=angle)
        processed_dist, processed_omega, processed_theta, processed_phi = processed
        tmp_dist = get_npz_from_pred_pdb(initial_npz, best_pdb, simga=sigma, tmp=True)
        labels = {
            "dist": processed_dist,
            "theta": processed_theta,
            "omega": processed_omega,
            "phi": processed_phi,
            "tmp": tmp_dist,
        }
    else:
        processed_dist = get_npz_from_pred_pdb(initial_npz, best_pdb, simga=sigma, angle=angle)
        tmp_dist = get_npz_from_pred_pdb(initial_npz, best_pdb, simga=sigma, tmp=True, angle=angle)
        labels = {"dist": processed_dist, "tmp": tmp_dist}
    new_tmp_dist = tmp_dist

    iter_n = begin_num
    processed_npz_pattern = os.path.join(processed_npz_dir, pdb_name + "{name}.npz")
    # Save the first processed NPZ (index begin_num+1)
    np.savez_compressed(processed_npz_pattern.format(name=begin_num+1), **labels)

    # Iteratively generate new structures until convergence or Nmax reached
    while True:
        iter_n += 1
        current_npz = processed_npz_pattern.format(name=iter_n)
        # Load previous tmp distribution for convergence check (if exists)
        if os.path.exists(current_npz):
            old_tmp_dist = np.load(current_npz)["tmp"]
        print(f"Start generating structure {iter_n}")
        folding_with_pred_pdb(
            base_npz=f'"{current_npz}"',
            base_fasta=f'"{fasta}"',
            base_out=f'"{pred_pdb_dir}"',
            out_name=pdb_name + str(iter_n),
            options=tta_opt,
        )
        print("Done generating structure", iter_n)
        iter_pdb = os.path.join(pred_pdb_dir, f"{pdb_name}{iter_n}.pdb")
        # Check iteration limit
        if iter_n - begin_num < Nmax:
            if angle:
                iter_geom = get_npz_from_pred_pdb(current_npz, iter_pdb, simga=sigma, angle=angle)
                iter_dist, iter_omega, iter_theta, iter_phi = iter_geom
                tmp_dist = get_npz_from_pred_pdb(current_npz, iter_pdb, simga=sigma, tmp=True, angle=angle)
                new_tmp_dist = tmp_dist
                iter_labels = {
                    "dist": iter_dist,
                    "theta": iter_theta,
                    "omega": iter_omega,
                    "phi": iter_phi,
                    "tmp": tmp_dist,
                }
            else:
                iter_dist = get_npz_from_pred_pdb(current_npz, iter_pdb, simga=sigma, angle=angle)
                tmp_dist = get_npz_from_pred_pdb(current_npz, iter_pdb, simga=sigma, tmp=True, angle=angle)
                new_tmp_dist = tmp_dist
                iter_labels = {"dist": iter_dist, "tmp": tmp_dist}
            # Save new processed NPZ (index iter_n+1)
            np.savez_compressed(processed_npz_pattern.format(name=iter_n+1), **iter_labels)
            # Check for convergence (difference in distance distributions)
            diff = np.max(np.abs(old_tmp_dist - new_tmp_dist))
            if diff < 0.01:
                break
        else:
            break

    # print("All structures generation finished.")
    # print(f"Total structures generated: {iter_n}")
    return iter_n

def move_and_delete_subfolders(parent_folder):
    """
    Flatten folder hierarchy: move files up to parent and remove empty subdirectories.
    """
    for root, dirs, files in os.walk(parent_folder, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            target_path = os.path.join(parent_folder, name)
            # Resolve name conflicts by appending a counter
            if os.path.exists(target_path):
                base, ext = os.path.splitext(name)
                counter = 1
                while os.path.exists(target_path):
                    target_path = os.path.join(parent_folder, f"{base}_{counter}{ext}")
                    counter += 1
            shutil.move(file_path, target_path)
        # Remove empty subdirectories
        for name in dirs:
            dir_path = os.path.join(root, name)
            if os.path.isdir(dir_path):
                try:
                    os.rmdir(dir_path)
                except OSError:
                    pass

def run_single(name, fasta_file, msa_file, save_dir, args):
    """
    Run prediction for a single sample given file paths and CLI arguments.
    """
    save_content = os.path.join(save_dir, name)
    save_npz_dir = os.path.join(save_content, "pred_npz/")
    save_pdb_dir = os.path.join(save_content, "pred_pdb/")
    npz_tmp_dir = os.path.join(save_content, "tmp_npz/")
    model_dir = "./trRosettaX2/model_pth"

    os.makedirs(save_npz_dir, exist_ok=True)
    os.makedirs(save_pdb_dir, exist_ok=True)
    os.makedirs(npz_tmp_dir, exist_ok=True)

    # Determine TTA options based on angle flag
    tta_opt = "-m 2 --orient -r no-idp" if args.angle else "-m 2 --no-orient -r no-idp"

    # Generate 2D geometry NPZ using one or two models
    if args.mult_two_models:
        model1 = os.path.join(model_dir, "trX2(NMR)_40.pth")
        model2 = os.path.join(model_dir, "trX2(X-ray)_40.pth")
        pred_2d_geometry(model1, msa_file, save_npz_dir, name + "_NMR", args.device)
        pred_2d_geometry(model2, msa_file, save_npz_dir, name + "_Xray", args.device)
        initial_npz1 = os.path.join(save_npz_dir, name + "_NMR.npz")
        initial_npz2 = os.path.join(save_npz_dir, name + "_Xray.npz")
        npz_tmp_dir1 = os.path.join(npz_tmp_dir, "NMR")
        npz_tmp_dir2 = os.path.join(npz_tmp_dir, "Xray")
        os.makedirs(npz_tmp_dir1, exist_ok=True)
        os.makedirs(npz_tmp_dir2, exist_ok=True)
        # Generate structures from NMR model
        num = generate_npz_and_pdb(
            name, npz_tmp_dir1, os.path.join(save_pdb_dir, "NMR/"), initial_npz1, fasta_file,
            N=args.init_num, Nmax=args.Nmax, angle=args.angle, tta_opt=tta_opt
        )
        # Generate structures from X-ray model, starting index after NMR
        total_num = generate_npz_and_pdb(
            name, npz_tmp_dir2, os.path.join(save_pdb_dir, "Xray/"), initial_npz2, fasta_file,
            N=args.init_num, Nmax=args.Nmax, begin_num=num, angle=args.angle, tta_opt=tta_opt
        )
        print("All structures generation finished.")
        print(f"Total structures generated: {total_num+2*args.init_num}")
    else:
        # Single model (NMR)
        model1 = os.path.join(model_dir, "trX2(NMR)_40.pth")
        pred_2d_geometry(model1, msa_file, save_npz_dir, name + "_NMR", args.device)
        initial_npz1 = os.path.join(save_npz_dir, name + "_NMR.npz")
        total_num = generate_npz_and_pdb(
            name, npz_tmp_dir, os.path.join(save_pdb_dir, "NMR/"), initial_npz1, fasta_file,
            N=args.init_num, Nmax=args.Nmax, angle=args.angle, tta_opt=tta_opt
        )
        print("All structures generation finished.")
        print(f"Total structures generated: {total_num+args.init_num}")
    # Clean up temporary NPZs and flatten PDB output directories
    shutil.rmtree(npz_tmp_dir)
    move_and_delete_subfolders(save_pdb_dir)
    print(f"Inference for sample '{name}' completed. Results in {save_content}")

def main(args):
    """Main function to run inference in single or batch mode based on arguments."""
    if args.name_lst:
        # Batch mode: process multiple sample names
        with open(args.name_lst, 'r') as f:
            names = [line.strip() for line in f if line.strip()]
        for name in names:
            fasta_file = os.path.join(args.fasta_dir, name + ".fasta")
            msa_file = os.path.join(args.msa_dir, name + ".a3m")
            run_single(name, fasta_file, msa_file, args.save_dir, args)
    else:
        # Single-sample mode
        name = args.name
        fasta_file = args.fasta
        msa_file = args.msa
        run_single(name, fasta_file, msa_file, args.save_dir, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Protein structure prediction (inference) using trRosettaX2."
    )
    # Modes: single vs batch
    parser.add_argument("--fasta", type=str, help="Path to single-sample FASTA (.fasta) file.")
    parser.add_argument("--msa", type=str, help="Path to single-sample MSA (.a3m) file.")
    parser.add_argument("--fasta_dir", type=str, help="Directory containing FASTA files for batch mode.")
    parser.add_argument("--msa_dir", type=str, help="Directory containing MSA files for batch mode.")
    parser.add_argument("--name", type=str, help="Sample name (basename without extension) for single mode.")
    parser.add_argument("--name_lst", type=str, help="File with list of sample names (one per line) for batch mode.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save output results.")

    # Optional parameters
    parser.add_argument("--init_num", type=int, default=10,
                        help="Initial number of structures to generate (default: 10).")
    parser.add_argument("--Nmax", type=int, default=300,
                        help="Maximum number of structures/iterations for each model (default: 300).")
    parser.add_argument("--angle", action=argparse.BooleanOptionalAction, default=True,
                        help="Include orientation angles (default: enabled).")
    parser.add_argument("--mult_two_models", action=argparse.BooleanOptionalAction, default=True,
                        help="Use two models (trX2(NMR) and trX2(X-ray)) to predict 2D geometry (default: enabled).")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (e.g., 'cuda:0' or 'cpu').")
    args = parser.parse_args()

    # Validate input mode
    if args.name_lst:
        if not args.fasta_dir or not args.msa_dir:
            parser.error("Batch mode requires --fasta_dir, --msa_dir, and --name_lst.")
    else:
        if not args.fasta or not args.msa or not args.name:
            parser.error("Single mode requires --fasta, --msa, and --name.")

    main(args)
