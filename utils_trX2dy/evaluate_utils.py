import os
import re
import subprocess
import numpy as np

def parse_compare_score_file(file_path=None, file=None, flag="file"):
    """
    Parse TM-score output and extract best RMSD and TM-score for each structure pair.
    """
    if flag == "file_path":
        with open(file_path, "r") as fh:
            content = fh.read()
    else:  # flag == "file"
        content = file
    blocks = content.split("--------------------------------------------------")
    results = {}
    for block in blocks:
        if "Structure1:" in block and "Structure2:" in block:
            chain1_name = re.search(r"Structure1:\s+(\S+)", block).group(1)
            chain2_name = re.search(r"Structure2:\s+(\S+)", block).group(1)
            rmsd = float(re.search(r"RMSD of  the common residues=\s+([\d.]+)", block).group(1))
            tm_score = float(re.search(r"TM-score    =\s+([\d.]+)", block).group(1))
            if chain1_name not in results:
                results[chain1_name] = {"best_rmsd": (rmsd, chain2_name), "best_tm_score": (tm_score, chain2_name)}
            else:
                # Update best RMSD and TM-score if found
                if rmsd < results[chain1_name]["best_rmsd"][0]:
                    results[chain1_name]["best_rmsd"] = (rmsd, chain2_name)
                if tm_score > results[chain1_name]["best_tm_score"][0]:
                    results[chain1_name]["best_tm_score"] = (tm_score, chain2_name)
    return results

def run_score(native_pdb_dir, pred_pdb_dir, align=False, save_summary=False, save_dir=None):
    """
    Compare native and predicted PDB structures using TM-score. Calculates RMSD and TM-score statistics.

    Args:
        native_pdb_dir (str): Directory of native (reference) PDB files.
        pred_pdb_dir (str): Directory of predicted PDB files.
        align (bool): If True, use sequence alignment in TM-score (adds '-seq').
        save_summary (bool): If True, save detailed summary to file.
        save_dir (str): Directory to save summary file. If None, defaults to pred_pdb_dir.

    Returns:
        tuple: (min_rmsd, max_tmscore, mean_rmsd, mean_tmscore).
    """
    results_str = ""
    for native_pdb in os.listdir(native_pdb_dir):
        if os.path.exists(pred_pdb_dir):
            for pred_pdb in os.listdir(pred_pdb_dir):
                if pred_pdb.endswith(".pdb") and native_pdb.endswith(".pdb"):
                    native = os.path.join(native_pdb_dir, native_pdb)
                    pred = os.path.join(pred_pdb_dir, pred_pdb)
                else:
                    continue
                # Run TM-score (with or without sequence alignment)
                if align:
                    command = ["./bin/TMscore", native, pred, "-seq"]
                else:
                    command = ["./bin/TMscore", native, pred]
                try:
                    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                except PermissionError:
                    os.system('chmod +x ./bin/TMscore')
                    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                results_str += "--------------------------------------------------" + result.stdout                    
    # Parse TM-score outputs
    final_result = parse_compare_score_file(file=results_str, flag="file")

    # Prepare summary lines
    lines = []
    for chain1, data in final_result.items():
        lines.append(
            f"{chain1.split('/')[-1].split('.')[0]} best_RMSD: {data['best_rmsd'][0]} model: "
            f"{data['best_rmsd'][1].split('/')[-1].split('.')[0]} best_TM_score: {data['best_tm_score'][0]} model: "
            f"{data['best_tm_score'][1].split('/')[-1].split('.')[0]}\n"
        )


    # Compute statistics
    RMSD = [float(line.split(" ")[2]) for line in lines]
    TMSCORE = [float(line.split(" ")[6]) for line in lines]
    mean_rsmd = np.mean(RMSD) if RMSD else None
    mean_tmscore = np.mean(TMSCORE) if TMSCORE else None
    min_rsmd = np.min(RMSD) if RMSD else None
    max_tmscore = np.max(TMSCORE) if TMSCORE else None
    lines.append(f"Mean RMSD: {round(mean_rsmd,2)}\n")
    lines.append(f"Mean TM-score: {round(mean_tmscore,2)}\n")
    lines.append(f"Min RMSD: {round(min_rsmd,2)}\n")
    lines.append(f"Max TM-score: {round(max_tmscore,2)}\n")
    # Save summary file if requested
    if save_summary:
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            summary_path = os.path.join(save_dir, "summary.txt")
        else:
            summary_path = os.path.join(pred_pdb_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("".join(lines))
    return min_rsmd, max_tmscore, mean_rsmd, mean_tmscore
