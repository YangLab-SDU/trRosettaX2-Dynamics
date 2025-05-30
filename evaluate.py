#!/usr/bin/env python
"""Evaluation script to compare predicted and native PDB structures using TM-score."""
import os
import argparse
import shutil
from utils_trX2dy.evaluate_utils import run_score

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predicted structures by comparing with native structures."
    )
    parser.add_argument("--native_dir", "-n", required=True, type=str,
                        help="Directory with native PDB structures.")
    parser.add_argument("--pred_dir", "-p", required=True, type=str,
                        help="Directory with predicted PDB structures.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output summary file or directory (default: predicted directory).")
    parser.add_argument("--align", action="store_true", default=False,
                        help="Align sequences during TM-score (use '-seq' option).")
    args = parser.parse_args()

    native_dir = args.native_dir
    pred_dir = args.pred_dir
    output = args.output

    # Determine where to save summary
    if output:
        if output.endswith(".txt"):
            # Output as a specific file path
            out_dir = os.path.dirname(output) or os.getcwd()
            out_file = os.path.basename(output)
        else:
            # Output as a directory
            out_dir = output
            out_file = "summary.txt"
        os.makedirs(out_dir, exist_ok=True)
        summary_path = os.path.join(out_dir, out_file)
    else:
        out_dir = pred_dir
        summary_path = os.path.join(pred_dir, "summary.txt")

    # Run scoring and get metrics
    min_rmsd, max_tmscore, mean_rmsd, mean_tmscore = run_score(
        native_dir, pred_dir, align=args.align, save_summary=True, save_dir=out_dir
    )
    # If output is a specific file, rename the default summary
    if output and output.endswith(".txt"):
        default_summary = os.path.join(out_dir, "summary.txt")
        if os.path.exists(default_summary):
            shutil.move(default_summary, summary_path)

    # Print summary statistics
    print("Evaluation Summary:")
    print(f"  Min RMSD: {round(min_rmsd,3)}")
    print(f"  Max TM-score: {round(max_tmscore,3)}")
    print(f"  Mean RMSD: {round(mean_rmsd,3)}")
    print(f"  Mean TM-score: {round(mean_tmscore,3)}")
    print(f"Full summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
