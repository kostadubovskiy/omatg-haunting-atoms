#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

from ghost_utils import remap_ghosts_in_xyz
from ase.io import read, write


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OMatG inference utilities.")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/root"),  # Default for Modal container; override for local use
        help="Root of the OMatG workspace.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML. Defaults to <repo>/code/ghosting-repo/ode_ghosted_local.yaml.",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Path to checkpoint. Defaults to <repo>/checkpoints/lightning_logs/version_0/checkpoints/best_val_loss_total.ckpt.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        help="Trainer accelerator to pass to omg (cpu, mps, gpu, etc.).",
    )
    parser.add_argument(
        "--xyz_out",
        type=Path,
        default=None,
        help="Where to write generated XYZ file. Defaults to <repo>/generated_local.xyz.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Output PDF path for visualize step. Defaults to <repo>/generated_distribution.pdf.",
    )
    parser.add_argument(
        "--csp_json",
        type=Path,
        default=None,
        help="Output JSON path for csp_metrics. Defaults to <repo>/csp_metrics.json.",
    )
    parser.add_argument(
        "--cif_out",
        type=Path,
        default=None,
        help="Output CIF path for sample structure. Defaults to <repo>/sample_structure.cif.",
    )
    parser.add_argument(
        "--limit_predict_batches",
        default=None,
        help="Optional Lightning Trainer limit_predict_batches value (int batches or fraction).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional override for DataModule batch_size.",
    )
    parser.add_argument(
        "--skip_visualize",
        action="store_true",
        help="Skip omg visualize step.",
    )
    parser.add_argument(
        "--skip_metrics",
        action="store_true",
        help="Skip omg csp_metrics step.",
    )
    return parser.parse_args()


def run(cmd, cwd: Path):
    pretty_cmd = " ".join(cmd)
    print(f"\n=== Running: {pretty_cmd} ===")
    subprocess.run(cmd, check=True, cwd=cwd)


def main():
    args = parse_args()
    repo = args.repo.expanduser().resolve()

    config = (
        args.config.expanduser().resolve()
        if args.config
        else repo / "code/ghosting-repo/ode_ghosted_local.yaml"
    )
    ckpt = (
        args.ckpt.expanduser().resolve()
        if args.ckpt
        else repo
        / "checkpoints/lightning_logs/version_0/checkpoints/best_val_loss_total.ckpt"
    )
    xyz_out = (
        args.xyz_out.expanduser().resolve()
        if args.xyz_out
        else repo / "generated_local.xyz"
    )
    plot_out = (
        args.plot.expanduser().resolve()
        if args.plot
        else repo / "generated_distribution.pdf"
    )
    csp_json = (
        args.csp_json.expanduser().resolve()
        if args.csp_json
        else repo / "csp_metrics.json"
    )
    cif_out = (
        args.cif_out.expanduser().resolve()
        if args.cif_out
        else repo / "sample_structure.cif"
    )

    accelerator_flag = f"--trainer.accelerator={args.accelerator}"

    # Delete existing output file to prevent appending from previous runs
    if xyz_out.exists():
        xyz_out.unlink()
        print(f"Deleted existing file: {xyz_out}")

    # Step 1: Generate structures
    predict_cmd = [
        "omg",
        "predict",
        f"--config={config}",
        f"--ckpt_path={ckpt}",
        f"--model.generation_xyz_filename={xyz_out}",
        accelerator_flag,
        "--trainer.log_every_n_steps=1",
        "--trainer.enable_progress_bar=true",
    ]
    if args.limit_predict_batches is not None:
        predict_cmd.append(
            f"--trainer.limit_predict_batches={args.limit_predict_batches}"
        )
    if args.batch_size is not None:
        predict_cmd.append(f"--data.batch_size={args.batch_size}")

    try:
        run(predict_cmd, cwd=repo / "ghosting-repo")
        remap_ghosts_in_xyz(xyz_out)
        print("✓ Successfully generated structures")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during prediction: {e}")
        raise  # Re-raise since we can't continue without generated structures

    # Step 2: Visualize (optional, can fail if training dataset unavailable)
    if not args.skip_visualize:
        try:
            # Run visualize from ghosting-repo directory so relative paths in config resolve correctly
            run(
                [
                    "omg",
                    "visualize",
                    f"--config={config}",
                    f"--xyz_file={xyz_out}",
                    f"--plot_name={plot_out}",
                    accelerator_flag,
                ],
                cwd=repo / "ghosting-repo",
            )
            print("✓ Successfully created visualization")
        except subprocess.CalledProcessError as e:
            print(
                f"⚠ Warning: Visualization failed (this is often due to missing training dataset): {e}"
            )
            print("  Continuing with metrics computation...")
            # Don't raise - continue to metrics

    # Step 3: Compute metrics (optional)
    if not args.skip_metrics:
        try:
            # Run csp_metrics from ghosting-repo directory so relative paths in config resolve correctly
            run(
                [
                    "omg",
                    "csp_metrics",
                    f"--config={config}",
                    f"--xyz_file={xyz_out}",
                    f"--result_name={csp_json}",
                    accelerator_flag,
                ],
                cwd=repo / "ghosting-repo",
            )
            print("✓ Successfully computed CSP metrics")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error during metrics computation: {e}")
            # Don't raise - at least we have the generated structures

    # Step 4: Save sample CIF
    try:
        atoms = read(xyz_out, index=0)
        write(cif_out, atoms)
        print(f"✓ Saved first structure to {cif_out}")
    except Exception as e:
        print(f"⚠ Warning: Could not save CIF file: {e}")

    # Step 5: Print metrics summary
    if not args.skip_metrics and csp_json.exists():
        try:
            metrics = json.loads(csp_json.read_text())
            print("\nCSP metrics sample:", json.dumps(metrics, indent=2)[:400], "...")
        except Exception as e:
            print(f"⚠ Warning: Could not read metrics: {e}")

    print("\n✓ Inference pipeline completed (some steps may have been skipped)")


if __name__ == "__main__":
    main()
