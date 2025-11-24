import os
import wandb

import subprocess
from pathlib import Path

from modal import App, Image, Secret, Volume

# Define the Modal App
app = App("omatg-training")

# Define paths
compiled_repo_path = Path(__file__).parent / "ghost-training-compiled"
omg_path = (
    Path(__file__).parent.parent / "omg-fork"
)  # path to forked OMG repo (ghosting-repo/omg-fork)

# Create a volume for checkpoints
checkpoints_vol = Volume.from_name("omatg-checkpoints", create_if_missing=True)

# 1. Define the BASE IMAGE (Heavy dependencies)
# This layer will be cached and not rebuilt unless you change these lines.
base_image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "python3.11-dev")
    .pip_install("wheel")
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
        # Use the correct index for PyTorch 2.8.0 (CUDA 12.x)
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "torch-scatter==2.1.2",
        "torch-sparse==0.6.18",
        "torch-cluster==1.6.3",
        "torch-spline-conv==1.2.2",
        "torch-geometric==2.7.0",
    )
)

# 2. Define the APP IMAGE (Code and lighter deps)
# This builds ON TOP of base_image. Changing code only rebuilds this part.
omatg_image = (
    base_image
    # Copy the omg source code
    .add_local_dir(omg_path, remote_path="/root/omg_source", copy=True)
    # Add the compiled repo (contains requirements.txt)
    .add_local_dir(compiled_repo_path, remote_path="/root/ghosting-repo", copy=True)
    # Install requirements with strict version control to avoid type-checking crashes
    # We force lightning 2.5.0 and jsonargparse 4.27.7 which are compatible and lenient.
    # We filter out conflicting pins AND the git version of omg from requirements.txt
    .run_commands(
        "pip install 'lightning==2.5.0' 'jsonargparse[signatures]==4.27.7'",
        "grep -vE 'lightning|jsonargparse|omg' /root/ghosting-repo/requirements.txt | pip install -r /dev/stdin",
    )
    # NOW install the local omg package LAST, but pin torch to prevent upgrading
    # Use --no-deps to prevent any dependency resolution that might upgrade torch
    .run_commands(
        # "pip uninstall -y omg || true",  # Uninstall if it exists (|| true prevents failure if not found)
        # "pip install --no-deps --force-reinstall /root/omg_source",  # force reinstall
        "pip install --no-deps /root/omg_source",  # Install ONLY our package, no deps
        # Now install the missing omg dependencies without upgrading torch
        "pip install ase loguru tqdm scipy pandas matplotlib plotly pydantic",
        "pip install 'pymatgen~=2025.6' 'matminer~=0.9' 'smact~=3.1' 'spglib~=2.6' 'wandb~=0.20'",
        "echo 'Force rebuild 2025-11-22-1745'",
    )
    # Symlink data
    .run_commands(
        "cd /root/ghosting-repo && ln -s processed_datasets/unweighted_v2 data"
    )
    # Upload helper scripts/configs last so Modal mounts them at runtime
    .add_local_file(
        Path(__file__).parent / "check_result.py",
        remote_path="/root/check_result.py",
        copy=True,
    )
    .add_local_file(
        Path(__file__).parent / "ghost_utils.py",
        remote_path="/root/ghost_utils.py",
        copy=True,
    )
    .add_local_file(
        Path(__file__).parent / "ghost-training-compiled" / "ode_ghosted.yaml",
        remote_path="/root/ghosting-repo/ode_ghosted_modal.yaml",
        copy=True,
    )
)


@app.function(
    image=omatg_image,
    gpu="A100",
    secrets=[Secret.from_dotenv(compiled_repo_path / ".env.local")],
    volumes={"/root/ghosting-repo/checkpoints": checkpoints_vol},
    timeout=86400,
)
def train():
    """
    Function to run the OMatG training.
    """
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY not set in environment.")
    wandb.login(key=api_key)
    wandb.init(project="omatg", name="kdelV0-modal-A100", reinit=True).finish()

    # DIAGNOSTIC: Check what version of omg_lightning.py is actually installed
    print("=" * 80)
    print("DIAGNOSTIC: Checking installed omg_lightning.py")
    print("=" * 80)

    result = subprocess.run(
        [
            "grep",
            "-n",
            "Force enable masked species",
            "/usr/local/lib/python3.11/site-packages/omg/omg_lightning.py",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("✓ FIX FOUND in installed package:")
        print(result.stdout)

        # Show the actual line to confirm it's the right fix
        subprocess.run(
            [
                "sed",
                "-n",
                "168,171p",
                "/usr/local/lib/python3.11/site-packages/omg/omg_lightning.py",
            ]
        )
    else:
        print("✗ FIX NOT FOUND - showing lines 148-152 instead:")
        subprocess.run(
            [
                "sed",
                "-n",
                "148,152p",
                "/usr/local/lib/python3.11/site-packages/omg/omg_lightning.py",
            ]
        )

    print("=" * 80)
    print()

    training_command = [
        "omg",
        "fit",
        "--config=ode_ghosted.yaml",
    ]

    # Set the working directory for the subprocess
    working_dir = "/root/ghosting-repo"

    # Execute the training command
    process = subprocess.Popen(
        training_command,
        cwd=working_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Stream the output
    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            print(line, end="")

    # Wait for the process to complete and get the return code
    return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, training_command)

    # Commit the volume to persist checkpoints
    checkpoints_vol.commit()

    print("Training finished successfully.")


@app.function(
    image=omatg_image,
    gpu="A100",
    secrets=[Secret.from_dotenv(compiled_repo_path / ".env.local")],
    volumes={"/root/ghosting-repo/checkpoints": checkpoints_vol},
    timeout=86400,
)
def run_inference():
    """
    Run the local check_result.py script inside the Modal environment to
    generate samples/metrics with the latest checkpoint.
    """

    results_dir = Path("/root/ghosting-repo/checkpoints/modal_results")
    os.makedirs(results_dir, exist_ok=True)

    cmd = [
        "python3",
        "/root/check_result.py",
        "--repo",
        "/root",
        "--config",
        "/root/ghosting-repo/ode_ghosted_modal.yaml",
        "--ckpt",
        "/root/ghosting-repo/checkpoints/lightning_logs/version_0/checkpoints/best_val_loss_total.ckpt",
        "--accelerator",
        "gpu",
        "--xyz_out",
        str(results_dir / "generated_modal.xyz"),
        "--plot",
        str(results_dir / "generated_distribution_modal.pdf"),
        "--csp_json",
        str(results_dir / "csp_metrics_modal.json"),
        "--cif_out",
        str(results_dir / "sample_structure_modal.cif"),
    ]

    subprocess.run(cmd, check=True, cwd="/root")
    checkpoints_vol.commit()
    print("Inference artifacts saved to", results_dir)


@app.function(
    image=omatg_image,
    gpu="A100",
    secrets=[Secret.from_dotenv(compiled_repo_path / ".env.local")],
    volumes={"/root/ghosting-repo/checkpoints": checkpoints_vol},
    timeout=86400,
)
def run_single_sample():
    """
    Generate a single-sample prediction (and supporting artifacts) for quick sanity checks.
    """

    results_dir = Path("/root/ghosting-repo/checkpoints/modal_results")
    os.makedirs(results_dir, exist_ok=True)

    cmd = [
        "python3",
        "/root/check_result.py",
        "--repo",
        "/root",
        "--config",
        "/root/ghosting-repo/ode_ghosted_modal.yaml",
        "--ckpt",
        "/root/ghosting-repo/checkpoints/lightning_logs/version_0/checkpoints/best_val_loss_total.ckpt",
        "--accelerator",
        "gpu",
        "--limit_predict_batches",
        "1",
        "--batch_size",
        "1",
        "--xyz_out",
        str(results_dir / "single_modal.xyz"),
        "--plot",
        str(results_dir / "single_modal_distribution.pdf"),
        "--csp_json",
        str(results_dir / "single_modal_metrics.json"),
        "--cif_out",
        str(results_dir / "single_modal.cif"),
        "--skip_visualize",
        "--skip_metrics",
    ]

    subprocess.run(cmd, check=True, cwd="/root")
    checkpoints_vol.commit()
    print("Single-sample inference artifacts saved to", results_dir)


@app.function(
    image=omatg_image,
    gpu="A100",
    secrets=[Secret.from_dotenv(compiled_repo_path / ".env.local")],
    volumes={"/root/ghosting-repo/checkpoints": checkpoints_vol},
    timeout=86400,
)
def evaluate_existing():
    """
    Run omg visualize + csp_metrics on the already-generated XYZ
    (skips the predict step to save time/compute).
    """
    results_dir = Path("/root/ghosting-repo/checkpoints/modal_results")
    xyz_path = results_dir / "generated_modal.xyz"

    if not xyz_path.exists():
        raise FileNotFoundError(f"{xyz_path} not found; run inference first.")

    # Run visualize
    subprocess.run(
        [
            "omg",
            "visualize",
            "--config=/root/ghosting-repo/ode_ghosted_modal.yaml",
            f"--xyz_file={xyz_path}",
            f"--plot_name={results_dir / 'generated_distribution_modal.pdf'}",
            "--trainer.accelerator=gpu",
        ],
        check=True,
        cwd="/root",
    )

    # Run csp_metrics
    subprocess.run(
        [
            "omg",
            "csp_metrics",
            "--config=/root/ghosting-repo/ode_ghosted_modal.yaml",
            f"--xyz_file={xyz_path}",
            f"--result_name={results_dir / 'csp_metrics_modal.json'}",
            "--trainer.accelerator=gpu",
        ],
        check=True,
        cwd="/root",
    )

    checkpoints_vol.commit()
    print(f"Evaluation artifacts saved to {results_dir}")


@app.local_entrypoint()
def main(action: str = "train"):
    """
    Local entrypoint to trigger Modal jobs.
    """
    if action == "train":
        print("Starting OMatG training on Modal...")
        train.remote()
        print("Training job started.")
    elif action == "inference":
        print("Starting OMatG inference on Modal...")
        run_inference.remote()
        print("Inference job started.")
    elif action == "single":
        print("Starting single-sample inference on Modal...")
        run_single_sample.remote()
        print("Single-sample job started.")
    elif action == "evaluate":
        print("Running evaluation on existing Modal XYZ...")
        evaluate_existing.remote()
        print("Evaluation job started.")
    else:
        raise ValueError(f"Unknown action '{action}'. Use 'train' or 'inference'.")
