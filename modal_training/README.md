# OMatG Modal Training

Containerized training, inference, and evaluation for crystal structure generation with ghost atom support.

## Prerequisites

1. **Install Modal CLI:**
   ```bash
   pip install modal
   modal token new  # authenticate
   ```

2. **Clone and install the patched OMatG fork:**
   ```bash
   # Clone into ghosting-repo directory (will be gitignored)
   cd /path/to/omatg_TomEgg/code/ghosting-repo
   git clone https://github.com/kostadubovskiy/OMatG-Fork omg-fork
   cd omg-fork
   pip install -e .
   ```
   
   **Note:** The fork is cloned at `ghosting-repo/omg-fork` and is already in `.gitignore`. 
   
   Alternatively, to apply patches manually to the upstream repo.

3. **Set up credentials:**
   ```bash
   cd modal_training
   cp .env.template .env.local
   # Edit .env.local and add your WANDB_API_KEY
   # Source env file
   set -a
   source .env.local
   set +a
   ```

4. **Datasets:**
   Ghosted datasets should be in `ghost-training-compiled/processed_datasets/`. This repo includes `unweighted_v2` (20-atom structures with Voronoi ghost atoms, 0.05 jitter).

## Quick Start

### Training
```bash
cd modal_training
modal run train.py --action train
```
- Trains on Modal A100 GPU by default // GPU choice flexible with modal.
- Checkpoints saved to Modal volume `omatg-checkpoints`
- Monitor: `modal [app/container] logs omatg-training.train`
- SSH into active VM: `modal container list` ==> `modal shell [CONTAINER_ID]`
- Config: `ghost-training-compiled/ode_ghosted.yaml`

### Full-Set Inference
```bash
modal run train.py --action inference
```
Generates predictions for entire test set, then:
```bash
RUN_TAG=$(date +"%Y%m%d-%H%M%S")
LOCAL_DIR=$PWD/results/$RUN_TAG
mkdir -p "$LOCAL_DIR"
modal volume get omatg-checkpoints modal_results "$LOCAL_DIR"
```

### Single-Sample Inference (quick sanity check)
```bash
modal run train.py --action single
```

### Evaluation (on existing predictions)
```bash
modal run train.py --action evaluate
```
Runs visualize/metrics on `generated_modal.xyz` already on the volume.

## Command Reference

| Action | Command |
|--------|---------|
| Train | `modal run train.py --action train` |
| Full inference | `modal run train.py --action inference` |
| Single sample | `modal run train.py --action single` |
| Evaluate existing | `modal run train.py --action evaluate` |
| Download results | `modal volume get omatg-checkpoints modal_results <local_dir>` |
| Download checkpoint | `modal volume get omatg-checkpoints lightning_logs/version_0/checkpoints <local_dir>` |
| Tail logs | `modal logs omatg-training.<function_name>` |
| List containers | `modal container list` |
| Shell into container | `modal shell <container_id>` |

## Configuration

**Training config:** `ghost-training-compiled/ode_ghosted.yaml`
- Adjust `max_epochs`, learning rate, batch size, etc.
- Dataset paths: `processed_datasets/unweighted_v2/*.lmdb` (relative to `/root/ghosting-repo` in Modal)
- Model: CSPNet with masked species enabled for ghost atoms

**Root-level configs** (`../ode_ghosted*.yaml`): These configs use `data/` symlinks and are for local development only. Modal uses the version in `ghost-training-compiled/`.

**Important:** Ghost atoms are stored as `-1` in LMDB, converted to `max_real_Z+1` by the dataloader, and exported as `max_real_Z+1` in XYZ/CIF files. Mask tokens (if used) are at index `0` - these are NEVER conflated.

## Inspecting Results Locally

Use `main_inspect_trained.ipynb` to visualize generated structures:

```python
# In the notebook, set:
XYZ_PATH = PROJECT_ROOT / "path/to/downloaded/generated_modal.xyz"
sample_idx = 0  # cycle through structures
```

The notebook uses `ghost_utils.ase_atoms_to_omg_data()` to handle ghost metadata automatically.

## Outputs

**Checkpoints** (`lightning_logs/version_X/checkpoints/`):
- `best_val_loss_total.ckpt` - best validation loss checkpoint
- `epoch=N-step=M.ckpt` - periodic checkpoints (every 100 epochs)

**Inference artifacts** (`modal_results/`):
- `generated_modal.xyz` - all predicted structures
- `generated_modal_init.xyz` - initial (t=0) structures
- `generated_distribution_modal.pdf` - distribution comparison plots
- `csp_metrics_modal.json` - validation metrics (may fail on ghost atoms)
- `sample_structure_modal.cif` - first structure in CIF format

## Known Issues

### CSP Metrics Crash
`omg csp_metrics` uses `smact` for chemistry validation, which cannot handle ghost atoms. The command will fail with:
```
TypeError: 'NoneType' object is not iterable
```

**Workaround:** The visualize step completes successfully, producing the PDF. Metrics can be skipped or wrapped in try/except.

### Detached Runs
If your local machine disconnects during a long-running job, the Modal container continues executing but you lose the log stream. Use:
```bash
modal run --detach train.py --action inference
```
Then check status with `modal container list` and reconnect logs with `modal logs ...`.

## File Structure

```
ghosting-repo/
├── omg-fork/              # Patched OMatG fork (gitignored, clone separately)
│   └── omg/               # Package with ghost atom patches applied
└── modal_training/
    ├── README.md          # This file
    ├── train.py           # Modal containerization script
    ├── check_result.py    # Inference/evaluation runner
    ├── ghost_utils.py     # Ghost atom handling utilities
    ├── main_inspect_trained.ipynb  # Local visualization notebook
    └── ghost-training-compiled/
        ├── ode_ghosted.yaml   # Training/inference config
        ├── requirements.txt   # Python dependencies
        └── processed_datasets/
            └── unweighted_v0/ # Voronoi Ghosted LMDB datasets. Jitter=0.0 - degenerate tessellations 
            └── unweighted_v2/ # V-Ghosted LMDB datasets. Jitter=0.05 - broke degeneracies
                ├── train_ghosted.lmdb
                ├── val_ghosted.lmdb
                └── test_ghosted.lmdb
            └── random_v0/ # Randomly Ghosted LMDB datasets
                ├── train_ghosted.lmdb
                ├── val_ghosted.lmdb
                └── test_ghosted.lmdb
```

## Troubleshooting

**"Modal image build takes 30+ minutes"**  
→ Normal on first build. Subsequent builds cache base image (PyTorch/CUDA) and only rebuild code layer (~2 min).

**"Training checkpoints not found locally"**  
→ They live on the Modal volume. Use `modal volume get` to download them.

## Cost Considerations

- **A100 GPU:** ~$1-2/hour depending on region
- **Storage:** Modal volumes are persistent and may incur storage costs
- **Tip:** Use `--detach` for overnight runs to avoid keeping your laptop awake

