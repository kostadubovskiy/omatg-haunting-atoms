# Crystal Structure Generation with Ghost Atoms

This repository contains tools for generating crystal structures using ghost atoms for padding and Voronoi tessellation methods.

## Directory Structure

### `modal_training/`
**Production Modal containerized training system.** See `modal_training/README.md` for:
- Training models on Modal (GPU cloud)
- Running inference on test sets
- Evaluation and visualization

This is the recommended way to train and run inference.

### Root-level Scripts & Configs

**Data generation utilities:**
- `voronoi.py`, `voronoi_weighted.py`, `voronoi_weighted_noise.py` - Generate ghost atoms via Voronoi tessellation
- `pete_data_gen.py` - Packaged End-to-End Data Generation wrapping everything needed to generate a Voronoi-ghosted dataset from a vanilla dataset. Use command line arguments to run multiple in parallel easily.

**Configs for local development:**
- `ode_ghosted.yaml`, `ode_ghosted_local.yaml`, `ode_ghosted_modal.yaml` - Training configs using `data/` symlink
- These are for local experimentation only; Modal uses `modal_training/ghost-training-compiled/ode_ghosted.yaml`

**Datasets:**
- `processed_datasets/` - Various LMDB datasets with different ghosting strategies
  - `unweighted_v2/` - Current production dataset (20-atom structures, 0.05 jitter). Ghost atoms dropped in via furthest-center-point 3x3x3 supercell Voronoi tessellations.
  - `random_v0/` - Randomly placing ghost atoms in the base dataset until 20 atoms are reached 

## Quick Start

### Modal Training (Recommended)

1. **Clone the patched fork:**
   ```bash
   cd ghosting-repo
   git clone https://github.com/kostadubovskiy/OMatG-Fork omg-fork
   cd omg-fork
   pip install -e .
   ```

2. **Follow Modal training guide:** See [`modal_training/README.md`](modal_training/README.md) for training, inference, and evaluation.

### Local Data Generation

For generating new ghosted datasets:
```bash
python voronoi_weighted_noise.py --help
```

## Ghost Atom Convention

- **Storage (LMDB):** `-1`
- **Dataloader → Model:** `max_real_Z + 1` (keeps distinct from mask=0)
- **Export (XYZ/CIF):** `max_real_Z + 1` (with metadata arrays)
- **Mask tokens:** `0` (reserved for model masking, NEVER used for ghosts)

See `modal_training/PATCHES.md` for details and required OMatG-fork modifications.

## Notebooks

- `main.ipynb` - Data exploration and visualization
- `histogram.ipynb` - Dataset statistics
- `modal_training/main_inspect_trained.ipynb` - Inspect model predictions

## Support

- Modal training issues: See `modal_training/README.md`
- Contact kostadubovskiy@gmail.com

