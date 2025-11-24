# OMatG Modal How-To

## 1. Prerequisites
- Install Modal CLI: `pip install modal` and log in thru terminal.
- `.env` inside `code/ghost-training-compiled` must contain valid WANDB.
- Datasets live under `code/ghosting-repo/processed_datasets`; no extra env setup needed locally because Modal builds the runtime image.

## 2. Training on Modal
1. `cd /Users/kosta/Documents/Research/martiniani/omatg_TomEgg/code`
2. Launch training: `modal run train.py --action train`
   - Uses `ode_ghosted.yaml` by default; edit `code/ghosting-repo/ode_ghosted.yaml` to change datasets/model.
   - Heavy deps are baked into the Modal image (PyTorch 2.8.0 + CUDA 12.6); don’t edit `pyproject.toml`.
   - Compute is flexible, but beware of how much you'll end up paying. 1 GPU is relatively cheap - multiple stack quickly, consider using HPC at that point.
3. Monitor: `modal logs omatg-training.train`
4. View active containers: `modal container list`
5. Launch SSH-ed terminal for container: `modal shell [CONTAINER_ID]` where you'll see the ID via the command in #4.

**Critical:** Training checkpoints land in the persistent Modal volume `omatg-checkpoints` under `ghosting-repo/checkpoints`.

## 3. Choosing Configs
- Base configs in `code/ghosting-repo/*.yaml`.
- Training: `ode_ghosted.yaml` (train/val/test splits w/ 0.05 jitter magnitude for breaking voronoi tessellation degeneracies = `processed_datasets/unweighted_v2`).
- Inference: `ode_ghosted_modal.yaml` points `predict_dataset` at `test_ghosted.lmdb`; edit paths in configs if you ghost new datasets and want to train on 'em.
- Keep CSPNet settings in sync with your dataset (ghost atoms are `-1` internally because we call `enable_masked_species`, but any exported XYZ/CIF stores them as `max_real_Z+1` to keep mask tokens distinct).

## 4. Full-Set Inference
1. `cd /Users/kosta/Documents/Research/martiniani/omatg_TomEgg/code`
2. Run: `modal run train.py --action inference`
    - Executes `check_result.py` with `ode_ghosted_modal.yaml`, `best_val_loss_total.ckpt`, and generates:
        - `generated_modal.xyz`
        - `generated_distribution_modal.pdf`
        - `csp_metrics_modal.json`
        - `sample_structure_modal.cif`
    - All saved to `/root/ghosting-repo/checkpoints/modal_results` on the volume.
3. After it finishes (safe to leave overnight; volume persists):
    ```RUN_TAG=$(date +"%Y%m%d-%H%M%S")
    LOCAL_DIR=/Users/kosta/Documents/Research/martiniani/omatg_TomEgg/checkpoints/modal_results/$RUN_TAG
    mkdir -p "$LOCAL_DIR"
    modal volume get omatg-checkpoints modal_results "$LOCAL_DIR"```

## 5. Single-Sample Inference (one structure)
1. `modal run train.py --action single`
   - Adds `--limit_predict_batches=1 --batch_size=1 --skip_visualize --skip_metrics`.
   - Outputs `single_modal.xyz`, `single_modal_distribution.pdf`, `single_modal_metrics.json`, `single_modal.cif`.

## 6. Inspecting Predictions Locally
- Use `code/data-gen/main_inspect_trained.ipynb` to load any `.xyz` produced above:
  - Set `XYZ_PATH` to the downloaded file.
  - Change `sample_idx` to cycle through structures; `plot_crystal_with_points` handles ghost atoms.

## 7. Quick Command Reference
| Action | Command |
|--------|---------|
| Train | `modal run train.py --action train` |
| Full inference | `modal run train.py --action inference` |
| Single sample | `modal run train.py --action single` |
| Copy latest Modal results | `modal volume get omatg-checkpoints modal_results <local_dir>` |
| Tail logs | `modal logs omatg-training.<function_name>` |

**Warnings**
- Do not modify `omg/` core files unless absolutely necessary; everything is packaged into the Modal image.
- Lightning ≥2.5.5 conflicts with our type hints; image pins `lightning==2.5.0` + `jsonargparse==4.27.7`. Don’t upgrade unless you know the implications.
```