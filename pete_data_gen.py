import os
import sys
import pickle
import torch
import lmdb
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

import numpy as np

# Force torch's multiprocessing tensor IPC to use named tempfiles instead of
# /dev/shm file descriptors. Otherwise each tensor pickled across a Pool
# boundary allocates a small /dev/shm/torch_* mmap; on the 27k-sample train
# split this exhausts /dev/shm and every subsequent dispatch fails with
# `unable to mmap ...: Cannot allocate memory`.
#
# Combined with the spawn start method in ghost_dataset(), workers re-execute
# this module-level call on import, so they reliably pick up the file_system
# strategy (fork inheritance was unreliable in practice).
torch.multiprocessing.set_sharing_strategy("file_system")

from omg.datamodule import StructureDataset
from voronoi_weighted_noise import VoronoiPhantomCellGenerator


DESIRED_ATOM_COUNT = 20

VoronoiGenerator = VoronoiPhantomCellGenerator(
    desired_atom_count=DESIRED_ATOM_COUNT,
    dist_eval="min",
    # dist_eval="avg_x_min",
    # num_min_distances=3,
    epsilon=1e-3,
    weight_distances=False,
    noise_magnitude=5e-2,  # 0.05 ideal for squashing low-interatomic distances and breaking voronoi degeneracies but not skewing the distribution
)

# v0
# => no jtter
# v1
# => 0.01 jitter
# v2
# => 0.05 jitter
# v3
# => 0.02 jitter
# v4
# => 0.1 jitter
# v5
# => 0.15 jitter
# v6
# => 0.2 jitter


def _worker_init():
    """Run once per worker process: keep numpy/freud single-threaded so N workers
    don't oversubscribe CPU cores."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def process_single_sample(args):
    """Ghost one structure given pure-numpy inputs; return pure-numpy outputs.

    IMPORTANT: this function must take and return ONLY numpy arrays (no torch
    tensors), so that nothing crossing the multiprocessing.Pool boundary
    triggers PyTorch's shared-memory ForkingPickler. The parent converts the
    returned numpy arrays into torch tensors at LMDB-write time.

    Returns a dict of numpy arrays:
        {"cell": (3, 3) float64, "pos": (N, 3) float64, "atomic_numbers": (N,) int64}
    Ghosts are encoded as ``-1`` in atomic_numbers; OMGData remaps them to the
    global ghost label (119) at training time.
    """
    i, cell, positions, atomic_numbers, voronoi_generator_config = args

    voronoi_generator = VoronoiPhantomCellGenerator(**voronoi_generator_config)

    try:
        x_vec, y_vec, z_vec = cell[0], cell[1], cell[2]

        new_points = positions.copy()
        new_atomic_numbers = atomic_numbers.copy()

        iterations = DESIRED_ATOM_COUNT - len(new_atomic_numbers)
        for _ in range(iterations):
            next_point = voronoi_generator._get_next_point(
                points=new_points,
                atomic_numbers=new_atomic_numbers,
                x_vec=x_vec,
                y_vec=y_vec,
                z_vec=z_vec,
            )

            if np.any(np.isnan(next_point)):
                print(f"NaN value detected for sample {i}, skipping.")
                return None

            new_atomic_numbers = np.append(new_atomic_numbers, -1)
            new_points = np.vstack([new_points, next_point])

        return {
            "cell": np.ascontiguousarray(cell, dtype=np.float64),
            "pos": np.ascontiguousarray(new_points, dtype=np.float64),
            "atomic_numbers": np.ascontiguousarray(new_atomic_numbers, dtype=np.int64),
        }
    except Exception as e:
        print(f"Error processing sample {i} in worker: {e}")
        return None


# Override with the MP20_DIR env var (e.g. on HPC where the layout is flat).
MP20_DIR = Path(os.environ.get("MP20_DIR", "./omg-fork/omg/data/mp_20"))


def load_data():
    """Open the three vanilla MP-20 LMDBs with the new omg.datamodule API.

    ``convert_to_fractional=False`` keeps positions Cartesian (required by the
    Voronoi generator) and ``niggli_reduce=False`` preserves the raw cells.
    ``lazy_storage=False`` loads the (small) MP-20 splits fully into memory so
    iteration is fast and avoids per-item LMDB reads across pool workers.
    """
    common = dict(
        lazy_storage=False,
        niggli_reduce=False,
        convert_to_fractional=False,
    )
    train_dataset = StructureDataset(str(MP20_DIR / "train.lmdb"), **common)
    val_dataset = StructureDataset(str(MP20_DIR / "val.lmdb"), **common)
    test_dataset = StructureDataset(str(MP20_DIR / "test.lmdb"), **common)

    assert len(train_dataset) == 27136
    assert len(val_dataset) == 9047
    assert len(test_dataset) == 9046

    return train_dataset, test_dataset, val_dataset


def save_records_to_lmdb(records: list[dict], lmdb_filename: Path) -> None:
    """Write a list of numpy-dict records to LMDB as torch-tensor records.

    Worker output schema (numpy):
        - cell:           (3, 3) float64
        - pos:            (N, 3) float64 (Cartesian)
        - atomic_numbers: (N,)   int64 (ghosts stored as -1)

    LMDB schema expected by ``omg.datamodule.StructureDataset._from_lmdb``
    (torch):
        - cell:           (3, 3) torch.float64
        - pos:            (N, 3) torch.float64
        - atomic_numbers: (N,)   torch.int64

    The numpy→torch conversion happens here in the parent so the workers never
    have to pickle torch tensors across the multiprocessing boundary.

    The ``OMGData`` constructor remaps any negative atomic numbers to the
    global ghost label (119) at training time, so storing -1 here keeps
    ghosts distinguishable from real atomic numbers in the LMDB itself.
    """
    lmdb_filename.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(lmdb_filename), subdir=False, map_size=int(1e12))
    with env.begin(write=True) as txn:
        for i, record in enumerate(records):
            torch_record = {
                "cell": torch.from_numpy(record["cell"]).to(dtype=torch.float64),
                "pos": torch.from_numpy(record["pos"]).to(dtype=torch.float64),
                "atomic_numbers": torch.from_numpy(record["atomic_numbers"]).to(
                    dtype=torch.int64
                ),
            }
            txn.put(str(i).encode("utf-8"), pickle.dumps(torch_record))
    env.close()
    print(f"Wrote {len(records)} structures to {lmdb_filename}")


def ghost_dataset(dataset: StructureDataset, dataset_type: str) -> list[dict]:
    """Generate ghost atoms for every structure in ``dataset``.

    Parallelizes across N worker processes in chunks of N so the original
    per-sample timeout semantics are preserved (a stuck sample in a chunk does
    not block the rest of the chunk, which has already been submitted).

    Uses the ``spawn`` start method so each worker re-imports this module and
    reliably picks up ``set_sharing_strategy("file_system")``; with fork-mode
    Pools the workers were inheriting ``file_descriptor`` strategy and
    eventually exhausting /dev/shm on the 27k-sample train split.

    ``maxtasksperchild`` recycles workers after a fixed number of samples so
    any per-process resource creep (torch caches, /dev/shm fragments, freud
    state) gets released before it can accumulate.

    Tunable via env vars:
      - GHOST_GEN_WORKERS:        number of worker processes (default 16)
      - GHOST_GEN_TIMEOUT:        per-sample timeout in seconds (default 20)
      - GHOST_GEN_MAXTASKS:       maxtasksperchild for workers (default 500)
    """
    ghosted_data_list = []
    n_workers = int(os.environ.get("GHOST_GEN_WORKERS", "16"))
    sample_timeout = float(os.environ.get("GHOST_GEN_TIMEOUT", "20"))
    maxtasks = int(os.environ.get("GHOST_GEN_MAXTASKS", "500"))

    voronoi_generator_config = {
        "desired_atom_count": VoronoiGenerator.desired_atom_count,
        "dist_eval": VoronoiGenerator.dist_eval,
        "epsilon": VoronoiGenerator.epsilon,
        "num_min_distances": VoronoiGenerator.num_min_distances,
        "weight_distances": VoronoiGenerator.weight_distances,
        "noise_magnitude": VoronoiGenerator.noise_magnitude,
    }

    print(
        f"Ghosting {dataset_type} dataset with {n_workers} spawn workers "
        f"(per-sample timeout: {sample_timeout}s, maxtasksperchild: {maxtasks})"
    )

    n_timeouts = 0
    n_errors = 0

    def _drain(chunk):
        nonlocal n_timeouts, n_errors
        for j, fut in chunk:
            try:
                processed = fut.get(timeout=sample_timeout)
                if processed is not None:
                    ghosted_data_list.append(processed)
            except mp.TimeoutError:
                n_timeouts += 1
                print(f"Skipping sample {j} due to timeout.")
            except Exception as e:
                n_errors += 1
                print(f"Error processing sample {j}: {e}")

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        maxtasksperchild=maxtasks,
    ) as pool:
        chunk: list = []
        for i, single_data in enumerate(
            tqdm(dataset, desc=f"Ghosting {dataset_type}")
        ):
            # Force-copy via np.array(..., copy=True) so the resulting buffer
            # has NO shared memory with the underlying torch storage. A simple
            # `.cpu().numpy()` returns a *view* of the torch tensor's data,
            # and even np.ascontiguousarray() returns that same view when the
            # dtype/layout already match. Pickling such a view can drag the
            # torch storage into the pickle stream (via PickleBuffer / buffer
            # protocol), which is what was triggering rebuild_storage_fd in
            # the parent's result handler and exhausting /dev/shm.
            cell_np = np.array(
                single_data.cell.cpu().numpy(), dtype=np.float64, copy=True
            )
            pos_np = np.array(
                single_data.pos.cpu().numpy(), dtype=np.float64, copy=True
            )
            an_np = np.array(
                single_data.atomic_numbers.cpu().numpy(),
                dtype=np.int64,
                copy=True,
            )
            args = (i, cell_np, pos_np, an_np, voronoi_generator_config)
            fut = pool.apply_async(process_single_sample, args=(args,))
            chunk.append((i, fut))
            if len(chunk) >= n_workers:
                _drain(chunk)
                chunk = []
        if chunk:
            _drain(chunk)

    print(
        f"\nGhost atom generation complete. "
        f"Processed: {len(ghosted_data_list)}, "
        f"timeouts: {n_timeouts}, errors: {n_errors}"
    )

    return ghosted_data_list


def main(inputs: list[str] | None = None):
    if not inputs and len(sys.argv) < 3:
        print("Usage: python pete_data_gen.py <dataset_type> <run_folder>")
        print("  dataset_type: train, test, or val")
        sys.exit(1)

    dataset_type = inputs[0] if inputs else sys.argv[1]
    run_folder = inputs[1] if inputs else sys.argv[2]

    train_dataset, test_dataset, val_dataset = load_data()

    if dataset_type == "train":
        dataset = train_dataset
    elif dataset_type == "test":
        dataset = test_dataset
    elif dataset_type == "val":
        dataset = val_dataset
    else:
        raise ValueError("Invalid dataset. Must be 'train', 'test', or 'val'")

    ghosted_records = ghost_dataset(dataset, dataset_type)

    processed_dir = Path(f"./processed_datasets/{run_folder}")
    processed_dir.mkdir(parents=True, exist_ok=True)

    full_lmdb_filepath = processed_dir / f"{dataset_type}_ghosted.lmdb"
    save_records_to_lmdb(ghosted_records, full_lmdb_filepath)


if __name__ == "__main__":
    main()
