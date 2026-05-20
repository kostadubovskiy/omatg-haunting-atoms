import os
import sys
import pickle
import torch
import lmdb
from typing import Union
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

import numpy as np

from ase import Atoms
from ase.io import read, write
from omg.datamodule import DataModule
from omg.datamodule.dataloader import OMGTorchDataset, OMGData
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
    """Processes a single sample to generate ghost atoms."""
    i, single_data, voronoi_generator_config = args

    # Re-create the generator inside the worker process
    voronoi_generator = VoronoiPhantomCellGenerator(**voronoi_generator_config)

    try:
        # --- 1. Extract initial data ---
        cell = single_data.cell[0].numpy()
        x_vec, y_vec, z_vec = cell[0], cell[1], cell[2]

        positions = single_data.pos.numpy()
        atomic_numbers = single_data.species.numpy()

        new_points = positions.copy()
        new_atomic_numbers = atomic_numbers.copy()

        # --- 2. Generate ghost atoms ---
        iterations = DESIRED_ATOM_COUNT - single_data.n_atoms.item()
        if iterations > 0:
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
                    return None  # Indicate failure

                new_atomic_numbers = np.append(new_atomic_numbers, -1)
                new_points = np.vstack([new_points, next_point])

        # --- 3. Update the data object ---
        single_data.pos = torch.from_numpy(new_points).to(dtype=torch.float64)
        single_data.n_atoms = torch.tensor(len(new_points), dtype=torch.long)
        single_data.batch = torch.zeros(len(new_points), dtype=torch.long)
        single_data.species = torch.from_numpy(new_atomic_numbers).long()

        return single_data
    except Exception as e:
        print(f"Error processing sample {i} in worker: {e}")
        return None  # Indicate failure


# Override with the MP20_DIR env var (e.g. on HPC where the layout is flat).
MP20_DIR = Path(os.environ.get("MP20_DIR", "./omg-fork/omg/data/mp_20"))


def load_data():
    train_data = DataModule()
    train_data.add_from_lmdb(str(MP20_DIR / "train.lmdb"))

    val_data = DataModule()
    val_data.add_from_lmdb(str(MP20_DIR / "val.lmdb"))

    test_data = DataModule()
    test_data.add_from_lmdb(str(MP20_DIR / "test.lmdb"))

    assert len(train_data) == 27136
    assert len(val_data) == 9047
    assert len(test_data) == 9046

    train_dataset = OMGTorchDataset(
        dataset=train_data, convert_to_fractional=False, niggli=False
    )
    assert len(train_dataset) == len(train_data)

    val_dataset = OMGTorchDataset(
        dataset=val_data, convert_to_fractional=False, niggli=False
    )
    assert len(val_dataset) == len(val_data)

    test_dataset = OMGTorchDataset(
        dataset=test_data, convert_to_fractional=False, niggli=False
    )
    assert len(test_dataset) == len(test_data)

    return train_dataset, test_dataset, val_dataset


def xyz_saver(data: Union[OMGData, list[OMGData]], filename: Path) -> None:
    """
    Save structures from OMGData instances to an extxyz file.
    Correctly handles a list of single data objects and stores ghost
    atom information (-1) in a separate array to avoid data loss.
    """
    if not filename.suffix == ".extxyz":
        raise ValueError("The filename must have the suffix '.extxyz'.")

    if not isinstance(data, list):
        data = [data]

    atoms_list = []
    for d in data:
        # Get species and convert to numpy array
        species_np = d.species.cpu().numpy().copy()

        # Create a boolean array to mark ghost atoms
        is_ghost = species_np == -1

        # Store ghosts as max(real_Z) + 1 to avoid mixing with mask tokens (0/-1)
        real_species = species_np[~is_ghost]
        max_real = int(real_species.max()) if real_species.size else 0
        ghost_label = max_real + 1
        species_np[is_ghost] = ghost_label

        atoms = Atoms(
            numbers=species_np,
            positions=d.pos.cpu().numpy(),
            cell=d.cell[0].cpu().numpy(),
            pbc=True,
        )

        # Attach the ghost atom information as a per-atom array
        atoms.set_array("is_ghost", is_ghost)
        atoms.set_array(
            "ghost_atomic_number",
            np.full(len(species_np), ghost_label, dtype=species_np.dtype),
        )
        atoms_list.append(atoms)

    # Overwrite the file with the new list of atoms.
    os.makedirs(filename.parent, exist_ok=True)
    # The format is inferred from the filename suffix '.extxyz'
    write(filename, atoms_list, append=False)


def convert_xyz_to_lmdb(input_xyz_file: Path, output_lmdb_file: Path) -> None:
    """Converts an ASE-readable extxyz file to a single-file LMDB database."""
    if not input_xyz_file.exists():
        print(f"Error: Input file not found at {input_xyz_file}")
        return

    try:
        # ASE automatically handles .extxyz and reads the extra arrays
        atoms_list = read(input_xyz_file, index=":")
        print(f"Read {len(atoms_list)} structures from '{input_xyz_file}'")
    except Exception as e:
        print(f"Error reading EXYZ file: {e}")
        return

    env = lmdb.open(str(output_lmdb_file), subdir=False, map_size=int(1e12))
    with env.begin(write=True) as txn:
        for i, atoms in enumerate(atoms_list):
            key = str(i).encode("utf-8")

            atomic_numbers = atoms.get_atomic_numbers()

            # Check if ghost atom info is present and restore -1
            if "is_ghost" in atoms.arrays:
                is_ghost = atoms.get_array("is_ghost")
                atomic_numbers[is_ghost] = -1

            value = {
                "atomic_numbers": torch.from_numpy(atomic_numbers),
                "pos": torch.from_numpy(atoms.get_positions()).to(dtype=torch.float64),
                "cell": torch.from_numpy(atoms.get_cell()[:]).to(dtype=torch.float64),
                "pbc": torch.tensor(atoms.get_pbc(), dtype=torch.bool),
            }
            serialized_value = pickle.dumps(value)
            txn.put(key, serialized_value)
    env.close()
    print(f"Successfully created single-file LMDB: '{output_lmdb_file}'")


def save_omgdata_direct_to_lmdb(
    data: list[OMGData], lmdb_filename: Path
) -> None:
    """
    Write the in-memory ghosted OMGData objects directly to LMDB, skipping the
    extxyz round-trip used by ``save_atoms_to_lmdb``. Produces the same record
    format that ``convert_xyz_to_lmdb`` writes:
        {atomic_numbers, pos, cell, pbc}
    with ghost atoms encoded as ``-1`` in ``atomic_numbers``.
    """
    lmdb_filename.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(lmdb_filename), subdir=False, map_size=int(1e12))
    with env.begin(write=True) as txn:
        for i, d in enumerate(data):
            atomic_numbers = d.species.cpu().numpy().astype(np.int64)
            value = {
                "atomic_numbers": torch.from_numpy(atomic_numbers),
                "pos": d.pos.cpu().to(dtype=torch.float64),
                "cell": d.cell[0].cpu().to(dtype=torch.float64),
                "pbc": torch.tensor(
                    [True, True, True], dtype=torch.bool
                ),
            }
            txn.put(str(i).encode("utf-8"), pickle.dumps(value))
    env.close()
    print(f"Wrote {len(data)} structures to {lmdb_filename}")


def save_atoms_to_lmdb(
    data: Union[OMGData, list[OMGData]], xyz_filename: Path, lmdb_filename: Path
) -> None:
    """Legacy: extxyz round-trip path. Kept for reproducibility / fallback.

    The fast path is :func:`save_omgdata_direct_to_lmdb`, which skips the
    intermediate extxyz file.
    """
    print(
        f"--- Starting save process from {xyz_filename} to {lmdb_filename}: Atoms -> XYZ -> LMDB ---"
    )
    xyz_saver(data, xyz_filename)
    print(f"Successfully saved intermediate file: '{xyz_filename}'")

    convert_xyz_to_lmdb(xyz_filename, lmdb_filename)
    print(
        f"--- Save process from {xyz_filename} to {lmdb_filename}: Atoms -> XYZ -> LMDB complete ---"
    )


def ghost_dataset(dataset: OMGTorchDataset, dataset_type: str) -> list[OMGData]:
    """Generate ghost atoms for every structure in ``dataset``.

    Parallelizes across N worker processes in chunks of N so the original
    per-sample timeout semantics are preserved (a stuck sample in a chunk does
    not block the rest of the chunk, which has already been submitted).

    Tunable via env vars:
      - GHOST_GEN_WORKERS: number of worker processes (default 16)
      - GHOST_GEN_TIMEOUT: per-sample timeout in seconds (default 20)
    """
    ghosted_data_list = []
    n_workers = int(os.environ.get("GHOST_GEN_WORKERS", "16"))
    sample_timeout = float(os.environ.get("GHOST_GEN_TIMEOUT", "20"))

    voronoi_generator_config = {
        "desired_atom_count": VoronoiGenerator.desired_atom_count,
        "dist_eval": VoronoiGenerator.dist_eval,
        "epsilon": VoronoiGenerator.epsilon,
        "num_min_distances": VoronoiGenerator.num_min_distances,
        "weight_distances": VoronoiGenerator.weight_distances,
        "noise_magnitude": VoronoiGenerator.noise_magnitude,
    }

    print(
        f"Ghosting {dataset_type} dataset with {n_workers} workers "
        f"(per-sample timeout: {sample_timeout}s)"
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

    with mp.Pool(processes=n_workers, initializer=_worker_init) as pool:
        chunk: list = []
        for i, single_data in enumerate(
            tqdm(dataset, desc=f"Ghosting {dataset_type}")
        ):
            args = (i, single_data, voronoi_generator_config)
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
    train_dataset, test_dataset, val_dataset = load_data()

    try:
        if not inputs and len(sys.argv) < 2:
            print("Usage: python parallelizable_data_gen.py <dataset_type>")
            print("  dataset_type: train, test, or val")
            sys.exit(1)

        dataset_type = inputs[0] if inputs else sys.argv[1]

        if dataset_type == "train":
            dataset = train_dataset
        elif dataset_type == "test":
            dataset = test_dataset
        elif dataset_type == "val":
            dataset = val_dataset
        else:
            raise ValueError("Invalid dataset. Must be 'train', 'test', or 'val'")

        ghosted_data_list = ghost_dataset(dataset, dataset_type)

        # --- Define File Paths for the full ghosted dataset ---
        run_folder = inputs[1] if inputs else sys.argv[2]
        processed_dir = Path(f"./processed_datasets/{run_folder}")
        processed_dir.mkdir(parents=True, exist_ok=True)

        full_lmdb_filepath = processed_dir / f"{dataset_type}_ghosted.lmdb"

        # Fast path: write OMGData directly to LMDB (no extxyz round-trip).
        # Set GHOST_GEN_WRITE_XYZ=1 to also emit the legacy extxyz file.
        save_omgdata_direct_to_lmdb(ghosted_data_list, full_lmdb_filepath)

        if os.environ.get("GHOST_GEN_WRITE_XYZ", "0") == "1":
            xyz_path = processed_dir / f"{dataset_type}_ghosted.extxyz"
            xyz_saver(ghosted_data_list, xyz_path)
            print(f"Also wrote extxyz mirror: {xyz_path}")
    finally:
        print("Cleaning up dataset resources...")
        train_dataset.dataset.cleanup()
        test_dataset.dataset.cleanup()
        val_dataset.dataset.cleanup()


if __name__ == "__main__":
    main()
