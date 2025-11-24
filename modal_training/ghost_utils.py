"""
Utilities for handling ghost atoms in crystal structure generation.

Ghost atoms are used to pad structures to a fixed atom count and are stored
as atomic number -1 internally, but exported as max_real_Z+1 to avoid conflicts
with mask tokens.
"""

import numpy as np
import torch
from pathlib import Path
from ase.io import read, write
from omg.datamodule.dataloader import OMGData


def remap_ghosts_in_xyz(xyz_path: Path):
    """
    Ensure ghost atoms in an XYZ use max(real_Z)+1 so plotting logic doesn't
    collide with masked species (0/-1). Stores helper arrays for recovery.

    Args:
        xyz_path: Path to XYZ file to process in-place
    """
    if not xyz_path.exists():
        return

    atoms_list = read(xyz_path, index=":")
    updated_structures = []
    for atoms in atoms_list:
        numbers = atoms.get_atomic_numbers()
        ghost_mask = numbers <= 0

        if "is_ghost" in atoms.arrays:
            ghost_mask |= atoms.get_array("is_ghost").astype(bool)

        if "ghost_atomic_number" in atoms.arrays:
            ghost_values = np.unique(atoms.get_array("ghost_atomic_number")).astype(
                numbers.dtype
            )
            ghost_mask |= np.isin(numbers, ghost_values)

        if not ghost_mask.any():
            updated_structures.append(atoms)
            continue

        real_numbers = numbers[~ghost_mask]
        max_real = int(real_numbers.max()) if real_numbers.size else 0
        ghost_label = max_real + 1
        numbers = numbers.astype(int, copy=True)
        numbers[ghost_mask] = ghost_label

        atoms.set_atomic_numbers(numbers)
        atoms.set_array("is_ghost", ghost_mask)
        atoms.set_array(
            "ghost_atomic_number",
            np.full(len(numbers), ghost_label, dtype=numbers.dtype),
        )
        updated_structures.append(atoms)

    write(xyz_path, updated_structures)


def ase_atoms_to_omg_data(atoms):
    """
    Convert ASE Atoms object to OMGData, handling ghost atom metadata.

    Ghost atoms are identified by:
    - is_ghost array (boolean mask)
    - ghost_atomic_number array (stores the remapped label)
    - atomic number <= 0

    All ghosts are converted to species=-1 for internal use.

    Args:
        atoms: ASE Atoms object

    Returns:
        OMGData object with species=-1 for ghost atoms
    """
    species_np = atoms.numbers.astype("int64")
    ghost_mask = np.zeros_like(species_np, dtype=bool)

    if "is_ghost" in atoms.arrays:
        ghost_mask |= atoms.get_array("is_ghost").astype(bool)

    if "ghost_atomic_number" in atoms.arrays:
        ghost_values = np.unique(atoms.get_array("ghost_atomic_number")).astype("int64")
        ghost_mask |= np.isin(species_np, ghost_values)

    if not ghost_mask.any():
        ghost_mask |= species_np <= 0

    species = torch.from_numpy(species_np)
    if ghost_mask.any():
        species = species.masked_fill(torch.from_numpy(ghost_mask), -1)

    data = OMGData()
    data.pos = torch.from_numpy(atoms.positions).to(dtype=torch.float64)
    data.species = species.long()
    data.cell = torch.from_numpy(atoms.cell.array).to(dtype=torch.float64).unsqueeze(0)
    data.n_atoms = torch.tensor(len(atoms), dtype=torch.long)
    data.batch = torch.zeros(len(atoms), dtype=torch.long)
    return data
