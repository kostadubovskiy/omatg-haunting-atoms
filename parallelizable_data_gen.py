import os
import pickle
import torch
import lmdb
from typing import Union
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

from ase import Atoms
from ase.io import read, write
from omg.datamodule import DataModule
from omg.datamodule.dataloader import OMGTorchDataset, OMGData
from voronoi_weighted import VoronoiPhantomCellGenerator


VoronoiGenerator = VoronoiPhantomCellGenerator(desired_atom_count=20, dist_eval="min")


def load_data():
    train_data = DataModule()
    train_data.add_from_lmdb("./OMatG/omg/data/mp_20/train.lmdb")

    val_data = DataModule()
    val_data.add_from_lmdb("./OMatG/omg/data/mp_20/val.lmdb")

    test_data = DataModule()
    test_data.add_from_lmdb("./OMatG/omg/data/mp_20/test.lmdb")

    assert len(train_data) == 27136
    assert len(val_data) == 9047

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


def plot_crystal_with_points(single_data, idx):
    """
    Plots a crystal structure with original and ghost points.

    Args:
        single_data: A data object containing cell, positions, and species.
        idx: The index of the sample being plotted.
    """
    cell = single_data.cell[0].numpy()
    x_vec, y_vec, z_vec = cell[0], cell[1], cell[2]

    # Assuming that positions are in cartesian coords not fractional coords
    positions = single_data.pos.numpy()
    species = single_data.species.numpy()

    # Create DataFrames for both original and ghost points
    df_points = pd.DataFrame(positions, columns=["x", "y", "z"])
    df_points["species"] = species
    df_points["type"] = df_points["species"].apply(
        lambda s: "Original Points" if s != -1 else "Ghost Points"
    )

    # Normalize atomic numbers to the range [0, 1] for the color scale
    # We add 1 to handle the ghost atom at -1 correctly
    norm = plt.Normalize(vmin=-1, vmax=df_points["species"].max())

    # Get a color from the 'Viridis' color scale
    colorscale = px.colors.sequential.Viridis
    df_points["color"] = [
        px.colors.sample_colorscale(colorscale, val)[0]
        for val in norm(df_points["species"])
    ]

    # Make ghost atoms red
    df_points.loc[df_points["species"] == -1, "color"] = "red"

    # Programmatically determine size (e.g., linear scaling)
    # Base size of 6, increasing with atomic number
    df_points["size"] = 6 + (df_points["species"] + 1) * 0.75
    # --- END of programmatic color and size mapping ---

    # Create the custom hover text
    df_points["hover_text"] = df_points.apply(
        lambda row: f"A_n: {row['species']}<br>Pos: ({row['x']:.2f}, {row['y']:.2f}, {row['z']:.2f})",
        axis=1,
    )

    # Create figure
    fig = go.Figure()

    # Add scatter points for all points, using mapped color and size
    fig.add_trace(
        go.Scatter3d(
            x=df_points["x"],
            y=df_points["y"],
            z=df_points["z"],
            mode="markers",
            marker=dict(size=df_points["size"], color=df_points["color"], opacity=0.6),
            name="Atoms",  # Combined legend entry
            hovertext=df_points["hover_text"],
            hoverinfo="text",
        )
    )

    # Define the 8 vertices of the parallelepiped
    origin = np.array([0, 0, 0])
    vertices = [
        origin,  # 0: (0,0,0)
        x_vec,  # 1: x_vec
        y_vec,  # 2: y_vec
        z_vec,  # 3: z_vec
        x_vec + y_vec,  # 4: x_vec + y_vec
        x_vec + z_vec,  # 5: x_vec + z_vec
        y_vec + z_vec,  # 6: y_vec + z_vec
        x_vec + y_vec + z_vec,  # 7: x_vec + y_vec + z_vec
    ]

    # Define the 12 edges (pairs of vertex indices to connect)
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),  # edges from origin
        (1, 4),
        (1, 5),  # edges from x_vec
        (2, 4),
        (2, 6),  # edges from y_vec
        (3, 5),
        (3, 6),  # edges from z_vec
        (4, 7),
        (5, 7),
        (6, 7),  # edges to opposite corner
    ]

    # Draw each edge
    for i, j in edges:
        v1, v2 = vertices[i], vertices[j]
        fig.add_trace(
            go.Scatter3d(
                x=[v1[0], v2[0]],
                y=[v1[1], v2[1]],
                z=[v1[2], v2[2]],
                mode="lines",
                line=dict(color="black", width=3),
                showlegend=False,
                hoverinfo="none",
            )
        )

    fig.update_layout(
        title=f"Sample {idx}: Newly Loaded Data",
        width=500,  # Width in pixels
        height=500,  # Height in pixels
        autosize=False,  # Disable autosize to use fixed dimensions
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
    )

    fig.show()


def xyz_saver(data: Union[OMGData, list[OMGData]], filename: Path) -> None:
    """
    Save structures from OMGData instances to an xyz file.
    Correctly handles a list of single data objects and converts ghost
    atom species from -1 to 0 for compatibility with the XYZ format.
    """
    if not filename.suffix == ".xyz":
        raise ValueError("The filename must have the suffix '.xyz'.")
    if not isinstance(data, list):
        data = [data]

    atoms_list = []
    for d in data:
        # Get species and convert to numpy array
        species_np = d.species.cpu().numpy()

        # Replace -1 with 0 for ghost atoms (XYZ format compatibility)
        # 0 corresponds to the 'X' dummy atom in ASE.
        # TODO: are we accidentally mixing ghost and mask atoms then?
        species_np[species_np == -1] = 0

        atoms_list.append(
            Atoms(
                numbers=species_np,
                positions=d.pos.cpu().numpy(),
                cell=d.cell[0].cpu().numpy(),
                pbc=True,
            )
        )

    # Overwrite the file with the new list of atoms.
    os.makedirs(filename.parent, exist_ok=True)
    write(filename, atoms_list, append=False)


def convert_xyz_to_lmdb(input_xyz_file: Path, output_lmdb_file: Path) -> None:
    """Converts an ASE-readable xyz file to a single-file LMDB database."""
    if not input_xyz_file.exists():
        print(f"Error: Input file not found at {input_xyz_file}")
        return

    try:
        atoms_list = read(input_xyz_file, index=":")
        print(f"Read {len(atoms_list)} structures from '{input_xyz_file}'")
    except Exception as e:
        print(f"Error reading XYZ file: {e}")
        return

    env = lmdb.open(str(output_lmdb_file), subdir=False, map_size=int(1e12))
    with env.begin(write=True) as txn:
        for i, atoms in enumerate(atoms_list):
            key = str(i).encode("utf-8")
            value = {
                "atomic_numbers": torch.from_numpy(atoms.get_atomic_numbers()),
                "pos": torch.from_numpy(atoms.get_positions()).to(dtype=torch.float64),
                "cell": torch.from_numpy(atoms.get_cell()[:]).to(dtype=torch.float64),
                "pbc": torch.tensor(atoms.get_pbc(), dtype=torch.bool),
            }
            serialized_value = pickle.dumps(value)
            txn.put(key, serialized_value)
    env.close()
    print(f"Successfully created single-file LMDB: '{output_lmdb_file}'")


def save_atoms_to_lmdb(
    data: Union[OMGData, list[OMGData]], xyz_filename: Path, lmdb_filename: Path
) -> None:
    """
    Saves atomic data to a single-file LMDB by first saving to an
    intermediate XYZ file and then converting.
    """
    print(
        f"--- Starting save process from {xyz_filename} to {lmdb_filename}: Atoms -> XYZ -> LMDB ---"
    )
    # Step 1: Save the data to an XYZ file
    xyz_saver(data, xyz_filename)
    print(f"Successfully saved intermediate file: '{xyz_filename}'")

    # Step 2: Convert the XYZ file to an LMDB file
    convert_xyz_to_lmdb(xyz_filename, lmdb_filename)
    print(
        f"--- Save process complete from {xyz_filename} to {lmdb_filename}: Atoms -> XYZ -> LMDB ---"
    )


def ghost_dataset(dataset: OMGTorchDataset, dataset_type: str) -> list[OMGData]:
    # This list will store all the processed data objects
    ghosted_data_list = []

    for single_data in tqdm(dataset, desc=f"Ghosting {dataset_type} dataset"):
        # --- 1. Extract initial data ---
        cell = single_data.cell[0].numpy()
        x_vec, y_vec, z_vec = cell[0], cell[1], cell[2]

        positions = single_data.pos.numpy()
        atomic_numbers = single_data.species.numpy()

        new_points = positions.copy()
        new_atomic_numbers = atomic_numbers.copy()

        # --- 2. Generate ghost atoms ---
        iterations = 20 - single_data.n_atoms.item()
        if iterations > 0:
            for _ in range(iterations):
                # --- Save the inputs for debugging ---
                debug_data = {
                    "points": new_points,
                    "atomic_numbers": new_atomic_numbers,
                    "x_vec": x_vec,
                    "y_vec": y_vec,
                    "z_vec": z_vec,
                }
                with open("crash_input.pkl", "wb") as f:
                    pickle.dump(debug_data, f)
                # --- End of debugging save ---

                next_point = VoronoiGenerator._get_next_point(
                    points=new_points,
                    atomic_numbers=new_atomic_numbers,
                    x_vec=x_vec,
                    y_vec=y_vec,
                    z_vec=z_vec,
                )

                if np.any(np.isnan(next_point)):
                    print(
                        f"NaN value detected for a sample, skipping ghost atom generation for it."
                    )
                    break

                new_atomic_numbers = np.append(new_atomic_numbers, -1)
                new_points = np.vstack([new_points, next_point])

        # --- 3. Update the data object with the new atom info ---
        single_data.pos = torch.from_numpy(new_points).to(dtype=torch.float64)
        single_data.n_atoms = torch.tensor(len(new_points), dtype=torch.long)
        single_data.batch = torch.zeros(len(new_points), dtype=torch.long)

        # Update species efficiently
        single_data.species = torch.from_numpy(new_atomic_numbers).long()

        ghosted_data_list.append(single_data)

    print(
        f"\n✅ Ghost atom generation complete. Total samples processed: {len(ghosted_data_list)}"
    )

    return ghosted_data_list


def main():
    train_dataset, test_dataset, val_dataset = load_data()

    dataset_type = input("Enter the dataset to ghost: train, test, or val:\n")
    if dataset_type == "train":
        dataset = train_dataset
    elif dataset_type == "test":
        dataset = test_dataset
    elif dataset_type == "val":
        dataset = val_dataset
    else:
        raise ValueError("Invalid dataset")

    ghosted_data_list = ghost_dataset(dataset, dataset_type)

    # --- Define File Paths for the full ghosted dataset ---
    processed_dir = Path("./processed_datasets/")
    processed_dir.mkdir(exist_ok=True)  # Ensure the directory exists

    full_xyz_filepath = processed_dir / f"{dataset_type}_ghosted.xyz"
    full_lmdb_filepath = processed_dir / f"{dataset_type}_ghosted.lmdb"

    # --- Execute the full saving pipeline for the new dataset ---
    save_atoms_to_lmdb(ghosted_data_list, full_xyz_filepath, full_lmdb_filepath)


if __name__ == "__main__":
    main()
