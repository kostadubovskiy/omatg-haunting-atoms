"""
Utility functions for plotting crystal structures with ghost atoms.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt


def plot_crystal_with_points(single_data, idx, title=None, highlight_points=None):
    """
    Plots a crystal structure with original and ghost points.

    Args:
        single_data: A data object containing cell, positions, and species.
        idx: The index of the sample being plotted.
        title (str, optional): The title for the plot. Defaults to None.
        highlight_points (np.ndarray, optional): Array of points to highlight.
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

    # Add highlighted points if provided
    if highlight_points is not None:
        # Ensure highlight_points is a 2D array
        points_to_plot = np.atleast_2d(highlight_points)
        fig.add_trace(
            go.Scatter3d(
                x=points_to_plot[:, 0],
                y=points_to_plot[:, 1],
                z=points_to_plot[:, 2],
                mode="markers",
                marker=dict(size=10, color="cyan", symbol="diamond", opacity=0.9),
                name="Highlighted Point",
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
        title=title if title else f"Sample {idx}: Newly Loaded Data",
        width=500,  # Width in pixels
        height=500,  # Height in pixels
        autosize=False,  # Disable autosize to use fixed dimensions
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
    )

    fig.show()
