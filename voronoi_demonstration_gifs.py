#!/usr/bin/env python3
"""
Generate demonstration animations for Voronoi-based ghost atom placement.

Produces MP4 (recommended; reliable playback speed) and/or GIF.

Usage:
  python voronoi_demonstration_gifs.py --format mp4 --output-dir ./demo_gifs_out
  python voronoi_demonstration_gifs.py --format both --speed-factor 2.0
  python voronoi_demonstration_gifs.py --format mp4 --fps 0.8   # constant 0.8 frames/sec
  python voronoi_demonstration_gifs.py --finale-duration 0.12   # 3D finale frame hold time
"""

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from typing import Literal

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial import Voronoi, voronoi_plot_2d

from voronoi_weighted import VoronoiPhantomCellGenerator

# ---------------------------------------------------------------------------
# Shared constants & styling
# ---------------------------------------------------------------------------

GHOST_COLOR = "#FF2222"
REAL_CMAP = plt.cm.viridis
FIG_BG = "white"
CELL_EDGE_COLOR = "#222222"
SUPERCELL_GRID_COLOR = "#BBBBBB"
CENTER_CELL_COLOR = "#2266CC"
VORONOI_EDGE_COLOR = "#E88833"
CAVITY_CIRCLE_COLOR = "#CC3333"
HIGHLIGHT_VERTEX_COLOR = "#FF6600"

FramePhase2D = Literal["base", "supercell", "tessellation", "highlight", "placed"]
FramePhase3D = Literal["cavity", "place", "clean"]


@dataclass
class GhostStep2D:
    """One ghost-placement iteration for the 2D demo."""

    vor: Voronoi
    tiled_points: np.ndarray
    vertices: np.ndarray
    best_vertex: np.ndarray
    cavity_radius: float
    ghost_position: np.ndarray


@dataclass
class GhostStep3D:
    """One ghost-placement iteration for the 3D demo."""

    best_vertex: np.ndarray
    cavity_radius: float
    ghost_position: np.ndarray


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _cell_polygon_2d(x_vec: np.ndarray, y_vec: np.ndarray, origin: np.ndarray | None = None) -> np.ndarray:
    o = np.zeros(2) if origin is None else np.asarray(origin, dtype=float)
    return np.array([o, o + x_vec, o + x_vec + y_vec, o + y_vec])


def _fractional_coords_2d(cart: np.ndarray, x_vec: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
    cell = np.column_stack([x_vec, y_vec])
    return np.linalg.solve(cell, np.asarray(cart, dtype=float).T).T


def _in_center_cell(
    cart: np.ndarray, x_vec: np.ndarray, y_vec: np.ndarray, epsilon: float = 1e-6
) -> np.ndarray:
    frac = _fractional_coords_2d(cart, x_vec, y_vec)
    return np.all(
        (frac >= -epsilon) & (frac <= 1.0 + epsilon),
        axis=1,
    )


def _supercell_view_limits(
    x_vec: np.ndarray, y_vec: np.ndarray, margin_frac: float = 0.08
) -> tuple[float, float, float, float]:
    """Axis limits spanning the 3x3 tiled supercell (notebook-style view)."""
    corners = []
    for i in (-1, 0, 1, 2):
        for j in (-1, 0, 1, 2):
            o = i * x_vec + j * y_vec
            corners.extend([o, o + x_vec, o + y_vec, o + x_vec + y_vec])
    corners = np.array(corners)
    span = max(np.linalg.norm(x_vec), np.linalg.norm(y_vec))
    m = margin_frac * span
    return (
        corners[:, 0].min() - m,
        corners[:, 0].max() + m,
        corners[:, 1].min() - m,
        corners[:, 1].max() + m,
    )


def _draw_supercell_grid(ax, x_vec: np.ndarray, y_vec: np.ndarray) -> None:
    """Draw 3x3 parallelogram cell outlines (dashed), highlight center cell."""
    for i in range(-1, 2):
        for j in range(-1, 2):
            origin = i * x_vec + j * y_vec
            poly = _cell_polygon_2d(x_vec, y_vec, origin)
            is_center = i == 0 and j == 0
            ax.add_patch(
                Polygon(
                    poly,
                    closed=True,
                    fill=False,
                    edgecolor=CENTER_CELL_COLOR if is_center else SUPERCELL_GRID_COLOR,
                    linewidth=2.5 if is_center else 0.9,
                    linestyle="-" if is_center else "--",
                    zorder=1,
                )
            )


def _cell_edges_3d(x_vec: np.ndarray, y_vec: np.ndarray, z_vec: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    origin = np.zeros(3)
    verts = [
        origin,
        x_vec,
        y_vec,
        z_vec,
        x_vec + y_vec,
        x_vec + z_vec,
        y_vec + z_vec,
        x_vec + y_vec + z_vec,
    ]
    edge_idx = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    ]
    return [(verts[i], verts[j]) for i, j in edge_idx]


def _sphere_mesh(center: np.ndarray, radius: float, n: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    uu, vv = np.meshgrid(u, v)
    x = center[0] + radius * np.cos(uu) * np.sin(vv)
    y = center[1] + radius * np.sin(uu) * np.sin(vv)
    z = center[2] + radius * np.cos(vv)
    return x, y, z


def _species_color(species: int) -> str:
    if species < 0:
        return GHOST_COLOR
    norm = plt.Normalize(vmin=1, vmax=30)
    return REAL_CMAP(norm(max(species, 1)))


def _species_size(species: int, dim: int = 2) -> float:
    if species < 0:
        return 90 if dim == 2 else 40
    base = 120 if dim == 2 else 55
    return base + species * 4


# ---------------------------------------------------------------------------
# 2D Voronoi engine (mirrors voronoi_weighted 3D logic in 2D)
# ---------------------------------------------------------------------------


def find_furthest_vertex_in_center(
    vor: Voronoi,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray | None, float, np.ndarray]:
    """
    Pick the Voronoi vertex in the center unit cell with largest min-distance
    to any generator in the tiled supercell (vor.points), including periodic
    images — matches 2D_tesselations.ipynb.
    """
    finite = vor.vertices[np.all(np.isfinite(vor.vertices), axis=1)]
    if len(finite) == 0:
        return None, 0.0, np.zeros((0, 2))

    mask = _in_center_cell(finite, x_vec, y_vec, epsilon)
    center_vertices = finite[mask]
    if len(center_vertices) == 0:
        return None, 0.0, np.zeros((0, 2))

    # All tiled generators (periodic images), not center-cell atoms only
    generators = vor.points

    best_v = center_vertices[0]
    best_r = 0.0
    for v in center_vertices:
        dists = np.linalg.norm(v - generators, axis=1)
        r = float(np.min(dists))
        if r > best_r:
            best_r = r
            best_v = v
    return best_v, best_r, center_vertices


class VoronoiPhantom2D:
    """2D Voronoi ghost placement (3x3 supercell + voronoi_plot_2d visualization)."""

    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def create_tiled_points(
        self, points: np.ndarray, x_vec: np.ndarray, y_vec: np.ndarray
    ) -> np.ndarray:
        """Tile the unit cell across a 3x3 grid using lattice translations."""
        tiled = []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                tiled.append(points + i * x_vec + j * y_vec)
        return np.concatenate(tiled)

    def compute_next_step(
        self,
        points: np.ndarray,
        x_vec: np.ndarray,
        y_vec: np.ndarray,
    ) -> GhostStep2D:
        tiled = self.create_tiled_points(points, x_vec, y_vec)
        vor = Voronoi(tiled)
        best_v, radius, center_vertices = find_furthest_vertex_in_center(
            vor, x_vec, y_vec, self.epsilon
        )
        if best_v is None:
            best_v = np.mean(points, axis=0)
            radius = 0.0
            center_vertices = np.zeros((0, 2))

        return GhostStep2D(
            vor=vor,
            tiled_points=tiled,
            vertices=center_vertices,
            best_vertex=best_v,
            cavity_radius=radius,
            ghost_position=best_v,
        )


# ---------------------------------------------------------------------------
# Synthetic demo cells
# ---------------------------------------------------------------------------


def make_demo_cell_2d() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (points, species, x_vec, y_vec) for a 10-atom 2D cell."""
    x_vec = np.array([8.0, 0.0])
    y_vec = np.array([1.2, 7.5])
    # Hand-placed fractional coords -> Cartesian
    frac = np.array([
        [0.12, 0.18],
        [0.35, 0.22],
        [0.62, 0.15],
        [0.78, 0.38],
        [0.25, 0.45],
        [0.48, 0.55],
        [0.70, 0.62],
        [0.18, 0.72],
        [0.42, 0.78],
        [0.58, 0.35],
    ])
    cell = np.column_stack([x_vec, y_vec])
    points = frac @ cell.T
    species = np.array([6, 8, 14, 6, 8, 14, 6, 8, 14, 6], dtype=int)
    return points, species, x_vec, y_vec


def make_demo_cell_3d() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (points, species, x_vec, y_vec, z_vec) for a 10-atom 3D cell."""
    # Triclinic / slanted cell for a more interesting visualization
    x_vec = np.array([5.4, 0.5, 0.3])
    y_vec = np.array([0.4, 5.2, 0.35])
    z_vec = np.array([0.6, 0.45, 5.6])
    frac = np.array([
        [0.15, 0.20, 0.25],
        [0.40, 0.18, 0.30],
        [0.65, 0.22, 0.20],
        [0.25, 0.45, 0.55],
        [0.50, 0.50, 0.50],
        [0.72, 0.48, 0.62],
        [0.20, 0.70, 0.35],
        [0.55, 0.75, 0.28],
        [0.35, 0.30, 0.72],
        [0.68, 0.68, 0.40],
    ])
    cell = np.column_stack([x_vec, y_vec, z_vec])
    points = frac @ cell.T
    species = np.array([6, 8, 14, 6, 8, 14, 6, 8, 14, 6], dtype=int)
    return points, species, x_vec, y_vec, z_vec


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------


def render_frame_2d(
    points: np.ndarray,
    species: np.ndarray,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    phase: FramePhase2D,
    step: GhostStep2D | None = None,
    title: str = "",
    dpi: int = 100,
    figsize: tuple[float, float] = (7, 7),
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor(FIG_BG)
    fig.patch.set_facecolor(FIG_BG)

    use_supercell_view = phase in ("supercell", "tessellation", "highlight")

    if use_supercell_view:
        xmin, xmax, ymin, ymax = _supercell_view_limits(x_vec, y_vec)
        _draw_supercell_grid(ax, x_vec, y_vec)

        if step is not None:
            # Periodic images (small gray dots)
            image_mask = ~_in_center_cell(step.tiled_points, x_vec, y_vec)
            if np.any(image_mask):
                imgs = step.tiled_points[image_mask]
                ax.scatter(
                    imgs[:, 0], imgs[:, 1],
                    s=22, c="#AAAAAA", alpha=0.55, zorder=2,
                )

            if phase in ("tessellation", "highlight"):
                voronoi_plot_2d(
                    step.vor,
                    ax=ax,
                    show_vertices=True,
                    line_colors=VORONOI_EDGE_COLOR,
                    line_width=1.4,
                    line_alpha=0.75,
                    point_size=2,
                )

            if phase == "highlight":
                if len(step.vertices) > 0:
                    ax.scatter(
                        step.vertices[:, 0],
                        step.vertices[:, 1],
                        s=28,
                        c="#999999",
                        marker="x",
                        zorder=4,
                        linewidths=0.9,
                    )
                ax.add_patch(
                    Circle(
                        step.best_vertex,
                        step.cavity_radius,
                        fill=False,
                        edgecolor=CAVITY_CIRCLE_COLOR,
                        linewidth=2.2,
                        linestyle="--",
                        zorder=5,
                    )
                )
                ax.scatter(
                    [step.best_vertex[0]],
                    [step.best_vertex[1]],
                    s=220,
                    c=HIGHLIGHT_VERTEX_COLOR,
                    marker="*",
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=7,
                )
    else:
        poly = _cell_polygon_2d(x_vec, y_vec)
        ax.add_patch(
            Polygon(
                poly, closed=True, fill=False,
                edgecolor=CELL_EDGE_COLOR, linewidth=2, zorder=1,
            )
        )
        margin = 0.6
        ax.set_xlim(poly[:, 0].min() - margin, poly[:, 0].max() + margin)
        ax.set_ylim(poly[:, 1].min() - margin, poly[:, 1].max() + margin)

    real_mask = species >= 0
    ghost_mask = species < 0

    if np.any(real_mask):
        for pt, sp in zip(points[real_mask], species[real_mask]):
            ax.scatter(
                pt[0], pt[1],
                s=_species_size(sp, 2),
                c=[_species_color(sp)],
                edgecolors="white",
                linewidths=0.6,
                zorder=6,
            )

    if np.any(ghost_mask):
        ax.scatter(
            points[ghost_mask, 0],
            points[ghost_mask, 1],
            s=_species_size(-1, 2),
            c=GHOST_COLOR,
            edgecolors="darkred",
            linewidths=0.6,
            zorder=6,
        )

    if use_supercell_view:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=FIG_BG)
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)


def render_frame_3d(
    points: np.ndarray,
    species: np.ndarray,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    z_vec: np.ndarray,
    phase: FramePhase3D,
    step: GhostStep3D | None = None,
    azimuth: float = 30.0,
    elevation: float = 22.0,
    title: str = "",
    dpi: int = 100,
    figsize: tuple[float, float] = (6, 6),
    sphere_alpha: float = 0.35,
) -> np.ndarray:
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(FIG_BG)

    for v0, v1 in _cell_edges_3d(x_vec, y_vec, z_vec):
        ax.plot(
            [v0[0], v1[0]], [v0[1], v1[1]], [v0[2], v1[2]],
            color=CELL_EDGE_COLOR, linewidth=1.5, zorder=1,
        )

    if phase in ("cavity", "place") and step is not None:
        alpha = sphere_alpha if phase == "cavity" else sphere_alpha * 0.5
        scale = 1.0 if phase == "cavity" else 0.35
        r = step.cavity_radius * scale
        sx, sy, sz = _sphere_mesh(step.best_vertex, r)
        ax.plot_surface(
            sx, sy, sz,
            color=GHOST_COLOR,
            alpha=alpha,
            linewidth=0,
            shade=True,
            zorder=2,
        )

    real_mask = species >= 0
    ghost_mask = species < 0

    if np.any(real_mask):
        ax.scatter(
            points[real_mask, 0],
            points[real_mask, 1],
            points[real_mask, 2],
            s=[_species_size(s, 3) for s in species[real_mask]],
            c=[_species_color(s) for s in species[real_mask]],
            edgecolors="white",
            linewidths=0.3,
            depthshade=True,
            zorder=4,
        )

    if np.any(ghost_mask):
        ax.scatter(
            points[ghost_mask, 0],
            points[ghost_mask, 1],
            points[ghost_mask, 2],
            s=_species_size(-1, 3),
            c=GHOST_COLOR,
            edgecolors="darkred",
            linewidths=0.4,
            depthshade=True,
            zorder=4,
        )

    if phase in ("place", "clean") and step is not None:
        ax.scatter(
            [step.ghost_position[0]],
            [step.ghost_position[1]],
            [step.ghost_position[2]],
            s=_species_size(-1, 3) * 1.2,
            c=GHOST_COLOR,
            edgecolors="darkred",
            linewidths=0.5,
            zorder=6,
        )

    all_pts = np.vstack([points, x_vec, y_vec, z_vec, x_vec + y_vec + z_vec])
    center = (x_vec + y_vec + z_vec) / 2
    span = np.max(np.linalg.norm(all_pts - center, axis=1)) + 1.0
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10, pad=4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=FIG_BG)
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)


# ---------------------------------------------------------------------------
# Animation export (MP4 + GIF)
# ---------------------------------------------------------------------------

# Timeline used when expanding per-frame durations into MP4 frames
_MP4_TIMELINE_FPS = 30


def _frames_to_gif(
    frames: list[np.ndarray],
    durations: list[float],
    output_path: str,
    loop: int = 0,
) -> None:
    """Write GIF with per-frame duration via Pillow (milliseconds)."""
    if len(frames) != len(durations):
        raise ValueError("frames and durations must have the same length")
    pil_frames = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in frames]
    # GIF stores delay in ms; enforce a minimum so browsers respect timing
    duration_ms = [max(20, int(round(d * 1000))) for d in durations]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=loop,
        disposal=2,
    )
    total_s = sum(durations)
    print(f"Wrote {output_path} ({len(frames)} frames, ~{total_s:.1f}s)")


def _frames_to_mp4_timed(
    frames: list[np.ndarray],
    durations: list[float],
    output_path: str,
    timeline_fps: int = _MP4_TIMELINE_FPS,
) -> None:
    """
    Expand each logical frame by its duration at timeline_fps, then encode MP4.
    This preserves per-step pacing and works reliably in browsers/VLC.
    """
    if len(frames) != len(durations):
        raise ValueError("frames and durations must have the same length")
    expanded: list[np.ndarray] = []
    for frame, dur in zip(frames, durations):
        n = max(1, int(round(dur * timeline_fps)))
        expanded.extend([frame] * n)
    try:
        writer = imageio.get_writer(
            output_path,
            fps=timeline_fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
        )
    except Exception as e:
        raise RuntimeError(
            "MP4 export requires ffmpeg. Install with: pip install imageio-ffmpeg\n"
            "On macOS you may also need: brew install ffmpeg"
        ) from e
    for frame in expanded:
        writer.append_data(np.asarray(frame, dtype=np.uint8))
    writer.close()
    total_s = sum(durations)
    print(
        f"Wrote {output_path} ({len(frames)} logical frames -> {len(expanded)} "
        f"video frames, ~{total_s:.1f}s @ {timeline_fps} fps timeline)"
    )


def _frames_to_mp4_constant_fps(
    frames: list[np.ndarray],
    output_path: str,
    fps: float,
) -> None:
    """One logical frame per video frame at a fixed FPS (lower fps = slower)."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    try:
        imageio.mimsave(output_path, frames, fps=fps, codec="libx264", pixelformat="yuv420p")
    except Exception as e:
        raise RuntimeError(
            "MP4 export requires ffmpeg. Install with: pip install imageio-ffmpeg\n"
            "On macOS you may also need: brew install ffmpeg"
        ) from e
    total_s = len(frames) / fps
    print(f"Wrote {output_path} ({len(frames)} frames @ {fps} fps, ~{total_s:.1f}s)")


def write_animation(
    frames: list[np.ndarray],
    durations: list[float],
    output_path: str,
    *,
    constant_fps: float | None = None,
) -> None:
    """Save frames to .mp4 or .gif based on file extension."""
    ext = output_path.rsplit(".", 1)[-1].lower()
    if ext == "gif":
        _frames_to_gif(frames, durations, output_path)
    elif ext in ("mp4", "webm", "mov"):
        if constant_fps is not None:
            _frames_to_mp4_constant_fps(frames, output_path, constant_fps)
        else:
            _frames_to_mp4_timed(frames, durations, output_path)
    else:
        raise ValueError(f"Unsupported output extension: {ext}")


class VoronoiDemo2D:
    """Build the 2D Voronoi ghost-placement demonstration GIF."""

    def __init__(
        self,
        n_real: int = 10,
        n_total: int = 20,
        speed_factor: float = 1.0,
    ):
        self.n_real = n_real
        self.n_total = n_total
        self.speed_factor = speed_factor
        self.engine = VoronoiPhantom2D()

    def _build_iteration_frames(
        self,
        points: np.ndarray,
        species: np.ndarray,
        x_vec: np.ndarray,
        y_vec: np.ndarray,
        step: GhostStep2D,
        ghost_idx: int,
    ) -> tuple[list[np.ndarray], list[float]]:
        frames: list[np.ndarray] = []
        durations: list[float] = []

        def add(phase: FramePhase2D, duration: float, title: str, step_arg: GhostStep2D | None):
            frames.append(
                render_frame_2d(
                    points, species, x_vec, y_vec,
                    phase=phase, step=step_arg, title=title,
                )
            )
            durations.append(duration)

        add("base", 0.5, f"Step {ghost_idx}: unit cell", None)
        add("supercell", 0.7, f"Step {ghost_idx}: 3×3 supercell", step)
        add("tessellation", 0.9, f"Step {ghost_idx}: Voronoi tessellation", step)
        add("highlight", 1.2, f"Step {ghost_idx}: largest empty cavity", step)
        # After placement
        new_points = np.vstack([points, step.ghost_position])
        new_species = np.append(species, -1)
        frames.append(
            render_frame_2d(
                new_points, new_species, x_vec, y_vec,
                phase="placed", step=step, title=f"Step {ghost_idx}: ghost placed",
            )
        )
        durations.append(0.7)
        return frames, durations

    def build_animation(self) -> tuple[list[np.ndarray], list[float]]:
        points, species, x_vec, y_vec = make_demo_cell_2d()
        assert len(points) == self.n_real

        all_frames: list[np.ndarray] = []
        all_durations: list[float] = []

        # Opening frame: base cell only
        all_frames.append(
            render_frame_2d(
                points, species, x_vec, y_vec,
                phase="base", step=None, title="Base cell (10 atoms)",
            )
        )
        all_durations.append(1.0)

        current_pts = points.copy()
        current_sp = species.copy()
        n_ghosts = self.n_total - self.n_real

        for i in range(1, n_ghosts + 1):
            step = self.engine.compute_next_step(current_pts, x_vec, y_vec)
            fr, dur = self._build_iteration_frames(
                current_pts, current_sp, x_vec, y_vec, step, ghost_idx=i
            )
            all_frames.extend(fr)
            all_durations.extend(dur)
            current_pts = np.vstack([current_pts, step.ghost_position])
            current_sp = np.append(current_sp, -1)

        # Final hold
        all_frames.append(
            render_frame_2d(
                current_pts, current_sp, x_vec, y_vec,
                phase="base", step=None, title=f"Final cell ({self.n_total} atoms)",
            )
        )
        all_durations.append(1.5)

        if self.speed_factor != 1.0:
            all_durations = [d * self.speed_factor for d in all_durations]
        return all_frames, all_durations

    def generate(self, output_path: str, constant_fps: float | None = None) -> None:
        frames, durations = self.build_animation()
        write_animation(frames, durations, output_path, constant_fps=constant_fps)


class VoronoiDemo3D:
    """Build the 3D Voronoi ghost-placement demonstration GIF."""

    def __init__(
        self,
        n_real: int = 10,
        n_total: int = 20,
        rotation_per_frame: float = 4.5,
        finale_rotation_deg: float = 540.0,
        finale_frame_duration: float = 0.06,
        speed_factor: float = 1.0,
    ):
        self.n_real = n_real
        self.n_total = n_total
        self.rotation_per_frame = rotation_per_frame
        self.finale_rotation_deg = finale_rotation_deg
        self.finale_frame_duration = finale_frame_duration
        self.speed_factor = speed_factor
        self.generator = VoronoiPhantomCellGenerator(
            desired_atom_count=n_total,
            dist_eval="min",
            epsilon=1e-3,
            weight_distances=False,
        )

    def _compute_step(
        self,
        points: np.ndarray,
        species: np.ndarray,
        x_vec: np.ndarray,
        y_vec: np.ndarray,
        z_vec: np.ndarray,
    ) -> GhostStep3D:
        # Generator uses 0 for phantoms; pass species with ghosts as 0
        atomic = species.copy().astype(float)
        atomic[atomic < 0] = 0

        ghost_pos = self.generator._get_next_point(
            points, atomic, x_vec, y_vec, z_vec
        )
        center = (x_vec + y_vec + z_vec) / 2
        centered_pts = points - center
        centered_ghost = ghost_pos - center
        # Cavity radius: nearest atom among all periodic images (3×3×3 supercell)
        supercell_pts = self.generator._create_supercell_from_points(
            centered_pts, atomic, x_vec, y_vec, z_vec
        )
        dists = np.linalg.norm(centered_ghost - supercell_pts, axis=1)
        radius = float(np.min(dists))

        return GhostStep3D(
            best_vertex=ghost_pos,
            cavity_radius=radius,
            ghost_position=ghost_pos,
        )

    def build_animation(self) -> tuple[list[np.ndarray], list[float]]:
        points, species, x_vec, y_vec, z_vec = make_demo_cell_3d()
        assert len(points) == self.n_real

        all_frames: list[np.ndarray] = []
        all_durations: list[float] = []
        azimuth = 25.0

        all_frames.append(
            render_frame_3d(
                points, species, x_vec, y_vec, z_vec,
                phase="clean", step=None, azimuth=azimuth,
                title="Base cell (10 atoms)",
            )
        )
        all_durations.append(1.0)

        current_pts = points.copy()
        current_sp = species.copy()
        n_ghosts = self.n_total - self.n_real

        for i in range(1, n_ghosts + 1):
            step = self._compute_step(
                current_pts, current_sp, x_vec, y_vec, z_vec
            )
            azimuth += self.rotation_per_frame

            for phase, dur, title in [
                ("cavity", 0.9, f"Step {i}: largest cavity"),
                ("place", 0.7, f"Step {i}: placing ghost"),
                ("clean", 0.6, f"Step {i}: ghost added"),
            ]:
                all_frames.append(
                    render_frame_3d(
                        current_pts, current_sp, x_vec, y_vec, z_vec,
                        phase=phase, step=step, azimuth=azimuth, title=title,
                    )
                )
                all_durations.append(dur)
                azimuth += self.rotation_per_frame

            current_pts = np.vstack([current_pts, step.ghost_position])
            current_sp = np.append(current_sp, -1)

        # Finale: 540° turntable rotation of the completed structure
        n_finale = int(self.finale_rotation_deg / self.rotation_per_frame)
        for k in range(n_finale):
            all_frames.append(
                render_frame_3d(
                    current_pts, current_sp, x_vec, y_vec, z_vec,
                    phase="clean", step=None,
                    azimuth=azimuth + k * self.rotation_per_frame,
                    title=f"Final cell ({self.n_total} atoms)" if k == 0 else "",
                )
            )
            all_durations.append(self.finale_frame_duration)

        if self.speed_factor != 1.0:
            all_durations = [d * self.speed_factor for d in all_durations]
        return all_frames, all_durations

    def generate(self, output_path: str, constant_fps: float | None = None) -> None:
        frames, durations = self.build_animation()
        write_animation(frames, durations, output_path, constant_fps=constant_fps)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Voronoi ghost-placement demonstration animations."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for output files (default: current directory)",
    )
    parser.add_argument(
        "--format",
        choices=("gif", "mp4", "both"),
        default="mp4",
        help="Output format (default: mp4 — most reliable playback speed)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="MP4 only: fixed frames per second (one logical frame = one video frame). "
        "Lower is slower, e.g. 0.5 = 2 seconds per frame. Ignores per-step timing.",
    )
    parser.add_argument(
        "--only-2d",
        action="store_true",
        help="Generate only the 2D GIF",
    )
    parser.add_argument(
        "--only-3d",
        action="store_true",
        help="Generate only the 3D GIF",
    )
    parser.add_argument(
        "--n-real",
        type=int,
        default=10,
        help="Number of real atoms in the demo cell",
    )
    parser.add_argument(
        "--n-total",
        type=int,
        default=20,
        help="Target total atom count (real + ghosts)",
    )
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=1.0,
        help="Multiply all frame durations by this value (>1 slows down, <1 speeds up). "
        "Example: 2.0 makes the GIF play at half speed.",
    )
    parser.add_argument(
        "--finale-duration",
        type=float,
        default=None,
        help="Seconds per frame for the 3D finale rotation only (default: 0.06). "
        "Still scaled by --speed-factor.",
    )
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    finale_dur = 0.06 if args.finale_duration is None else args.finale_duration

    demo_2d = VoronoiDemo2D(
        n_real=args.n_real,
        n_total=args.n_total,
        speed_factor=args.speed_factor,
    )
    demo_3d = VoronoiDemo3D(
        n_real=args.n_real,
        n_total=args.n_total,
        finale_frame_duration=finale_dur,
        speed_factor=args.speed_factor,
    )

    if not args.only_3d:
        print("Generating 2D demonstration...")
        if args.format in ("gif", "both"):
            demo_2d.generate(
                os.path.join(args.output_dir, "voronoi_2d_demo.gif"),
                constant_fps=args.fps,
            )
        if args.format in ("mp4", "both"):
            demo_2d.generate(
                os.path.join(args.output_dir, "voronoi_2d_demo.mp4"),
                constant_fps=args.fps,
            )

    if not args.only_2d:
        print("Generating 3D demonstration...")
        if args.format in ("gif", "both"):
            demo_3d.generate(
                os.path.join(args.output_dir, "voronoi_3d_demo.gif"),
                constant_fps=args.fps,
            )
        if args.format in ("mp4", "both"):
            demo_3d.generate(
                os.path.join(args.output_dir, "voronoi_3d_demo.mp4"),
                constant_fps=args.fps,
            )

    print("Done.")


if __name__ == "__main__":
    main()
