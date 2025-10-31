import numpy as np
from numpy.typing import NDArray
import freud
from typing import Generator


class VoronoiPhantomCellGenerator:
    """
    A generator to add phantom atoms to a cell configuration using Voronoi analysis.
    """

    def __init__(self, desired_atom_count: int, dist_eval: str = 'min'):
        """
        Initializes the VoronoiPhantomCellGenerator.

        Args:
            desired_atom_count (int): The target number of atoms for the cell.
            dist_eval (str): The distance evaluation metric to use.
                Options: 'min', 'avg', 'root_mean_avg', 'median', 'reciprocal'.
        """
        if desired_atom_count <= 0:
            raise ValueError("desired_atom_count must be a positive integer.")
        self.desired_atom_count = desired_atom_count
        self.dist_eval = dist_eval

    def generate_phantoms(self, points: NDArray, x_vec: NDArray, y_vec: NDArray, z_vec: NDArray) -> Generator[NDArray, None, None]:
        """
        Generates phantom atoms for a given cell until the desired atom count is reached.

        This is a generator that yields each new phantom atom's coordinates.

        Args:
            points (NDArray): Initial points in the unit cell, shape (n, 3).
            x_vec (NDArray): First lattice vector.
            y_vec (NDArray): Second lattice vector.
            z_vec (NDArray): Third lattice vector.

        Yields:
            NDArray: The coordinates of the next phantom atom, shape (3,).
        """
        current_points = np.copy(points)
        while len(current_points) < self.desired_atom_count:
            next_point = self._get_next_point(current_points, x_vec, y_vec, z_vec)
            yield next_point
            current_points = np.vstack([current_points, next_point])

    def _create_supercell_from_points(self, points: NDArray, x_vec: NDArray, y_vec: NDArray, z_vec: NDArray) -> NDArray:
        """
        Create a 3x3x3 supercell from points in a unit cell defined by three vectors.
        """
        v1 = np.array(x_vec)
        v2 = np.array(y_vec)
        v3 = np.array(z_vec)

        supercell_points = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    translation = i * v1 + j * v2 + k * v3
                    translated_points = points + translation
                    supercell_points.append(translated_points)

        return np.concatenate(supercell_points)

    def _get_farthest_point(self, vertices: NDArray, points: NDArray) -> NDArray:
        """
        Finds the Voronoi vertex that is farthest from the existing points.
        """
        max_distance = 0
        farthest_vertice = None

        for vertice in vertices:
            distances = np.linalg.norm(vertice - points, axis=1)
            if self.dist_eval == 'min':
                eval_distance = np.min(distances)
            elif self.dist_eval == 'avg':
                eval_distance = np.mean(distances)
            elif self.dist_eval == 'root_mean_avg':
                eval_distance = np.sqrt(np.mean(distances ** 2))
            elif self.dist_eval == 'median':
                eval_distance = np.median(distances)
            elif self.dist_eval == 'reciprocal':
                eval_distance = 1 / np.mean(1 / distances)
            else:
                raise ValueError(f"Unknown dist_eval: {self.dist_eval}")

            if eval_distance > max_distance:
                max_distance = eval_distance
                farthest_vertice = vertice
        
        if farthest_vertice is None and len(vertices) > 0:
            return vertices[0]
        if farthest_vertice is None:
            raise RuntimeError("Could not determine the farthest point.")

        return farthest_vertice

    def _get_voronoi_vertices(self, supercell_points: NDArray, x_vec: NDArray, y_vec: NDArray, z_vec: NDArray) -> NDArray:
        """
        Computes the Voronoi vertices for the central cell of a supercell.
        """
        box_matrix = np.column_stack([3 * np.array(x_vec), 3 * np.array(y_vec), 3 * np.array(z_vec)])
        box = freud.box.Box.from_matrix(box_matrix)
        voro = freud.locality.Voronoi()
        cells = voro.compute((box, supercell_points)).polytopes

        all_vertices = box.wrap(np.concatenate(cells))
        unique_vertices = np.unique(np.round(all_vertices, decimals=10), axis=0)
        freud_matrix = box.to_matrix()
        voronoi_vertices = (box_matrix @ np.linalg.solve(freud_matrix, unique_vertices.T)).T
        
        fractional_coords = np.linalg.solve(box_matrix/3, voronoi_vertices.T).T
        mask = np.all((fractional_coords >= -0.5) & (fractional_coords < 0.5), axis=1)
        center_cell_vertices = voronoi_vertices[mask]

        return center_cell_vertices

    def _get_next_point(self, points: NDArray, x_vec: NDArray, y_vec: NDArray, z_vec: NDArray) -> NDArray:
        """
        Calculates the next phantom atom to add based on Voronoi analysis.
        """
        center_vector = (np.array(x_vec) + np.array(y_vec) + np.array(z_vec)) / 2
        centered_points = points - center_vector
        
        supercell_points = self._create_supercell_from_points(centered_points, x_vec, y_vec, z_vec)
        
        voronoi_vertices = self._get_voronoi_vertices(supercell_points, x_vec, y_vec, z_vec)
        
        next_point = self._get_farthest_point(voronoi_vertices, centered_points) + center_vector
        return next_point
