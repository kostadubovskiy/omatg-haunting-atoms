import numpy as np
from numpy.typing import NDArray
import freud
from typing import Generator


class VoronoiPhantomCellGenerator:
    """
    A generator to add phantom atoms to a cell configuration using Voronoi analysis.
    """

    valid_dist_eval_options = [
        "min",
        "avg",
        "root_mean_avg",
        "median",
        "reciprocal",
        "avg_x_min",
    ]

    def __init__(
        self,
        desired_atom_count: int,
        dist_eval: str = "min",
        epsilon: float = 1e-3,
        num_min_distances: int = None,
        weight_distances: bool = False,
    ):
        """
        Initializes the VoronoiPhantomCellGenerator.

        Args:
            desired_atom_count (int): The target number of atoms for each structure.
            dist_eval (str): The method for evaluating vertex distances.
                Options: "min", "avg", "root_mean_avg", "median", "reciprocal", "avg_x_min".
            epsilon (float): A small value for floating point comparisons.
            num_min_distances (int, optional): The number of smallest distances to average.
                Required when dist_eval is "avg_x_min". Defaults to None.
            weight_distances (bool): Whether to weight the distances by the atomic numbers.
                Defaults to False.
        """
        self.desired_atom_count = desired_atom_count
        self.dist_eval = dist_eval
        self.epsilon = epsilon
        self.num_min_distances = num_min_distances
        self.weight_distances = weight_distances

        if self.dist_eval == "avg_x_min" and (
            self.num_min_distances is None or self.num_min_distances <= 0
        ):
            raise ValueError(
                "num_min_distances must be a positive integer for 'avg_x_min' dist_eval."
            )

        if self.dist_eval not in self.valid_dist_eval_options:
            raise ValueError(
                f"Invalid dist_eval: {self.dist_eval}. Valid options are: {self.valid_dist_eval_options}"
            )

    def generate_phantoms(
        self,
        points: NDArray,
        atomic_numbers: NDArray,
        x_vec: NDArray,
        y_vec: NDArray,
        z_vec: NDArray,
    ) -> Generator[NDArray, None, None]:
        """
        Generates phantom atoms for a given cell until the desired atom count is reached.

        This is a generator that yields each new phantom atom's coordinates.

        Args:
            points (NDArray): Initial points in the unit cell, shape (n, 3).
            atomic_numbers (NDArray): Atomic numbers for each point, shape (n,).
            x_vec (NDArray): First lattice vector.
            y_vec (NDArray): Second lattice vector.
            z_vec (NDArray): Third lattice vector.

        Yields:
            NDArray: The coordinates of the next phantom atom, shape (3,).
        """
        current_points = np.copy(points)
        current_atomic_numbers = np.copy(atomic_numbers)
        while len(current_points) < self.desired_atom_count:
            next_point = self._get_next_point(
                current_points, current_atomic_numbers, x_vec, y_vec, z_vec
            )
            yield next_point
            current_points = np.vstack([current_points, next_point])
            current_atomic_numbers = np.append(current_atomic_numbers, 0)

    def _create_supercell_from_points(
        self,
        points: NDArray,
        atomic_numbers: NDArray,
        x_vec: NDArray,
        y_vec: NDArray,
        z_vec: NDArray,
    ) -> NDArray:
        """
        Create a 3x3x3 supercell from points in a unit cell defined by three vectors.
        This ensures that the central cell is surrounded by 26 neighboring cells,
        which is necessary for correct periodic Voronoi tessellation.
        """
        v1 = np.array(x_vec)
        v2 = np.array(y_vec)
        v3 = np.array(z_vec)
        noise_magnitude = 0.02

        supercell_points = []
        # Iterate over the 27 cells in a 3x3x3 grid (-1, 0, 1 in each dimension)
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    noise = (np.random.rand(3) - 0.5) * noise_magnitude
                    # Calculate the translation vector for each cell
                    translation = (i + noise[0]) * v1 + (j + noise[1]) * v2 + (k + noise[2]) * v3
                    translated_points = points + translation
                    supercell_points.append(translated_points)

        # Concatenate all points from all 27 cells
        supercell_points = np.concatenate(supercell_points)

        # Verification: Ensure all 27 subcells are populated
        expected_points = 27 * len(points)
        if len(supercell_points) != expected_points:
            raise ValueError(
                f"Supercell creation failed: expected {expected_points} points, but got {len(supercell_points)}."
            )

        return supercell_points

    def _get_farthest_point(
        self, vertices: NDArray, points: NDArray, atomic_numbers: NDArray
    ) -> NDArray:
        """
        Finds the Voronoi vertex that is farthest from the existing points. Weights the distances by the atomic numbers.
        Weighting works by giving more weight to the points with higher atomic numbers (e.g. larger atoms should be further away from the center of the cell).
        """
        max_distance = 0
        farthest_vertice = None

        if len(vertices) == 0:
            return np.array([0.0, 0.0, 0.0])  # TODO: tricky case.

        for vertice in vertices:
            distances = np.linalg.norm(vertice - points, axis=1)
            # Avoid division by zero: use absolute values and replace zeros with 1
            safe_atomic_numbers = np.abs(atomic_numbers)
            safe_atomic_numbers[safe_atomic_numbers == 0] = 1

            if self.weight_distances:
                # Use log1p(x) which is log(1+x) to avoid issues where log(1)=0
                weighted_distances = distances * np.log1p(safe_atomic_numbers)
            else:
                weighted_distances = distances

            if self.dist_eval == "min":
                eval_distance = np.min(weighted_distances)
            elif self.dist_eval == "avg":
                eval_distance = np.mean(weighted_distances)
            elif self.dist_eval == "root_mean_avg":
                eval_distance = np.sqrt(np.mean(weighted_distances**2))
            elif self.dist_eval == "median":
                eval_distance = np.median(weighted_distances)
            elif self.dist_eval == "reciprocal":
                eval_distance = 1 / np.mean(1 / weighted_distances)
            elif self.dist_eval == "avg_x_min":
                sorted_distances = np.sort(weighted_distances)
                # Ensure we don't try to average more distances than available
                x = min(self.num_min_distances, len(sorted_distances))
                eval_distance = np.mean(sorted_distances[:x])
            else:
                raise ValueError(f"Unknown dist_eval: {self.dist_eval}")

            if eval_distance > max_distance:
                max_distance = eval_distance
                farthest_vertice = vertice

        if farthest_vertice is None and len(vertices) > 0:
            return vertices[0]
        if farthest_vertice is None:
            return vertices[0]

        return farthest_vertice

    def _get_voronoi_vertices(
        self, supercell_points: NDArray, x_vec: NDArray, y_vec: NDArray, z_vec: NDArray
    ) -> NDArray:
        """
        Computes the Voronoi vertices for the central cell of a supercell.
        """
        # Add jitter to break geometric degeneracies
        # jitter_magnitude = 1e-6
        # jitter = np.random.rand(*supercell_points.shape) * jitter_magnitude
        # jittered_points = supercell_points + jitter

        box_matrix = np.column_stack(
            [3 * np.array(x_vec), 3 * np.array(y_vec), 3 * np.array(z_vec)]
        )
        box = freud.box.Box.from_matrix(box_matrix)
        voro = freud.locality.Voronoi()
        try:
            cells = voro.compute((box, supercell_points)).polytopes
        except Exception as e:
            print(f"Error computing Voronoi vertices: {e}")
            return np.array([1e-5, 1e-5, 1e-5])

        all_vertices = box.wrap(np.concatenate(cells))
        unique_vertices = np.unique(np.round(all_vertices, decimals=10), axis=0)
        # freud_matrix = box.to_matrix()
        # voronoi_vertices = (box_matrix @ np.linalg.solve(freud_matrix, unique_vertices.T)).T # redundant, messes up coordinates by doing an extra linear transform.
        voronoi_vertices = unique_vertices

        fractional_coords = np.linalg.solve(box_matrix / 3, voronoi_vertices.T).T
        mask = np.all(
            (fractional_coords >= -0.5 - self.epsilon)
            & (fractional_coords < 0.5 + self.epsilon),
            axis=1,
        )
        center_cell_vertices = voronoi_vertices[mask]

        return center_cell_vertices

    def _get_next_point(
        self,
        points: NDArray,
        atomic_numbers: NDArray,
        x_vec: NDArray,
        y_vec: NDArray,
        z_vec: NDArray,
    ) -> NDArray:
        """
        Calculates the next phantom atom to add based on Voronoi analysis.
        """
        center_vector = (np.array(x_vec) + np.array(y_vec) + np.array(z_vec)) / 2
        centered_points = points - center_vector

        supercell_points = self._create_supercell_from_points(
            centered_points, atomic_numbers, x_vec, y_vec, z_vec
        )

        voronoi_vertices = self._get_voronoi_vertices(
            supercell_points, x_vec, y_vec, z_vec
        )

        next_point = (
            self._get_farthest_point(voronoi_vertices, centered_points, atomic_numbers)
            + center_vector
        )
        return next_point
