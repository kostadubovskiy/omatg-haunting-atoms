import freud
import numpy as np


def create_supercell_from_points(points, x_vec, y_vec, z_vec):
    """
    Create a 3x3x3 supercell from points in a unit cell defined by three vectors.

    Args:
        points (np.ndarray): Points in the unit cell, shape (n, 3)
        x_vec (array-like): First lattice vector [x1, x2, x3]
        y_vec (array-like): Second lattice vector [y1, y2, y3]
        z_vec (array-like): Third lattice vector [z1, z2, z3]

    Returns:
        np.ndarray: The points in the supercell, shape (27*n, 3)
    """
    # Convert to numpy arrays
    v1 = np.array(x_vec)
    v2 = np.array(y_vec)
    v3 = np.array(z_vec)

    supercell_points = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                # Translation is a linear combination of lattice vectors
                translation = i * v1 + j * v2 + k * v3
                translated_points = points + translation
                supercell_points.append(translated_points)

    return np.concatenate(supercell_points)


def get_farthest_point(vertices, points, dist_eval = 'min'):
  max_distance = 0
  farthest_vertice = None

  for vertice in vertices:
    distances = np.linalg.norm(vertice - points, axis=1)
    if dist_eval == 'min':
       eval_distance = np.min(distances)
    elif dist_eval == 'avg':
       eval_distance = np.mean(distances)
    elif dist_eval == 'root_mean_avg':
       eval_distance = np.sqrt(np.mean(distances ** 2))
    elif dist_eval == 'median':
       eval_distance = np.median(distances)
    elif dist_eval == 'reciprocal':
       eval_distance = 1/np.mean(1/distances)

    if eval_distance > max_distance:
      max_distance = eval_distance
      farthest_vertice = vertice

  return farthest_vertice

def get_voronoi_vertices(supercell_points, x_vec, y_vec, z_vec):
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

def get_next_point(points, x_vec, y_vec, z_vec, dist_eval = 'min'):
   '''
   Get the next point to add to the set of points.

   Args:
      points (np.ndarray): point in the parallelpiped defined by the three vectors in cartesian coordinates
            - shape (n, 3), assuming that (0,0,0) is the corner of the parallelpiped
      x_vec (array-like): First lattice vector [x1, x2, x3]
      y_vec (array-like): Second lattice vector [y1, y2, y3]
      z_vec (array-like): Third lattice vector [z1, z2, z3]
      dist_eval (str): Evaluation metric for distance, default is 'min'

   Returns:
      np.ndarray: The next point to add to the set of points, shape (1, 3)
   '''
   

   # translate the points so that they are centered around (0,0,0) assuming they were not already centered
   center_vector = (x_vec + y_vec + z_vec) / 2
   points = points - center_vector
   
   # Create a 3x3x3 supercell from the points
   supercell_points = create_supercell_from_points(points, x_vec, y_vec, z_vec)

   # Get the Voronoi vertices
   voronoi_vertices = get_voronoi_vertices(supercell_points, x_vec, y_vec, z_vec)
   # Get the next point
   next_point = get_farthest_point(voronoi_vertices, points, dist_eval) + center_vector
   return next_point
