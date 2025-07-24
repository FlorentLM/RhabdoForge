from pathlib import Path
from typing import Optional, List
import numpy as np
from dataclasses import dataclass, field
from scipy.spatial import KDTree
from graphics.utils import DTYPE


@dataclass
class Ommatidium:
    """ A single ommatidium and its properties """

    id: int
    direction: np.ndarray = field(repr=False)  # 3D pointing vector
    azimuth_rad: float      # Horizontal angle (longitude)
    elevation_rad: float    # Vertical angle (latitude)
    acceptance_angle_rad: float = 0.0

    @property
    def azimuth(self):
        return self.azimuth_rad

    @property
    def elevation(self):
        return self.elevation_rad

    # And some more aliases
    longitude = azimuth
    latitude = elevation
    lon = longitude
    lat = latitude

    def __post_init__(self):

        # Ensure direction is a normalized np array
        self.direction = np.asarray(self.direction, dtype=DTYPE)

        norm = np.linalg.norm(self.direction)
        if not np.isclose(norm, 1.0):
            self.direction /= norm


class EyeModel:
    """ Container for a single eye's ommatidia with spatial query capabilities """

    def __init__(self, ommatidia: List[Ommatidium]):

        self.ommatidia: List[Ommatidium] = ommatidia
        self.num_ommatidia: int = len(self.ommatidia)

        # Build the KD-Tree for fast spatial queries
        self.directions = np.array([om.direction for om in self.ommatidia], dtype=DTYPE)
        self.kdtree = KDTree(self.directions)

    @classmethod
    def generate_uniform_eye(cls, num_ommatidia: int, acceptance_angle_deg: Optional[float] = None):
        """ Factory to create an EyeModel with a uniform spherical distribution based on a subdivided icosahedron """

        lod = estimate_lod(num_ommatidia)

        # Recalculate the true number of ommatidia for this LOD
        true_num_ommatidia = 10 * lod ** 2 + 2
        if num_ommatidia != true_num_ommatidia:
            print(
                f"Warning: Requested {num_ommatidia} ommatidia, but the closest valid count for LOD {lod} is {true_num_ommatidia}. Using {true_num_ommatidia}.")

        om_dirs = subdivide_icosahedron(lod)
        om_lons = np.arctan2(om_dirs[:, 0], -om_dirs[:, 2])
        om_lats = np.arcsin(om_dirs[:, 1])

        acceptance_angle_rad = np.deg2rad(acceptance_angle_deg or 0.0, dtype=DTYPE)
        # TODO: this will break if acceptance_angle_deg is a numpy array

        ommatidia_list = [
            Ommatidium(id=i, direction=dir, azimuth_rad=lon, elevation_rad=lat, acceptance_angle_rad=acceptance_angle_rad)
            for i, (dir, lon, lat) in enumerate(zip(om_dirs, om_lons, om_lats))
        ]
        model = cls(ommatidia_list)

        # If no acceptance angle was passed, it needs to be computed
        if not acceptance_angle_deg:
            model.estimate_acceptance_angles()

        return model

    @classmethod
    def from_file(cls, file_path: str | Path):
        """
        Factory to create an EyeModel from a data file
        For now, assumes a .npy file with shape (N, 3) for the direction vectors
        # TODO: More parsers
        """

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Cannot find eye data file: {file_path}")

        if file_path.suffix == '.npy':
            data = np.load(file_path).astype(DTYPE)
            num_ommatidia = data.shape[0]
            num_cols = data.shape[1]

            if num_cols == 3:  # Directions only
                om_dirs = data
                om_lons = np.arctan2(om_dirs[:, 0], -om_dirs[:, 2])
                om_lats = np.arcsin(om_dirs[:, 1])
                ommatidia_list = [Ommatidium(id=i, direction=d, azimuth_rad=lon, elevation_rad=lat) for i, (d, lon, lat)
                                  in enumerate(zip(om_dirs, om_lons, om_lats))]

                # Create the model and then calculate the missing acceptance angles
                model = cls(ommatidia_list)

                print("Loaded 3-column data. Estimating acceptance angles from local density.")
                model.estimate_acceptance_angles()
                return model

            elif num_cols == 4:  # Directions + acceptance angles
                om_dirs = data[:, :3]
                acceptance_angles = data[:, 3]
                om_lons = np.arctan2(om_dirs[:, 0], -om_dirs[:, 2])
                om_lats = np.arcsin(om_dirs[:, 1])
                ommatidia_list = [
                    Ommatidium(id=i, direction=d, azimuth_rad=lon, elevation_rad=lat, acceptance_angle_rad=angle) for
                    i, (d, lon, lat, angle) in enumerate(zip(om_dirs, om_lons, om_lats, acceptance_angles))]
                return cls(ommatidia_list)

            else:
                raise ValueError(f"Expected 3 or 4 columns, but got {num_cols}")
        else:
            raise NotImplementedError(f"File format '{file_path.suffix}' not supported yet.")

    def find_neighbors(self, ommatidium_id: int, k: int = 6) -> list[tuple[float, Ommatidium]]:
        """
        Finds the k nearest neighbours to a given ommatidium
        Returns a list of tuples: (distance, Ommatidium)
        """

        if not (0 <= ommatidium_id < self.num_ommatidia):
            raise IndexError("ommatidium_id out of range.")

        distances, indices = self.kdtree.query(self.ommatidia[ommatidium_id].direction, k=k + 1)

        neighbors = []
        for dist, index in zip(distances[1:], indices[1:]):
            neighbors.append((dist, self.ommatidia[index]))

        return neighbors

    def estimate_acceptance_angles(self, k: int = 6):
        """
        Calculates acceptance angles for each ommatidium based on the average angular
        distance to its k-nearest neighbours
        """

        if self.num_ommatidia <= k:
            raise ValueError("Cannot estimate angles when k is >= number of ommatidia.")

        distances, _ = self.kdtree.query(self.directions, k=k + 1)

        # first column is distance to self
        neighbor_distances = distances[:, 1:]

        avg_euclidean_dist = np.mean(neighbor_distances, axis=1)

        # Convert to angular distance (radians)
        # d^2 = 2 - 2*cos(phi), which gives phi = arccos(1 - d^2/2)
        term = 1 - (avg_euclidean_dist ** 2) / 2.0
        avg_angular_dist_rad = np.arccos(np.clip(term, -1.0, 1.0))

        # Assign the calculated angle to each ommatidium
        for om, angle in zip(self.ommatidia, avg_angular_dist_rad):
            om.acceptance_angle_rad = angle

    def pack(self) -> np.ndarray:
        # TODO: Maybe the packed version could be stored and accessed with cool accessor properties?

        directions = np.array([om.direction for om in self.ommatidia], dtype=DTYPE)
        acceptance_angles = np.array([om.acceptance_angle_rad for om in self.ommatidia], dtype=DTYPE).reshape(-1, 1)

        return np.hstack([directions, acceptance_angles])


def estimate_lod(num_ommatidia: int) -> int:
    """ Calculates the Level of Division (lod) needed to produce a number of ommatidia """

    if num_ommatidia < 12:
        return 1

    # y = 10 * n^2 + 2 for n
    n = np.sqrt((num_ommatidia - 2) / 10.0)
    return int(np.round(n))


def icosahedron_faces() -> np.ndarray:
    """
    Defines the base (z-axis aligned) icosahedron and returns the vertices for the 20 triangular faces
     Shape: (20, 3, 3)
    """
    # TODO: Move this to the primitives file maybe?

    # Golden ratio
    G = (1 + np.sqrt(5)) / 2.0

    # Three mutually perpendicular golden ratio rectangles make the icosahedron's vertices :)
    p = np.array([
        [   G,  -G,   -G,    G, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0,  0.0,  0.0 ],
        [ 0.0, 0.0,  0.0,  0.0,   G,  -G,   -G,    G, 1.0, 1.0, -1.0, -1.0 ],
        [ 1.0, 1.0, -1.0, -1.0, 0.0, 0.0,  0.0,  0.0,   G,  -G,   -G,    G ]
    ]).T
    p /= np.sqrt(np.sum(p ** 2, axis=1))[0]

    # Rotate top point to the z-axis
    ang = np.arctan(p[0, 0] / p[0, 2])
    ca, sa = np.cos(ang), np.sin(ang)
    rotation = np.array([[ca, 0.0, -sa], [0.0, 1.0, 0.0], [sa, 0.0, ca]])
    p = np.inner(rotation, p).T

    # Reorder in a downward spiral
    reorder_index = [0, 3, 4, 8, -1, 5, -2, -3, 7, 1, 6, 2]
    p = p[reorder_index]

    # 20 triangular faces
    tri_indices = np.array([
        [ 1, 2, 3, 4, 5, 6, 2, 7, 2, 8, 3,  9, 10, 10, 6,  6,  7,  8,  9, 10 ],
        [ 2, 3, 4, 5, 1, 7, 1, 8, 8, 9, 9, 10,  5,  6, 1, 11, 11, 11, 11, 11 ],
        [ 0, 0, 0, 0, 0, 1, 7, 2, 3, 3, 4,  4,  4,  5, 5,  7,  8,  9,  10, 6 ]
    ]).T

    return p[tri_indices]


def barycentric_coords(n: int) -> np.ndarray:
    """
    Generates a matrix of barycentric coordinates (u, v, w) inside a reference triangle where u + v + w = 1
    Any point inside the triangle can be represented by a unique set of these coordinates (weights)

    Args:
        n: number of subdivisions along each edge of the triangle

    Returns:
        A numpy array of shape (n*(n+1)/2, 3) containing the barycentric coordinates
        for all the points in the subdivided triangle
    """

    vals = np.linspace(0, 1, n)

    # Total number of points in a triangle subdivided n times
    num_points = int(n * (n + 1) / 2)
    bcmat = np.zeros((num_points, 3))

    # Builds the points 'row by row' inside the ref triangle
    shifts = np.arange(n, 0, -1)
    starts = np.zeros(n, dtype=int)
    starts[1:] = np.cumsum(shifts[:-1])
    stops = starts + shifts

    # along each row: u decreases, v increases, w stays conatant
    for i, (start, stop, shift) in enumerate(zip(starts, stops, shifts)):
        bcmat[start:stop, 0] = vals[shift - 1::-1]
        bcmat[start:stop, 1] = vals[:shift]
        bcmat[start:stop, 2] = vals[i]

    return bcmat


def subdivide_icosahedron(n: int) -> np.ndarray:
    """ Subdivides icosahedron using barycentric coordinates """

    verts = icosahedron_faces()
    bary = barycentric_coords(n)

    # Barycentric interpolation to each of the 20 triangles
    # 'ij,kjl->kil': i=bary_idx, j=bary_coord, k=tri_idx, l=vertex_coord
    all_new_verts = np.einsum('ij,kjl->kil', bary, verts)

    # Reshape, normalize to unit sphere, and find unique vertices
    all_new_verts = all_new_verts.reshape(-1, 3)
    all_new_verts /= np.linalg.norm(all_new_verts, axis=1)[:, np.newaxis]
    _, iunique = np.unique(np.round(all_new_verts, 6), axis=0, return_index=True)

    return all_new_verts[iunique].astype(DTYPE)