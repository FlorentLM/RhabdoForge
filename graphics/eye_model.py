from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from dataclasses import dataclass, field
from scipy.spatial import KDTree
from graphics.utils import VEC_DTYPE, WORLD_UP


@dataclass
class Ommatidium:
    """ A single ommatidium and its properties """

    id: int

    azimuth_rad: float      # Horizontal angle (longitude)
    elevation_rad: float    # Vertical angle (latitude)

    direction: np.ndarray = field(repr=False)  # 3D pointing vector
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=VEC_DTYPE), repr=False)  # 3D origin point

    # Acceptance angle(s)
    acceptance_angle_h_rad: float = 0.0
    acceptance_angle_v_rad: float = 0.0

    @property
    def azimuth(self):
        return self.azimuth_rad

    @property
    def elevation(self):
        return self.elevation_rad

    @property
    def acceptance_angles_rad(self) -> Tuple[float, float]:
        """ Returns (horizontal, vertical) acceptance angles in radians """
        return self.acceptance_angle_h_rad, self.acceptance_angle_v_rad

    @property
    def acceptance_angles_deg(self) -> Tuple[float, float]:
        """ Returns (horizontal, vertical) acceptance angles in degrees """
        return np.rad2deg(self.acceptance_angle_h_rad), np.rad2deg(self.acceptance_angle_v_rad)

    # And some more aliases
    longitude = azimuth
    latitude = elevation
    lon = longitude
    lat = latitude

    def __post_init__(self):

        # Ensure origin and direction is are np arrays
        self.direction = np.asarray(self.direction, dtype=VEC_DTYPE)
        self.origin = np.asarray(self.origin, dtype=VEC_DTYPE)

        # normalise direction
        norm = np.linalg.norm(self.direction)
        if not np.isclose(norm, 1.0):
            self.direction /= norm


class EyeModel:
    """ Container for a single eye's ommatidia with spatial query capabilities """

    def __init__(self, ommatidia: List[Ommatidium]):

        self.ommatidia: List[Ommatidium] = ommatidia
        self.num_ommatidia: int = len(self.ommatidia)

        # Build the KD-Tree for fast spatial queries
        self.directions = np.array([om.direction for om in self.ommatidia], dtype=VEC_DTYPE)
        self.kdtree = KDTree(self.directions)

    @classmethod
    def generate_uniform_eye(cls, num_ommatidia: int, acceptance_angle_deg: Optional[float] = None, eye_radius: float = 0.0):
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
        om_origins = om_dirs * eye_radius

        # Handle acceptance angle input
        angle_h_rad, angle_v_rad = 0.0, 0.0
        need_estimation = True
        if acceptance_angle_deg is not None:
            need_estimation = False
            if isinstance(acceptance_angle_deg, (list, tuple)):
                angle_h_rad = np.deg2rad(acceptance_angle_deg[0], dtype=VEC_DTYPE)
                angle_v_rad = np.deg2rad(acceptance_angle_deg[1], dtype=VEC_DTYPE)
            else:  # single float
                angle_h_rad = angle_v_rad = np.deg2rad(acceptance_angle_deg, dtype=VEC_DTYPE)

        ommatidia_list = [
            Ommatidium(id=i,
                       direction=dir,
                       origin=origin,
                       azimuth_rad=lon,
                       elevation_rad=lat,
                       acceptance_angle_h_rad=angle_h_rad,
                       acceptance_angle_v_rad=angle_v_rad)
            for i, (dir, origin, lon, lat) in enumerate(zip(om_dirs, om_origins, om_lons, om_lats))
        ]
        model = cls(ommatidia_list)

        if need_estimation:
            # for a uniform eye estimate one angle and apply it to both H and V
            print("Estimating acceptance angles for uniform eye...")
            model.estimate_acceptance_angles(assume_circul=True)

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
            data = np.load(file_path).astype(VEC_DTYPE)
            num_ommatidia, num_cols = data.shape

            om_dirs = data[:, :3]
            om_lons = np.arctan2(om_dirs[:, 0], -om_dirs[:, 2])
            om_lats = np.arcsin(om_dirs[:, 1])

            if num_cols == 3:  # Directions only
                ommatidia_list = [Ommatidium(id=i, direction=d, azimuth_rad=lon, elevation_rad=lat)
                                  for i, (d, lon, lat) in enumerate(zip(om_dirs, om_lons, om_lats))]
                model = cls(ommatidia_list)
                print("Loaded 3-column data. Estimating non-uniform acceptance angles.")
                model.estimate_acceptance_angles(assume_circular=False)
                return model

            elif num_cols == 4:  # Directions + 1 acceptance angle (circular)
                angles = data[:, 3]
                ommatidia_list = [Ommatidium(id=i, direction=d, azimuth_rad=lon, elevation_rad=lat,
                                             acceptance_angle_h_rad=a, acceptance_angle_v_rad=a)
                                  for i, (d, lon, lat, a) in enumerate(zip(om_dirs, om_lons, om_lats, angles))]
                return cls(ommatidia_list)

            elif num_cols == 5:  # Directions + 2 acceptance angles (elliptical)
                angle_h = data[:, 3]
                angle_v = data[:, 4]
                ommatidia_list = [Ommatidium(id=i, direction=d, azimuth_rad=lon, elevation_rad=lat,
                                             acceptance_angle_h_rad=ah, acceptance_angle_v_rad=av)
                                  for i, (d, lon, lat, ah, av) in
                                  enumerate(zip(om_dirs, om_lons, om_lats, angle_h, angle_v))]
                return cls(ommatidia_list)

            else:
                raise ValueError(f"Expected 3, 4, or 5 columns, but got {num_cols}")
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

    def estimate_acceptance_angles(self, k: int = 8, assume_circular: bool = False):
        """
        Calculates acceptance angles based on nearest neighbours
        """
        if self.num_ommatidia <= k:
            raise ValueError("Cannot estimate angles when k is >= number of ommatidia.")

        distances, indices = self.kdtree.query(self.directions, k=k + 1)
        neighbour_dirs = self.directions[indices[:, 1:]]

        # For each ommatidium the local 'up' is derived from the global 'up'
        local_y_axes = WORLD_UP - self.directions * np.sum(self.directions * WORLD_UP, axis=1, keepdims=True)
        local_y_axes /= np.linalg.norm(local_y_axes, axis=1, keepdims=True)
        local_x_axes = np.cross(local_y_axes, self.directions)

        neighbour_vectors = neighbour_dirs - self.directions[:, np.newaxis, :] # shape (num_om, k, 3)

        # Project neighbour vectors onto the local x and y axes to get h and v components of the separation
        sep_h = np.sum(neighbour_vectors * local_x_axes[:, np.newaxis, :], axis=2)
        sep_v = np.sum(neighbour_vectors * local_y_axes[:, np.newaxis, :], axis=2)

        # angular distance is ~ Euclidean distance in the tangent plane
        angular_dist_h = np.abs(sep_h)
        angular_dist_v = np.abs(sep_v)

        # For each ommatidium find the 2 "most horizontal" and 2 "most vertical" neighbours
        # (heuristic: horizontal neighbours have high 'angular_dist_h' and low 'angular_dist_v')
        # TODO: not sure this is the most robust way to do it
        h_likeness = angular_dist_h / (angular_dist_v + 1e-9)
        v_likeness = angular_dist_v / (angular_dist_h + 1e-9)
        h_neighbor_indices = np.argsort(h_likeness, axis=1)[:, -2:]
        v_neighbor_indices = np.argsort(v_likeness, axis=1)[:, -2:]

        # Get actual angular distances for these neighbours
        h_distances = np.take_along_axis(distances[:, 1:], h_neighbor_indices, axis=1)
        v_distances = np.take_along_axis(distances[:, 1:], v_neighbor_indices, axis=1)

        # Average distances to get characteristic interommatidial distance
        avg_h_dist = np.mean(h_distances, axis=1)
        avg_v_dist = np.mean(v_distances, axis=1)

        # Convert Euclidean distance on unit sphere to angular distance
        term_h = 1 - (avg_h_dist ** 2) / 2.0
        term_v = 1 - (avg_v_dist ** 2) / 2.0
        final_angles_h = np.arccos(np.clip(term_h, -1.0, 1.0))
        final_angles_v = np.arccos(np.clip(term_v, -1.0, 1.0))

        # And assign to ommatidia
        for i, om in enumerate(self.ommatidia):
            if assume_circular:
                # For uniform eyes just average the two
                avg_angle = (final_angles_h[i] + final_angles_v[i]) / 2.0
                om.acceptance_angle_h_rad = avg_angle
                om.acceptance_angle_v_rad = avg_angle
            else:
                om.acceptance_angle_h_rad = final_angles_h[i]
                om.acceptance_angle_v_rad = final_angles_v[i]

    def max_gap(self):
        """
        Finds the maximum angular distance between any ommatidium and its
        single nearest neighbor, which represents the largest "gap" in the eye
        """

        if self.num_ommatidia == 1:
            return 0.0

        # Find the single nearest neighbor for each ommatidium
        distances, _ = self.kdtree.query(self.directions, k=2)
        nearest_neighbor_dists = distances[:, 1]
        max_euclidean_dist = np.max(nearest_neighbor_dists)

        # Convert the maximum Euclidean distance to an angular distance in radians
        # This is the largest inter-ommatidial angle in the entire eye
        # angle = arccos(1 - dist^2 / 2)
        term = 1.0 - (max_euclidean_dist ** 2) / 2.0
        max_angular_dist = np.arccos(np.clip(term, -1.0, 1.0))

        return max_angular_dist

    def pack(self) -> np.ndarray:
        # Packs ommatidia data into a numpy array compatible with the new GLSL struct.
        #
        # GLSL struct layout (std430):
        # struct Ommatidium {
        #     vec4 origin;            // offset 0,  size 16
        #     vec4 direction;         // offset 16, size 16
        #     vec2 acceptance_angles; // offset 32, size 8
        #     float pad0, pad1;       // offset 40, size 8
        # };                          // total size = 48 bytes

        num_om = self.num_ommatidia

        # empty array with 12 columns 4 (origin) + 4 (dir) + 2 (angles) + 2 (pad)
        packed_data = np.zeros((num_om, 12), dtype=VEC_DTYPE)

        # Fill with data
        packed_data[:, 0:3] = np.array([om.origin for om in self.ommatidia])
        packed_data[:, 4:7] = self.directions
        packed_data[:, 8] = np.array([om.acceptance_angle_h_rad for om in self.ommatidia])
        packed_data[:, 9] = np.array([om.acceptance_angle_v_rad for om in self.ommatidia])

        return packed_data


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

    vals = np.linspace(0, 1, n + 1)

    # Total number of points in a triangle subdivided n times
    num_points = int((n + 1) * (n + 2) / 2)
    bcmat = np.zeros((num_points, 3))

    # Builds the points 'row by row' inside the ref triangle
    shifts = np.arange(n + 1, 0, -1)
    starts = np.zeros(n + 1, dtype=int)
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

    return all_new_verts[iunique].astype(VEC_DTYPE)