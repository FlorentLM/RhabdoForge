from pathlib import Path
from typing import Optional, List, Tuple, Union
import numpy as np
from dataclasses import dataclass, field
from scipy.spatial import KDTree
from graphics.utils import VEC_DTYPE, WORLD_UP, WORLD_RIGHT


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
    """
    Container for a single eye's ommatidia with spatial query capabilities
    """

    def __init__(self,
                 directions: Optional[np.ndarray] = None,
                 num_ommatidia: Optional[int] = None,
                 acceptance_angles_rad: Optional[Union[np.ndarray, Tuple, float]] = None,
                 eye_parameter: Optional[Union[float, Tuple]] = None,
                 lens_diameter: Optional[Union[float, Tuple]] = None,
                 rhabdom_diameter: Optional[Union[float, Tuple]] = None,
                 focal_length: Optional[Union[float, Tuple]] = None,
                 wavelength: float = 500e-9,    # TODO: this is a nice temporary value, but the shaders will compute the 3 channels independently
                 eye_radius: float = 0.0
                 ):
        """
        The primary constructor for creating an EyeModel

        Args:
            directions: An (N, 3) numpy array of ommatidial direction vectors
            acceptance_angles_rad: (Optional) The acceptance angles (Δρ). Can be:
                - An (N, 2) array for individual H/V angles
                - A tuple (h, v) for global H/V angles
                - A float for a global circular angle
                - None: If not provided, will be estimated using the eye_parameter
            eye_parameter: (Optional) The eye parameter 'p' value (Δρ / Δφ). Used to estimate acceptance
                angles if they are not provided directly (defaults to 1.0)
            eye_radius: (Optional) Physical radius of the eye for setting ommatidial origins
        """

        # Directions
        if directions is not None:
            # Priority 1: Direct directions are provided
            print("Using provided direction vectors.")
            self.directions = directions
        elif num_ommatidia is not None:
            # Priority 2: Generate directions from num_ommatidia
            print(f"Generating uniform direction vectors for approx. {num_ommatidia} ommatidia.")
            lod = estimate_lod(num_ommatidia)
            self.directions = subdivide_icosahedron(lod)
            true_num_ommatidia = len(self.directions)
            if abs(num_ommatidia - true_num_ommatidia) > 1:
                print(f"Note: Using {true_num_ommatidia} ommatidia to match subdivision level {lod}.")
        else:
            raise ValueError("EyeModel requires either 'directions' or 'num_ommatidia' to be provided.")

        # Create the base ommatidia with their geometric properties
        self.num_ommatidia = len(self.directions)

        lons = np.arctan2(self.directions[:, 0], -self.directions[:, 2])
        lats = np.arcsin(self.directions[:, 1])
        origs = self.directions * eye_radius if eye_radius > 0 else np.zeros_like(self.directions)

        self.ommatidia: List[Ommatidium] = [
            Ommatidium(id=i, direction=d, origin=o, azimuth_rad=lon, elevation_rad=lat)
            for i, (d, o, lon, lat) in enumerate(zip(self.directions, origs, lons, lats))
        ]
        self.kdtree = KDTree(self.directions)

        # Interommatidial angles
        self.interommatidial_angle_h_rad, self.interommatidial_angle_v_rad = self.estimate_interommatidial_angles()

        # Determine and set acceptance angles
        if acceptance_angles_rad is not None:
            # Priority 1: Direct acceptance angles are provided
            print("Using directly provided acceptance angles (Δρ).")
            self._set_acceptance_angles(acceptance_angles_rad)

        elif lens_diameter is not None and rhabdom_diameter is not None and focal_length is not None:
            # Priority 2: Estimate acceptance angles from optical parameters
            print("Calculating acceptance angles (Δρ) from physical optical parameters.")

            Dh, Dv = self._unpack(lens_diameter, "lens_diameter")
            dh, dv = self._unpack(rhabdom_diameter, "rhabdom_diameter")
            fh, fv = self._unpack(focal_length, "focal_length")

            # Calculate acceptance angles
            # Horizontal
            delta_phi_optics_h = wavelength / Dh
            delta_phi_receptor_h = dh / fh
            angles_h_rad = np.sqrt(delta_phi_optics_h ** 2 + delta_phi_receptor_h ** 2)

            # Vertical
            delta_phi_optics_v = wavelength / Dv
            delta_phi_receptor_v = dv / fv
            angles_v_rad = np.sqrt(delta_phi_optics_v ** 2 + delta_phi_receptor_v ** 2)

            estimated_angles = np.vstack([angles_h_rad, angles_v_rad]).T
            self._set_acceptance_angles(estimated_angles)

        else:
            # Priority 3: Estimate acceptance angles from geometry using eye parameter 'p'
            p = eye_parameter if eye_parameter is not None else 1.0
            print(f"Estimating acceptance angles (Δρ) from interommatidial angles (Δφ) with eye parameter p={p}.")

            if isinstance(p, (list, tuple)):
                p_h, p_v = p
            else:
                p_h = p_v = p

            # acceptance angles: Δρ = p * Δφ
            delta_rho_h = p_h * self.interommatidial_angle_h_rad
            delta_rho_v = p_v * self.interommatidial_angle_v_rad

            estimated_angles = np.vstack([delta_rho_h, delta_rho_v]).T
            self._set_acceptance_angles(estimated_angles)

        # Now that Δρ and Δφ are known, calculate and store the resulting eye parameter p
        all_rho_h = np.array([om.acceptance_angle_h_rad for om in self.ommatidia])
        all_rho_v = np.array([om.acceptance_angle_v_rad for om in self.ommatidia])

        with np.errstate(divide='ignore', invalid='ignore'):
            self.p_h = all_rho_h / self.interommatidial_angle_h_rad
            self.p_v = all_rho_v / self.interommatidial_angle_v_rad

        # Replace any non-finite values
        self.eye_parameter_h = np.nan_to_num(self.p_h, nan=0.0, posinf=0.0, neginf=0.0)
        self.eye_parameter_v = np.nan_to_num(self.p_v, nan=0.0, posinf=0.0, neginf=0.0)

    def _prepare_param(self, param, name="param"):
        """ Ensures parameter is a numpy array of the correct shape """

        arr = np.asarray(param, dtype=VEC_DTYPE)

        # if scalar, broadcast to number of ommatidia
        if arr.ndim == 0:
            return np.full(self.num_ommatidia, arr.item())

        # if 1D array, check if it has the right length
        elif arr.ndim == 1:
            if len(arr) != self.num_ommatidia:
                raise ValueError(
                    f"Per-ommatidium parameter '{name}' has length {len(arr)}, but expected {self.num_ommatidia}.")
            return arr
        else:
            raise ValueError(f"Parameter '{name}' has invalid shape. Must be scalar or 1D array.")

    def _unpack(self, param, name="param"):
        if isinstance(param, (list, tuple)):
            return self._prepare_param(param[0], f"{name}_h"), self._prepare_param(param[1], f"{name}_v")
        else:
            p_scalar = self._prepare_param(param, name)
            return p_scalar, p_scalar

    def _set_acceptance_angles(self, angles_rad: Union[np.ndarray, Tuple[float, float], float]):
        """ Helper to assign acceptance angles to all ommatidia """

        if isinstance(angles_rad, np.ndarray) and angles_rad.ndim == 2:
            # Per-ommatidium H/V angles
            for i, om in enumerate(self.ommatidia):
                om.acceptance_angle_h_rad = float(angles_rad[i, 0])
                om.acceptance_angle_v_rad = float(angles_rad[i, 1])
        else:
            # single global angle (or tuple of angles) for all ommatidia
            if isinstance(angles_rad, (list, tuple)):
                angle_h, angle_v = angles_rad
            else:  # is a float
                angle_h = angle_v = angles_rad
            for om in self.ommatidia:
                om.acceptance_angle_h_rad = angle_h
                om.acceptance_angle_v_rad = angle_v

    @property
    def interommatidial_angles_rad(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns the estimated (horizontal, vertical) interommatidial angles (Δφ) in radians
        (or None if they have not been estimated yet)
        """
        if self.interommatidial_angle_h_rad is not None:
            return self.interommatidial_angle_h_rad, self.interommatidial_angle_v_rad
        return None

    @classmethod
    def from_file(cls, file_path: str | Path, **kwargs):
        """ Creates an eye model from a .npy file
        - (N, 3): Directions only (angles will be estimated)
        - (N, 4): Directions + circular acceptance angles
        - (N, 5): Directions + ellipsoid acceptance angles (h != v)
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Cannot find eye data file: {path}")

        data = np.load(path).astype(VEC_DTYPE)
        directions = data[:, :3]
        num_cols = data.shape[1]

        acceptance_angles_rad = None
        if num_cols == 4:  # Directions + 1 circular angle (in radians)
            acceptance_angles_rad = np.vstack([data[:, 3], data[:, 3]]).T
        elif num_cols == 5:  # Directions + H and V angles (in radians)
            acceptance_angles_rad = data[:, 3:5]

        return cls(directions=directions, acceptance_angles_rad=acceptance_angles_rad, **kwargs)

    def estimate_interommatidial_angles(self, k: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimates the horizontal and vertical interommatidial angles (Δφ) for each ommatidium
        from the local geometry of its neighbours

        Returns:
            A tuple of two numpy arrays: (delta_phi_h, delta_phi_v) in radians
        """
        if self.num_ommatidia <= k:
            raise ValueError("Cannot estimate angles when k is >= number of ommatidia.")

        # Check for ommatidia at the poles
        dot_products = np.abs(np.sum(self.directions * WORLD_UP, axis=1))
        is_polar = dot_products > 0.9999

        # Choose a reference 'up' vector (WORLD_RIGHT for polar ommatidia, WORLD_UP for others)
        ref_up_vectors = np.where(is_polar[:, np.newaxis], WORLD_RIGHT, WORLD_UP)

        # Now the Gram-Schmidt process is safe for all ommatidia
        local_y_axes = ref_up_vectors - self.directions * np.sum(self.directions * ref_up_vectors, axis=1,
                                                                 keepdims=True)
        local_y_axes /= np.linalg.norm(local_y_axes, axis=1, keepdims=True)
        local_x_axes = np.cross(local_y_axes, self.directions)

        distances, indices = self.kdtree.query(self.directions, k=k + 1)
        # Check if any ommatidium has fewer than k+1 neighbors (can happen in weird geometries)
        if isinstance(indices, int) or indices.shape[1] < 2:
            return np.zeros(self.num_ommatidia), np.zeros(self.num_ommatidia)

        neighbour_dirs = self.directions[indices[:, 1:]]
        neighbour_vectors = neighbour_dirs - self.directions[:, np.newaxis, :]

        # Project neighbour vectors onto the local x and y axes to get h and v components of the separation
        # small-angle approximation: angular distance is ~ Euclidean distance in the tangent plane
        sep_h = np.sum(neighbour_vectors * local_x_axes[:, np.newaxis, :], axis=2)
        sep_v = np.sum(neighbour_vectors * local_y_axes[:, np.newaxis, :], axis=2)
        abs_sep_h, abs_sep_v = np.abs(sep_h), np.abs(sep_v)

        # Horizontal neighbours = the 2 neighbours with smallest vertical separation
        h_neighbour_indices = np.argsort(abs_sep_v, axis=1)[:, :2]

        # Vertical neighbours = the 2 neighbours with smallest horizontal separation
        v_neighbour_indices = np.argsort(abs_sep_h, axis=1)[:, :2]

        # Get actual angular distances for these neighbours
        horizontal_angles = np.take_along_axis(abs_sep_h, h_neighbour_indices, axis=1)
        vertical_angles = np.take_along_axis(abs_sep_v, v_neighbour_indices, axis=1)

        # The characteristic interommatidial angle is the mean of these two neighbours
        delta_phi_h = np.mean(horizontal_angles, axis=1)
        delta_phi_v = np.mean(vertical_angles, axis=1)

        return delta_phi_h, delta_phi_v

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

    def __repr__(self):

        # Basic info
        summary = [f"<EyeModel with {self.num_ommatidia} ommatidia>"]

        # Interommatidial Angles (Δφ)
        if self.interommatidial_angles_rad is not None:
            d_phi_h_deg = np.rad2deg(self.interommatidial_angle_h_rad)
            d_phi_v_deg = np.rad2deg(self.interommatidial_angle_v_rad)
            summary.append(f"  Interommatidial Angles (Δφ):")
            summary.append(
                f"    Horizontal: {np.mean(d_phi_h_deg):.3f}° (mean), {np.min(d_phi_h_deg):.3f}° (min), {np.max(d_phi_h_deg):.3f}° (max)")
            summary.append(
                f"    Vertical:   {np.mean(d_phi_v_deg):.3f}° (mean), {np.min(d_phi_v_deg):.3f}° (min), {np.max(d_phi_v_deg):.3f}° (max)")

        # Acceptance Angles (Δρ)
        all_rho_h = np.array([om.acceptance_angle_h_rad for om in self.ommatidia])
        all_rho_v = np.array([om.acceptance_angle_v_rad for om in self.ommatidia])
        rho_h_deg = np.rad2deg(all_rho_h)
        rho_v_deg = np.rad2deg(all_rho_v)
        summary.append(f"  Acceptance Angles (Δρ):")
        summary.append(
            f"    Horizontal: {np.mean(rho_h_deg):.3f}° (mean), {np.min(rho_h_deg):.3f}° (min), {np.max(rho_h_deg):.3f}° (max)")
        summary.append(
            f"    Vertical:   {np.mean(rho_v_deg):.3f}° (mean), {np.min(rho_v_deg):.3f}° (min), {np.max(rho_v_deg):.3f}° (max)")

        # Eye Parameter (p)
        if self.eye_parameter_h is not None:  # Check if it has been calculated
            summary.append(f"  Eye Parameter (p = Δρ/Δφ):")
            summary.append(f"    Horizontal: {np.mean(self.p_h):.2f} (mean)")
            summary.append(f"    Vertical:   {np.mean(self.p_v):.2f} (mean)")

        return "\n".join(summary)


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