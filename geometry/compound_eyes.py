from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

from numpy.typing import ArrayLike
from scipy.spatial import KDTree
from graphics.utils import VEC_DTYPE, WORLD_UP, WORLD_RIGHT

GPU_OMMATIDIUM_DTYPE = np.dtype([
    ('origin', VEC_DTYPE, 4),               # vec4 (4 * float32, x, y, z coords and w pad)
    ('direction', VEC_DTYPE, 4),            # vec4 (4 * float32, x, y, z coords and w pad)
    ('acceptance_angles', VEC_DTYPE, 2),    # vec2 (2 * float32)
    ('_padding', VEC_DTYPE, 2)      # 8 bytes (2 * float32) of padding
])  # total 48 bytes


class OmmatidiumView:
    """
    A proxy object that provides a convenient, dataclass-like view
    into a single row of the main CompoundEye data array.

    This object does not store any data itself, it reads from and
    writes to the underlying numpy array.
    """

    def __init__(self, data_array: np.ndarray, index: int, changed_set: set):
        self._data = data_array
        self._index = index
        self._changed_set = changed_set

    def _mark_changed(self):
        self._changed_set.add(self._index)

    @property
    def id(self) -> int:
        return self._index

    @property
    def origin(self) -> np.ndarray:
        return self._data[self._index]['origin'][:3]

    @origin.setter
    def origin(self, value: ArrayLike):
        self._data[self._index]['origin'][:3] = np.asarray(value, dtype=VEC_DTYPE)
        self._data[self._index]['origin'][3] = 1.0  # The w component for origin should be 1.0
        self._mark_changed()

    @property
    def direction(self) -> np.ndarray:
        return self._data[self._index]['direction'][:3]

    @direction.setter
    def direction(self, value: ArrayLike):
        vec = np.asarray(value, dtype=VEC_DTYPE)
        norm = np.linalg.norm(vec)
        if not np.isclose(norm, 1.0):
            vec /= norm
        self._data[self._index]['direction'][:3] = vec
        self._data[self._index]['direction'][3] = 0.0  # The w component for a direction vector should be 0.0
        self._mark_changed()

    @property
    def acceptance_h(self) -> float:
        return self._data[self._index]['acceptance_angles'][0].item()

    @acceptance_h.setter
    def acceptance_h(self, value: float):
        self._data[self._index]['acceptance_angles'][0] = value
        self._mark_changed()

    @property
    def acceptance_v(self) -> float:
        return self._data[self._index]['acceptance_angles'][1].item()

    @acceptance_v.setter
    def acceptance_v(self, value: float):
        self._data[self._index]['acceptance_angles'][1] = value
        self._mark_changed()

    @property
    def acceptance_rad(self) -> np.ndarray:
        """
        Horizontal and Vertical acceptance angles in radians.
        """
        return self._data[self._index]['acceptance_angles']

    @acceptance_rad.setter
    def acceptance_rad(self, values: Union[float, ArrayLike]):
        self._data[self._index]['acceptance_angles'][:] = values
        self._mark_changed()

    @property
    def acceptance(self) -> np.ndarray:
        """
        Horizontal and Vertical acceptance angles in degrees.
        """
        return np.rad2deg(self.acceptance_rad)

    @acceptance.setter
    def acceptance(self, values: Union[float, ArrayLike]):
        self.acceptance_rad = np.deg2rad(np.asarray(values, dtype=VEC_DTYPE))

    @property
    def azimuth_rad(self) -> float:
        x, z = self._data[self._index]['direction'][[0, 2]]
        return np.arctan2(x, -z).item()

    @property
    def elevation_rad(self) -> float:
        return np.arcsin(self._data[self._index]['direction'][1]).item()

    # And some more aliases
    lon = longitude = azimuth = azimuth_rad
    lat = latitude = elevation = elevation_rad

    def __repr__(self):
        return (f"OmmatidiumView(id={self.id}, "
                f"direction=[{self.direction[0]:.3f}, {self.direction[1]:.3f}, {self.direction[2]:.3f}])")


class OmmatidiaCollection:
    """
    A wrapper that makes the structured numpy array behave like a list of OmmatidiumView objects.
    """

    def __init__(self, data_array: np.ndarray, changed_set: set):
        self._data = data_array
        self._changed_set = changed_set

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, index: int) -> OmmatidiumView:
        if not isinstance(index, int):
            raise TypeError("OmmatidiaCollection only supports integer indexing.")
        if not (-len(self) <= index < len(self)):
            raise IndexError("Ommatidium index out of range")
        return OmmatidiumView(self._data, index, self._changed_set)

    def __iter__(self):
        for i in range(len(self)):
            yield OmmatidiumView(self._data, i, self._changed_set)


class CompoundEye:
    """
    Container for a single eye's ommatidia with spatial query capabilities.
    """

    def __init__(self,
                 directions: Optional[ArrayLike] = None,
                 origins: Optional[ArrayLike] = None,
                 num_ommatidia: Optional[int] = None,
                 acceptance_angles_rad: Optional[Union[ArrayLike, Tuple, float]] = None,
                 eye_parameter: Optional[Union[float, Tuple]] = None,
                 lens_diameter: Optional[Union[float, Tuple]] = None,
                 rhabdom_diameter: Optional[Union[float, Tuple]] = None,
                 focal_length: Optional[Union[float, Tuple]] = None,
                 wavelength: float = 500e-9,  # TODO: this is a nice temporary value, but the shaders will compute the 3 channels independently
                 eye_radius: float = 0.0,
                 force_isotropic: bool = False
                 ):
        """
        The primary constructor for creating a Compound Eye.

        Args:
            directions: An (N, 3) numpy array of ommatidial direction vectors
            origins: An (N, 3) or (3,) array of ommatidial origin positions
            num_ommatidia: If directions are not provided, this number is used to generate a uniform sphere of directions.
            acceptance_angles_rad: (Optional) The acceptance angles (Δρ). Can be:
                - An (N, 2) array for individual H/V angles
                - A tuple (h, v) for global H/V angles
                - A float for a global circular angle
                - None: If not provided, will be estimated using other optical or geometric parameters.
            eye_parameter: (Optional) The eye parameter 'p' value (Δρ / Δφ). Used to estimate acceptance
                angles if they are not provided directly (defaults to 1.0)
            eye_radius: (Optional) Physical radius of the eye for setting ommatidial origins on a sphere.
            force_isotropic: (Optional) If True, ensures that the final acceptance angles
                are circular (Δρ_h = Δρ_v) by averaging any estimated anisotropic values.
        """

        if directions is None and num_ommatidia is None:
            raise ValueError("CompoundEye requires either 'directions' or 'num_ommatidia' to be provided.")

        # Determine ommatidial directions
        if directions is not None:
            # Priority 1: Direct directions are provided
            print("Using provided direction vectors.")
            directions = np.asarray(directions, dtype=VEC_DTYPE)
            nb_effective_dirs = len(directions)

        else:
            # Priority 2: Generate directions from num_ommatidia
            print(f"Generating uniform direction vectors for approx. {num_ommatidia} ommatidia.")
            lod = estimate_lod(num_ommatidia)
            directions = subdivide_icosahedron(lod)
            nb_effective_dirs = len(directions)
            if abs(num_ommatidia - nb_effective_dirs) > 1:
                print(f"Note: Using {nb_effective_dirs} ommatidia to match subdivision level {lod}.")

        self.num_ommatidia = nb_effective_dirs
        self.data = np.zeros(self.num_ommatidia, dtype=GPU_OMMATIDIUM_DTYPE)

        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        self.data['direction'][:, :3] = directions / norms
        self.data['direction'][:, 3] = 0.0  # w=0 for directions

        # Ommatidia origins
        if origins is not None:
            origins_arr = np.asarray(origins, dtype=VEC_DTYPE)
            if origins_arr.ndim == 1 and origins_arr.shape[0] == 3:
                print(f"Using a single origin {origins_arr} for all {self.num_ommatidia} ommatidia.")
                self.data['origin'][:, :3] = origins_arr  # Broadcast single origin

            elif origins_arr.shape == (self.num_ommatidia, 3):
                self.data['origin'][:, :3] = origins_arr
            else:
                raise ValueError(
                    f"Invalid shape for 'origins': {origins_arr.shape}. Expected ({self.num_ommatidia}, 3) or (3,).")
        elif eye_radius > 0:
            self.data['origin'][:, :3] = self.data['direction'][:, :3] * eye_radius
        # else: origins are already (0, 0, 0)
        self.data['origin'][:, 3] = 1.0  # w=1 for positions

        self.dirty_indices = set()
        self.ommatidia = OmmatidiaCollection(self.data, self.dirty_indices)

        self.kdtree = KDTree(self.data['direction'][:, :3])

        # Interommatidial angles (Δφ)
        self.interommatidial_angle_h_rad, self.interommatidial_angle_v_rad = self.estimate_interommatidial_angles(isotropic=force_isotropic)

        # Acceptance angles (Δρ)

        if acceptance_angles_rad is not None:
            # Priority 1: Direct acceptance angles are provided
            print("Using provided acceptance angles (Δρ).")
            estimated_angles = acceptance_angles_rad
        elif all(p is not None for p in [lens_diameter, rhabdom_diameter, focal_length]):
            # Priority 2: Estimate from optical parameters
            print("Calculating acceptance angles (Δρ) from physical optical parameters.")
            Dh, Dv = self._unpack(lens_diameter, "lens_diameter")
            dh, dv = self._unpack(rhabdom_diameter, "rhabdom_diameter")
            fh, fv = self._unpack(focal_length, "focal_length")

            delta_phi_optics_h = wavelength / Dh
            delta_phi_receptor_h = dh / fh
            angles_h_rad = np.sqrt(delta_phi_optics_h ** 2 + delta_phi_receptor_h ** 2)

            delta_phi_optics_v = wavelength / Dv
            delta_phi_receptor_v = dv / fv
            angles_v_rad = np.sqrt(delta_phi_optics_v ** 2 + delta_phi_receptor_v ** 2)
            estimated_angles = np.vstack([angles_h_rad, angles_v_rad]).T
        else:
            # Priority 3: Estimate from geometry using eye parameter 'p'
            p = eye_parameter if eye_parameter is not None else 1.0
            print(f"Estimating acceptance angles (Δρ) from interommatidial angles (Δφ) with eye parameter p={p}.")
            p_h, p_v = (p, p) if isinstance(p, (int, float)) else p
            delta_rho_h = p_h * self.interommatidial_angle_h_rad
            delta_rho_v = p_v * self.interommatidial_angle_v_rad
            estimated_angles = np.vstack([delta_rho_h, delta_rho_v]).T

        # Apply isotropic constraint if requested
        if force_isotropic and estimated_angles is not None:
            angles = np.asarray(estimated_angles)
            if angles.ndim >= 1:
                angles_2d = np.atleast_2d(angles)
                mean_angles = np.mean(angles_2d, axis=1)
                estimated_angles = np.vstack([mean_angles, mean_angles]).T

        # Set acceptance angles
        self._set_acceptance_angles(estimated_angles)

        # Now that Δρ and Δφ are known, calculate the resulting eye parameter p
        with np.errstate(divide='ignore', invalid='ignore'):
            self.eye_parameter_h = self.data['acceptance_angles'][:, 0] / self.interommatidial_angle_h_rad
            self.eye_parameter_v = self.data['acceptance_angles'][:, 1] / self.interommatidial_angle_v_rad

        # and clean non-finite values
        np.nan_to_num(self.eye_parameter_h, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(self.eye_parameter_v, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def _prepare_param(self, param, name="param"):
        """
        Ensures parameter is a numpy array of the correct shape.
        """
        arr = np.asarray(param, dtype=VEC_DTYPE)
        if arr.ndim == 0:
            return np.full(self.num_ommatidia, arr.item())
        if arr.ndim == 1 and len(arr) == self.num_ommatidia:
            return arr
        raise ValueError(
            f"Parameter '{name}' has invalid shape. Must be scalar or 1D array of length {self.num_ommatidia}.")

    def _unpack(self, param, name="param"):
        """
        Unpacks a parameter into horizontal and vertical components.
        """
        if isinstance(param, (list, tuple)):
            return self._prepare_param(param[0], f"{name}_h"), self._prepare_param(param[1], f"{name}_v")
        p_scalar = self._prepare_param(param, name)
        return p_scalar, p_scalar

    def _set_acceptance_angles(self, angles_rad: Union[np.ndarray, Tuple, float, None]):
        """
        Helper to assign acceptance angles to all ommatidia.
        """
        if angles_rad is None:
            print("Warning: No acceptance angles were provided or could be estimated.")
            return

        angles_arr = np.asarray(angles_rad, dtype=VEC_DTYPE)

        # This logic handles scalar, (2,), (N,), and (N,2) cases via broadcasting
        if angles_arr.shape == (self.num_ommatidia,):
            # If shape is (N,), broadcast to (N,2) for isotropic angles
            self.data['acceptance_angles'] = angles_arr[:, np.newaxis]
        else:
            # Handles scalar -> (1,) -> (N,2) broadcasting
            # Handles (2,) -> (N,2) broadcasting
            # Handles (N,2) -> (N,2) direct assignment
            self.data['acceptance_angles'] = angles_arr

    @property
    def interommatidial_angles_rad(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the estimated (horizontal, vertical) interommatidial angles (Δφ) in radians.
        """
        return self.interommatidial_angle_h_rad, self.interommatidial_angle_v_rad

    @classmethod
    def from_file(cls, file_path: Union[str, Path], **kwargs):
        """
        Creates an eye model from a .npy file
        - (N, 3): Directions only (angles will be estimated)
        - (N, 4): Directions + circular acceptance angles
        - (N, 5): Directions + ellipsoid acceptance angles (h != v)
        """
        # TODO: This needs to be replaced by a proper loader + proper file format
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

    def estimate_interommatidial_angles(self, k: int = 8, isotropic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimates the horizontal and vertical interommatidial angles (Δφ) for each ommatidium
        from the local geometry of its neighbours

        Args:
            k: The number of nearest neighbors to consider
            isotropic: If True, calculates a single representative angle by averaging the
                       raw geometry of all k neighbors, ignoring H/V distinctions
                       If False (default), estimates horizontal and vertical angles independently

        Returns:
            A tuple of two numpy arrays: (delta_phi_h, delta_phi_v) in radians
        """

        if self.num_ommatidia <= k:
            zeros = np.zeros(self.num_ommatidia, dtype=VEC_DTYPE)
            return zeros, zeros

        all_dirs = self.data['direction'][:, :3]
        distances, indices = self.kdtree.query(all_dirs, k=k + 1)

        neighbor_indices = indices[:, 1:]
        if neighbor_indices.size == 0:
            zeros = np.zeros(self.num_ommatidia, dtype=VEC_DTYPE)
            return zeros, zeros

        if isotropic:
            neighbor_distances = distances[:, 1:]
            avg_euclidean_dist = np.mean(neighbor_distances, axis=1)
            term = 1.0 - (avg_euclidean_dist ** 2) / 2.0
            delta_phi = np.arccos(np.clip(term, -1.0, 1.0))
            return delta_phi, delta_phi
        else:
            # Check for ommatidia at the poles
            dot_products = np.abs(np.dot(all_dirs, WORLD_UP))
            is_polar = dot_products > 0.9999
            ref_up_vectors = np.where(is_polar[:, np.newaxis], WORLD_RIGHT, WORLD_UP)

            # Gram-Schmidt to find local tangent 'up' vector (local_y)
            local_y_axes = ref_up_vectors - all_dirs * np.sum(all_dirs * ref_up_vectors, axis=1, keepdims=True)
            local_y_axes /= np.linalg.norm(local_y_axes, axis=1, keepdims=True)
            local_x_axes = np.cross(local_y_axes, all_dirs)

            # Project neighbour vectors onto the local tangent plane
            neighbour_dirs = all_dirs[neighbor_indices]
            neighbour_vectors = neighbour_dirs - all_dirs[:, np.newaxis, :]

            sep_h = np.sum(neighbour_vectors * local_x_axes[:, np.newaxis, :], axis=2)
            sep_v = np.sum(neighbour_vectors * local_y_axes[:, np.newaxis, :], axis=2)
            abs_sep_h, abs_sep_v = np.abs(sep_h), np.abs(sep_v)

            # Horizontal neighbours = the 2 neighbours with smallest vertical separation
            h_neighbour_indices = np.argsort(abs_sep_v, axis=1)[:, :2]
            # Vertical neighbours = the 2 neighbours with smallest horizontal separation
            v_neighbour_indices = np.argsort(abs_sep_h, axis=1)[:, :2]

            # Use angular distances of these neighbours for final estimate
            horizontal_angles = np.take_along_axis(abs_sep_h, h_neighbour_indices, axis=1)
            vertical_angles = np.take_along_axis(abs_sep_v, v_neighbour_indices, axis=1)

            delta_phi_h = np.mean(horizontal_angles, axis=1)
            delta_phi_v = np.mean(vertical_angles, axis=1)

            return delta_phi_h, delta_phi_v

    def max_gap(self):
        """
        Finds the maximum angular distance between any ommatidium and its
        single nearest neighbor, which represents the largest "gap" in the eye.
        """

        if self.num_ommatidia == 1:
            return 0.0

        distances, _ = self.kdtree.query(self.data['direction'][:, :3], k=2)
        max_euclidean_dist = np.max(distances[:, 1])
        term = 1.0 - (max_euclidean_dist ** 2) / 2.0
        return np.arccos(np.clip(term, -1.0, 1.0))

    def __repr__(self):
        summary = [f"<CompoundEye with {self.num_ommatidia} ommatidia>"]

        # Interommatidial Angles (Δφ)
        d_phi_h_deg = np.rad2deg(self.interommatidial_angle_h_rad)
        d_phi_v_deg = np.rad2deg(self.interommatidial_angle_v_rad)
        summary.append("  Interommatidial Angles (Δφ):")
        summary.append(
            f"    Horizontal: {np.mean(d_phi_h_deg):.3f}° (mean), {np.min(d_phi_h_deg):.3f}° (min), {np.max(d_phi_h_deg):.3f}° (max)")
        summary.append(
            f"    Vertical:   {np.mean(d_phi_v_deg):.3f}° (mean), {np.min(d_phi_v_deg):.3f}° (min), {np.max(d_phi_v_deg):.3f}° (max)")

        # Acceptance Angles (Δρ)
        angles_deg = np.rad2deg(self.data['acceptance_angles'])
        means = np.mean(angles_deg, axis=0)
        mins = np.min(angles_deg, axis=0)
        maxs = np.max(angles_deg, axis=0)
        summary.append("  Acceptance Angles (Δρ):")
        summary.append(f"    Horizontal: {means[0]:.3f}° (mean), {mins[0]:.3f}° (min), {maxs[0]:.3f}° (max)")
        summary.append(f"    Vertical:   {means[1]:.3f}° (mean), {mins[1]:.3f}° (min), {maxs[1]:.3f}° (max)")

        # Eye Parameter (p)
        summary.append("  Eye Parameter (p = Δρ/Δφ):")
        summary.append(f"    Horizontal: {np.mean(self.eye_parameter_h):.2f} (mean)")
        summary.append(f"    Vertical:   {np.mean(self.eye_parameter_v):.2f} (mean)")

        return "\n".join(summary)


def estimate_lod(num_ommatidia: int) -> int:
    """
    Calculates the Level of Division (LoD) needed to produce a number of ommatidia.
    """
    if num_ommatidia < 12:
        return 1

    # LoD: y = 10 * n^2 + 2 for n
    n = np.sqrt((num_ommatidia - 2) / 10.0)
    return int(np.round(n))


def icosahedron_faces() -> np.ndarray:
    """
    Defines the base (z-axis aligned) icosahedron and returns the vertices for the 20 triangular faces
    """
    # TODO: Move this to the primitives file maybe?

    # Golden ratio
    G = (1 + np.sqrt(5)) / 2.0

    # Three mutually perpendicular golden ratio rectangles make the icosahedron's vertices :)
    p = np.array([
        [G, -G, -G, G, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, G, -G, -G, G, 1.0, 1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, G, -G, -G, G]
    ]).T
    # Rotate top point to the z-axis
    p /= np.linalg.norm(p[0])
    ang = np.arctan(p[0, 0] / p[0, 2])
    ca, sa = np.cos(ang), np.sin(ang)
    rotation = np.array([[ca, 0.0, -sa], [0.0, 1.0, 0.0], [sa, 0.0, ca]])
    p = np.inner(rotation, p).T

    # Reorder in a downward spiral
    reorder_index = [0, 3, 4, 8, -1, 5, -2, -3, 7, 1, 6, 2]
    p = p[reorder_index]

    # 20 triangular faces
    tri_indices = np.array([
        [1, 2, 3, 4, 5, 6, 2, 7, 2, 8, 3, 9, 10, 10, 6, 6, 7, 8, 9, 10],
        [2, 3, 4, 5, 1, 7, 1, 8, 8, 9, 9, 10, 5, 6, 1, 11, 11, 11, 11, 11],
        [0, 0, 0, 0, 0, 1, 7, 2, 3, 3, 4, 4, 4, 5, 5, 7, 8, 9, 10, 6]
    ]).T
    return p[tri_indices]


def barycentric_coords(n_subdiv: int) -> np.ndarray:
    """
    Generates a matrix of barycentric coordinates (u, v, w)
    inside a reference triangle where u + v + w = 1
    """

    vals = np.linspace(0, 1, n_subdiv + 1)

    # Total number of points in a triangle subdivided n times
    num_points = int((n_subdiv + 1) * (n_subdiv + 2) / 2)
    bcmat = np.zeros((num_points, 3))

    # Builds the points 'row by row' inside the ref triangle
    shifts = np.arange(n_subdiv + 1, 0, -1)
    starts = np.zeros(n_subdiv + 1, dtype=int)
    starts[1:] = np.cumsum(shifts[:-1])
    stops = starts + shifts

    # along each row: u decreases, v increases, w stays constant
    for i, (start, stop, shift) in enumerate(zip(starts, stops, shifts)):
        bcmat[start:stop, 0] = vals[shift - 1::-1]
        bcmat[start:stop, 1] = vals[:shift]
        bcmat[start:stop, 2] = vals[i]

    return bcmat


def subdivide_icosahedron(n_subdiv: int) -> np.ndarray:
    """ Subdivides icosahedron using barycentric coordinates """

    verts = icosahedron_faces()
    bary = barycentric_coords(n_subdiv)

    # Barycentric interpolation to each of the 20 triangles
    # 'ij,kjl->kil': i=bary_idx, j=bary_coord, k=tri_idx, l=vertex_coord
    all_new_verts = np.einsum('ij,kjl->kil', bary, verts)
    # Normalize to unit sphere and find unique vertices
    all_new_verts = all_new_verts.reshape(-1, 3)
    all_new_verts /= np.linalg.norm(all_new_verts, axis=1)[:, np.newaxis]
    _, iunique = np.unique(np.round(all_new_verts, 6), axis=0, return_index=True)

    return all_new_verts[iunique].astype(VEC_DTYPE)