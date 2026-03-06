from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union, Sequence
import numpy as np

from numpy.typing import ArrayLike
from scipy.spatial import KDTree
from graphics.utils import WORLD_UP, WORLD_RIGHT, DeltaTimeTransformer


GPU_OMMATIDIUM_DTYPE = np.dtype([
    ('origin', np.float32, 4),                  # 16 bytes (4 * float32): x, y, z coords and w for homogeneous
    ('direction', np.float32, 4),               # 16 bytes (4 * float32): x, y, z coords and w for homogeneous
    ('acceptance_angles', np.float32, 2),       #  8 bytes (2 * float32): minor and major axes
    ('interommatidial_angles', np.float32, 2),  #  8 bytes (2 * float32): minor and major axes
    ('tilt', np.float32),                       #  4 bytes (1 * float32): ellipse tilt
    ('sensitivity', np.float32),                #  4 bytes (1 * float32): receptor sensitivity
    ('packed_data', np.uint32),                 #  4 bytes (1 * uint32): Packed additional data, see below
    ('padding', np.uint32)                      #  4 bytes padding
])  # total = 64 bytes

# packed_data layout:
# - bits 0-2: eye ID (0-7)
# - bits 3-6: receptor type (0-15)
# - bits 7-10: number of immediate neighbours (0-15)
# - bits 11-26: custom ID (0-65535)
# - bits 27-31: padding


DEFAULT_ANGLE = 'deg'
# DEFAULT_ANGLE = 'rad'


@dataclass
class RhabdomereLayout:
    """Configuration for a specific biological rhabdomere layout."""
    name: str
    offsets_um: np.ndarray  # X and Y offsets in micrometres, shape (R, 2)
    focal_length_um: float  # focal length (micrometres)
    diameters_um: Optional[np.ndarray]  # diameter of each receptor, shape (R,)


# # Data from Juusola et al. (Drosophila)
# DROSOPHILA_RHABDOMERES = RhabdomereLayout(
#     name="Drosophila_R1_R7",
#     focal_length_um=21.0,
#     offsets_um=np.array([
#         [-1.6881, 1.0273],  # R1
#         [-1.8046, -0.9934],  # R2
#         [-1.7111, -2.9717],  # R3
#         [-0.0025, -1.9261],  # R4
#         [1.6690, -0.9493],  # R5
#         [1.6567, 0.9762],  # R6
#         [0.0045, -0.0113]  # R7/8 (Central)
#     ]),
#     diameters_um=np.array([1.86, 1.86, 1.86, 1.86, 1.86, 1.86, 1.57])
# )


def rotate_vectors(vectors: np.ndarray, axes: np.ndarray, angles: np.ndarray, degrees: bool = True) -> np.ndarray:
    """
    Rotates batches of vectors around corresponding axes using Rodrigues' formula.
    """

    angles_arr = np.asarray(angles)
    angles_rad = np.deg2rad(angles_arr) if degrees else angles_arr

    if angles_rad.ndim == 0:
        cos_a = np.cos(angles_rad)
        sin_a = np.sin(angles_rad)
    else:
        cos_a = np.cos(angles_rad)[:, np.newaxis]
        sin_a = np.sin(angles_rad)[:, np.newaxis]

    term1 = vectors * cos_a
    term2 = np.cross(axes, vectors) * sin_a
    term3 = axes * np.sum(axes * vectors, axis=1, keepdims=True) * (1 - cos_a)

    return term1 + term2 + term3


class Ommatidium:
    """
    A proxy that provides a view into the CompoundEye ommatidia data array.
    """

    def __init__(self, data_array: np.ndarray, item, parent_eye: 'CompoundEye'):
        self._data = data_array
        self._item = item
        self._parent_eye = parent_eye

    @property
    def origin(self) -> np.ndarray:
        return self._data[self._item]['origin'][..., :3]

    @origin.setter
    def origin(self, value: Union[float, ArrayLike]):
        self._data['origin'][self._item, :3] = np.asarray(value, dtype=np.float32)
        self._data['origin'][self._item, 3] = 1.0  # The w component for origins should be 1.0
        self._parent_eye.dirty_mask[self._item] = True
        self._parent_eye.needs_rebuild['origin'] = True

    @property
    def direction(self) -> np.ndarray:
        return self._data[self._item]['direction'][..., :3]

    @direction.setter
    def direction(self, value: Union[float, ArrayLike]):

        new_dirs = np.atleast_2d(value)
        norms = np.linalg.norm(new_dirs, axis=-1, keepdims=True)
        normalized_dirs = np.divide(new_dirs, norms, out=new_dirs, where=norms != 0)

        self._data['direction'][self._item, :3] = normalized_dirs
        self._data['direction'][self._item, 3] = 0.0   # The w component for a direction vector should be 0.0

        self._parent_eye.dirty_mask[self._item] = True
        self._parent_eye.needs_rebuild['direction'] = True

    def dt(self, delta_time: float) -> DeltaTimeTransformer:
        """
        Enables framerate-independent transformations for a chain of method calls
        """
        return DeltaTimeTransformer(self, delta_time)

    def rotate(self, yaw_delta: Union[float, ArrayLike] = 0.0, pitch_delta: Union[float, ArrayLike] = 0.0, roll_delta : Union[float, ArrayLike] = 0.0, degrees: bool = True):
        """
        Rotates the ommatidium's direction in its local tangent space.
        - 'yaw_delta' rotates horizontally (accepts scalar or array).
        - 'pitch_delta' rotates vertically (accepts scalar or array).
        - 'roll_delta' is ignored.
        """
        current_dirs = np.atleast_2d(self._data[self._item]['direction'][..., :3])

        dots = np.abs(current_dirs @ WORLD_UP)
        is_polar = dots > 0.9999
        reference_ups = np.where(is_polar[:, np.newaxis], WORLD_RIGHT, WORLD_UP)

        local_tangents = np.cross(current_dirs, reference_ups)
        norms_t = np.linalg.norm(local_tangents, axis=1, keepdims=True)
        np.divide(local_tangents, norms_t, out=local_tangents, where=norms_t != 0)

        local_bitangents = np.cross(local_tangents, current_dirs)
        rotated_dirs = current_dirs

        yaw_delta_arr = np.asarray(yaw_delta)
        pitch_delta_arr = np.asarray(pitch_delta)

        if np.any(yaw_delta_arr != 0.0):
            rotated_dirs = rotate_vectors(rotated_dirs, local_bitangents, yaw_delta_arr, degrees=degrees)

        if np.any(pitch_delta_arr != 0.0):
            rotated_dirs = rotate_vectors(rotated_dirs, local_tangents, pitch_delta_arr, degrees=degrees)

        self.direction = rotated_dirs
        return self

    def translate(self, distance: Union[float, ArrayLike]):
        """
        Moves the ommatidium's origin along its own direction vector.
        """

        current_origins = self._data[self._item]['origin'][..., :3]
        current_dirs = self._data[self._item]['direction'][..., :3]

        distances_arr = np.asarray(distance, dtype=np.float32)
        if distances_arr.ndim == 1:
            distances_arr = distances_arr[:, np.newaxis]

        self.origin = current_origins + current_dirs * distances_arr
        return self

    @property
    def eye_id(self) -> np.ndarray:
        """ Unpacks eye ID from bits 0-2 """
        return self._data[self._item]['packed_data'] & 0x07

    @eye_id.setter
    def eye_id(self, value: Union[int, ArrayLike]):
        """ Packs eye ID into bits 0-2 """
        value_arr = np.asarray(value, dtype=np.uint32)

        current_data = self._data['packed_data'][self._item]
        cleared_data = current_data & ~0x07
        new_data = cleared_data | (value_arr & 0x07)

        self._data['packed_data'][self._item] = new_data
        self._parent_eye.dirty_mask[self._item] = True

    @property
    def acceptance_major(self) -> np.ndarray:
        return self._data[self._item]['acceptance_angles'][..., 0]

    @acceptance_major.setter
    def acceptance_major(self, value: Union[float, ArrayLike]):
        self._data['acceptance_angles'][self._item, 0] = value
        self._parent_eye.dirty_mask[self._item] = True

    @property
    def acceptance_minor(self) -> np.ndarray:
        return self._data[self._item]['acceptance_angles'][..., 1]

    @acceptance_minor.setter
    def acceptance_minor(self, value: Union[float, ArrayLike]):
        self._data['acceptance_angles'][self._item, 1] = value
        self._parent_eye.dirty_mask[self._item] = True

    @property
    def acceptance_rad(self) -> np.ndarray:
        return self._data[self._item]['acceptance_angles']

    @acceptance_rad.setter
    def acceptance_rad(self, values: Union[float, ArrayLike]):
        self._data['acceptance_angles'][self._item] = values
        self._parent_eye.dirty_mask[self._item] = True

    @property
    def acceptance_deg(self) -> np.ndarray:
        return np.rad2deg(self.acceptance_rad)

    @acceptance_deg.setter
    def acceptance_deg(self, values: Union[float, ArrayLike]):
        self.acceptance_rad = np.deg2rad(np.asarray(values, dtype=np.float32))

    @property
    def sensitivity(self) -> np.ndarray:
        return self._data[self._item]['sensitivity']

    @sensitivity.setter
    def sensitivity(self, value: Union[float, ArrayLike]):
        self._data['sensitivity'][self._item] = np.asarray(value, dtype=np.float32)
        self._parent_eye.dirty_mask[self._item] = True

    @property
    def receptor_type(self) -> np.ndarray:
        """ Unpacks receptor type from bits 3-6 """
        return (self._data[self._item]['packed_data'] >> 3) & 0x0F

    @receptor_type.setter
    def receptor_type(self, value: Union[int, ArrayLike]):
        """ Packs receptor type into bits 3-6 """
        value_arr = np.asarray(value, dtype=np.uint32)

        current_data = self._data['packed_data'][self._item]
        cleared_data = current_data & ~(0x0F << 3)
        new_data = cleared_data | ((value_arr & 0x0F) << 3)

        self._data['packed_data'][self._item] = new_data
        self._parent_eye.dirty_mask[self._item] = True

    @property
    def neighbours_count(self) -> np.ndarray:
        """ Unpacks number of neighbours from bits 7-10 """
        return (self._data[self._item]['packed_data'] >> 7) & 0x0F

    @neighbours_count.setter
    def neighbours_count(self, value: Union[int, ArrayLike]):
        """ Packs number of neighbours into bits 7-10 """
        value_arr = np.asarray(value, dtype=np.uint32)

        current_data = self._data['packed_data'][self._item]
        cleared_data = current_data & ~(0x0F << 7)
        new_data = cleared_data | ((value_arr & 0x0F) << 7)

        self._data['packed_data'][self._item] = new_data
        self._parent_eye.dirty_mask[self._item] = True

    @property
    def custom_id(self) -> np.ndarray:
        """ Unpacks custom ID from bits 11-26 """
        return (self._data[self._item]['packed_data'] >> 11) & 0xFFFF

    @custom_id.setter
    def custom_id(self, value: Union[int, ArrayLike]):
        """ Packs custom ID into bits 11-26 """
        value_arr = np.asarray(value, dtype=np.uint32)

        current_data = self._data['packed_data'][self._item]
        cleared_data = current_data & ~(0xFFFF << 11)
        new_data = cleared_data | ((value_arr & 0xFFFF) << 11)

        self._data['packed_data'][self._item] = new_data
        self._parent_eye.dirty_mask[self._item] = True

    @property
    def azimuth_rad(self) -> np.ndarray:
        return np.arctan2(self._data[self._item]['direction'][..., 0], -self._data[self._item]['direction'][..., 2])

    @property
    def azimuth_deg(self):
        return np.rad2deg(self.azimuth_rad)

    @property
    def elevation_rad(self) -> np.ndarray:
        return np.arcsin(self._data[self._item]['direction'][..., 1])

    @property
    def elevation_deg(self) -> np.ndarray:
        return np.rad2deg(self.elevation_rad)

    # And some more aliases
    lon = longitude = azimuth = azimuth_rad if DEFAULT_ANGLE == 'rad' else azimuth_deg
    lat = latitude = elevation = elevation_rad if DEFAULT_ANGLE == 'rad' else elevation_deg
    rho = acceptance = acceptance_rad if DEFAULT_ANGLE == 'rad' else acceptance_deg
    rho_minor = acceptance_minor
    rho_major = acceptance_major

    def __len__(self):
        return 1 if self._data[self._item].ndim == 0 else self._data[self._item].shape[0]

    def __repr__(self):
        if isinstance(self._item, (int, np.int_)):
            origin_str = np.array2string(self.origin, precision=3, suppress_small=True)
            direction_str = np.array2string(self.direction, precision=3, suppress_small=True)
            return f"<Ommatidium(id={int(self._item)}, origin={origin_str}, direction={direction_str})>"
        else:
            return f"<OmmatidiumProxy(key={self._item}, count={len(self)})>"


class OmmatidiaCollection:

    def __init__(self, data_array: np.ndarray, parent_eye: 'CompoundEye'):
        self._data = data_array
        self._parent_eye = parent_eye
        self._len = len(self._data)

    def __getitem__(self, key: Union[int, slice, Sequence[int]]):
        if isinstance(key, (int, np.int_, slice, list, tuple, np.ndarray)):
            return Ommatidium(self._data, key, self._parent_eye)
        else:
            raise IndexError("Ommatidia indices must be integers, slices, or lists.")

    def __len__(self):
        return self._len

    def __repr__(self):
        return f"<OmmatidiaCollection for {self._len} ommatidia>"


class CompoundEye:
    """
    Container for a single eye's ommatidia with spatial query capabilities.
    """

    def __init__(self,
                 directions: Optional[ArrayLike] = None,
                 origins: Optional[ArrayLike] = None,
                 num_ommatidia: Optional[int] = None,
                 acceptance_angles_rad: Optional[Union[ArrayLike, Tuple, float]] = None,
                 interommatidial_angles_rad: Optional[Union[ArrayLike, Tuple, float]] = None,
                 sensitivities: Optional[Union[ArrayLike, float]] = None,
                 receptor_types: Optional[Union[ArrayLike, int]] = None,
                 eye_id: Optional[Union[int, ArrayLike]] = None,
                 custom_ids: Optional[Union[ArrayLike, int]] = None,
                 eye_parameter: Optional[Union[float, Tuple]] = None,
                 lens_diameter_nm: Optional[Union[float, Tuple]] = None,
                 rhabdom_diameter_nm: Optional[Union[float, Tuple]] = None,
                 focal_length_nm: Optional[Union[float, Tuple]] = None,
                 wavelength_nm: float = 500,  # TODO: this is a temporary value, the shaders should eventually compute the 3 channels independently
                 eye_radius: float = 0.01,
                 force_isotropic: bool = False
                 ):
        """
        The primary constructor for creating a Compound Eye.

        Args:
            directions: An (N, 3) numpy array of ommatidial direction vectors
            origins: An (N, 3) or (3,) array of ommatidial origin positions
            num_ommatidia: If directions are not provided, this is used to generate a uniform sphere of directions.
            acceptance_angles_rad: (Optional) The acceptance angles (Δρ), minor and major axes. Can be an (N, 2) array, a tuple (minor, major),
                a float, or None to estimate from other parameters.
            interommatidial_angles_rad: (Optional) The interommatidial angles (Δφ), minor and major axes. Can be an (N, 2) array, a tuple (minor, major),
                a float, or None to estimate from other parameters.
            sensitivities: (Optional) A scalar or (N,) array for ommatidial sensitivity. Defaults to 1.0.
            receptor_types: (Optional) A scalar or (N,) array of integer receptor types. Defaults to 0.
            eye_id: (Optional) A scalar or (N,) encoding which eye the ommatidium (or ommatidia) belongs to. 0 to 7.
            custom_ids: (Optional) A scalar or (N,) array of integer custom IDs. Defaults to 0.
            eye_parameter: (Optional) The eye parameter 'p' value (Δρ / Δφ). Used to estimate acceptance
                angles if they are not provided directly (defaults to 1.0)
            eye_radius: (Optional) Physical radius of the eye for setting ommatidial origins on a sphere.
            force_isotropic: (Optional) If True, ensures acceptance angles are circular.
        """

        if directions is None and num_ommatidia is None:
            raise ValueError("CompoundEye requires either 'directions' or 'num_ommatidia' to be provided.")

        # Determine ommatidial directions
        if directions is not None:
            # Priority 1: directions are provided
            print("Using provided direction vectors.")
            directions = np.asarray(directions, dtype=np.float32)
            nb_effective_dirs = len(directions)

        else:
            # Priority 2: Generate directions from ommatidia_count
            print(f"Generating uniform direction vectors for approx. {num_ommatidia} ommatidia.")
            lod = estimate_lod(num_ommatidia)
            directions = subdivide_icosahedron(lod)
            nb_effective_dirs = len(directions)
            if abs(num_ommatidia - nb_effective_dirs) > 1:
                print(f"Note: Using {nb_effective_dirs} ommatidia to match subdivision level {lod}.")

        self.ommatidia_count = nb_effective_dirs
        self.data = np.zeros(self.ommatidia_count, dtype=GPU_OMMATIDIUM_DTYPE)

        self._is_rhabdomeres = False
        self._lens_data = None
        self.receptor_count = 1

        # Directions
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        self.data['direction'][:, :3] = directions / norms
        self.data['direction'][:, 3] = 0.0  # w=0 for directions

        # Origins
        if origins is not None:
            origins_arr = np.asarray(origins, dtype=np.float32)
            if origins_arr.ndim == 1 and origins_arr.shape[0] == 3:
                print(f"Using a single origin {origins_arr} for all {self.ommatidia_count} ommatidia.")
                self.data['origin'][:, :3] = origins_arr

            elif origins_arr.shape == (self.ommatidia_count, 3):
                self.data['origin'][:, :3] = origins_arr

            else:
                raise ValueError(
                    f"Invalid shape for 'origins': {origins_arr.shape}. Expected ({self.ommatidia_count}, 3) or (3,).")
        elif eye_radius > 0:
            self.data['origin'][:, :3] = self.data['direction'][:, :3] * eye_radius

        self.data['origin'][:, 3] = 1.0  # w=1 for positions

        # Metadata and packing
        self.ommatidia = OmmatidiaCollection(self.data, self)   # temporary proxy for packing metadata
        self.ommatidia.sensitivity = sensitivities if sensitivities is not None else 1.0

        id_arr = np.zeros(self.ommatidia_count, dtype=np.uint32)
        if eye_id is not None:
            prepared_ids = self._prepare_param(eye_id, "eye_id")
            if np.any(prepared_ids > 7) or np.any(prepared_ids < 0):
                raise ValueError("Eye ID must be in the range [0, 7].")
            id_arr = prepared_ids.astype(np.uint32)

        types_arr = np.zeros(self.ommatidia_count, dtype=np.uint32)
        if receptor_types is not None:
            prepared_types = self._prepare_param(receptor_types, "receptor_types")
            if np.any(prepared_types > 15) or np.any(prepared_types < 0):
                print("Warning: Receptor types should be in [0, 15]. Clamping values.")
            types_arr = prepared_types.astype(np.uint32)

        custom_ids_arr = np.zeros(self.ommatidia_count, dtype=np.uint32)
        if custom_ids is not None:
            prepared_custom_ids = self._prepare_param(custom_ids, "custom_ids")
            if np.any(prepared_custom_ids > 65535) or np.any(prepared_custom_ids < 0):
                raise ValueError("Custom IDs must be in the range [0, 65535].")
            custom_ids_arr = prepared_custom_ids.astype(np.uint32)

        packed_data = (id_arr.astype(np.uint32) & 0x07) | \
                      ((types_arr.astype(np.uint32) & 0x0F) << 3) | \
                      ((custom_ids_arr.astype(np.uint32) & 0xFFFF) << 11)
        self.data['packed_data'] = packed_data

        # Prep spatial queries
        self.dirty_mask = np.zeros(self.ommatidia_count, dtype=bool)
        self.needs_rebuild = {'direction': False, 'origin': True}
        self.kdtree_directions = KDTree(self.data['direction'][:, :3])
        self.kdtree_positions = KDTree(self.data['origin'][:, :3])

        # Lattice geometry (Δφ and Tilt)
        # (check if origins overlap: indicates loading a pre-expanded eye)  # TODO: Too fragile, maybe should just save th flag
        is_pre_expanded = False
        if self.ommatidia_count > 1:
            if np.allclose(self.data['origin'][0], self.data['origin'][1], atol=1e-7):
                is_pre_expanded = True

        if interommatidial_angles_rad is not None:
            # Case A: User provided ground-truth angles
            print("Using provided interommatidial angles.")
            angles_arr = np.asarray(interommatidial_angles_rad, dtype=np.float32)

            if angles_arr.shape == (self.ommatidia_count,):
                self.ioa_minor_rad = angles_arr
                self.ioa_major_rad = angles_arr
            else:
                angles_broadcast = np.broadcast_to(angles_arr, (self.ommatidia_count, 2))
                self.ioa_minor_rad = np.minimum(angles_broadcast[:, 0], angles_broadcast[:, 1])
                self.ioa_major_rad = np.maximum(angles_broadcast[:, 0], angles_broadcast[:, 1])

            self.data['interommatidial_angles'][:, 0] = self.ioa_minor_rad
            self.data['interommatidial_angles'][:, 1] = self.ioa_major_rad

        elif not is_pre_expanded:
            # Case B: Standard lens eye, run estimation
            print("Estimating lattice properties (tilt, neighbours, IOA) from ommatidia origins...")
            est_minor, est_major, tilts, counts = self._compute_lattice_properties()

            self.ioa_minor_rad, self.ioa_major_rad = est_minor, est_major
            self.data['interommatidial_angles'][:, 0] = est_minor
            self.data['interommatidial_angles'][:, 1] = est_major
            self.data['tilt'] = tilts
            self.ommatidia.neighbours_count = counts
        else:
            # Case C: Pre-expanded (rhabdomeres) eye, read existing data
            print("Detected pre-expanded data. Skipping geometric estimation.")
            self.ioa_minor_rad = self.data['interommatidial_angles'][:, 0]
            self.ioa_major_rad = self.data['interommatidial_angles'][:, 1]

        # Acceptance angles (Δρ)
        if acceptance_angles_rad is not None:
            # Priority 1: Direct acceptance angles are provided
            print("Using provided acceptance angles (Δρ).")
            estimated_angles = acceptance_angles_rad

        elif all(p is not None for p in [lens_diameter_nm, rhabdom_diameter_nm, focal_length_nm]):
            # Priority 2: Estimate from optical parameters
            print("Calculating acceptance angles (Δρ) from physical optical parameters.")
            D_minor, D_major = self._unpack(lens_diameter_nm, "lens_diameter")
            d_minor, d_major = self._unpack(rhabdom_diameter_nm, "rhabdom_diameter")
            f_minor, f_major = self._unpack(focal_length_nm, "focal_length")

            acc_min = np.sqrt((wavelength_nm / D_minor) ** 2 + (d_minor / f_minor) ** 2)
            acc_maj = np.sqrt((wavelength_nm / D_major) ** 2 + (d_major / f_major) ** 2)
            estimated_angles = np.vstack([acc_min, acc_maj]).T
        else:
            # Priority 3: Estimate from geometry using eye parameter 'p'
            p = eye_parameter if eye_parameter is not None else 1.0
            print(f"Estimating acceptance angles (Δρ) from interommatidial angles (Δφ) with eye parameter p={p}.")
            p_min, p_maj = (p, p) if isinstance(p, (int, float)) else p
            estimated_angles = np.vstack([p_min * self.ioa_minor_rad, p_maj * self.ioa_major_rad]).T

        # Apply isotropic constraint if requested
        if force_isotropic:
            mean_angles = np.mean(np.atleast_2d(estimated_angles), axis=1)
            estimated_angles = np.vstack([mean_angles, mean_angles]).T

        # Assign acceptance angles Δρ
        if estimated_angles is None:
            raise AttributeError("Warning: No acceptance angles were provided or could be estimated.")

        angles_arr = np.asarray(estimated_angles, dtype=np.float32)
        if angles_arr.shape == (self.ommatidia_count,):
            self.data['acceptance_angles'] = angles_arr[:, np.newaxis]
        else:
            self.data['acceptance_angles'] = angles_arr

        # Now that Δρ and Δφ are known, calculate the resulting eye parameter p
        with np.errstate(divide='ignore', invalid='ignore'):
            self.eye_parameter_minor = self.data['acceptance_angles'][:, 0] / self.ioa_minor_rad
            self.eye_parameter_major = self.data['acceptance_angles'][:, 1] / self.ioa_major_rad

        # clean non-finite values
        np.nan_to_num(self.eye_parameter_minor, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(self.eye_parameter_major, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Final refresh of the proxy to ensure it captures everythign
        self.ommatidia = OmmatidiaCollection(self.data, self)

    def _prepare_param(self, param, name="param"):
        """
        Ensures parameter is a numpy array of the correct shape
        """
        arr = np.asarray(param, dtype=np.float32)
        if arr.ndim == 0:
            return np.full(self.ommatidia_count, arr.item())
        if arr.ndim == 1 and len(arr) == self.ommatidia_count:
            return arr
        raise ValueError(
            f"Parameter '{name}' has invalid shape. Must be scalar or 1D array of length {self.ommatidia_count}.")

    def _unpack(self, param, name="param"):
        """
        Unpacks a parameter into minor and major components
        """
        if isinstance(param, (list, tuple)):
            return self._prepare_param(param[0], f"{name}_minor"), self._prepare_param(param[1], f"{name}_major")
        p_scalar = self._prepare_param(param, name)
        return p_scalar, p_scalar

    @property
    def interommatidial_angles_rad(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the estimated (minor, major) interommatidial angles (Δφ) in radians.
        """
        return self.ioa_minor_rad, self.ioa_major_rad

    @classmethod
    def from_file(cls, file_path: Union[str, Path], **kwargs):
        """
        Creates an eye model from a .npz archive file.

        The .npz file is expected to contain at least a 'directions' array.
        It can optionally contain 'origins', 'acceptance_angles_rad', 'interommatidial_angles_rad',
        'sensitivities', 'receptor_types', 'eye_id', and 'custom_ids'.
        Any arguments passed via **kwargs will override the data from the file.

        Args:
            file_path: Path to the .npz file
            **kwargs: Additional arguments to pass to the CompoundEye constructor,
                      which will override file data.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Cannot find eye data file: {path}")

        data = np.load(path)

        if 'directions' not in data:
            raise ValueError(f"Eye data file '{path}' is missing the required 'directions' array.")

        constructor_args = {
            'directions': data['directions'],
            'origins': data.get('origins'),
            'acceptance_angles_rad': data.get('acceptance_angles_rad'),
            'interommatidial_angles_rad': data.get('interommatidial_angles_rad'),
            'sensitivities': data.get('sensitivities'),
            'receptor_types': data.get('receptor_types'),
            'eye_id': data.get('eye_id'),
            'custom_ids': data.get('custom_ids'),
        }

        constructor_args.update(kwargs)

        print(f"Loaded eye model from '{path}'.")
        return cls(**constructor_args)

    def _compute_lattice_properties(self, k: int = 8, neighbour_dist_factor: float = 1.5) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimates local lattice properties for each ommatidium using its nearest neighbours.
        This includes interommatidial angles (minor and major axes), the lattice tilt angle,
        and the number of immediate neighbours.

        Args:
            k (int): Number of neighbours to consider for the analysis.
            neighbour_dist_factor (float): A factor to determine immediate neighbours. A neighbour is
                considered "immediate" if its angular separation is less than or equal to
                (neighbour_dist_factor * angular_separation_to_the_closest_neighbour).
                A value of 1.5 is generally robust for hexagonal-like lattices.

        Returns:
            A tuple of numpy arrays: (ioa_minor_rad, ioa_major_rad, tilts_rad, neighbour_counts)
        """
        if self.ommatidia_count <= k:
            zeros = np.zeros(self.ommatidia_count, dtype=np.float32)
            return zeros, zeros, zeros, np.zeros(self.ommatidia_count, dtype=np.uint32)

        # Calculate physical direction vectors from a common center
        all_origins = self.data['origin'][:, :3]
        eye_center = np.mean(all_origins, axis=0)
        phys_dirs = all_origins - eye_center
        phys_dirs /= np.linalg.norm(phys_dirs, axis=1, keepdims=True)

        # Query for k+1 neighbours (point itself is the first)
        phys_kdtree = KDTree(phys_dirs)
        distances, indices = phys_kdtree.query(phys_dirs, k=k + 1)
        neighbour_indices = indices[:, 1:]
        neighbour_distances = distances[:, 1:]

        if neighbour_indices.size == 0:
            zeros = np.zeros(self.ommatidia_count, dtype=np.float32)
            return zeros, zeros, zeros, np.zeros(self.ommatidia_count, dtype=np.uint32)

        # Convert Euclidean distance on unit sphere to angular separation
        angular_separations = 2.0 * np.arcsin(np.clip(neighbour_distances / 2.0, -1.0, 1.0))

        # Count immediate neighbours
        dist_to_closest = angular_separations[:, 0]
        is_immediate_neighbour = angular_separations <= dist_to_closest[:, np.newaxis] * neighbour_dist_factor
        neighbour_counts = np.sum(is_immediate_neighbour, axis=1)

        # Determine tilt and minor/major IOA
        # Define local coordinate systems (tangent planes) for each ommatidium
        dot_products = np.abs(np.dot(phys_dirs, WORLD_UP))
        is_polar = dot_products > 0.9999
        ref_up_vectors = np.where(is_polar[:, np.newaxis], WORLD_RIGHT, WORLD_UP)
        local_y_axes = ref_up_vectors - phys_dirs * np.sum(phys_dirs * ref_up_vectors, axis=1, keepdims=True)
        local_y_axes /= np.linalg.norm(local_y_axes, axis=1, keepdims=True)
        local_x_axes = np.cross(local_y_axes, phys_dirs)

        # Project neighbour vectors onto the local tangent planes
        neighbour_phys_dirs = phys_dirs[neighbour_indices]
        delta_vectors = neighbour_phys_dirs - phys_dirs[:, np.newaxis, :]
        proj_x = np.sum(delta_vectors * local_x_axes[:, np.newaxis, :], axis=2)
        proj_y = np.sum(delta_vectors * local_y_axes[:, np.newaxis, :], axis=2)

        tilts_rad = np.zeros(self.ommatidia_count, dtype=np.float32)
        ioa_major_arr = np.zeros(self.ommatidia_count, dtype=np.float32)
        ioa_minor_arr = np.zeros(self.ommatidia_count, dtype=np.float32)

        for i in range(self.ommatidia_count):
            # Use only immediate neighbours for PCA
            immediate_mask = is_immediate_neighbour[i]
            points = np.vstack([proj_x[i, immediate_mask], proj_y[i, immediate_mask]]).T

            if points.shape[0] < 2:  # Not enough neighbours for PCA
                # Fallback to a simple average if PCA is not possible
                avg_angle = np.mean(angular_separations[i, immediate_mask]) if np.any(immediate_mask) else 0.0
                ioa_major_arr[i], ioa_minor_arr[i], tilts_rad[i] = avg_angle, avg_angle, 0.0
                continue

            # Compute covariance and find the principal axis
            cov_matrix = np.cov(points, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            primary_axis_vector = eigenvectors[:, np.argmax(eigenvalues)]

            # The tilt is the angle of this principal axis
            tilts_rad[i] = np.arctan2(primary_axis_vector[1], primary_axis_vector[0])

            # Rotate projected points to align with the new principal axes
            cos_tilt, sin_tilt = np.cos(-tilts_rad[i]), np.sin(-tilts_rad[i])
            aligned_proj_x = proj_x[i, immediate_mask] * cos_tilt - proj_y[i, immediate_mask] * sin_tilt
            aligned_proj_y = proj_x[i, immediate_mask] * sin_tilt + proj_y[i, immediate_mask] * cos_tilt
            neighbour_angles_aligned = np.arctan2(aligned_proj_y, aligned_proj_x)

            # Find neighbours closest to the new major (aligned x) and minor (aligned y) axes
            masked_angular_seps = angular_separations[i, immediate_mask]
            major_indices = np.argsort(np.abs(np.sin(neighbour_angles_aligned)))[:2]
            minor_indices = np.argsort(np.abs(np.cos(neighbour_angles_aligned)))[:2]

            # Average the angles of the two best-matching neighbours for each axis
            ioa_major_arr[i] = np.mean(masked_angular_seps[major_indices])
            ioa_minor_arr[i] = np.mean(masked_angular_seps[minor_indices])

        # Ensure minor is always the smaller value for consistency
        final_ioa_minor = np.minimum(ioa_minor_arr, ioa_major_arr)
        final_ioa_major = np.maximum(ioa_minor_arr, ioa_major_arr)

        return final_ioa_minor, final_ioa_major, tilts_rad, neighbour_counts.astype(np.uint32)

    def rebuild_spatial(self):
        """
        Rebuilds the internal KDTrees for positional and directional queries.
        No-op if rebuild is not needed.
        """
        if self.needs_rebuild['direction']:
            self.kdtree_directions = KDTree(self.data['direction'][:, :3])
            self.needs_rebuild['direction'] = False

        if self.needs_rebuild['origin']:
            self.kdtree_positions = KDTree(self.data['origin'][:, :3])
            self.needs_rebuild['origin'] = False

    def max_gap(self):
        """
        Finds the maximum angular distance between any ommatidium and its
        single nearest neighbour, which represents the largest "gap" in the eye.
        """

        if self.ommatidia_count == 1:
            return 0.0

        distances, _ = self.kdtree_directions.query(self.data['direction'][:, :3], k=2)
        max_euclidean_dist = np.max(distances[:, 1])
        term = 1.0 - (max_euclidean_dist ** 2) / 2.0
        return np.arccos(np.clip(term, -1.0, 1.0))

    def query_directions(self, directions: ArrayLike, k: int = 1) -> np.ndarray:
        """
        Finds ommatidia whose viewing direction is closest to the given direction vector(s).

        Args:
            directions: A (3,) vector or an (N, 3) array of direction vectors.
            k: The number of nearest matches to return for each input direction.

        Returns:
            If input is a single vector: An integer index (if k=1) or a (k,) array of indices.
            If input is an array: A (N,) array of indices (if k=1) or an (N, k) array.
        """
        if k < 1:
            raise ValueError("k must be a positive integer.")

        self.rebuild_spatial()

        query_dirs = np.asarray(directions, dtype=np.float32)
        is_single_query = query_dirs.ndim == 1
        query_dirs_2d = np.atleast_2d(query_dirs)

        # normalise
        norms = np.linalg.norm(query_dirs_2d, axis=-1, keepdims=True)
        np.divide(query_dirs_2d, norms, out=query_dirs_2d, where=norms != 0)

        distances, indices = self.kdtree_directions.query(query_dirs_2d, k=k)

        if is_single_query and k == 1:
            return indices.item()
        return indices.squeeze()

    def query_position(self, positions: ArrayLike, k: int = 1) -> np.ndarray:
        """
        Finds ommatidia whose origin is closest to the given point(s) in space.

        Args:
            positions: A (3,) vector or an (N, 3) array of points.
            k: The number of nearest ommatidia to return for each input point.

        Returns:
            If input is a single point: An integer index (if k=1) or a (k,) array of indices.
            If input is an array: A (N,) array of indices (if k=1) or an (N, k) array.
        """
        if k < 1:
            raise ValueError("k must be a positive integer.")

        self.rebuild_spatial()

        query_pos = np.asarray(positions, dtype=np.float32)
        is_single_query = query_pos.ndim == 1
        query_pos_2d = np.atleast_2d(query_pos)

        distances, indices = self.kdtree_positions.query(query_pos_2d, k=k)

        if is_single_query and k == 1:
            return indices.item()
        return indices.squeeze()

    def query_lookat(self, targets: ArrayLike, k: int = 1) -> np.ndarray:
        """
        Finds ommatidia whose viewing direction best aligns with one or several target points.
        """
        if k < 1:
            raise ValueError("k must be a positive integer.")

        self.rebuild_spatial()

        query_targets = np.asarray(targets, dtype=np.float32)
        is_single_query = query_targets.ndim == 1
        query_targets_2d = np.atleast_2d(query_targets)

        # Calculate desired direction from each ommatidium to each target
        desired_vectors = query_targets_2d[:, np.newaxis, :] - self.data['origin'][:, :3][np.newaxis, :, :]
        norms = np.linalg.norm(desired_vectors, axis=-1, keepdims=True)
        np.divide(desired_vectors, norms, out=desired_vectors, where=norms != 0)

        # higher dot product == smaller angle == better alignment
        dot_products = np.einsum('jk,ijk->ij', self.data['direction'][:, :3], desired_vectors)

        # Get top k indices for each target (each row)
        partition_indices = np.argpartition(dot_products, -k, axis=1)[:, -k:]

        # Sort (only the top k) indices based on their dot product values
        top_k_dots = np.take_along_axis(dot_products, partition_indices, axis=1)
        sorted_top_k_indices = np.argsort(top_k_dots, axis=1)[:, ::-1]  # sort descending
        best_indices = np.take_along_axis(partition_indices, sorted_top_k_indices, axis=1)

        if is_single_query and k == 1:
            return best_indices.item()

        return best_indices.squeeze()

    def query_directions_angle(self, center_direction: ArrayLike, angle: float, degrees: bool = True) -> np.ndarray:
        """
        Finds all ommatidia whose viewing direction is within a given angle of a center direction.
        """

        self.rebuild_spatial()

        # Normalize the input direction to be safe
        center_direction = np.asarray(center_direction, dtype=np.float32)
        center_direction /= np.linalg.norm(center_direction)

        # Convert the search angle (cone radius) to a Euclidean distance (chord length) on the unit sphere
        angle_rad = np.deg2rad(angle) if degrees else angle
        radius = 2.0 * np.sin(angle_rad / 2.0)

        indices = self.kdtree_directions.query_ball_point(center_direction, r=radius)
        return indices

    def query_positions_radius(self, center_position: ArrayLike, radius: float) -> np.ndarray:
        """
        Finds all ommatidia whose origin is within a given radius of a centre point.
        """
        self.rebuild_spatial()

        center_position = np.asarray(center_position, dtype=np.float32)

        indices = self.kdtree_positions.query_ball_point(center_position, r=radius)
        return indices

    def __repr__(self):

        nb_om = self.ommatidia_count
        nb_rhab = self.receptor_count

        if self.is_rhabdomeres:

            summary = [f"<CompoundEye with {nb_om/nb_rhab}x{nb_rhab} rhabdomeres>"]
        else:
            summary = [f"<CompoundEye with {nb_om} ommatidia>"]

        # TODO: Add orientation?

        # Interommatidial Angles (Δφ)
        d_phi_minor_deg = np.rad2deg(self.ioa_minor_rad)
        d_phi_major_deg = np.rad2deg(self.ioa_major_rad)
        summary.append("  Interommatidial Angles (Δφ):")
        summary.append(
            f"    Minor: {np.mean(d_phi_minor_deg):.3f}° (mean), {np.min(d_phi_minor_deg):.3f}° (min), {np.max(d_phi_minor_deg):.3f}° (max)")
        summary.append(
            f"    Major: {np.mean(d_phi_major_deg):.3f}° (mean), {np.min(d_phi_major_deg):.3f}° (min), {np.max(d_phi_major_deg):.3f}° (max)")

        # Acceptance Angles (Δρ)
        angles_deg = np.rad2deg(self.data['acceptance_angles'])
        means = np.mean(angles_deg, axis=0)
        mins = np.min(angles_deg, axis=0)
        maxs = np.max(angles_deg, axis=0)
        summary.append("  Acceptance Angles (Δρ):")
        summary.append(f"    Minor: {means[0]:.3f}° (mean), {mins[0]:.3f}° (min), {maxs[0]:.3f}° (max)")
        summary.append(f"    Major:   {means[1]:.3f}° (mean), {mins[1]:.3f}° (min), {maxs[1]:.3f}° (max)")

        # Eye Parameter (p)
        summary.append("  Eye Parameter (p = Δρ/Δφ):")
        summary.append(f"    Minor: {np.mean(self.eye_parameter_minor):.2f} (mean)")
        summary.append(f"    Major:   {np.mean(self.eye_parameter_major):.2f} (mean)")

        return "\n".join(summary)

    @property
    def is_rhabdomeres(self) -> bool:
        return self._is_rhabdomeres

    def to_rhabdomeres(self, layout: 'RhabdomereLayout'):

        if self._is_rhabdomeres:
            return

        # Store original lens lattice for collapse/referencing
        self._lens_data = self.data.copy()
        N = self.ommatidia_count
        R = len(layout.offsets_um)

        new_data = np.zeros(N * R, dtype=GPU_OMMATIDIUM_DTYPE)

        lens_fwd = self.data['direction'][:, :3]
        lens_pos = self.data['origin'][:, :3]

        # tangent space
        dots = np.abs(lens_fwd @ WORLD_UP)
        ref_ups = np.where(dots[:, np.newaxis] > 0.9999, WORLD_RIGHT, WORLD_UP)
        local_right = np.cross(lens_fwd, ref_ups)
        local_right /= np.linalg.norm(local_right, axis=1, keepdims=True)
        local_up = np.cross(local_right, lens_fwd)

        # dx, dy = physical micrometres behind the lens
        dx = layout.offsets_um[:, 0]
        dy = layout.offsets_um[:, 1]
        f = layout.focal_length_um

        is_ventral = lens_pos[:, 1] < 0 # TODO: This should be a property maybe?

        # Local vectors (from lens centre to receptor tip)
        local_tip_offsets = np.tile(np.column_stack([dx, dy, np.full(R, -f)]), (N, 1, 1))
        local_tip_offsets[is_ventral, :, 1] *= -1  # equator flip

        # to world space
        world_tip_offsets = (
                local_tip_offsets[..., 0:1] * local_right[:, None, :] +
                local_tip_offsets[..., 1:2] * local_up[:, None, :] +
                local_tip_offsets[..., 2:3] * lens_fwd[:, None, :]
        ).reshape(-1, 3)

        # Origin is receptor tip
        new_data['origin'][:, :3] = np.repeat(lens_pos, R, axis=0) + world_tip_offsets
        new_data['origin'][:, 3] = 1.0

        # Direction is the vector from the tip through the lens centre
        # (which is -world_tip_offsets normalised)
        new_dirs = -world_tip_offsets
        new_dirs /= np.linalg.norm(new_dirs, axis=1, keepdims=True)
        new_data['direction'][:, :3] = new_dirs

        # inherit the Δφ and Tilt from the parent lens lattice
        # (allows the L-cell neurons to know the lens spacing even in rhabdomere mode)
        new_data['interommatidial_angles'] = np.repeat(self.data['interommatidial_angles'], R, axis=0)
        new_data['tilt'] = np.repeat(self.data['tilt'], R)
        new_data['acceptance_angles'][:, 0] = np.tile(layout.diameters_um / f, N)
        new_data['acceptance_angles'][:, 1] = np.tile(layout.diameters_um / f, N)

        eye_ids = np.repeat(self.ommatidia.eye_id, R)
        receptor_types = np.tile(np.arange(R), N)
        parent_ids = np.repeat(np.arange(N), R)
        new_data['packed_data'] = (eye_ids & 0x07) | ((receptor_types & 0x0F) << 3) | ((parent_ids & 0xFFFF) << 11)

        self.data = new_data
        self.ommatidia_count = N * R    # TODO: maybe this should always store the lens count
        self.receptor_count = R
        self._is_rhabdomeres = True

        self.ommatidia = OmmatidiaCollection(self.data, self)

        self.needs_rebuild['direction'] = True
        self.needs_rebuild['origin'] = True
        self.rebuild_spatial()

    def from_rhabdomeres(self):
        """
        Collapses the receptor array back into a single lens lattice.
        """
        if not self._is_rhabdomeres or self._lens_data is None:
            print("Eye is not in rhabdomere mode.")
            return

        # Restore cached lens lattice
        self.data = self._lens_data
        self.ommatidia_count = len(self.data)
        self.receptor_count = 1
        self._is_rhabdomeres = False
        self._lens_data = None

        self.ommatidia = OmmatidiaCollection(self.data, self)

        self.needs_rebuild['direction'] = True
        self.needs_rebuild['origin'] = True
        self.rebuild_spatial()


##

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
    Defines the base (z-axis aligned) icosahedron and returns the vertices for the 20 triangular faces.
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
    """Subdivides icosahedron using barycentric coordinates."""

    verts = icosahedron_faces()
    bary = barycentric_coords(n_subdiv)

    # Barycentric interpolation to each of the 20 triangles
    # 'ij,kjl->kil': i=bary_idx, j=bary_coord, k=tri_idx, l=vertex_coord
    all_new_verts = np.einsum('ij,kjl->kil', bary, verts)
    # Normalize to unit sphere and find unique vertices
    all_new_verts = all_new_verts.reshape(-1, 3)
    all_new_verts /= np.linalg.norm(all_new_verts, axis=1)[:, np.newaxis]
    _, iunique = np.unique(np.round(all_new_verts, 6), axis=0, return_index=True)

    return all_new_verts[iunique].astype(np.float32)