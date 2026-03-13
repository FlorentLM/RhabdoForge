import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union, Sequence, List
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import KDTree

from insectvision.engine.utils import WORLD_UP, WORLD_RIGHT


GPU_RECEPTOR_DTYPE = np.dtype([
    ('origin', np.float32, 4),                  # 16 bytes: x, y, z, w=1
    ('direction', np.float32, 4),               # 16 bytes: x, y, z, w=0
    ('acceptance_angles', np.float32, 2),       #  8 bytes: minor, major
    ('interommatidial_angles', np.float32, 2),  #  8 bytes: minor, major (from parent lens)
    ('tilt', np.float32),                       #  4 bytes: ellipse tilt (lattice geometry)
    ('sensitivity', np.float32),                #  4 bytes: receptor sensitivity
    ('packed_data', np.uint32),                 #  4 bytes: see below
    ('padding', np.uint32)                      #  4 bytes
])  # total = 64 bytes

# packed_data layout:
# bits 0-2: eye ID (0-7)
# bits 3-6: receptor type (0-15) R1=0, R2=1, ...
# bits 7-10: neighbour count (0-15)
# bits 11-26: lens index (0-65535) parent ommatidium
# bits 27-31: unused for now

# TODO: Receptor dtype coul dbe 48 bytes if IOA and tilt were a separate Lens struct


_CLEAR_EYE_ID = np.uint32(0xFFFFFFF8)
_CLEAR_RECEPTOR_TYPE = np.uint32(0xFFFFFF87)
_CLEAR_NEIGHBOURS = np.uint32(0xFFFFF87F)
_CLEAR_LENS_INDEX = np.uint32(0xF80007FF)

DEFAULT_ANGLE = 'deg'   # TODO: get rid of this, and ensure unit consistency everywhere


@dataclass
class RhabdomereKernel:
    """
    Model of thr rhabdomere bundle inside a single ommatidium.

    The offsets describe each rhabdomere's position behind the lens in the focal plane (micrometres).

    - `nodal_distance_um`: physical distance from the lens inner surface (nodal point) to the rhabdomere tips (at rest).
        This is the lever arm that converts a lateral displacement (μm) into an angular shift
        For Drosophila this is ~20-21 μm (Kemppainen et al. 2022, 10.1073/pnas.2109717119; Stavenga 2003, 10.1007/s00359-002-0370-2)
        https://github.com/JuusolaLab/Hyperacute_Stereopsis_paper/blob/main/CG-Compound-Eye/model_init.py#L213
    - `main_axis` is the main axis of the bundle, defined as the R3-R6 axis
    - `saccade_axis_deg`: angular offset (from the main axis) of the microsaccade actuation direction
        In world space the full actuation angle is `chi + main_axis + saccade_axis_deg`
    """
    name: str
    offsets_um: np.ndarray          # (R, 2) XY in focal plane
    nodal_distance_um: float        # lens to rhabdomere tip distance (at rest)
    diameters_um: np.ndarray        # (R,) waveguide diameter
    lens_diameter_um: float         # facet lens diameter (for diffraction term)
    saccade_axis_deg: float = 0.0   # microsaccade axis delta (relative to main_axis)

    @property
    def count(self) -> int:
        return len(self.offsets_um)

    @property
    def center(self) -> np.ndarray:
        """Central rhabdomere position (R7/R8)."""
        return self.offsets_um[6]

    @property
    def main_axis_deg(self) -> float:
        """Bundle axis angle (R3->R6 line) in degrees."""
        delta = self.offsets_um[5] - self.offsets_um[2]
        return float(np.degrees(np.arctan2(delta[1], delta[0])))

    @property
    def actuation_angle_deg(self) -> float:
        """Full actuation angle in the kernel's local frame (degrees)."""
        return self.main_axis_deg + self.saccade_axis_deg

    def plot(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        for r, pos in enumerate(self.offsets_um):
            col = plt.cm.tab10(r / 8)
            txt = "R7/R8" if r == 6 else f"R{r + 1}"
            circle = plt.Circle(pos, self.diameters_um[r] / 2,
                                alpha=0.3, color=col, label=txt)
            ax.add_patch(circle)
            ax.scatter(*pos, color=col, s=10)

        angle_rad = np.radians(self.actuation_angle_deg)
        dx, dy = np.cos(angle_rad), np.sin(angle_rad)
        ax.axline(self.center, slope=dy / dx, color='black', linestyle='--', alpha=0.7, label="Microsaccade axis")

        ax.set_aspect('equal')
        ax.set_title(f'Rhabdomere layout ({self.name})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.show()

## Move this to math utils

def rotate_vectors(vectors: np.ndarray, axes: np.ndarray, angles: np.ndarray, degrees: bool = True) -> np.ndarray:
    """Rotate vectors around axes (Rodrigues formula)."""

    angles_arr = np.asarray(angles)
    angles_rad = np.deg2rad(angles_arr) if degrees else angles_arr

    if angles_rad.ndim == 0:
        cos_a = np.cos(angles_rad)
        sin_a = np.sin(angles_rad)
    else:
        cos_a = np.cos(angles_rad)[:, np.newaxis]
        sin_a = np.sin(angles_rad)[:, np.newaxis]

    rotated = (
            vectors * cos_a
            + np.cross(axes, vectors) * sin_a
            + axes * np.sum(axes * vectors, axis=1, keepdims=True) * (1 - cos_a)
    )
    return rotated

##
# TODO: Move these to a geometric utils module

# Lattice property estimation (used by both construction paths)

def _compute_lattice_properties(
        directions: np.ndarray,
        origins: np.ndarray,
        k: int = 8,
        neighbour_dist_factor: float = 1.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate local lattice properties from lens positions.
    """

    N = len(directions)
    if N <= k:
        z = np.zeros(N, dtype=np.float32)
        return z, z, z, np.zeros(N, dtype=np.uint32)

    # Physical direction vectors from common centre
    eye_center = np.mean(origins, axis=0)
    phys_dirs = origins - eye_center
    norms = np.linalg.norm(phys_dirs, axis=1, keepdims=True)
    np.divide(phys_dirs, norms, out=phys_dirs, where=norms != 0)

    phys_kdtree = KDTree(phys_dirs)
    distances, indices = phys_kdtree.query(phys_dirs, k=k + 1)
    nb_indices = indices[:, 1:]
    nb_distances = distances[:, 1:]

    if nb_indices.size == 0:
        z = np.zeros(N, dtype=np.float32)
        return z, z, z, np.zeros(N, dtype=np.uint32)

    angular_sep = 2.0 * np.arcsin(np.clip(nb_distances / 2.0, -1.0, 1.0))
    dist_to_closest = angular_sep[:, 0]
    is_immediate = angular_sep <= dist_to_closest[:, np.newaxis] * neighbour_dist_factor
    nb_counts = np.sum(is_immediate, axis=1)

    # Local tangent planes
    dot_up = np.abs(phys_dirs @ WORLD_UP)
    is_polar = dot_up > 0.9999
    ref_ups = np.where(is_polar[:, np.newaxis], WORLD_RIGHT, WORLD_UP)

    local_y = ref_ups - phys_dirs * np.sum(phys_dirs * ref_ups, axis=1, keepdims=True)
    local_y /= np.linalg.norm(local_y, axis=1, keepdims=True)
    local_x = np.cross(local_y, phys_dirs)

    nb_phys = phys_dirs[nb_indices]
    delta = nb_phys - phys_dirs[:, np.newaxis, :]

    proj_x = np.sum(delta * local_x[:, np.newaxis, :], axis=2)
    proj_y = np.sum(delta * local_y[:, np.newaxis, :], axis=2)

    tilts = np.zeros(N, dtype=np.float32)
    ioa_major = np.zeros(N, dtype=np.float32)
    ioa_minor = np.zeros(N, dtype=np.float32)

    for i in range(N):

        mask = is_immediate[i]
        pts = np.column_stack([proj_x[i, mask], proj_y[i, mask]])

        if pts.shape[0] < 2:
            avg = np.mean(angular_sep[i, mask]) if np.any(mask) else 0.0
            ioa_major[i], ioa_minor[i], tilts[i] = avg, avg, 0.0
            continue

        cov = np.cov(pts, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        primary = evecs[:, np.argmax(evals)]
        tilts[i] = np.arctan2(primary[1], primary[0])

        ct, st = np.cos(-tilts[i]), np.sin(-tilts[i])
        ax = proj_x[i, mask] * ct - proj_y[i, mask] * st
        ay = proj_x[i, mask] * st + proj_y[i, mask] * ct
        angles_aligned = np.arctan2(ay, ax)

        sep = angular_sep[i, mask]

        maj_idx = np.argsort(np.abs(np.sin(angles_aligned)))[:2]
        min_idx = np.argsort(np.abs(np.cos(angles_aligned)))[:2]

        ioa_major[i] = np.mean(sep[maj_idx])
        ioa_minor[i] = np.mean(sep[min_idx])

    final_minor = np.minimum(ioa_minor, ioa_major)
    final_major = np.maximum(ioa_minor, ioa_major)

    return final_minor, final_major, tilts, nb_counts.astype(np.uint32)


# Tangent frame computation (shared by from_build and actuate)

def _compute_tangent_frames(forward: np.ndarray):
    """
    Compute per-lens orthonormal tangent frames.
    """
    # TODO: Can this gimbal lock??

    dots = np.abs(forward @ WORLD_UP)
    ref_ups = np.where(dots[:, np.newaxis] > 0.9999, WORLD_RIGHT, WORLD_UP)
    local_right = np.cross(forward, ref_ups)
    local_right /= np.linalg.norm(local_right, axis=1, keepdims=True)
    local_up = np.cross(local_right, forward)
    return local_right, local_up


##

# Proxy interfaces

class Receptor:
    """
    View into one or more elements of a ReceptorArray.
    """

    def __init__(self, data_array: np.ndarray, item, parent_array: 'ReceptorArray'):
        self._data = data_array
        self._item = item
        self._parent = parent_array

    def __len__(self):
        return 1 if self._data[self._item].ndim == 0 else self._data[self._item].shape[0]

    def __repr__(self):
        if isinstance(self._item, (int, np.int_)):
            o = np.array2string(self.origin, precision=3, suppress_small=True)
            d = np.array2string(self.direction, precision=3, suppress_small=True)
            return f"<Receptor(idx={int(self._item)}, origin={o}, direction={d})>"
        return f"<Receptors(key={self._item}, count={len(self)})>"

    # Spatial properties

    @property
    def origin(self) -> np.ndarray:
        return self._data[self._item]['origin'][..., :3]

    @origin.setter
    def origin(self, value: Union[float, ArrayLike]):

        self._data['origin'][self._item, :3] = np.asarray(value, dtype=np.float32)
        self._data['origin'][self._item, 3] = 1.0

        self._parent.dirty_mask[self._item] = True
        self._parent._stale_receptor_spatial = True

    @property
    def direction(self) -> np.ndarray:
        return self._data[self._item]['direction'][..., :3]

    @direction.setter
    def direction(self, value: Union[float, ArrayLike]):

        new_dirs = np.atleast_2d(value)
        norms = np.linalg.norm(new_dirs, axis=-1, keepdims=True)
        np.divide(new_dirs, norms, out=new_dirs, where=norms != 0)

        self._data['direction'][self._item, :3] = new_dirs
        self._data['direction'][self._item, 3] = 0.0

        self._parent.dirty_mask[self._item] = True
        self._parent._stale_receptor_spatial = True

    # Acceptance / sensitivity

    @property
    def acceptance_major(self) -> np.ndarray:
        return self._data[self._item]['acceptance_angles'][..., 0]

    @acceptance_major.setter
    def acceptance_major(self, value: Union[float, ArrayLike]):
        self._data['acceptance_angles'][self._item, 0] = value
        self._parent.dirty_mask[self._item] = True

    @property
    def acceptance_minor(self) -> np.ndarray:
        return self._data[self._item]['acceptance_angles'][..., 1]

    @acceptance_minor.setter
    def acceptance_minor(self, value: Union[float, ArrayLike]):
        self._data['acceptance_angles'][self._item, 1] = value
        self._parent.dirty_mask[self._item] = True

    @property
    def acceptance_rad(self) -> np.ndarray:
        return self._data[self._item]['acceptance_angles']

    @acceptance_rad.setter
    def acceptance_rad(self, values: Union[float, ArrayLike]):
        self._data['acceptance_angles'][self._item] = values
        self._parent.dirty_mask[self._item] = True

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
        self._parent.dirty_mask[self._item] = True

    # Metadata

    @property
    def eye_id(self) -> np.ndarray:
        return self._data[self._item]['packed_data'] & 0x07

    @eye_id.setter
    def eye_id(self, value: Union[int, ArrayLike]):
        v = np.asarray(value, dtype=np.uint32)
        cur = self._data['packed_data'][self._item]
        self._data['packed_data'][self._item] = (cur & _CLEAR_EYE_ID) | (v & 0x07)
        self._parent.dirty_mask[self._item] = True

    @property
    def receptor_type(self) -> np.ndarray:
        return (self._data[self._item]['packed_data'] >> 3) & 0x0F

    @receptor_type.setter
    def receptor_type(self, value: Union[int, ArrayLike]):
        v = np.asarray(value, dtype=np.uint32)
        cur = self._data['packed_data'][self._item]
        self._data['packed_data'][self._item] = (cur & _CLEAR_RECEPTOR_TYPE) | ((v & 0x0F) << 3)
        self._parent.dirty_mask[self._item] = True

    @property
    def neighbours_count(self) -> np.ndarray:
        return (self._data[self._item]['packed_data'] >> 7) & 0x0F

    @neighbours_count.setter
    def neighbours_count(self, value: Union[int, ArrayLike]):
        v = np.asarray(value, dtype=np.uint32)
        cur = self._data['packed_data'][self._item]
        self._data['packed_data'][self._item] = (cur & _CLEAR_NEIGHBOURS) | ((v & 0x0F) << 7)
        self._parent.dirty_mask[self._item] = True

    @property
    def lens_index(self) -> np.ndarray:
        """Index of the parent ommatidium in the lens-level array."""
        return (self._data[self._item]['packed_data'] >> 11) & 0xFFFF

    @lens_index.setter
    def lens_index(self, value: Union[int, ArrayLike]):
        v = np.asarray(value, dtype=np.uint32)
        cur = self._data['packed_data'][self._item]
        self._data['packed_data'][self._item] = (cur & _CLEAR_LENS_INDEX) | ((v & 0xFFFF) << 11)
        self._parent.dirty_mask[self._item] = True

    # Convenience angular accessors

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

    lon = longitude = azimuth = azimuth_rad if DEFAULT_ANGLE == 'rad' else azimuth_deg
    lat = latitude = elevation = elevation_rad if DEFAULT_ANGLE == 'rad' else elevation_deg
    rho = acceptance = acceptance_rad if DEFAULT_ANGLE == 'rad' else acceptance_deg
    rho_minor = acceptance_minor
    rho_major = acceptance_major


class Ommatidium:
    """
    Grouping of the R receptors behind a single lens.
    """

    def __init__(self, array: 'ReceptorArray', lens_index: int):
        self._array = array
        self._lens_index = int(lens_index)

        R = array.receptor_count

        self._start = self._lens_index * R
        self._stop = self._start + R
        self._slice = slice(self._start, self._stop)

    def __getitem__(self, receptor_idx) -> Receptor:
        """``omm[r]`` returns the Receptor proxy for receptor type r"""

        if isinstance(receptor_idx, (int, np.integer)):
            return Receptor(self._array.data,
                            self._start + int(receptor_idx),
                            self._array)

        indices = np.arange(self._start, self._stop)[receptor_idx]
        return Receptor(self._array.data, indices, self._array)

    def __len__(self) -> int:
        return self._array.receptor_count

    def __iter__(self):
        for k in range(len(self)):
            yield self[k]

    def __repr__(self):
        return (f"<Ommatidium(eye={self.eye_id}, lens={self._lens_index}, {len(self)} receptors)>")

    @property
    def optical_axis(self) -> np.ndarray:
        """Unit direction of the lens (R7/R8 axis)."""
        return self._array._lens_directions[self._lens_index]

    @property
    def position(self) -> np.ndarray:
        """Lens centre position."""
        return self._array._lens_positions[self._lens_index]

    @property
    def eye_id(self) -> int:
        return int(self._array.data['packed_data'][self._start] & 0x07)

    @property
    def bundle_orientation(self) -> float:
        """Rotation of rhabdomere bundle in tangent plane (radians).""" # TODO: maybe return degrees instead?
        return float(self._array._bundle_orientation[self._lens_index])

    @property
    def receptors(self) -> Receptor:
        """Proxy spanning all R receptors of this ommatidium."""
        return Receptor(self._array.data, self._slice, self._array)

    def actuate(self, displacement_um: float, axial_um: float = 0.0):
        """
        Displace all receptors via microsaccade actuation.

        Args:
            displacement_um: lateral shift in focal plane (um), absolute from rest.
            axial_um: axial contraction toward lens (um), positive.
        """
        self._array.actuate(
            np.float32(displacement_um),
            axial_um=np.float32(axial_um),
            lens_mask=np.array([self._lens_index])
        )


class Cartridge:
    """
    Neural superposition unit: The 6 outer receptors (R1-R6) from 6 different ommatidia
    that converge onto one lamina column.
    """

    def __init__(self, array: 'ReceptorArray', lens_index: int):
        self._array = array
        self._lens_index = int(lens_index)

        if array._cartridge_map is None:
            raise RuntimeError("Cartridge map not built. Call array.build_cartridge_map() first.")  # TODO: may eventually be done automatically

    def __getitem__(self, receptor_type: int) -> Receptor:
        """``cartridge[k]`` returns the Receptor R{k+1} from the appropriate ommatidium."""
        source_lens = self._array._cartridge_map[self._lens_index, receptor_type]
        global_idx = source_lens * self._array.receptor_count + receptor_type
        return Receptor(self._array.data, global_idx, self._array)

    @property
    def receptor_indices(self) -> np.ndarray:
        """Global indices into ReceptorArray.data"""
        sources = self._array._cartridge_map[self._lens_index]
        R = self._array.receptor_count
        return sources * R + np.arange(min(6, R))

    @property
    def optical_axis(self) -> np.ndarray:
        return self._array._lens_directions[self._lens_index]

    def __len__(self) -> int:
        return min(6, self._array.receptor_count)

    def __repr__(self):
        return f"<Cartridge(lens={self._lens_index}, inputs={len(self)})>"


class Eye:
    """
    View into a ReceptorArray scoped to a a single eye_id.

    Spatial queries operate at the **lens** level and return **lens-local** indices (0 ... len-1).
    Use `.global_lens_indices` to map back to the parent ReceptorArray's lens numbering.
    """

    def __init__(self, array: 'ReceptorArray', eye_id: int):
        self._array = array
        self._eye_id = eye_id
        R = array.receptor_count

        # lenses belonging to this eye
        first_ids = array.data['packed_data'][::R] & 0x07
        self._lens_mask = first_ids == eye_id
        self._lens_indices = np.where(self._lens_mask)[0]  # global lens indices

        # All receptor indices for this eye
        li = self._lens_indices
        self._receptor_indices = (
            li[:, np.newaxis] * R + np.arange(R)[np.newaxis, :]
        ).ravel()

        # Spatial structures are lazy
        self._kdtree_directions = None
        self._kdtree_positions = None
        self._neighbour_graph = None

    @property
    def eye_id(self) -> int:
        return self._eye_id

    def __len__(self) -> int:
        """Number of ommatidia (lenses) in this eye."""
        return len(self._lens_indices)

    def __repr__(self):
        return f"<Eye(id={self._eye_id}, ommatidia={len(self)})>"

    def __iter__(self):
        for i in range(len(self)):
            yield self.ommatidium(i)

    def __getitem__(self, key) -> Union[Ommatidium, List[Ommatidium]]:
        """
        Index by lens-local index. Returns an Ommatidium group view.
        `eye[i]` -> the i-th ommatidium in this eye.
        """

        if isinstance(key, (int, np.integer)):
            return Ommatidium(self._array, int(self._lens_indices[key]))

        # Slice returns list of Ommatidium  # TODO: maybe returning a list is a bit dumb
        indices = self._lens_indices[key]
        return [Ommatidium(self._array, int(li)) for li in indices]

    def ommatidium(self, local_index: int) -> Ommatidium:
        """Explicit access: Lens-local index -> Ommatidium group."""
        return Ommatidium(self._array, int(self._lens_indices[local_index]))

    def cartridge(self, local_index: int) -> Cartridge:
        """Explicit access: Lens-local index -> Cartridge (neural superposition unit)."""
        return Cartridge(self._array, int(self._lens_indices[local_index]))

    # Spatial struct helpers

    def _ensure_kdtree_directions(self):
        if self._kdtree_directions is None or self._array._stale_lens_spatial:
            self._array._resolve_lens_spatial()
            self._kdtree_directions = KDTree(self.directions)
        return self._kdtree_directions

    def _ensure_kdtree_positions(self):
        if self._kdtree_positions is None or self._array._stale_lens_spatial:
            self._array._resolve_lens_spatial()
            self._kdtree_positions = KDTree(self.positions)
        return self._kdtree_positions

    def _invalidate(self):
        self._kdtree_directions = None
        self._kdtree_positions = None
        self._neighbour_graph = None

    # Bulk data views

    @property
    def directions(self) -> np.ndarray:
        """Optical axes for all lenses in this eye."""
        return self._array._lens_directions[self._lens_indices]

    @property
    def positions(self) -> np.ndarray:
        """Lens positions for all lenses in this eye."""
        return self._array._lens_positions[self._lens_indices]

    @property
    def global_lens_indices(self) -> np.ndarray:
        """Maps lens-local indices to global lens indices."""
        return self._lens_indices

    @property
    def global_indices(self) -> np.ndarray:
        """Maps to global receptor indices in ReceptorArray.data"""
        return self._receptor_indices

    @property
    def bundle_orientations(self) -> np.ndarray:
        """Receptors bundle orientations (chi) for each ommatidium in this eye."""
        return self._array._bundle_orientation[self._lens_indices]

    # Spatial queries (lens-level, return lens-local indices)
    # TODO: These are duplicated, could be taken out as pure functions

    def query_directions(self, directions: ArrayLike, k: int = 1) -> np.ndarray:
        """
        Find lenses with optical axis best aligned with some directions. Lens-local indices.
        """

        if k < 1:
            raise ValueError("k must be >= 1")

        kd = self._ensure_kdtree_directions()
        q = np.atleast_2d(np.asarray(directions, dtype=np.float32))

        norms = np.linalg.norm(q, axis=-1, keepdims=True)
        np.divide(q, norms, out=q, where=norms != 0)

        is_single = np.asarray(directions).ndim == 1
        _, idx = kd.query(q, k=k)

        if is_single and k == 1:
            return idx.item()

        return idx.squeeze()

    def query_position(self, positions: ArrayLike, k: int = 1) -> np.ndarray:
        """
        Find lenses closest to some positions (on the eye surface). Lens-local indices.
        """

        if k < 1:
            raise ValueError("k must be >= 1")

        kd = self._ensure_kdtree_positions()
        q = np.atleast_2d(np.asarray(positions, dtype=np.float32))

        is_single = np.asarray(positions).ndim == 1
        _, idx = kd.query(q, k=k)

        if is_single and k == 1:
            return idx.item()

        return idx.squeeze()

    def query_lookat(self, targets: ArrayLike, k: int = 1) -> np.ndarray:
        """
        Find lenses looking at some target points (world-space). Lens-local indices.
        """

        if k < 1:
            raise ValueError("k must be >= 1")

        q = np.atleast_2d(np.asarray(targets, dtype=np.float32))
        is_single = np.asarray(targets).ndim == 1

        dirs = self.directions
        origs = self.positions

        desired = q[:, np.newaxis, :] - origs[np.newaxis, :, :]
        norms = np.linalg.norm(desired, axis=-1, keepdims=True)
        np.divide(desired, norms, out=desired, where=norms != 0)

        dots = np.einsum('jk,ijk->ij', dirs, desired)

        part = np.argpartition(dots, -k, axis=1)[:, -k:]
        top = np.take_along_axis(dots, part, axis=1)

        order = np.argsort(top, axis=1)[:, ::-1]
        best = np.take_along_axis(part, order, axis=1)

        if is_single and k == 1:
            return best.item()

        return best.squeeze()

    def query_cone(self, center_direction: ArrayLike, angle: float, degrees: bool = True) -> np.ndarray:
        """
        Find all lenses within angle of a center direction. Lens-local indices.
        """

        kd = self._ensure_kdtree_directions()
        c = np.asarray(center_direction, dtype=np.float32)
        c = c / np.linalg.norm(c)
        a = np.deg2rad(angle) if degrees else angle
        return kd.query_ball_point(c, r=2.0 * np.sin(a / 2.0))

    def max_gap(self) -> float:
        """
        Largest angular gap between any lens and its nearest neighbour.
        """

        if len(self) <= 1:
            return 0.0
        kd = self._ensure_kdtree_directions()
        d, _ = kd.query(self.directions, k=2)
        return float(np.arccos(np.clip(1.0 - (np.max(d[:, 1]) ** 2) / 2.0, -1, 1)))

    # Directed neighbours graph (lens-level)

    def _build_neighbour_graph(self, k_search: int = 8):
        N = len(self)
        if N <= 1:
            self._neighbour_graph = {
                'proj_x': np.zeros((N, 0), np.float32),
                'proj_y': np.zeros((N, 0), np.float32),
                'angular_sep': np.zeros((N, 0), np.float32),
                'neighbour_local_indices': np.zeros((N, 0), np.intp),
                'local_x': np.zeros((N, 3), np.float32),
                'local_y': np.zeros((N, 3), np.float32),
                'k_search': 0
            }
            return

        k_eff = min(k_search, N - 1)
        kd = self._ensure_kdtree_directions()
        dirs = self.directions

        dists, kd_idx = kd.query(dirs, k=k_eff + 1)
        nb_idx = kd_idx[:, 1:]
        nb_dist = dists[:, 1:]
        angular_sep = 2.0 * np.arcsin(np.clip(nb_dist / 2.0, -1.0, 1.0))

        dot_up = np.abs(dirs @ WORLD_UP)
        is_polar = dot_up > 0.9999
        ref_ups = np.where(is_polar[:, np.newaxis], WORLD_RIGHT, WORLD_UP)
        local_y = ref_ups - dirs * np.sum(dirs * ref_ups, axis=1, keepdims=True)
        local_y /= np.linalg.norm(local_y, axis=1, keepdims=True)
        local_x = np.cross(local_y, dirs)

        nb_dirs = dirs[nb_idx]
        delta = nb_dirs - dirs[:, np.newaxis, :]
        proj_x = np.sum(delta * local_x[:, np.newaxis, :], axis=2)
        proj_y = np.sum(delta * local_y[:, np.newaxis, :], axis=2)

        self._neighbour_graph = {
            'proj_x': proj_x, 'proj_y': proj_y,
            'angular_sep': angular_sep,
            'neighbour_local_indices': nb_idx,
            'local_x': local_x, 'local_y': local_y,
            'k_search': k_eff
        }

    def directed_neighbours(self,
            direction: ArrayLike,
            k: int = 1,
            coordinate: str = 'spherical',
            return_weights: bool = False
        ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        For every lens in this eye, find k nearest neighbours along a *direction*.
        Returns (N,) of lens-local indices when k=1, otherwise returns (N, k).
        """

        if self._neighbour_graph is None:
            self._build_neighbour_graph()

        graph = self._neighbour_graph
        N = len(self)
        if N <= 1 or graph['k_search'] == 0:
            return np.zeros(N, dtype=np.intp) if k == 1 else np.zeros((N, k), dtype=np.intp)

        direction = np.asarray(direction, dtype=np.float32)
        local_x_bases = graph['local_x']
        local_y_bases = graph['local_y']

        if coordinate == 'spherical':
            d_az, d_el = direction[0], direction[1]
            dirs = self.directions.astype(np.float32)

            az = np.arctan2(dirs[:, 0], -dirs[:, 2])
            el = np.arcsin(np.clip(dirs[:, 1], -1.0, 1.0))
            cos_az, sin_az = np.cos(az), np.sin(az)
            cos_el, sin_el = np.cos(el), np.sin(el)

            az_grad = np.column_stack([cos_az * cos_el, np.zeros(N), sin_az * cos_el])
            el_grad = np.column_stack([-sin_az * sin_el, cos_el, cos_az * sin_el])

            target_world = d_az * az_grad + d_el * el_grad
            target_dx = np.sum(target_world * local_x_bases, axis=1)
            target_dy = np.sum(target_world * local_y_bases, axis=1)

        elif coordinate == 'cartesian':
            target_dx = local_x_bases @ direction
            target_dy = local_y_bases @ direction

        else:
            raise ValueError(f"Unknown coordinate '{coordinate}'. Use 'spherical' or 'cartesian'.")

        target_norms = np.sqrt(target_dx ** 2 + target_dy ** 2)
        zero = target_norms < 1e-12

        target_dx = np.where(zero, 1.0, target_dx / np.where(zero, 1.0, target_norms))
        target_dy = np.where(zero, 0.0, target_dy / np.where(zero, 1.0, target_norms))
        target_angle = np.arctan2(target_dy, target_dx)

        nb_angles = np.arctan2(graph['proj_y'], graph['proj_x'])
        angle_diff = (nb_angles - target_angle[:, np.newaxis] + np.pi) % (2 * np.pi) - np.pi

        score = np.abs(angle_diff)
        score[score > (np.pi / 2.0)] = 1e6

        nb_local = graph['neighbour_local_indices']
        if k == 1:
            best = np.argmin(score, axis=1)
            indices = nb_local[np.arange(N), best]
        else:
            k_eff = min(k, score.shape[1])
            top_k = np.argpartition(score, k_eff, axis=1)[:, :k_eff]
            top_scores = np.take_along_axis(score, top_k, axis=1)
            order = np.argsort(top_scores, axis=1)
            top_k_sorted = np.take_along_axis(top_k, order, axis=1)
            indices = nb_local[np.arange(N)[:, np.newaxis], top_k_sorted]

            if k > k_eff:
                indices = np.hstack([indices, np.zeros((N, k - k_eff), dtype=np.intp)])

        if return_weights:
            w = target_norms if k == 1 else np.tile(target_norms[:, None], (1, k))
            return indices, w

        return indices

    def actuate(self, displacement_um: Union[float, ArrayLike],
                axial_um: Union[float, ArrayLike] = 0.0):
        """
        Displace rhabdomeres for all ommatidia in this eye.

        Args:
            displacement_um: scalar or (N_eye,) lateral shift (um)
            axial_um: scalar or (N_eye,) axial contraction (um)
        """
        self._array.actuate(displacement_um, axial_um=axial_um,
                            lens_mask=self._lens_indices)

##

class ReceptorArray:
    """
    Flat (GPU-friendly) structured array of receptors for the renderer.
    Every element is one rhabdomere. The GPU traces rays for `len(data)` receptors.

    Construction modes:
    # TODO: This will probably change

    *Full model (R receptors per lens):
        array = ReceptorArray.from_build(directions=dirs,
                                        origins=origins,
                                        kernel=DROSOPHILA_KERNEL,
                                        bundle_orientation=chi, ...)

    *Simplified (1 receptor per lens, more or less just R7/8):
        array = ReceptorArray(directions=dirs, origins=origins, ...)
    """

    @classmethod
    def from_build(cls,
                   directions: ArrayLike,
                   origins: ArrayLike,
                   kernel: RhabdomereKernel,
                   bundle_orientation: ArrayLike,
                   eye_ids: Optional[ArrayLike] = None,
                   eye_parameter: Optional[Union[float, Tuple]] = None,
                   interommatidial_angles_rad: Optional[ArrayLike] = None,
                   sensitivities: Optional[Union[ArrayLike, float]] = None,
                   wavelength_nm: float = 500.0,
                   ) -> 'ReceptorArray':
        """
        Construct a full receptor array from lens-level geometry and a rhabdomere kernel.

        Each of the *N* lenses contains *R* receptors whose world-space directions are
        determined by the kernel offsets and rotated by the per-lens bundle orientation (chi).

        Acceptance angles are computed from the full optical model (Snyder 1979):

            Δρ = sqrt( (λ/D)² + (d_rhab/f)² )

        where λ = wavelength_nm, D = kernel.lens_diameter_um, d_rhab = kernel.diameters_um, f = kernel.nodal_distance_um

        This can be overridden with `eye_parameter`: p = delta_rho / delta_phi

        Args:
            directions: (N, 3) lens optical axes
            origins: (N, 3) lens positions in head space
            kernel: Species-level rhabdomere geometry
            bundle_orientation: (N,) chi per lens (radians in tangent plane)
            eye_ids: (N,) integer eye id per lens, 0-7
            eye_parameter: Optional p = delta_rho / delta_phi override
                Bypasses the optical formula and computes acceptance as p * IOA (as in the simplified path)
            interommatidial_angles_rad: (N,) or (N,2) if known, otherwise estimated
            sensitivities: scalar or (N,) per lens (tiled to receptors) # TODO: maybe this should be a receptor-level prop?
            wavelength_nm: light wavelength for diffraction term (default 500)
        """

        dirs = np.asarray(directions, dtype=np.float32)
        origs = np.asarray(origins, dtype=np.float32)
        chi = np.asarray(bundle_orientation, dtype=np.float32)
        N = len(dirs)
        R = kernel.count
        M = N * R
        d = kernel.nodal_distance_um  # lens to rhabdomere tips (at rest) = lever arm

        # Normalise lens directions
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        lens_dirs = dirs / norms
        lens_origins = origs

        # Lens-level lattice properties
        if interommatidial_angles_rad is not None:
            ioa_arr = np.asarray(interommatidial_angles_rad, dtype=np.float32)
            if ioa_arr.ndim == 1:
                ioa_minor = ioa_arr
                ioa_major = ioa_arr
            else:
                ioa_minor = np.minimum(ioa_arr[:, 0], ioa_arr[:, 1])
                ioa_major = np.maximum(ioa_arr[:, 0], ioa_arr[:, 1])
            lattice_tilts = np.zeros(N, dtype=np.float32)
            nb_counts = np.zeros(N, dtype=np.uint32)
        else:
            ioa_minor, ioa_major, lattice_tilts, nb_counts = \
                _compute_lattice_properties(lens_dirs, lens_origins)

        # Tangent frames
        local_right, local_up = _compute_tangent_frames(lens_dirs)

        # Rotate kernel offsets by chi (that is per lens)
        cos_chi = np.cos(chi)[:, np.newaxis]  # (N, 1)
        sin_chi = np.sin(chi)[:, np.newaxis]

        dx = kernel.offsets_um[:, 0]  # (R,)
        dy = kernel.offsets_um[:, 1]

        rot_dx = cos_chi * dx[np.newaxis, :] - sin_chi * dy[np.newaxis, :]  # (N, R)
        rot_dy = sin_chi * dx[np.newaxis, :] + cos_chi * dy[np.newaxis, :]

        # Local tip vectors in lens frame: (N, R, 3) as [right, up, fwd]
        local_tip = np.stack([rot_dx, rot_dy,
                              np.full((N, R), -d, dtype=np.float32)], axis=-1)

        world_tip = (
            local_tip[..., 0:1] * local_right[:, np.newaxis, :] +
            local_tip[..., 1:2] * local_up[:, np.newaxis, :] +
            local_tip[..., 2:3] * lens_dirs[:, np.newaxis, :]
        ).reshape(M, 3)

        # Receptor direction: from tip through lens centre = -world_tip
        rec_dirs = -world_tip
        rec_dirs /= np.linalg.norm(rec_dirs, axis=1, keepdims=True)

        # Receptor position: lens + tip offset
        rec_positions = np.repeat(lens_origins, R, axis=0) + world_tip

        data = np.zeros(M, dtype=GPU_RECEPTOR_DTYPE)
        data['origin'][:, :3] = rec_positions
        data['origin'][:, 3] = 1.0
        data['direction'][:, :3] = rec_dirs
        data['direction'][:, 3] = 0.0

        wavelength_um = wavelength_nm * 1e-3
        diffraction = wavelength_um / kernel.lens_diameter_um     # λ/D (scalar)
        geometric = kernel.diameters_um / d                       # d_rhab/f (R,)
        full_acceptance = np.sqrt(diffraction**2 + geometric**2)  # (R,)

        if eye_parameter is not None:
            p_min, p_maj = (eye_parameter, eye_parameter) if isinstance(eye_parameter, (float, int, np.number)) else eye_parameter
            data['acceptance_angles'][:, 0] = np.repeat(p_min * ioa_minor, R)
            data['acceptance_angles'][:, 1] = np.repeat(p_maj * ioa_major, R)

        else:
            data['acceptance_angles'][:, 0] = np.tile(full_acceptance, N)
            data['acceptance_angles'][:, 1] = np.tile(full_acceptance, N)

        data['interommatidial_angles'][:, 0] = np.repeat(ioa_minor, R)
        data['interommatidial_angles'][:, 1] = np.repeat(ioa_major, R)
        data['tilt'] = np.repeat(lattice_tilts, R)

        sens = 1.0 if sensitivities is None else sensitivities
        data['sensitivity'] = np.repeat(
            np.broadcast_to(np.float32(sens), N), R
        )

        # Packed metadata
        if eye_ids is not None:
            eid = np.repeat(np.asarray(eye_ids, dtype=np.uint32), R)
        else:
            eid = np.zeros(M, dtype=np.uint32)

        rtypes = np.tile(np.arange(R, dtype=np.uint32), N)
        lindex = np.repeat(np.arange(N, dtype=np.uint32), R)

        data['packed_data'] = (
            (eid & 0x07) |
            ((rtypes & 0x0F) << 3) |
            ((np.repeat(nb_counts, R) & 0x0F) << 7) |
            ((lindex & 0xFFFF) << 11)
        )

        # assemble object
        obj = object.__new__(cls)
        obj.data = data
        obj.lens_count = N
        obj.receptor_count = R

        obj._kernel = kernel
        obj._bundle_orientation = chi.copy()
        obj._lens_directions = lens_dirs.copy()
        obj._lens_positions = lens_origins.copy()
        obj._actuation_state = np.zeros(N, dtype=np.float32)
        obj._wavelength_nm = wavelength_nm

        obj._ioa_minor_rad = ioa_minor
        obj._ioa_major_rad = ioa_major
        obj._lattice_tilts = lattice_tilts

        obj._local_right = local_right
        obj._local_up = local_up

        with np.errstate(divide='ignore', invalid='ignore'):
            obj.eye_parameter_minor = data['acceptance_angles'][:, 0] / np.repeat(ioa_minor, R)
            obj.eye_parameter_major = data['acceptance_angles'][:, 1] / np.repeat(ioa_major, R)

        np.nan_to_num(obj.eye_parameter_minor, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(obj.eye_parameter_major, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        obj.dirty_mask = np.zeros(M, dtype=bool)
        obj._stale_receptor_spatial = False
        obj._stale_lens_spatial = False
        obj._kdtree_directions = None   # lazy
        obj._kdtree_positions = None    # also lazy
        obj._eye_cache = {}
        obj._cartridge_map = None

        return obj

    # Single-receptor constructor

    def __init__(self,
                 directions: Optional[ArrayLike] = None,
                 origins: Optional[ArrayLike] = None,
                 num_ommatidia: Optional[int] = None,
                 acceptance_angles_rad: Optional[Union[ArrayLike, Tuple, float]] = None,
                 interommatidial_angles_rad: Optional[Union[ArrayLike, Tuple, float]] = None,
                 sensitivities: Optional[Union[ArrayLike, float]] = None,
                 receptor_types: Optional[Union[ArrayLike, int]] = None,
                 eye_id: Optional[Union[int, ArrayLike]] = None,
                 eye_parameter: Optional[Union[float, Tuple]] = None,
                 lens_diameter_nm: Optional[Union[float, Tuple]] = None,
                 rhabdom_diameter_nm: Optional[Union[float, Tuple]] = None,
                 focal_length_nm: Optional[Union[float, Tuple]] = None,
                 wavelength_nm: float = 500,
                 eye_radius: float = 0.01,
                 force_isotropic: bool = False,
                 icosphere_method: bool = True,
                 ):
        """
        Simplified construction (single receptor per lens).
        """

        if directions is None and num_ommatidia is None:
            raise ValueError("Requires either 'directions' or 'num_ommatidia'.")

        if directions is not None:
            directions = np.asarray(directions, dtype=np.float32)
            N = len(directions)
        else:
            if icosphere_method:
                lod = estimate_lod(num_ommatidia)
                directions = subdivide_icosahedron(lod)
                N = len(directions)
                if abs(num_ommatidia - N) > 1:
                    print(f"Note: {N} ommatidia for subdivision level {lod}.")
            else:
                directions = fibonacci_sphere(num_ommatidia)
                N = len(directions)

        self.lens_count = N
        self.receptor_count = 1
        self.data = np.zeros(N, dtype=GPU_RECEPTOR_DTYPE)

        self._kernel = None
        self._cartridge_map = None
        self._wavelength_nm = wavelength_nm

        # Directions
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        self.data['direction'][:, :3] = directions / norms
        self.data['direction'][:, 3] = 0.0

        # Origins
        if origins is not None:
            origins_arr = np.asarray(origins, dtype=np.float32)

            if origins_arr.ndim == 1 and origins_arr.shape[0] == 3:
                self.data['origin'][:, :3] = origins_arr
            elif origins_arr.shape == (N, 3):
                self.data['origin'][:, :3] = origins_arr
            else:
                raise ValueError(f"Invalid 'origins' shape {origins_arr.shape}. Expected ({N}, 3) or (3,).")

        elif eye_radius > 0:
            self.data['origin'][:, :3] = self.data['direction'][:, :3] * eye_radius

        self.data['origin'][:, 3] = 1.0

        self.data['sensitivity'] = np.asarray(
            sensitivities if sensitivities is not None else 1.0, dtype=np.float32)

        # Packed metadata
        id_arr = np.zeros(N, dtype=np.uint32)
        if eye_id is not None:
            prepared = self._prepare_param(eye_id, "eye_id")
            if np.any(prepared > 7) or np.any(prepared < 0):
                raise ValueError("eye_id must be in [0, 7].")
            id_arr = prepared.astype(np.uint32)

        types_arr = np.zeros(N, dtype=np.uint32)
        if receptor_types is not None:
            prepared = self._prepare_param(receptor_types, "receptor_types")
            types_arr = np.clip(prepared, 0, 15).astype(np.uint32)

        lens_idx_arr = np.arange(N, dtype=np.uint32)

        self.data['packed_data'] = (
            (id_arr & 0x07) |
            ((types_arr & 0x0F) << 3) |
            ((lens_idx_arr & 0xFFFF) << 11)
        )

        self.dirty_mask = np.zeros(N, dtype=bool)
        self._stale_receptor_spatial = False
        self._stale_lens_spatial = False

        # Can eagerly build since receptor=lens, it's cheap
        self._kdtree_directions = KDTree(self.data['direction'][:, :3])
        self._kdtree_positions = KDTree(self.data['origin'][:, :3])
        self._eye_cache = {}

        # Lattice properties
        is_pre_expanded = False
        if N > 1:
            if np.allclose(self.data['origin'][0], self.data['origin'][1], atol=1e-7):
                is_pre_expanded = True

        if interommatidial_angles_rad is not None:
            angles_arr = np.asarray(interommatidial_angles_rad, dtype=np.float32)

            if angles_arr.shape == (N,):
                ioa_min = angles_arr
                ioa_maj = angles_arr
            else:
                broad = np.broadcast_to(angles_arr, (N, 2))
                ioa_min = np.minimum(broad[:, 0], broad[:, 1])
                ioa_maj = np.maximum(broad[:, 0], broad[:, 1])

            self.data['interommatidial_angles'][:, 0] = ioa_min
            self.data['interommatidial_angles'][:, 1] = ioa_maj

            self._ioa_minor_rad = ioa_min
            self._ioa_major_rad = ioa_maj
            self._lattice_tilts = np.zeros(N, dtype=np.float32)

        elif not is_pre_expanded:
            ioa_min, ioa_maj, tilts, counts = _compute_lattice_properties(
                self.data['direction'][:, :3],
                self.data['origin'][:, :3]
            )

            self._ioa_minor_rad = ioa_min
            self._ioa_major_rad = ioa_maj
            self._lattice_tilts = tilts

            self.data['interommatidial_angles'][:, 0] = ioa_min
            self.data['interommatidial_angles'][:, 1] = ioa_maj
            self.data['tilt'] = tilts

            cleared = self.data['packed_data'] & _CLEAR_NEIGHBOURS
            self.data['packed_data'] = cleared | ((counts & 0x0F) << 7)
        else:
            self._ioa_minor_rad = self.data['interommatidial_angles'][:, 0]
            self._ioa_major_rad = self.data['interommatidial_angles'][:, 1]
            self._lattice_tilts = self.data['tilt'].copy()

        # Bundle orientation: in single receptor mode this defaults to lattice tilt
        # TODO: This is probably not the best approximation

        self._bundle_orientation = self._lattice_tilts.copy()
        self._lens_directions = self.data['direction'][:, :3].copy()
        self._lens_positions = self.data['origin'][:, :3].copy()
        self._actuation_state = np.zeros(N, dtype=np.float32)
        self._local_right = None
        self._local_up = None

        # Acceptance angles
        if acceptance_angles_rad is not None:
            estimated_angles = acceptance_angles_rad

        elif all(p is not None for p in [lens_diameter_nm, rhabdom_diameter_nm, focal_length_nm]):

            D_min, D_maj = self._unpack(lens_diameter_nm, "lens_diameter")
            d_min, d_maj = self._unpack(rhabdom_diameter_nm, "rhabdom_diameter")
            f_min, f_maj = self._unpack(focal_length_nm, "focal_length")

            acc_min = np.sqrt((wavelength_nm / D_min) ** 2 + (d_min / f_min) ** 2)
            acc_maj = np.sqrt((wavelength_nm / D_maj) ** 2 + (d_maj / f_maj) ** 2)
            estimated_angles = np.vstack([acc_min, acc_maj]).T

        else:
            p = eye_parameter if eye_parameter is not None else 1.0
            p_min, p_maj = (p, p) if isinstance(p, (int, float)) else p

            estimated_angles = np.vstack([
                p_min * self._ioa_minor_rad,
                p_maj * self._ioa_major_rad
            ]).T

        if force_isotropic:
            mean_a = np.mean(np.atleast_2d(estimated_angles), axis=1)
            estimated_angles = np.vstack([mean_a, mean_a]).T

        angles_arr = np.asarray(estimated_angles, dtype=np.float32)

        if angles_arr.shape == (N,):
            self.data['acceptance_angles'] = angles_arr[:, np.newaxis]
        else:
            self.data['acceptance_angles'] = angles_arr

        with np.errstate(divide='ignore', invalid='ignore'):
            self.eye_parameter_minor = self.data['acceptance_angles'][:, 0] / self._ioa_minor_rad
            self.eye_parameter_major = self.data['acceptance_angles'][:, 1] / self._ioa_major_rad

        np.nan_to_num(self.eye_parameter_minor, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(self.eye_parameter_major, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    @classmethod
    def from_file(cls, file_path: Union[str, Path], **kwargs):
        """
        Load (a R=1 simple) model from .npz archive.
        """
        # TODO: This should also load a full model

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Cannot find: {path}")

        data = np.load(path)
        if 'directions' not in data:
            raise ValueError(f"'{path}' missing required 'directions' array.")

        args = {
            'directions': data['directions'],
            'origins': data.get('origins'),
            'acceptance_angles_rad': data.get('acceptance_angles_rad'),
            'interommatidial_angles_rad': data.get('interommatidial_angles_rad'),
            'sensitivities': data.get('sensitivities'),
            'receptor_types': data.get('receptor_types'),
            'eye_id': data.get('eye_id'),
        }
        args.update(kwargs)
        return cls(**args)

    # Overrides and internal helpers

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"<ReceptorArray(lenses={self.lens_count}, R={self.receptor_count}, total={len(self.data)})>"

    def _prepare_param(self, param, name="param"):
        arr = np.asarray(param, dtype=np.float32)
        if arr.ndim == 0:
            return np.full(self.lens_count, arr.item())
        if arr.ndim == 1 and len(arr) == self.lens_count:
            return arr
        raise ValueError(
            f"'{name}' shape invalid. Need scalar or length-{self.lens_count}.")

    def _unpack(self, param, name="param"):
        if isinstance(param, Sequence):
            return self._prepare_param(param[0], f"{name}_min"), self._prepare_param(param[1], f"{name}_maj")
        p = self._prepare_param(param, name)
        return p, p

    # Eye / Ommatidium / Cartridge access

    def eye(self, eye_id: int) -> Eye:
        """Eye view for eye_id (0-7)."""
        if eye_id not in self._eye_cache:
            self._eye_cache[eye_id] = Eye(self, eye_id)
        return self._eye_cache[eye_id]

    @property
    def eyes(self) -> list:
        """List of Eye views for all eye_ids present."""
        unique = np.unique(self.data['packed_data'][::self.receptor_count] & 0x07)
        return [self.eye(int(eid)) for eid in unique]

    @property
    def eye_ids(self) -> np.ndarray:
        return np.unique(self.data['packed_data'][::self.receptor_count] & 0x07)

    def ommatidium(self, lens_index: int) -> Ommatidium:
        """Global lens index -> Ommatidium group view."""
        return Ommatidium(self, lens_index)

    def cartridge(self, lens_index: int) -> Cartridge:
        """Global lens index -> Cartridge (neural superposition unit)."""
        return Cartridge(self, lens_index)

    @property
    def kernel(self) -> Optional[RhabdomereKernel]:
        return self._kernel

    @property
    def bundle_orientation(self) -> np.ndarray:
        """Bundle orientation (chi), per lens."""
        return self._bundle_orientation

    @property
    def is_full_model(self) -> bool:
        """True if built with a rhabdomere kernel (R > 1)."""
        return self._kernel is not None

    @property
    def interommatidial_angles_rad(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._ioa_minor_rad, self._ioa_major_rad

    # Neural superposition wiring

    def build_cartridge_map(self) -> np.ndarray:
        """
        For each lens, compute which neighbour's receptor points at this lens's optical axis.
        Returns (N, R_outer) array of global lens indices, where R_outer = min(receptor_count, 6) (R1-R6 only)
        """

        if self.receptor_count < 2:
            warnings.warn("Cartridge map requires a full model (R > 1).")
            return np.zeros((self.lens_count, 0), dtype=np.intp)

        N = self.lens_count
        R = self.receptor_count
        R_outer = min(R, 6)
        lens_dirs = self._lens_directions

        cartridge = np.zeros((N, R_outer), dtype=np.intp)

        for k in range(R_outer):
            # Directions of receptor type k across all lenses
            type_k_dirs = self.data['direction'][k::R, :3]
            tree = KDTree(type_k_dirs)
            _, indices = tree.query(lens_dirs)
            cartridge[:, k] = indices

        self._cartridge_map = cartridge
        return cartridge

    @property
    def cartridge_global_indices(self) -> np.ndarray:
        # TODO: rename this
        """
        Returns (N, R_outer) array of global receptor indices (for neural superposition).
        """
        if self._cartridge_map is None:
            self.build_cartridge_map()

        R = self.receptor_count
        R_outer = self._cartridge_map.shape[1]

        type_offsets = np.arange(R_outer)

        return self._cartridge_map * R + type_offsets

    # Actuation

    def actuate(self, displacement_um: Union[float, ArrayLike],
                axial_um: Union[float, ArrayLike] = 0.0,
                lens_mask: Optional[ArrayLike] = None):
        """
        Displace rhabdomeres via microsaccades.

        Models the two components of rhabdomere actuation observed in vivo:
        (Juusola et al. 2017, 10.7554/eLife.26117; Kemppainen et al. 2022, 10.1073/pnas.2109717119)

        * Lateral: rhabdomere tips move sideways in the focal plane along
            the actuation axis (chi + kernel.actuation_angle_deg).
            Shifts the sampling direction.
            Typical range: 0.0 to ~1.7 μm in Drosophila

        * Axial: rhabdomeres contract away from the lens.
            This narrows the acceptance angle and widens the angular subtense of lateral offsets.
            Typical range: from ~8.1° to ~4.0° in Drosophila

        The rhabdomeres are mechanically coupled, activating one receptor likely contracts and tilts its neighbours.

        Both parameters are absolute from rest: calling `actuate(0.0, 0.0)` resets to the rest configuration.

        Args:
            displacement_um: Lateral focal-plane displacement (μm).
                Scalar for uniform, or (n_mask,) per-lens.
            axial_um: Axial contraction toward lens (μm), positive.
                Scalar for uniform, or (n_mask,) per-lens.
                Default 0 (lateral only).
            lens_mask: Global lens indices to actuate. None = all.
        """

        if self._kernel is None:
            raise RuntimeError("Actuation requires a full model (use from_build).")

        kernel = self._kernel
        R = self.receptor_count
        d_rest = kernel.nodal_distance_um  # nodal distance at rest

        dx = kernel.offsets_um[:, 0]
        dy = kernel.offsets_um[:, 1]

        if lens_mask is None:
            lens_mask = np.arange(self.lens_count)

        lens_mask = np.asarray(lens_mask)
        n_act = len(lens_mask)

        lat = np.broadcast_to(np.float32(displacement_um), n_act).copy()
        axi = np.broadcast_to(np.float32(axial_um), n_act).copy()
        self._actuation_state[lens_mask] = lat

        chi = self._bundle_orientation[lens_mask]
        cos_chi = np.cos(chi)[:, np.newaxis]
        sin_chi = np.sin(chi)[:, np.newaxis]

        # Actuation direction: chi + kernel intrinsic angle (main_axis + saccade offset)
        act_angle = chi + np.radians(kernel.actuation_angle_deg)
        cos_act = np.cos(act_angle)[:, np.newaxis]
        sin_act = np.sin(act_angle)[:, np.newaxis]

        # Effective nodal distance after axial contraction
        d_eff = d_rest - axi  # (n_act,)
        d_eff = np.maximum(d_eff, 1.0)  # clamp to 1 μm minimum

        # Rotate kernel offsets by chi then add lateral displacement along actuation axis
        rot_dx = cos_chi * dx[np.newaxis, :] - sin_chi * dy[np.newaxis, :]
        rot_dy = sin_chi * dx[np.newaxis, :] + cos_chi * dy[np.newaxis, :]
        rot_dx += lat[:, np.newaxis] * cos_act
        rot_dy += lat[:, np.newaxis] * sin_act

        if self._local_right is None:
            self._local_right, self._local_up = _compute_tangent_frames(self._lens_directions)

        lr = self._local_right[lens_mask]
        lu = self._local_up[lens_mask]
        fwd = self._lens_directions[lens_mask]

        # tip vectors (using per-lens effective nodal distance)
        local_tip = np.stack([
            rot_dx,
            rot_dy,
            np.broadcast_to(-d_eff[:, np.newaxis], (n_act, R)).copy()
        ], axis=-1)

        world_tip = (
            local_tip[..., 0:1] * lr[:, np.newaxis, :] +
            local_tip[..., 1:2] * lu[:, np.newaxis, :] +
            local_tip[..., 2:3] * fwd[:, np.newaxis, :]
        )  # (n_act, R, 3)

        new_dirs = -world_tip
        norms = np.linalg.norm(new_dirs, axis=-1, keepdims=True)
        new_dirs /= norms

        new_origins = self._lens_positions[lens_mask, np.newaxis, :] + world_tip

        # build global receptor indices for all affected lenses
        receptor_indices = (
            lens_mask[:, np.newaxis] * R + np.arange(R)[np.newaxis, :]
        ).ravel()  # (n_act * R,)

        self.data['direction'][receptor_indices, :3] = new_dirs.reshape(-1, 3)
        self.data['direction'][receptor_indices, 3] = 0.0
        self.data['origin'][receptor_indices, :3] = new_origins.reshape(-1, 3)
        self.data['origin'][receptor_indices, 3] = 1.0

        # Also change acceptance angles for any with axial displacement
        has_axial = np.any(axi != 0.0)

        if has_axial:
            wavelength_um = self._wavelength_nm * 1e-3
            diffraction = wavelength_um / kernel.lens_diameter_um
            geometric = kernel.diameters_um[np.newaxis, :] / d_eff[:, np.newaxis]
            new_acc = np.sqrt(diffraction ** 2 + geometric ** 2)  # (n_act, R)

            self.data['acceptance_angles'][receptor_indices, 0] = new_acc.ravel()
            self.data['acceptance_angles'][receptor_indices, 1] = new_acc.ravel()

        self.dirty_mask[receptor_indices] = True

        self._stale_receptor_spatial = True

    # Spatial structures: 2 levels tracked independently
    #
    # Lens-level (_stale_lens_spatial):
    #   - Eye KDTrees (built from _lens_directions / _lens_origins)
    #   - Eye neighbour graphs
    #   - Only invalidated by scale(), translate(), or direct lens mutation
    #   - Not invalidated by actuate() (lens axes are immutable after construction)
    #
    # Receptor-level (_stale_receptor_spatial):
    #   - Global receptor KDTrees (built lazily from data['direction'] / data['origin'])
    #   - Invalidated by actuate(), Receptor property setters, scale(), translate()
    #   - Built lazily

    def _resolve_lens_spatial(self):
        """Clear lens-stale flag and invalidate Eye caches."""

        if self._stale_lens_spatial:
            self._stale_lens_spatial = False

            for ev in self._eye_cache.values():
                ev._invalidate()

    def _ensure_global_kdtree_directions(self):
        """Lazy build of receptor-level direction KDTree."""

        if self._stale_receptor_spatial or self._kdtree_directions is None:
            self._kdtree_directions = KDTree(self.data['direction'][:, :3])
            self._stale_receptor_spatial = False  # directions rebuilt

        return self._kdtree_directions

    def _ensure_global_kdtree_positions(self):
        """Lazy build of receptor-level position KDTree."""

        if self._stale_receptor_spatial or self._kdtree_positions is None:
            self._kdtree_positions = KDTree(self.data['origin'][:, :3])

        return self._kdtree_positions

    @property
    def kdtree_directions(self):
        """Global (receptor-level) direction KDTree (lazy)."""
        return self._ensure_global_kdtree_directions()

    @property
    def kdtree_positions(self):
        """Global (receptor-level) position KDTree (lazy)."""
        return self._ensure_global_kdtree_positions()

    # Global spatial queries (receptor-level, return global data indices)
    # TODO: These are duplicated, could be taken out as pure functions
    #
    #   These search over all receptors (N*R elements).
    #   In the full model (R>1), Eye-level queries (which operate on lens optical axes) should be preferred.

    def query_directions(self, directions: ArrayLike, k: int = 1) -> np.ndarray:
        """
        Find receptors with optical axis best aligned with some directions. Global indices.
        """

        if k < 1:
            raise ValueError("k must be >= 1")
        kd = self._ensure_global_kdtree_directions()
        q = np.atleast_2d(np.asarray(directions, dtype=np.float32))
        norms = np.linalg.norm(q, axis=-1, keepdims=True)
        np.divide(q, norms, out=q, where=norms != 0)
        is_single = np.asarray(directions).ndim == 1
        _, idx = kd.query(q, k=k)
        if is_single and k == 1:
            return idx.item()
        return idx.squeeze()

    def query_position(self, positions: ArrayLike, k: int = 1) -> np.ndarray:
        """
        Find receptors closest to some positions (on the eye surface). Global indices.
        """

        if k < 1:
            raise ValueError("k must be >= 1")

        kd = self._ensure_global_kdtree_positions()
        q = np.atleast_2d(np.asarray(positions, dtype=np.float32))
        is_single = np.asarray(positions).ndim == 1
        _, idx = kd.query(q, k=k)

        if is_single and k == 1:
            return idx.item()

        return idx.squeeze()

    def query_lookat(self, targets: ArrayLike, k: int = 1) -> np.ndarray:
        """
        Find receptors looking at some target points (world-space). Global indices.
        """

        if k < 1:
            raise ValueError("k must be >= 1")

        q = np.atleast_2d(np.asarray(targets, dtype=np.float32))
        is_single = np.asarray(targets).ndim == 1

        desired = q[:, np.newaxis, :] - self.data['origin'][:, :3][np.newaxis, :, :]

        norms = np.linalg.norm(desired, axis=-1, keepdims=True)
        np.divide(desired, norms, out=desired, where=norms != 0)
        dots = np.einsum('jk,ijk->ij', self.data['direction'][:, :3], desired)

        part = np.argpartition(dots, -k, axis=1)[:, -k:]
        top = np.take_along_axis(dots, part, axis=1)

        order = np.argsort(top, axis=1)[:, ::-1]
        best = np.take_along_axis(part, order, axis=1)

        if is_single and k == 1:
            return best.item()

        return best.squeeze()

    def query_cone(self, center_direction: ArrayLike, angle: float, degrees: bool = True) -> np.ndarray:
        """
        Find all receptors within angle of a center direction. Global indices.
        """

        kd = self._ensure_global_kdtree_directions()
        c = np.asarray(center_direction, dtype=np.float32)
        c /= np.linalg.norm(c)
        a = np.deg2rad(angle) if degrees else angle

        return kd.query_ball_point(c, r=2.0 * np.sin(a / 2.0))

    def query_ball(self, center_position: ArrayLike, radius: float) -> np.ndarray:
        """
        Find all receptors within radius of a center position. Global indices.
        """

        kd = self._ensure_global_kdtree_positions()
        c = np.asarray(center_position, dtype=np.float32)
        return kd.query_ball_point(c, r=radius)

    def max_gap(self) -> float:
        """
        Largest angular gap between any receptor and its nearest neighbour.
        """

        if len(self.data) <= 1:
            return 0.0

        kd = self._ensure_global_kdtree_directions()
        d, _ = kd.query(self.data['direction'][:, :3], k=2)
        return float(np.arccos(np.clip(1.0 - (np.max(d[:, 1]) ** 2) / 2.0, -1, 1)))

    # Whole-array transforms (initial unit scaling, agent setup, etc)

    def scale(self, factor: float):
        """
        Scale all receptor origins by a factor.
        """

        self.data['origin'][:, :3] *= factor
        self._lens_positions *= factor

        # both levels stale: lens positions changed, receptor positions changed
        self._stale_receptor_spatial = True
        self._stale_lens_spatial = True
        self._kdtree_directions = None
        self._kdtree_positions = None
        self._resolve_lens_spatial()

        return self

    def translate(self, vector: ArrayLike):
        """
        Translate all receptor origins by a vector.
        """
        v = np.asarray(vector, dtype=np.float32)

        self.data['origin'][:, :3] += v

        self._lens_positions += v
        self._stale_receptor_spatial = True
        self._stale_lens_spatial = True
        self._kdtree_directions = None
        self._kdtree_positions = None
        self._resolve_lens_spatial()

        return self


class VisualOutput:
    """
    Wrapper around raw GPU readback to provide meaningful views.
    """

    def __init__(self, raw_data: np.ndarray, receptor_array):
        self.raw = raw_data    # raw_data is (batch_size, total_receptors, 4) or (total_receptors, 4)
        self._array = receptor_array

    def __repr__(self):
        return f"VisualOutput([{'×'.join(str(s) for s in self.raw.shape)}])"

    @property
    def per_ommatidium(self) -> np.ndarray:
        """
        Groups receptors by their physical lens.
        Returns (..., N_lenses, R_receptors, 4) array.
        """
        batch_shape = self.raw.shape[:-2]
        new_shape = (*batch_shape, self._array.lens_count, self._array.receptor_count, self.raw.shape[-1])
        return self.raw.reshape(new_shape).squeeze() if not self._array.is_full_model else self.raw.reshape(new_shape)

    @property
    def per_cartridge(self) -> np.ndarray:
        """
        Groups receptors by neural superposition (Lamina cartridge).
        Returns (..., N_lenses, R_outer, 4) array.
        """
        if not self._array.is_full_model:
            return self.per_ommatidium  # fallback for R=1 model

        indices = self._array.cartridge_global_indices
        return self.raw[..., indices, :]


## TODO: Move these to geometry utils module

def estimate_lod(n_vertices: int) -> int:
    """
    LoD for icosphere subdivision to approximate `nb_vertices`.
    """
    if n_vertices < 12:
        return 1
    return int(np.round(np.sqrt((n_vertices - 2) / 10.0)))


def icosahedron_faces() -> np.ndarray:
    """
    Base z-axis-aligned icosahedron.
    Returns (20, 3, 3) face vertices.
    """

    G = (1 + np.sqrt(5)) / 2.0

    p = np.array([
        [G, -G, -G, G, 1, 1, -1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, G, -G, -G, G, 1, 1, -1, -1],
        [1, 1, -1, -1, 0, 0, 0, 0, G, -G, -G, G]
    ], dtype=np.float32).T

    p /= np.linalg.norm(p[0])
    ang = np.arctan(p[0, 0] / p[0, 2])

    ca, sa = np.cos(ang), np.sin(ang)
    rot = np.array([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]])
    p = np.inner(rot, p).T
    p = p[[0, 3, 4, 8, -1, 5, -2, -3, 7, 1, 6, 2]]

    tri = np.array([
        [1, 2, 3, 4, 5, 6, 2, 7, 2, 8, 3, 9, 10, 10, 6, 6, 7, 8, 9, 10],
        [2, 3, 4, 5, 1, 7, 1, 8, 8, 9, 9, 10, 5, 6, 1, 11, 11, 11, 11, 11],
        [0, 0, 0, 0, 0, 1, 7, 2, 3, 3, 4, 4, 4, 5, 5, 7, 8, 9, 10, 6]
    ]).T
    return p[tri]


def barycentric_coords(n_subdiv: int) -> np.ndarray:
    """
    Barycentric coordinates for subdivided reference triangle.
    """

    vals = np.linspace(0, 1, n_subdiv + 1)
    num = int((n_subdiv + 1) * (n_subdiv + 2) / 2)
    bc = np.zeros((num, 3))

    shifts = np.arange(n_subdiv + 1, 0, -1)
    starts = np.zeros(n_subdiv + 1, dtype=int)
    starts[1:] = np.cumsum(shifts[:-1])
    stops = starts + shifts

    for i, (s, e, sh) in enumerate(zip(starts, stops, shifts)):
        bc[s:e, 0] = vals[sh - 1::-1]
        bc[s:e, 1] = vals[:sh]
        bc[s:e, 2] = vals[i]
    return bc


def subdivide_icosahedron(n_subdiv: int) -> np.ndarray:
    """
    Subdivide icosahedron via barycentric interpolation onto unit sphere.
    """

    verts = icosahedron_faces()
    bary = barycentric_coords(n_subdiv)

    all_v = np.einsum('ij,kjl->kil', bary, verts).reshape(-1, 3)
    all_v /= np.linalg.norm(all_v, axis=1)[:, np.newaxis]
    _, iu = np.unique(np.round(all_v, 6), axis=0, return_index=True)

    return all_v[iu].astype(np.float32)


def fibonacci_sphere(samples: int) -> np.ndarray:
    """
    Uniform points on unit sphere (Fibonacci method).
    """

    phi = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(samples)
    y = 1 - (i / float(samples - 1)) * 2
    r = np.sqrt(1 - y * y)
    theta = phi * i

    return np.column_stack([np.cos(theta) * r, y, np.sin(theta) * r])