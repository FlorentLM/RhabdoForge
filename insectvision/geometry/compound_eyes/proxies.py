from typing import Tuple, Union, List
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import KDTree

from insectvision.engine.world_utils import WORLD_UP, WORLD_RIGHT
from insectvision.geometry.compound_eyes.datatypes import (
    DEFAULT_ANGLE, _CLEAR_EYE_ID, _CLEAR_RECEPTOR_TYPE, _CLEAR_NEIGHBOURS, _CLEAR_LENS_INDEX, _CLEAR_CHIRALITY
)


# read write: can be got or set at any scope
_RW_FIELDS = (
    'tau', 'sensitivity',
    'acceptance_tilt', 'acceptance_major', 'acceptance_minor',
    'acceptance_rad', 'acceptance_deg',
    'eye_id', 'receptor_type', 'neighbours_count', 'lens_id', 'chirality'
)

# derived angular quantities (no meaningful setter)
_RO_FIELDS = (
    'azimuth_rad', 'azimuth_deg',
    'elevation_rad', 'elevation_deg',
)


def _make_rw(name):
    """Create a read-write property that delegates to the scoped Receptor proxy."""

    def fget(self):
        return getattr(self._receptor_proxy, name)

    def fset(self, value):
        setattr(self._receptor_proxy, name, value)

    return property(fget, fset, doc=f"Forwarded receptor field: {name}")


def _make_ro(name):
    """Create a read-only property that delegates to the scoped Receptor proxy."""

    def fget(self):
        return getattr(self._receptor_proxy, name)

    return property(fget, doc=f"Forwarded receptor field (read-only): {name}")


class _ReceptorProxyMixin:
    """
    Mixin that gives any hierarchy level accessors to receptor fields.

    (subclasses implement `_receptor_proxy` returning a `Receptor``
        which indexes cover the receptors in scope)

    Concerned properties are whitelisted so lens-level overrides
    like `Ommatidium.position` keep their semantics
    """

    @property
    def _receptor_proxy(self) -> 'Receptor':
        raise NotImplementedError

    @property
    def receptors(self) -> 'Receptor':
        """Receptor view spanning every receptor in this scope."""
        return self._receptor_proxy


for _name in _RW_FIELDS:
    setattr(_ReceptorProxyMixin, _name, _make_rw(_name))

for _name in _RO_FIELDS:
    setattr(_ReceptorProxyMixin, _name, _make_ro(_name))

if DEFAULT_ANGLE == 'rad':
    _ReceptorProxyMixin.lon = _ReceptorProxyMixin.longitude = _ReceptorProxyMixin.azimuth = _make_ro('azimuth_rad')
    _ReceptorProxyMixin.lat = _ReceptorProxyMixin.latitude = _ReceptorProxyMixin.elevation = _make_ro('elevation_rad')
    _ReceptorProxyMixin.rho = _ReceptorProxyMixin.acceptance = _make_rw('acceptance_rad')
else:
    _ReceptorProxyMixin.lon = _ReceptorProxyMixin.longitude = _ReceptorProxyMixin.azimuth = _make_ro('azimuth_deg')
    _ReceptorProxyMixin.lat = _ReceptorProxyMixin.latitude = _ReceptorProxyMixin.elevation = _make_ro('elevation_deg')
    _ReceptorProxyMixin.rho = _ReceptorProxyMixin.acceptance = _make_rw('acceptance_deg')

_ReceptorProxyMixin.rho_minor = _make_rw('acceptance_minor')
_ReceptorProxyMixin.rho_major = _make_rw('acceptance_major')


class Receptor:
    """
    View into one or several elements of a ReceptorArray.
    """

    def __init__(self, data_array: np.ndarray, item, parent_array: 'ReceptorArray'):
        self._data = data_array
        self._item = item
        self._parent = parent_array

    def __len__(self):
        return 1 if self._data[self._item].ndim == 0 else self._data[self._item].shape[0]

    def __repr__(self):

        if isinstance(self._item, (int, np.int_)):
            p = np.array2string(self.position, precision=3, suppress_small=True)
            d = np.array2string(self.direction, precision=3, suppress_small=True)

            return f"<Receptor(idx={int(self._item)}, position={p}, direction={d})>"
        return f"<Receptors(key={self._item}, count={len(self)})>"

    # Spatial properties

    @property
    def position(self) -> np.ndarray:
        return self._data[self._item]['position']

    @position.setter
    def position(self, value: Union[float, ArrayLike]):
        self._data['position'][self._item] = np.asarray(value, dtype=np.float32)
        self._parent.dirty_mask[self._item] = True
        self._parent._stale_receptor_spatial = True

    @property
    def direction(self) -> np.ndarray:
        return self._data[self._item]['direction']

    @direction.setter
    def direction(self, value: Union[float, ArrayLike]):
        new_dirs = np.atleast_2d(value)
        norms = np.linalg.norm(new_dirs, axis=-1, keepdims=True)
        np.divide(new_dirs, norms, out=new_dirs, where=norms != 0)

        self._data['direction'][self._item] = new_dirs
        self._parent.dirty_mask[self._item] = True
        self._parent._stale_receptor_spatial = True

    # Optics / Temporal

    @property
    def acceptance_tilt(self) -> np.ndarray:
        return self._data[self._item]['acc_tilt']

    @acceptance_tilt.setter
    def acceptance_tilt(self, value: Union[float, ArrayLike]):
        self._data['acc_tilt'][self._item] = np.asarray(value, dtype=np.float32)
        self._parent.dirty_mask[self._item] = True

    @property
    def acceptance_major(self) -> np.ndarray:
        return self._data[self._item]['acc_axes'][..., 1]

    @acceptance_major.setter
    def acceptance_major(self, value: Union[float, ArrayLike]):
        self._data['acc_axes'][self._item, 1] = value
        self._parent.dirty_mask[self._item] = True

    @property
    def acceptance_minor(self) -> np.ndarray:
        return self._data[self._item]['acc_axes'][..., 0]

    @acceptance_minor.setter
    def acceptance_minor(self, value: Union[float, ArrayLike]):
        self._data['acc_axes'][self._item, 0] = value
        self._parent.dirty_mask[self._item] = True

    @property
    def acceptance_rad(self) -> np.ndarray:
        return self._data[self._item]['acc_axes']

    @acceptance_rad.setter
    def acceptance_rad(self, values: Union[float, ArrayLike]):
        self._data['acc_axes'][self._item] = values
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

    @property
    def tau(self) -> np.ndarray:
        return self._data[self._item]['tau']

    @tau.setter
    def tau(self, value: Union[float, ArrayLike]):
        self._data['tau'][self._item] = np.asarray(value, dtype=np.float32)
        self._parent.dirty_mask[self._item] = True

    # Metadata

    @property
    def eye_id(self) -> np.ndarray:
        return self._data[self._item]['metadata'] & 0x07

    @eye_id.setter
    def eye_id(self, value: Union[int, ArrayLike]):
        v = np.asarray(value, dtype=np.uint32)
        cur = self._data['metadata'][self._item]
        self._data['metadata'][self._item] = (cur & _CLEAR_EYE_ID) | (v & 0x07)
        self._parent.dirty_mask[self._item] = True

    @property
    def receptor_type(self) -> np.ndarray:
        return (self._data[self._item]['metadata'] >> 3) & 0x0F

    @receptor_type.setter
    def receptor_type(self, value: Union[int, ArrayLike]):
        v = np.asarray(value, dtype=np.uint32)
        cur = self._data['metadata'][self._item]
        self._data['metadata'][self._item] = (cur & _CLEAR_RECEPTOR_TYPE) | ((v & 0x0F) << 3)
        self._parent.dirty_mask[self._item] = True

    @property
    def neighbours_count(self) -> np.ndarray:
        return (self._data[self._item]['metadata'] >> 7) & 0x0F

    @neighbours_count.setter
    def neighbours_count(self, value: Union[int, ArrayLike]):
        v = np.asarray(value, dtype=np.uint32)
        cur = self._data['metadata'][self._item]
        self._data['metadata'][self._item] = (cur & _CLEAR_NEIGHBOURS) | ((v & 0x0F) << 7)
        self._parent.dirty_mask[self._item] = True

    @property
    def chirality(self) -> np.ndarray:
        """
        Returns +1 for normal, -1 for mirrored.
        """
        # TODO: Maybe just return bools instead?
        is_mirrored = (self._data[self._item]['metadata'] >> 27) & 0x01
        return np.where(is_mirrored, -1, 1)

    @chirality.setter
    def chirality(self, value: Union[float, ArrayLike]):
        v_arr = np.asarray(value)
        is_mirrored = (v_arr < 0).astype(np.uint32)
        cur = self._data['metadata'][self._item]
        self._data['metadata'][self._item] = (cur & _CLEAR_CHIRALITY) | ((is_mirrored & 0x01) << 27)
        self._parent.dirty_mask[self._item] = True

    @property
    def lens_id(self) -> np.ndarray:
        """
        Index of the parent ommatidium in the lens-level array.
        """
        return (self._data[self._item]['metadata'] >> 11) & 0xFFFF

    @lens_id.setter
    def lens_id(self, value: Union[int, ArrayLike]):
        v = np.asarray(value, dtype=np.uint32)
        cur = self._data['metadata'][self._item]
        self._data['metadata'][self._item] = (cur & _CLEAR_LENS_INDEX) | ((v & 0xFFFF) << 11)
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


class Ommatidium(_ReceptorProxyMixin):
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

    @property
    def _receptor_proxy(self) -> Receptor:
        return Receptor(self._array.receptor_data, self._slice, self._array)

    def __getitem__(self, receptor_idx) -> Receptor:
        """``omm[r]`` returns the Receptor proxy for receptor type r"""

        if isinstance(receptor_idx, (int, np.integer)):
            return Receptor(self._array.receptor_data,
                            self._start + int(receptor_idx),
                            self._array)

        indices = np.arange(self._start, self._stop)[receptor_idx]
        return Receptor(self._array.receptor_data, indices, self._array)

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
        """Eye ID for this ommatidium."""
        return int(self._array.receptor_data['metadata'][self._start] & 0x07)

    @eye_id.setter
    def eye_id(self, value):
        """Set eye_id for all receptors in this ommatidium."""
        self._receptor_proxy.eye_id = value

    @property
    def bundle_orientation(self) -> float:
        """Rotation of rhabdomere bundle in tangent plane (radians).""" # TODO: maybe return degrees instead?
        return float(self._array._bundle_orientation[self._lens_index])

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


class Cartridge(_ReceptorProxyMixin):
    """
    Neural superposition unit: The 6 outer receptors (R1-R6) from 6 different ommatidia
    that converge onto one lamina column.
    """

    def __init__(self, array: 'ReceptorArray', lens_index: int):
        self._array = array
        self._lens_index = int(lens_index)

        if array._cartridge_map is None:
            raise RuntimeError("Cartridge map not built. Call array.build_cartridge_map() first.")  # TODO: may eventually be done automatically

    @property
    def _receptor_proxy(self) -> Receptor:
        return Receptor(self._array.receptor_data, self.receptor_indices, self._array)

    def __getitem__(self, receptor_type: int) -> Receptor:
        """``cartridge[k]`` returns the Receptor R{k+1} from the appropriate ommatidium."""
        source_lens = self._array._cartridge_map[self._lens_index, receptor_type]
        global_idx = source_lens * self._array.receptor_count + receptor_type
        return Receptor(self._array.receptor_data, global_idx, self._array)

    @property
    def receptor_indices(self) -> np.ndarray:
        """Global indices into ReceptorArray.receptor_data"""
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


class Eye(_ReceptorProxyMixin):
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
        first_ids = array.receptor_data['metadata'][::R] & 0x07
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
    def _receptor_proxy(self) -> Receptor:
        return Receptor(self._array.receptor_data, self._receptor_indices, self._array)

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

    # Bulk data views (lens-level)

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
        """Maps to global receptor indices in ReceptorArray.receptor_data"""
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

        desired = q[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        norms = np.linalg.norm(desired, axis=-1, keepdims=True)
        np.divide(desired, norms, out=desired, where=norms != 0)

        dots = np.einsum('jk,ijk->ij', self.directions, desired)

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
        # squeeze if simplified model and batch size is 1 # TODO: or not, idk
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
