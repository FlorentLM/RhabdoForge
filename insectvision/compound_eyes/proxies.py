from dataclasses import dataclass
from typing import Tuple, Union, List, Optional
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import KDTree

from insectvision.utils.math import tangent_frames, normalise_vectors
from insectvision.compound_eyes.datatypes import (
    _CLEAR_EYE_ID, _CLEAR_RECEPTOR_TYPE, _CLEAR_NEIGHBOURS, _CLEAR_LENS_INDEX, _CLEAR_CHIRALITY
)


@dataclass
class EyeNeighbours:
    """
    Neighbour query result from Eye.neighbours().

    All index arrays (indices, same_chirality, is_immediate) use eye-local indexing
    (indices into the subset selected by mask).
    """
    eye_id: int
    mask: np.ndarray                                # animal-level lens indices for this eye
    indices: np.ndarray                             # eye-local neighbour indices
    distances: np.ndarray                           # query distances
    same_chirality: Optional[np.ndarray] = None     # True where chirality matches
    is_immediate: Optional[np.ndarray] = None       # True for first-ring neighbours


class LensView:
    """
    Lens-level data, scoped to the whole animal or one eye.
    Provides property access to lens geometry and metadata.
    """

    def __init__(self,
             ra: 'ReceptorArray',
             lens_indices: np.ndarray,
             single_eye: bool = False
         ):

        self._ra = ra
        self._gi = np.asarray(lens_indices, dtype=np.intp)
        self._i = np.arange(len(self._gi))
        self._single_eye = single_eye
        self._neighbour_graph = None

    def __len__(self):
        return len(self._gi)

    def __eq__(self, other: 'LensView'):
        if not isinstance(other, LensView):
            return False
        return other._gi == self._gi

    def __repr__(self):
        scope = 'eye' if self._single_eye else 'animal'
        return f"<Lenses(n={len(self)}, scope={scope})>"

    def __getitem__(self, key):
        new = self._gi[key]
        if isinstance(new, np.integer):
            new = np.array([int(new)], dtype=np.intp)
        return LensView(self._ra, np.atleast_1d(new))

    # Identity

    @property
    def count(self) -> int:
        return len(self)

    @property
    def indices(self) -> np.ndarray:
        """Scope-local indices."""
        return self._i

    @property
    def global_indices(self) -> np.ndarray:
        """Animal-level lens indices for every lens in this scope."""
        return self._gi

    @property
    def eye(self):
        return self._ra._eye_cache_by_lens(self._gi[0])

    # Geometry

    @property
    def positions(self) -> np.ndarray:
        """Lens centre positions in head/world space."""

        return self._ra._lens_positions[self._gi]

    @property
    def directions(self) -> np.ndarray:
        """Optical axis unit vectors."""

        return self._ra._lens_directions[self._gi]

    @property
    def bundle_orientations(self) -> np.ndarray:
        """Rhabdomere bundle rotation chi (radians)."""

        return self._ra._bundle_orientation[self._gi]

    @property
    def chirality(self) -> np.ndarray:
        """+1 normal, -1 mirrored (one value per lens)."""

        first_rec = self._gi * self._ra.receptor_count
        is_mirrored = (self._ra.receptor_data['metadata'][first_rec] >> 27) & 0x01

        return np.where(is_mirrored, -1, 1)

    @property
    def interommatidial_angles(self) -> Tuple[np.ndarray, np.ndarray]:
        """IOA arrays in radians."""

        minor = self._ra.lens_data['ioa_axes'][self._gi, 0]
        major = self._ra.lens_data['ioa_axes'][self._gi, 1]

        return minor, major

    @property
    def lattice_tilts(self) -> np.ndarray:
        """Local lattice tilt (Ψ6 angle, radians)."""

        return self._ra.lens_data['tilt'][self._gi]

    # Directed-neighbour graph (eye-level only)

    def _build_neighbour_graph(self, k_search: int = 8):
        if not self._single_eye:
            raise RuntimeError(
                "directed_neighbours is only available on eye-level LensView, "
                "not at the animal level."
            )

        N = len(self)
        if N <= 1:
            self._neighbour_graph = {
                'proj_x': np.zeros((N, 0), np.float32),
                'proj_y': np.zeros((N, 0), np.float32),
                'angular_sep': np.zeros((N, 0), np.float32),
                'neighbour_local_indices': np.zeros((N, 0), np.intp),
                'local_x': np.zeros((N, 3), np.float32),
                'local_y': np.zeros((N, 3), np.float32),
                'k_search': 0,
            }
            return

        k_eff = min(k_search, N - 1)

        # Use parent eye's direction tree
        dirs = self.directions
        dists, kd_idx = self.eye._directions_tree.query(self.directions, k=k_eff + 1)

        nb_idx = kd_idx[:, 1:]
        nb_dist = dists[:, 1:]
        angular_sep = 2.0 * np.arcsin(np.clip(nb_dist / 2.0, -1.0, 1.0))

        right, up = tangent_frames(dirs)
        local_x = -right
        local_y = up

        nb_dirs = dirs[nb_idx]
        delta = nb_dirs - dirs[:, None, :]
        proj_x = np.sum(delta * local_x[:, None, :], axis=2)
        proj_y = np.sum(delta * local_y[:, None, :], axis=2)

        self._neighbour_graph = {
            'proj_x': proj_x, 'proj_y': proj_y,
            'angular_sep': angular_sep,
            'neighbour_local_indices': nb_idx,
            'local_x': local_x, 'local_y': local_y,
            'k_search': k_eff,
        }

    def directed_neighbours(self,
        direction: ArrayLike,
        k: int = 1,
        coordinate: str = 'spherical',
        return_weights: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        For every lens, find k nearest neighbours along a direction.
        """

        if self._neighbour_graph is None:
            self._build_neighbour_graph()

        graph = self._neighbour_graph
        N = len(self)

        if N <= 1 or graph['k_search'] == 0:
            zero = np.zeros(N, dtype=np.intp) if k == 1 else np.zeros((N, k), dtype=np.intp)
            return (zero, np.zeros_like(zero, dtype=np.float32)) if return_weights else zero

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
        angle_diff = (nb_angles - target_angle[:, None] + np.pi) % (2 * np.pi) - np.pi
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
            indices = nb_local[np.arange(N)[:, None], top_k_sorted]

            if k > k_eff:
                indices = np.hstack([indices, np.zeros((N, k - k_eff), dtype=np.intp)])

        if return_weights:
            w = target_norms if k == 1 else np.tile(target_norms[:, None], (1, k))
            return indices, w
        return indices

    def max_gap(self) -> float:
        """Largest angular gap between any lens and its nearest neighbour."""

        if len(self) <= 1:
            return 0.0

        if self._single_eye:
            return self.eye.max_gap()

        # Animal level: worst across all eyes
        return max(eye.max_gap() for eye in self._ra.eyes)

    def _invalidate(self):
        """Clear cached neighbour graph (after geometry changes)."""
        self._neighbour_graph = None


class ReceptorView:
    """
    Receptor-level data, scoped to the whole animal, one eye, one
    ommatidium, or an arbitrary selection.

    Subscripting (view[i], view[mask]) returns a narrower ReceptorView.
    Setters write through to the parent array.
    """

    def __init__(self, ra: 'ReceptorArray', receptor_indices: np.ndarray):
        self._ra = ra
        self._gi = np.asarray(receptor_indices, dtype=np.intp)
        self._i = np.arange(len(self._gi))

    def __len__(self):
        return len(self._gi)

    def __eq__(self, other: 'ReceptorView'):
        if not isinstance(other, ReceptorView):
            return False
        return other._gi == self._gi

    def __repr__(self):
        n = len(self)
        if n == 1:
            p = np.array2string(self.positions.squeeze(), precision=3, suppress_small=True)
            d = np.array2string(self.directions.squeeze(), precision=3, suppress_small=True)
            return f"<Receptors(n=1, pos={p}, dir={d})>"
        return f"<Receptors(n={n})>"

    def __getitem__(self, key):
        new = self._gi[key]
        if isinstance(new, np.integer):
            new = np.array([int(new)], dtype=np.intp)
        return ReceptorView(self._ra, np.atleast_1d(new))

    # Identity

    @property
    def count(self) -> int:
        return len(self)

    @property
    def indices(self) -> np.ndarray:
        """Scope-local indices."""
        return self._i

    @property
    def global_indices(self) -> np.ndarray:
        """Animal-level receptor indices."""
        return self._gi

    @property
    def eye(self):
        return self._ra._eye_cache_by_lens(self._gi[0])

    # Spatial

    @property
    def positions(self) -> np.ndarray:
        return self._ra.receptor_data['position'][self._gi]

    @positions.setter
    def positions(self, value):
        self._ra.receptor_data['position'][self._gi] = np.asarray(value, dtype=np.float32)
        self._ra.dirty_mask[self._gi] = True

    @property
    def directions(self) -> np.ndarray:
        return self._ra.receptor_data['direction'][self._gi]

    @directions.setter
    def directions(self, value):
        d = normalise_vectors(np.atleast_2d(np.asarray(value, dtype=np.float32)))
        self._ra.receptor_data['direction'][self._gi] = d
        self._ra.dirty_mask[self._gi] = True

    # Optics

    @property
    def acceptance_minor(self) -> np.ndarray:
        return self._ra.receptor_data['acc_axes'][self._gi, 0]

    @acceptance_minor.setter
    def acceptance_minor(self, value):
        self._ra.receptor_data['acc_axes'][self._gi, 0] = value
        self._ra.dirty_mask[self._gi] = True

    @property
    def acceptance_major(self) -> np.ndarray:
        return self._ra.receptor_data['acc_axes'][self._gi, 1]

    @acceptance_major.setter
    def acceptance_major(self, value):
        self._ra.receptor_data['acc_axes'][self._gi, 1] = value
        self._ra.dirty_mask[self._gi] = True

    @property
    def acceptance_rad(self) -> np.ndarray:
        """(N, 2) acceptance axes in radians: [minor, major]."""
        return self._ra.receptor_data['acc_axes'][self._gi]

    @acceptance_rad.setter
    def acceptance_rad(self, value):
        self._ra.receptor_data['acc_axes'][self._gi] = value
        self._ra.dirty_mask[self._gi] = True

    @property
    def acceptance_deg(self) -> np.ndarray:
        return np.rad2deg(self.acceptance_rad)

    @acceptance_deg.setter
    def acceptance_deg(self, value):
        self.acceptance_rad = np.deg2rad(np.asarray(value, dtype=np.float32))

    @property
    def acceptance_tilt(self) -> np.ndarray:
        return self._ra.receptor_data['acc_tilt'][self._gi]

    @acceptance_tilt.setter
    def acceptance_tilt(self, value):
        self._ra.receptor_data['acc_tilt'][self._gi] = np.asarray(value, dtype=np.float32)
        self._ra.dirty_mask[self._gi] = True

    @property
    def sensitivity(self) -> np.ndarray:
        return self._ra.receptor_data['sensitivity'][self._gi]

    @sensitivity.setter
    def sensitivity(self, value):
        self._ra.receptor_data['sensitivity'][self._gi] = np.asarray(value, dtype=np.float32)
        self._ra.dirty_mask[self._gi] = True

    @property
    def tau(self) -> np.ndarray:
        return self._ra.receptor_data['tau'][self._gi]

    @tau.setter
    def tau(self, value):
        self._ra.receptor_data['tau'][self._gi] = np.asarray(value, dtype=np.float32)
        self._ra.dirty_mask[self._gi] = True

    # Metadata

    @property
    def eye_id(self) -> np.ndarray:

        return self._ra.receptor_data['metadata'][self._gi] & 0x07

    @eye_id.setter
    def eye_id(self, value):

        value = np.asarray(value, dtype=np.uint32)
        indices = self._gi
        cur = self._ra.receptor_data['metadata'][indices]

        self._ra.receptor_data['metadata'][indices] = (cur & _CLEAR_EYE_ID) | (value & 0x07)
        self._ra.dirty_mask[indices] = True

    @property
    def receptor_type(self) -> np.ndarray:
        return (self._ra.receptor_data['metadata'][self._gi] >> 3) & 0x0F

    @receptor_type.setter
    def receptor_type(self, value):

        value = np.asarray(value, dtype=np.uint32)
        indices = self._gi
        cur = self._ra.receptor_data['metadata'][indices]

        self._ra.receptor_data['metadata'][indices] = (cur & _CLEAR_RECEPTOR_TYPE) | ((value & 0x0F) << 3)
        self._ra.dirty_mask[indices] = True

    @property
    def neighbours_count(self) -> np.ndarray:

        return (self._ra.receptor_data['metadata'][self._gi] >> 7) & 0x0F

    @neighbours_count.setter
    def neighbours_count(self, value):

        value = np.asarray(value, dtype=np.uint32)
        indices = self._gi
        cur = self._ra.receptor_data['metadata'][indices]

        self._ra.receptor_data['metadata'][indices] = (cur & _CLEAR_NEIGHBOURS) | ((value & 0x0F) << 7)
        self._ra.dirty_mask[indices] = True

    @property
    def lens_id(self) -> np.ndarray:
        """Parent ommatidium index in the animal-level lens array."""

        return (self._ra.receptor_data['metadata'][self._gi] >> 11) & 0xFFFF

    @lens_id.setter
    def lens_id(self, value):

        value = np.asarray(value, dtype=np.uint32)
        indices = self._gi
        cur = self._ra.receptor_data['metadata'][indices]

        self._ra.receptor_data['metadata'][indices] = (cur & _CLEAR_LENS_INDEX) | ((value & 0xFFFF) << 11)
        self._ra.dirty_mask[indices] = True

    @property
    def chirality(self) -> np.ndarray:
        """+1 normal, -1 mirrored."""

        is_mirrored = (self._ra.receptor_data['metadata'][self._gi] >> 27) & 0x01

        return np.where(is_mirrored, -1, 1)

    @chirality.setter
    def chirality(self, value):

        value = np.asarray(value)
        indices = self._gi
        cur = self._ra.receptor_data['metadata'][indices]
        is_mirrored = (value < 0).astype(np.uint32)

        self._ra.receptor_data['metadata'][indices] = (cur & _CLEAR_CHIRALITY) | ((is_mirrored & 0x01) << 27)
        self._ra.dirty_mask[indices] = True

    # Angular convenience

    @property
    def azimuth_rad(self) -> np.ndarray:
        d = self.directions
        return np.arctan2(d[..., 0], -d[..., 2])

    @property
    def azimuth_deg(self) -> np.ndarray:
        return np.rad2deg(self.azimuth_rad)

    @property
    def elevation_rad(self) -> np.ndarray:
        return np.arcsin(np.clip(self.directions[..., 1], -1, 1))

    @property
    def elevation_deg(self) -> np.ndarray:
        return np.rad2deg(self.elevation_rad)


class Ommatidium:
    """
    Grouping of the R receptors behind a single lens.
    """

    def __init__(self, ra: 'ReceptorArray', lens_index: int):
        self._ra = ra
        self._lens_index = int(lens_index)

        R = ra.receptor_count
        self._rec_start = self._lens_index * R
        self._rec_animal = np.arange(self._rec_start, self._rec_start + R, dtype=np.intp)

    @property
    def receptors(self) -> 'ReceptorView':
        """All R receptors behind this lens."""

        return ReceptorView(self._ra, self._rec_animal)

    def __getitem__(self, idx) -> 'ReceptorView':
        """omm[r] -> ReceptorView for receptor(s) r."""

        return self.receptors[idx]

    def __len__(self) -> int:
        return self._ra.receptor_count

    def __iter__(self):
        for k in range(len(self)):
            yield self[k]

    def __repr__(self):
        return f"<Ommatidium(lens={self._lens_index}, R={len(self)})>"

    # Lens-level props

    @property
    def optical_axis(self) -> np.ndarray:
        """Unit direction of the lens."""

        return self._ra._lens_directions[self._lens_index]

    @property
    def position(self) -> np.ndarray:
        """Lens centre position."""

        return self._ra._lens_positions[self._lens_index]

    @property
    def eye_id(self) -> int:
        return int(self._ra.receptor_data['metadata'][self._rec_start] & 0x07)

    @property
    def bundle_orientation(self) -> float:
        """Rotation of rhabdomere bundle in tangent plane (radians)."""

        return float(self._ra._bundle_orientation[self._lens_index])

    def actuate(self, lateral_um: float = 0.0, axial_um: float = 0.0):

        self._ra.actuate(
            lateral_um=lateral_um,
            axial_um=axial_um,
            to_actuate=np.array([self._lens_index]),
        )


class Cartridge:
    """
    Neural superposition unit: peripheral receptors from neighbouring ommatidia converging on one lamina column
    (plus the central receptors from the home ommatidium).
    """

    def __init__(self, ra: 'ReceptorArray', lens_index: int):
        self._ra = ra
        self._lens_index = int(lens_index)

        if ra._cartridge_map is None:
            raise RuntimeError("Cartridge map not built. Call array.wire_cartridges() first.")

        R = ra.receptor_count
        sources = ra._cartridge_map[lens_index]
        self._rec_animal = (sources * R + np.arange(R)).astype(np.intp)

    @property
    def receptors(self) -> 'ReceptorView':
        return ReceptorView(self._ra, self._rec_animal)

    def __getitem__(self, idx) -> 'ReceptorView':
        """cart[r] -> ReceptorView for receptor r (from the appropriate neighbour)."""

        return self.receptors[idx]

    def __len__(self) -> int:
        return self._ra.receptor_count

    def __repr__(self):
        return f"<Cartridge(lens={self._lens_index}, R={len(self)})>"


class Eye:
    """
    View into a ReceptorArray scoped to a single eye_id.
    """

    def __init__(self, ra: 'ReceptorArray', eye_id: int, lens_indices: np.ndarray):
        self._ra = ra
        self._eye_id = eye_id
        self._lens_indices = np.asarray(lens_indices, dtype=np.intp)

        R = ra.receptor_count
        self._receptor_indices = (
            self._lens_indices[:, None] * R + np.arange(R)[None, :]
        ).ravel()

        self._lenses_view: Optional['LensView'] = None
        self._receptors_view: Optional['ReceptorView'] = None

        self._rebuild_trees()

    def _rebuild_trees(self):
        """Build (or rebuild) KD-trees from current lens geometry."""

        self._positions_tree = KDTree(self._ra._lens_positions[self._lens_indices])
        self._directions_tree = KDTree(self._ra._lens_directions[self._lens_indices])

    # Namespaces

    @property
    def lenses(self) -> 'LensView':
        """Lens-level data (and directed-neighbour graph), eye-local."""

        if self._lenses_view is None:
            self._lenses_view = LensView(
                ra=self._ra,
                lens_indices=self._lens_indices,
                single_eye=True,
            )

        return self._lenses_view

    @property
    def receptors(self) -> 'ReceptorView':
        """Receptor-level data, eye-local."""

        if self._receptors_view is None:
            self._receptors_view = ReceptorView(self._ra, self._receptor_indices)

        return self._receptors_view

    # Basic stuff

    @property
    def eye_id(self) -> int:
        return self._eye_id

    @property
    def lens_indices(self) -> np.ndarray:
        """Animal-level lens indices belonging to this eye."""
        return self._lens_indices

    @property
    def lens_count(self) -> int:
        return len(self._lens_indices)

    def __len__(self) -> int:
        """Number of ommatidia in this eye."""

        return len(self._lens_indices)

    def __repr__(self):
        return f"<Eye(id={self._eye_id}, ommatidia={len(self)})>"

    def __iter__(self):
        for i in range(len(self)):
            yield self.ommatidium(i)

    def __getitem__(self, key) -> Union['Ommatidium', List['Ommatidium']]:
        """Index by eye-local lens index  -> Ommatidium."""

        if isinstance(key, (int, np.integer)):
            return Ommatidium(self._ra, int(self._lens_indices[key]))

        indices = self._lens_indices[key]
        return [Ommatidium(self._ra, int(li)) for li in indices]

    def ommatidium(self, local_index: int) -> 'Ommatidium':
        """Eye-local index -> Ommatidium."""

        return Ommatidium(self._ra, int(self._lens_indices[local_index]))

    def cartridge(self, local_index: int) -> 'Cartridge':
        """Eye-local index -> Cartridge."""

        return Cartridge(self._ra, int(self._lens_indices[local_index]))

    def actuate(self, lateral_um=0.0, axial_um=0.0):
        """Displace all rhabdomeres in this eye."""

        self._ra.actuate(lateral_um=lateral_um, axial_um=axial_um, to_actuate=self._lens_indices)

    def _query_knn(self,
           points: ArrayLike,
           k: int,
           tree: KDTree,
           normalise: bool = False,
           return_distances: bool = False
       ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """kNN against one of this eye's trees. Returns eye-local indices."""

        is_single = np.asarray(points).ndim == 1
        q = np.atleast_2d(np.asarray(points, dtype=np.float32))
        if normalise:
            q = normalise_vectors(q)

        distances, indices = tree.query(q, k=k)

        if is_single and k == 1:
            return int(indices.squeeze())

        if return_distances:
            return np.asarray(distances).squeeze(), np.asarray(indices).squeeze()

        return np.asarray(indices).squeeze()

    def _query_ball(self, center: ArrayLike, radius: float, tree: KDTree) -> np.ndarray:
        """Ball query against one of this eye's trees. Returns eye-local indices."""

        c = np.asarray(center, dtype=np.float32)
        hits = tree.query_ball_point(c, r=radius)

        return np.atleast_1d(np.asarray(hits, dtype=np.intp))

    def query_directions(self, directions: ArrayLike, k: int = 1, return_distances: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Find lenses whose optical axis best matches query directions (eye-local indices)."""

        return self._query_knn(directions, k, self._directions_tree, normalise=True, return_distances=return_distances)

    def query_positions(self, positions: ArrayLike, k: int = 1, return_distances: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Find lenses closest to query positions on the eye surface (eye-local indices)."""

        return self._query_knn(positions, k, self._positions_tree, return_distances=return_distances)

    def query_lookat(self, targets: ArrayLike, k: int = 1) -> np.ndarray:
        """
        Find lenses looking toward world-space target points (eye-local indices).
        (this accounts for lens position, unlike query_directions)
        """
        if k < 1:
            raise ValueError("k must be >= 1")

        is_single = np.asarray(targets).ndim == 1
        q = np.atleast_2d(np.asarray(targets, dtype=np.float32))

        pos = self._ra._lens_positions[self._lens_indices]
        dirs = self._ra._lens_directions[self._lens_indices]

        desired = q[:, None, :] - pos[None, :, :]
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
        """Find all lenses within angle of a centre direction (eye-local indices)."""

        c = np.asarray(center_direction, dtype=np.float32)
        c /= np.linalg.norm(c)
        a = np.deg2rad(angle) if degrees else angle
        r = 2.0 * np.sin(a / 2.0)

        return self._query_ball(c, r, self._directions_tree)

    def query_ball(self, center_position: ArrayLike, radius: float) -> np.ndarray:
        """Find all lenses within radius of a centre position (eye-local indices)."""

        return self._query_ball(center_position, radius, self._positions_tree)

    def max_gap(self) -> float:
        """Largest angular gap between any lens and its nearest neighbour."""

        if len(self) <= 1:
            return 0.0

        d, _ = self._directions_tree.query(
            self._ra._lens_directions[self._lens_indices], k=2)

        return float(np.arccos(np.clip(1.0 - (np.max(d[:, 1]) ** 2) / 2.0, -1, 1)))

    def neighbours(
            self,
            points: np.ndarray,
            k: int,
            chirality: np.ndarray = None,
            include_self: bool = False,
            immediate_only: bool = False,
            neighbour_dist_factor: float = 1.3,
            tree: str = 'positions'
    ) -> Optional[EyeNeighbours]:
        """
        Query k nearest neighbours within this eye.

        Args:
            points: (N_animal, D) array indexed at animal level.
            k: Number of neighbours.
            chirality: (N_animal,) optional chirality for same-chirality masking.
            include_self: Whether the query point counts as its own neighbour.
            immediate_only: Tag first-ring neighbours.
            neighbour_dist_factor: Distance ratio threshold for first-ring detection.
            tree: 'positions' or 'directions'.

        Returns:
            EyeNeighbours with eye-local indices, or None if eye has < 2 lenses.
        """

        mask = self._lens_indices
        n = len(mask)
        if n < 2:
            return None

        kdtree = self._positions_tree if tree == 'positions' else self._directions_tree
        k_query = min(k + (0 if include_self else 1), n)

        dists, idx = kdtree.query(points[mask], k=k_query)

        if not include_self and k_query > 1:
            dists = dists[:, 1:]
            idx = idx[:, 1:]

        result = EyeNeighbours(eye_id=self._eye_id, mask=mask, indices=idx, distances=dists)

        if chirality is not None:
            g_chiral = chirality[mask]
            result.same_chirality = (g_chiral[idx] == g_chiral[:, None])

        if immediate_only:
            closest = dists[:, 0]
            result.is_immediate = dists <= closest[:, None] * neighbour_dist_factor

        return result

    def _invalidate(self):
        """Clear cached views and rebuild trees (called after geometry changes)."""

        self._rebuild_trees()
        self._lenses_view = None
        self._receptors_view = None


class VisualOutput:
    """
    Wrapper around raw GPU readback with grouping.
    """

    def __init__(self, raw_data: np.ndarray, ra: 'ReceptorArray'):
        self.raw = raw_data
        self._ra = ra

    def __repr__(self):
        return f"<VisualOutput({'×'.join(str(s) for s in self.raw.shape)})>"

    def __getitem__(self, eye: 'Eye') -> 'EyeVisualOutput':
        """Scope to an eye."""

        if not isinstance(eye, Eye):
            raise TypeError(f"Index with an Eye instance, got {type(eye).__name__}")

        return EyeVisualOutput(self.raw, self._ra, eye)

    @property
    def receptors(self) -> np.ndarray:
        """Flat receptor data."""

        return self.raw

    @property
    def lenses(self) -> np.ndarray:
        """Grouped by physical lens."""

        arr = self.raw.reshape(
            *self.raw.shape[:-2],       # batch
            self._ra.lens_count,        # lenses
            self._ra.receptor_count,    # receptors
            self.raw.shape[-1]          # cartridges
        )

        return arr

    @property
    def cartridges(self) -> np.ndarray:
        """Grouped by neural superposition wiring."""

        if self._ra.receptor_count == 1:
            return self.lenses

        indices = self._ra.cartridge_indices
        return self.raw[..., indices, :]


class EyeVisualOutput:
    """
    Eye-scoped visual output.
    """

    def __init__(self, raw_data: np.ndarray, ra: 'ReceptorArray', eye: 'Eye'):
        self._raw = raw_data
        self._ra = ra
        self._eye = eye

    def __repr__(self):
        return f"<EyeVisualOutput(eye={self._eye.eye_id}, lenses={len(self._eye)})>"

    @property
    def receptors(self) -> np.ndarray:
        """Flat receptor data for this eye."""

        return self._raw[..., self._eye.receptors.global_indices, :]

    @property
    def lenses(self) -> np.ndarray:
        """Grouped by physical lens."""

        arr = self.receptors.reshape(
            *self._raw.shape[:-2],       # batch
            len(self._eye),              # lenses
            self._ra.receptor_count,     # receptors
            self._raw.shape[-1]          # cartridges
        )
        return arr

    @property
    def cartridges(self) -> np.ndarray:
        """Grouped by neural superposition."""

        if self._ra.receptor_count == 1:
            return self.lenses

        indices = self._ra.cartridge_indices[self._eye.lenses.global_indices]

        arr = self._raw[..., indices, :].reshape(
            *self._raw.shape[:-2],       # batch
            len(self._eye),              # lenses
            self._ra.receptor_count,     # receptors
            self._raw.shape[-1]          # cartridges
        )
        return arr