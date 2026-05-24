import logging
from typing import List, Optional, Tuple, Union
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

from insectvision.engine.world_utils import WORLD_FORWARD
from insectvision.utils.math import (
    normalise_vectors,
    tangent_frames,
    fibonacci_sphere,
    icosphere,
)

from insectvision.compound_eyes.datatypes import (
    LENS_STATIC_DTYPE,
    LENS_DYNAMIC_DTYPE,
    RCPT_STATIC_DTYPE,
    RCPT_DYNAMIC_DTYPE,
    pack_metadata,
    set_metadata_field,
)
from insectvision.compound_eyes.kernel import RhabdomereKernel
from insectvision.compound_eyes.orientation import (
    BundlesAligner,
    OrientationResult,
    trivial_orientation,
    apply_chirality,
)
from insectvision.compound_eyes.proxies import (
    LensView,
    ReceptorView,
    Cartridge,
    Eye,
    VisualOutput
)

logger = logging.getLogger(__name__)


class ReceptorArray:
    """
    A compound eye specified as N lens positions / directions and a kernel
    of R rhabdomeres per ommatidium.

    The data lives in four numpy structured arrays.
    Views (LensView, ReceptorView, Cartridge, Eye) wrap subsets of these arrays for typed access.

    Args:
        - directions: (N, 3) array_like, Lens optical-axis (forward) directions. Will be normalised.
        - positions: (N, 3) array_like, Lens world positions.
        - kernel: RhabdomereKernel (optional), Per-species bundle model. Defaults to a single panchromatic receptor.
        - eye_ids: (N,) array_like of uint (optional), Eye membership for each lens (0-7).
            If None, splits in two eyes on x: x < 0 -> 0 (left), x ≥ 0 -> 1 (right).
        - lens_diameter_um: float or (N,) array_like or None, Lens aperture (μm).
            If None (default), auto-derived per-lens from the local lattice spacing.
        - interommatidial_angles_rad: (2,), (N, 2), or None, Per-lens (minor, major) IOA.
            If None, computed from the local lattice (sorted mean over first ring,
            tilt from the hexatic Ψ6 order parameter).
        - acceptance_angles_rad: (R,), (N, R), (N, R, 2), or None, Per-receptor acceptance half-widths.
            If None, computed from optics (Snyder, when nodal distance is known) or
            from the local IOA (lattice-convention fallback), scaled by 'eye_parameter'.
        - eye_parameter: float or 2-tuple (p_min, p_maj) or None,
            Multiplier on the optical/lattice baseline that controls how wide the
            RFs are relative to physical optics.
            p=1.0 (default) is pure Snyder / pure IOA
            p>1 makes RFs wider, p<1 narrower.
            A 2-tuple gives anisotropic scaling per (minor, major).
        - bundle_orientations: (N,) array_like or None, Override chi from the orientation pipeline (rad).
        - chiralities: (N,) array_like or None, Override chirality from the orientation pipeline (±1).
        - orientation: BundlesAligner or None, Explicit pipeline configuration.
            Required when R > 1 unless both 'bundle_orientations' and 'chiralities' are supplied.
        - flow_direction: (3,) array_like or None, Shorthand for 'orientation=BundlesAligner(flow_direction)'.
            Defaults to forward-facing optic flow.
    """

    HEX_PACKING_FACTOR = 1.0 # Fraction spacing that the lens diameter occupies (1.0 = fully
    # touching, 0.9 leaves a small interommatidial cuticle gap, etc)

    def __init__(self,
                 directions: ArrayLike,
                 positions: ArrayLike,
                 kernel: Optional[RhabdomereKernel] = None,
                 eye_ids: Optional[ArrayLike] = None,
                 lens_diameter_um: Optional[Union[float, ArrayLike]] = None,
                 interommatidial_angles_rad: Optional[ArrayLike] = None,
                 acceptance_angles_rad: Optional[ArrayLike] = None,
                 eye_parameter: Optional[Union[float, Tuple[float, float]]] = None,
                 bundle_orientations: Optional[ArrayLike] = None,
                 chiralities: Optional[ArrayLike] = None,
                 orientation: Optional[BundlesAligner] = None,
                 flow_direction: Optional[ArrayLike] = None
                 ):

        # Validate geometry

        dirs = np.asarray(directions, dtype=np.float32).reshape(-1, 3)
        pos = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        if dirs.shape != pos.shape:
            raise ValueError(
                f"Directions and Positions must have the same shape, "
                f"got {dirs.shape} and {pos.shape}"
            )
        N = dirs.shape[0]
        if N == 0:
            raise ValueError("ReceptorArray needs at least 1 lens")

        self._lens_directions = normalise_vectors(dirs).astype(np.float32)
        self._lens_positions = pos.astype(np.float32).copy()

        # Kernel
        if kernel is None:
            kernel = RhabdomereKernel()  # default R=1
        self._kernel = kernel
        R = kernel.count

        # Tangent frames
        self._local_right, self._local_up = tangent_frames(self._lens_directions)
        self._local_right = self._local_right.astype(np.float32)
        self._local_up = self._local_up.astype(np.float32)

        # Eye membership (island detection if eye_ids is None)
        self._lens_eye_ids = self._resolve_eye_ids(self._lens_positions, eye_ids, N)

        # Allocate structured arrays
        self.lens_static_data = np.zeros(N, dtype=LENS_STATIC_DTYPE)
        self.lens_dynamic_data = np.zeros(N, dtype=LENS_DYNAMIC_DTYPE)
        self.rcpt_static_data = np.zeros(N * R, dtype=RCPT_STATIC_DTYPE)
        self.rcpt_dynamic_data = np.zeros(N * R, dtype=RCPT_DYNAMIC_DTYPE)

        # Fill lens static data
        self.lens_static_data['right'] = self._local_right
        self.lens_static_data['up'] = self._local_up
        self.lens_static_data['forward'] = self._lens_directions
        self.lens_static_data['nodal_distance_um'] = kernel.nodal_distance_um or 1.0

        # Broadcast kernel-level photomechanical biophysics to every lens
        self.lens_static_data['tau_rise'] = kernel.tau_rise
        self.lens_static_data['tau_relax'] = kernel.tau_relax
        self.lens_static_data['tau_fast'] = kernel.tau_fast
        self.lens_static_data['tau_adapt'] = kernel.tau_adapt
        self.lens_static_data['gain_lat_um'] = kernel.gain_lat_um
        self.lens_static_data['gain_ax_um'] = kernel.gain_ax_um

        self._eyes: List[Eye] = []
        self._build_eyes()

        # Lattice properties: IOA (minor, major), tilt, and per-lens lens spacing
        baseline_axes, baseline_tilts, lens_spacing = self._compute_ioa_baseline()
        if interommatidial_angles_rad is not None:
            ioa_axes, ioa_tilts = self._broadcast_ioa(interommatidial_angles_rad, N)
        else:
            ioa_axes, ioa_tilts = baseline_axes, baseline_tilts
        self.lens_static_data['ioa_axes'] = ioa_axes
        self.lens_static_data['ioa_tilt'] = ioa_tilts

        # Lens diameter: if caller supplied a value, use it.
        # Otherwise derive from lattice spacing.
        if lens_diameter_um is None:
            ld_arr = (self.HEX_PACKING_FACTOR * lens_spacing).astype(np.float32)
            # Sparse lattices (single-lens eye, etc): fallback to a reasonable default of 20 μm
            # TODO: Maybe just raise instead? Why would a single lens be any useful?
            ld_arr = np.where(ld_arr > 0, ld_arr, np.float32(20.0))
            self.lens_static_data['lens_diameter_um'] = ld_arr
        else:
            ld = np.atleast_1d(np.asarray(lens_diameter_um, dtype=np.float32))
            if ld.size == 1:
                self.lens_static_data['lens_diameter_um'] = ld.item()
            elif ld.size == N:
                self.lens_static_data['lens_diameter_um'] = ld
            else:
                raise ValueError(f"lens_diameter_um size {ld.size} must be 1 or N={N}")

        # Fill receptors static data
        self.rcpt_static_data['position'] = np.repeat(self._lens_positions, R, axis=0)
        self.rcpt_static_data['sensitivity'] = np.tile(kernel.sensitivity, (N, 1))
        self.rcpt_static_data['tau_membrane'] = kernel.tau_membrane
        self.rcpt_static_data['rhab_diameter_um'] = np.tile(kernel.diameters_um, N)
        self.rcpt_static_data['wavelength_um'] = np.tile(kernel.wavelengths_nm * 1e-3, N)

        # Acceptance angles (rest + initial dynamic)
        if acceptance_angles_rad is not None:
            acc = self._broadcast_acceptance(acceptance_angles_rad, N, R)
        else:
            acc = self._compute_acceptance_baseline(eye_parameter=eye_parameter)

        self.rcpt_static_data['rest_acc'] = acc
        self.rcpt_dynamic_data['acc_axes'] = acc

        # Pack metadata bits (chirality_neg filled in _apply_orientation)
        lens_ids = np.repeat(np.arange(N, dtype=np.uint32), R)
        rcpt_types = np.tile(np.arange(R, dtype=np.uint32), N)
        eye_ids_per_rcpt = np.repeat(self._lens_eye_ids, R)
        neighbour_counts = np.zeros(N * R, dtype=np.uint32)

        self.rcpt_static_data['metadata'] = pack_metadata(
            eye_id=eye_ids_per_rcpt,
            receptor_types=rcpt_types,
            neighbour_counts=neighbour_counts,
            lens_id=lens_ids,
            chirality_neg=0,
        )

        # Fill neighbours count from neighbour graph
        for eye in self._eyes:
            graph = eye._ensure_neighbour_graph(k=6)
            n_in_eye_per_lens = graph.shape[1]
            rcpt_indices_eye = (
                eye._lens_indices[:, None] * R + np.arange(R, dtype=np.intp)[None, :]
            ).ravel()

            self.rcpt_static_data['metadata'][rcpt_indices_eye] = set_metadata_field(
                self.rcpt_static_data['metadata'][rcpt_indices_eye],
                'neighbour_count',
                n_in_eye_per_lens,
            )

        # Orientation pipeline

        # placeholders (_apply_orientation will overwrite)
        self._bundle_orientation = np.zeros(N, dtype=np.float32)
        self._chirality_arr = np.ones(N, dtype=np.float32)
        self._saccade_cache = np.zeros((N, 3), dtype=np.float32)

        if R == 1:
            result = trivial_orientation(N)
        elif bundle_orientations is not None and chiralities is not None:
            # User supplied both so no flow pipeline needed
            result = apply_chirality(self, bundle_orientations, chiralities)
        else:
            # Pipeline needed
            if orientation is None:
                orientation = BundlesAligner(flow_direction or -WORLD_FORWARD)
            result = orientation.compute(
                self,
                override_chi=bundle_orientations,
                override_chirality=chiralities,
            )

        self._apply_orientation(result)

        # Cartridge mapping and diagnostics
        self._cartridges_wired = False
        self._cartridge_map = np.tile(np.arange(N)[:, None], (1, R))

        self.donation_conflicts = np.zeros(N, dtype=bool)
        self.receiving_conflicts = np.zeros(N, dtype=bool)
        self.have_conflicts = np.zeros(N, dtype=bool)
        self._lens_dirty = True

        if R > 1:
            self.wire_cartridges()
        else:
            self.rcpt_static_data['cartridge_src'] = np.arange(N * R, dtype=np.uint32)


    # Factory methods

    @classmethod
    def from_sphere(cls,
        n: int = 2000,
        eye_radius: float = 0.01,
        method: str = 'icosphere',
        **kwargs,
        ) -> 'ReceptorArray':
        """
        Construct a uniform spherical compound eye.

        Args:
            - n: int, Approximate number of ommatidia.
            - eye_radius: float, Sphere radius in world units (default 0.01 m = 10 mm).
            - method: {'icosphere', 'fibonacci'}, Spherical sampling method.
            - **kwargs: Forwarded to __init__.
        """
        method = method.lower()

        if method == 'icosphere':
            dirs = icosphere(n)
        elif method == 'fibonacci':
            dirs = fibonacci_sphere(n)
        else:
            raise ValueError("Method must be 'icosphere' or 'fibonacci'")
        positions = (dirs * float(eye_radius)).astype(np.float32)
        return cls(directions=dirs, positions=positions, **kwargs)

    @classmethod
    def from_lenses(cls,
        directions: ArrayLike,
        positions: ArrayLike,
        **kwargs,
        ) -> 'ReceptorArray':
        """
        Explicit lens placement. Forwards to __init__.
        """
        return cls(directions=directions, positions=positions, **kwargs)

    @classmethod
    def from_file(cls, path: str, **kwargs) -> 'ReceptorArray':
        """
        Load a species model from a .npz archive.

        Required fields (any of these names accepted, in order of preference):
            - positions: (N, 3), lens world positions
                aliases: 'positions', 'pos', 'lens_positions'
            - directions: (N, 3), lens optical axes
                aliases: 'directions', 'dirs', 'lens_directions', 'forward'

        Optional fields (used when present, otherwise defaults apply):
            - eye_ids: 'eye_ids', 'eye_id'
            - left / right: (N,) bool, fallback if no eye_ids
            - lens_diameter_um: 'lens_diameter_um', 'lens_diameter', 'aperture_um'
            - interommatidial_angles_rad: 'interommatidial_angles_rad', 'ioa', 'ioa_axes'
            - acceptance_angles_rad: 'acceptance_angles_rad', 'acceptance', 'rho'
            - bundle_orientations: 'bundle_orientations', 'chi'
            - chiralities: 'chiralities', 'chirality'

        Any keyword argument forwarded explicitly via **kwargs overrides what's found in the file.
        """

        with np.load(path, allow_pickle=False) as data:
            available = [s.lower() for s in data.files]

            def first_present(candidates):
                for name in candidates:
                    if name in available:
                        return np.asarray(data[name])
                return None

            positions = first_present(['positions', 'pos', 'lens_positions'])
            directions = first_present(['directions', 'dirs', 'lens_directions', 'forward'])
            if positions is None or directions is None:
                raise ValueError(
                    f"{path}: required 'positions' and 'directions' not found. "
                    f"Available keys: {available}"
                )

            # Resolve eye ids: explicit field, else infer from is_left / is_right.
            # If neither is present, falls back to island detection, and auto-assign sides from centroid x.
            if 'eye_ids' not in kwargs:
                eye_ids = first_present(['eye_ids', 'eye_id'])
                if eye_ids is None:
                    is_left = first_present(['l', 'left', 'is_left'])
                    if is_left is not None:
                        # Convention: 0 = left, 1 = right
                        eye_ids = (~is_left.astype(bool)).astype(np.uint32)
                    else:
                        is_right = first_present(['r', 'right', 'is_right'])
                        if is_right is not None:
                            eye_ids = is_right.astype(np.uint32)
                if eye_ids is not None:
                    kwargs['eye_ids'] = eye_ids

            # Optional geometry fields: only set if not overridden in kwargs.
            optional_field_aliases = {
                'lens_diameter_um': ['lens_diameter_um', 'lens_diameter', 'aperture_um'],
                'interommatidial_angles_rad': ['interommatidial_angles_rad', 'ioa', 'ioa_axes'],
                'acceptance_angles_rad': ['acceptance_angles_rad', 'acceptance', 'rho'],
                'bundle_orientations': ['bundle_orientations', 'chi'],
                'chiralities': ['chiralities', 'chirality'],
            }
            for param, aliases in optional_field_aliases.items():
                if param in kwargs:
                    continue
                value = first_present(aliases)
                if value is not None:
                    kwargs[param] = value

        logger.info(
            "Loaded %d lenses from %s (fields: %s)",
            positions.shape[0], path, available,
        )
        return cls(directions=directions, positions=positions, **kwargs)

    # Some basic stuff

    def __repr__(self) -> str:
        return (
            f"ReceptorArray(N={self.lens_count}, R={self.receptors_per_lens}, "
            f"eyes={len(self._eyes)}, kernel={self._kernel.name!r})"
        )

    def __len__(self) -> int:
        return int(self._lens_directions.shape[0])

    @property
    def lens_count(self) -> int:
        return len(self)

    @property
    def receptors_per_lens(self) -> int:
        return self._kernel.count

    @property
    def total_receptors(self) -> int:
        return self.lens_count * self.receptors_per_lens

    @property
    def kernel(self) -> RhabdomereKernel:
        return self._kernel

    @property
    def eyes(self) -> List[Eye]:
        return list(self._eyes)

    def eye(self, eye_id: int) -> Eye:
        for e in self._eyes:
            if e.eye_id == int(eye_id):
                return e
        raise KeyError(f"no eye with id {eye_id}")

    def eyes_by_side(self, side: str) -> List[Eye]:
        """All eyes on a given side ('left', 'right', or 'midline')."""
        s = str(side)
        return [e for e in self._eyes if e.side == s]

    @property
    def left_eyes(self) -> List[Eye]:
        return self.eyes_by_side('left')

    @property
    def right_eyes(self) -> List[Eye]:
        return self.eyes_by_side('right')

    @property
    def midline_eyes(self) -> List[Eye]:
        return self.eyes_by_side('midline')

    def lenses_by_side(self, side: str) -> LensView:
        """All lenses belonging to eyes on the given side."""
        eyes = self.eyes_by_side(side)
        if not eyes:
            return LensView(self, np.empty(0, dtype=np.intp))
        idx = np.concatenate([e.lens_indices for e in eyes])
        return LensView(self, idx)

    @property
    def left_lenses(self) -> LensView:
        return self.lenses_by_side('left')

    @property
    def right_lenses(self) -> LensView:
        return self.lenses_by_side('right')

    @property
    def midline_lenses(self) -> LensView:
        return self.lenses_by_side('midline')

    @property
    def lenses(self) -> LensView:
        return LensView(self, np.arange(self.lens_count, dtype=np.intp))

    @property
    def ommatidia(self) -> LensView:
        """Alias of .lenses, iterating yields Ommatidia."""
        return self.lenses

    @property
    def receptors(self) -> ReceptorView:
        return ReceptorView(self, np.arange(self.total_receptors, dtype=np.intp))

    @property
    def cartridges(self) -> List[Cartridge]:
        if not self._cartridges_wired:
            return []
        return [Cartridge(self, i) for i in range(self.lens_count)]

    @property
    def lens_dirty(self) -> bool:
        """True if lens-level data has been modified since the last renderer upload."""
        return self._lens_dirty

    @lens_dirty.setter
    def lens_dirty(self, value: bool) -> None:
        self._lens_dirty = bool(value)

    @property
    def cartridge_indices(self) -> np.ndarray:
        """(N, R) global receptor indices grouped by cartridge mapping."""
        if not self._cartridges_wired:
            raise RuntimeError("Cartridges not wired")
        return self._cartridge_map * self.receptors_per_lens + np.arange(self.receptors_per_lens)

    # Animal-wide spatial queries (dispatch across eyes)

    def query_directions(self,
         directions: ArrayLike,
         k: int = 1,
         exclude_conflicts: bool = False
         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the k lenses (across all eyes) whose optical axes lie closest to
        each query direction.

        Returns (indices, distances) of shape (Q, k).
        """
        dirs = np.asarray(directions, dtype=np.float32).reshape(-1, 3)
        Q = dirs.shape[0]
        best_idx = np.full((Q, k), -1, dtype=np.intp)
        best_dist = np.full((Q, k), np.inf, dtype=np.float32)

        for eye in self._eyes:
            idx, dist = eye.query_directions(dirs, k=k, exclude_conflicts=exclude_conflicts)
            if idx.size == 0:
                continue

            combined_idx = np.concatenate([best_idx, idx], axis=1)
            combined_dist = np.concatenate([best_dist, dist], axis=1)
            order = np.argsort(combined_dist, axis=1)[:, :k]
            rows = np.arange(Q)[:, None]
            best_idx = combined_idx[rows, order]
            best_dist = combined_dist[rows, order]

        return best_idx, best_dist

    def query_positions(self,
        positions: ArrayLike,
        k: int = 1,
        exclude_conflicts: bool = False
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the k lenses (across all eyes) closest in world-space position
        to each query point.

        Returns (indices, distances) of shape (Q, k).
        """
        pts = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        Q = pts.shape[0]
        best_idx = np.full((Q, k), -1, dtype=np.intp)
        best_dist = np.full((Q, k), np.inf, dtype=np.float32)

        for eye in self._eyes:
            idx, dist = eye.query_positions(pts, k=k, exclude_conflicts=exclude_conflicts)
            if idx.size == 0:
                continue

            combined_idx = np.concatenate([best_idx, idx], axis=1)
            combined_dist = np.concatenate([best_dist, dist], axis=1)
            order = np.argsort(combined_dist, axis=1)[:, :k]
            rows = np.arange(Q)[:, None]
            best_idx = combined_idx[rows, order]
            best_dist = combined_dist[rows, order]

        return best_idx, best_dist

    def query_lookat(self,
         targets: ArrayLike,
         k: int = 1,
         exclude_conflicts: bool = False
         ) -> np.ndarray:
        """
        Find the k lenses (across all eyes) best looking at each world-space
        target point. See Eye.query_lookat for the scoring.

        Returns global lens indices of shape (Q, k), best-first.
        """
        if k < 1:
            raise ValueError("k must be >= 1")

        q = np.asarray(targets, dtype=np.float32).reshape(-1, 3)
        Q = q.shape[0]
        if self.lens_count == 0:
            return np.empty((Q, k), dtype=np.intp)

        # Score against all lenses (across all eyes)
        pos = self._lens_positions
        dirs = self._lens_directions
        desired = q[:, None, :] - pos[None, :, :]
        norms = np.linalg.norm(desired, axis=-1, keepdims=True)
        np.divide(desired, norms, out=desired, where=norms > 0)
        dots = np.einsum('jk,ijk->ij', dirs, desired)

        if exclude_conflicts:
            dots[:, self.have_conflicts] = -np.inf

        k_eff = min(k, dots.shape[1])
        part = np.argpartition(dots, -k_eff, axis=1)[:, -k_eff:]
        top = np.take_along_axis(dots, part, axis=1)
        order = np.argsort(top, axis=1)[:, ::-1]
        best = np.take_along_axis(part, order, axis=1)

        return best.astype(np.intp)

    def query_cone(self,
        center_direction: ArrayLike,
        angle: float,
        degrees: bool = True,
        exclude_conflicts: bool = False
        ) -> np.ndarray:
        """
        All lenses (across all eyes) whose optical axis lies within 'angle'
        of 'center_direction'.

        Returns global lens indices (union across eyes, arbitrary order).
        """
        hits = [eye.query_cone(center_direction, angle, degrees, exclude_conflicts) for eye in self._eyes]
        hits = [h for h in hits if h.size > 0]
        return np.concatenate(hits) if hits else np.empty(0, dtype=np.intp)

    def query_ball(self,
        center_position: ArrayLike,
        radius: float,
        exclude_conflicts: bool = False
        ) -> np.ndarray:
        """
        All lenses (across all eyes) whose world position lies within 'radius'
        of 'center_position'.

        Returns global lens indices (union across eyes, arbitrary order).
        """
        hits = [eye.query_ball(center_position, radius, exclude_conflicts) for eye in self._eyes]
        hits = [h for h in hits if h.size > 0]
        return np.concatenate(hits) if hits else np.empty(0, dtype=np.intp)

    @property
    def max_gap(self) -> float:
        """
        Largest angular gap (in radians) between any lens and its nearest neighbour,
        over all eyes.
        """
        return float(max((e.max_gap() for e in self._eyes), default=0.0))

    def saccade_field(self) -> np.ndarray:
        """Per-lens microsaccade actuation axis in world coordinates. Shape (N, 3)."""
        return self._saccade_cache.copy()

    # Geometry transforms (on the whole array)

    def translate(self, offset: ArrayLike) -> 'ReceptorArray':
        """
        Translate all lens (and receptor) positions by 'offset'.
        """
        off = np.asarray(offset, dtype=np.float32).reshape(3)
        self._lens_positions += off
        self.rcpt_static_data['position'] += off
        self._invalidate_spatial()
        return self

    def scale(self, factor: float) -> 'ReceptorArray':
        """
        Scale all positions about the origin by 'factor'.

        Note: only world-scale geometry scales. Lens and rhabdomere
        diameters (μm) and the nodal distance (μm) are at ommatidium
        scale and are unchanged.
        """
        f = float(factor)
        self._lens_positions *= f
        self.rcpt_static_data['position'] *= f
        self._invalidate_spatial()
        return self

    def rotate(self, R: ArrayLike) -> 'ReceptorArray':
        """
        Rotate all positions, directions, and tangent frames by the 3x3
        rotation matrix 'R'.

        Cartridge wiring is preserved (relative directions are unchanged).
        """

        Rmat = np.asarray(R, dtype=np.float32)
        if Rmat.shape != (3, 3):
            raise ValueError(f"R must be 3x3, got {Rmat.shape}")

        if not np.allclose(Rmat @ Rmat.T, np.eye(3), atol=1e-3):
            logger.warning("rotate() called with non-orthonormal matrix, results may be off.")

        Rt = Rmat.T.astype(np.float32)

        self._lens_positions = self._lens_positions @ Rt
        self._lens_directions = self._lens_directions @ Rt
        self._local_right = self._local_right @ Rt
        self._local_up = self._local_up @ Rt

        self.lens_static_data['right'] = self._local_right
        self.lens_static_data['up'] = self._local_up
        self.lens_static_data['forward'] = self._lens_directions

        self.rcpt_static_data['position'] = self.rcpt_static_data['position'] @ Rt
        self.rcpt_dynamic_data['direction'] = self.rcpt_dynamic_data['direction'] @ Rt
        self._saccade_cache = self._saccade_cache @ Rt

        self._invalidate_spatial()
        return self

    # Orientation pipeline backwrite

    def _apply_orientation(self, result: OrientationResult) -> None:
        """
        Write an OrientationResult into corresponding data fields.
        Called by 'BundlesAligner.apply()' and during construction.
        """
        N, R = self.lens_count, self.receptors_per_lens
        chi = result.chi.astype(np.float32)
        chirality = result.chirality.astype(np.float32)
        sacc = result.saccade_phasor.astype(np.float32)

        if chi.shape != (N,) or chirality.shape != (N,) or sacc.shape != (N, 3):
            raise ValueError(
                f"OrientationResult shapes (chi {chi.shape}, chirality "
                f"{chirality.shape}, saccade {sacc.shape}) do not match N={N}"
            )

        # Per-lens
        self._bundle_orientation = chi
        self._chirality_arr = chirality
        self.lens_static_data['sacc_x'] = np.einsum('ij,ij->i', sacc, self._local_right)
        self.lens_static_data['sacc_y'] = np.einsum('ij,ij->i', sacc, self._local_up)

        # Per-receptor: rotated focal-plane offsets
        rot_dx, rot_dy = self._kernel.rotated_offsets(chi, chirality)
        rot_offset = np.stack([rot_dx.ravel(), rot_dy.ravel()], axis=-1).astype(np.float32)
        self.rcpt_static_data['rot_offset'] = rot_offset

        # Per-receptor: acceptance ellipse tilt = chi (broadcast)
        self.rcpt_static_data['acc_tilt'] = np.repeat(chi, R).astype(np.float32)

        # Per-receptor: chirality_neg bit in metadata
        is_mirrored = (np.repeat(chirality, R) < 0).astype(np.uint32)
        self.rcpt_static_data['metadata'] = set_metadata_field(
            self.rcpt_static_data['metadata'], 'chirality_neg', is_mirrored
        )

        # Per-receptor: actuated direction (rest direction post-orientation)
        rec_dirs, _ = self._compute_receptor_geometry()
        self.rcpt_dynamic_data['direction'] = rec_dirs

        self._saccade_cache = sacc.copy()
        self._lens_dirty = True

    def _compute_receptor_geometry(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-receptor world-space direction and position.

        Direction is the unit vector from the rotated rhabdomere tip back
        through the nodal point (i.e. the receptor's line of sight on the
        outside world).
        """
        kernel = self._kernel
        N, R = self.lens_count, self.receptors_per_lens
        nd = kernel.nodal_distance_um
        if nd is None:
            if R > 1:
                raise ValueError("kernel.nodal_distance_um is None but R > 1")
            nd = 1.0

        rot_dx, rot_dy = kernel.rotated_offsets(self._bundle_orientation, self._chirality_arr)

        local_tip = np.stack(
            [rot_dx, rot_dy, np.full((N, R), -nd, dtype=np.float32)],
            axis=-1,
        )
        world_tip = (
            local_tip[..., 0:1] * self._local_right[:, None, :]
            + local_tip[..., 1:2] * self._local_up[:, None, :]
            + local_tip[..., 2:3] * self._lens_directions[:, None, :]
        ).reshape(N * R, 3)

        rec_pos = np.repeat(self._lens_positions, R, axis=0)
        rec_dirs = normalise_vectors(-world_tip).astype(np.float32)
        return rec_dirs, rec_pos.astype(np.float32)

    # Cartridges (neural-superposition wiring)

    def wire_cartridges(self,
            snap_radius: float = 0.2,
            assign_radius: float = 1.0,
            angular_dev: float = 25.0,
            scale_dev: float = 0.3,
            pre_cull: bool = False,
            first_ring_only: bool = False,
            neighbour_dist_factor: float = 1.3,
            min_snap_matches: int = 2,
            k_search: int = 30,
            snap_priority_bonus: Optional[float] = None,
        ) -> None:
        """
        Lattice-aware neural-superposition wiring via template snapping.

        For each home lens i (= its central R7/8 hosts the cartridge),
        the kernel's rhabdomere offsets and orientation act as a template in i's tangent plane,
        scaled to local lattice units.
        Search for the best similarity transform that snaps the template onto the
        positions of i's nearest neighbours, and use Hungarian assignment to
        match one-to-one between peripheral rhabdomere positions and neighbour lenses.

        Matches outside 'assign_radius' or with mismatched chirality are ignored.

        Args:
            - snap_radius: Max lattice-unit distance for template-to-neighbour
                pair to count as a successful snap.
            - assign_radius: Max lattice-unit distance allowed in the final
                Hungarian assignment. Matches beyond this radius are dropped.
            - angular_dev: Max angular deviation (degrees) of candidate rotations from identity.
            - scale_dev: Max scale deviation of candidate from 1.0.
            - pre_cull: If True, only consider template anchors with at least
                one immediate neighbour within 'assign_radius' before
                generating candidates.
            - first_ring_only: If True, restrict snap targets to first-ring neighbours.
            - neighbour_dist_factor: A neighbour is first-ring if its distance
                is <= 'neighbour_dist_factor * closest_neighbour_dist'.
            - min_snap_matches: Minimum successful snaps for a candidate to survive scoring.
            - k_search: Number of nearest neighbours to query per lens (up to a few rings).
            - snap_priority_bonus: Penalty added to non-snap cells in the
                Hungarian cost matrix. Forces Hungarian to prefer assignments
                that preserve snap-quality matches over assignments that
                spread the cost uniformly across larger distances.
                If None, defaults to '100 * assign_radius'.
        """
        N = self.lens_count
        R = self.receptors_per_lens

        if R == 1:
            self._cartridges_wired = False
            return

        center = self._kernel.center_index
        periph_rhab = np.array([i for i in range(R) if i != center], dtype=np.intp)
        P = periph_rhab.size

        cartridge_map = np.tile(np.arange(N)[:, None], (1, R))

        if P == 0:
            self._cartridge_map = cartridge_map
            self.rcpt_static_data['cartridge_src'] = np.arange(N * R, dtype=np.uint32)
            self._cartridges_wired = True
            self._lens_dirty = True
            return

        kernel_periph = self._kernel.offsets_um[periph_rhab] - self._kernel.offsets_um[center]
        kernel_scale = float(np.mean(np.linalg.norm(kernel_periph, axis=1)))

        if kernel_scale < 1e-12:
            # Degenerate kernel: peripherals = centre
            self._cartridge_map = cartridge_map
            self.rcpt_static_data['cartridge_src'] = np.arange(N * R, dtype=np.uint32)
            self._cartridges_wired = True
            self._lens_dirty = True
            return

        # Apply per-lens chi + chirality
        rot_dx_all, rot_dy_all = self._kernel.rotated_offsets(self._bundle_orientation, self._chirality_arr)

        # Centre on central rhab, normalise to lattice units
        template_dx_all = (rot_dx_all - rot_dx_all[:, center:center + 1]) / kernel_scale
        template_dy_all = (rot_dy_all - rot_dy_all[:, center:center + 1]) / kernel_scale

        min_required = max(1, min(min_snap_matches, P))
        angular_dev_rad = float(np.radians(angular_dev))

        # Snap priority penalty
        if snap_priority_bonus is None:
            snap_priority_bonus = 100.0 * float(assign_radius)
        snap_priority_bonus = float(snap_priority_bonus)

        for eye in self._eyes:
            n_in_eye = len(eye)
            if n_in_eye < 2:
                continue

            neighb = eye.neighbours(
                lens_indices=eye._lens_indices,
                k=min(k_search, n_in_eye - 1),
                immediate_only=False,
                neighbour_dist_factor=neighbour_dist_factor,
                chirality=self._chirality_arr
            )
            if not neighb:
                continue

            for i_loc in range(n_in_eye):
                i_glob = int(eye._lens_indices[i_loc])
                row_immediate = neighb.is_immediate[i_loc]

                if not np.any(row_immediate):
                    continue

                neighb_global = neighb.indices[i_loc].astype(np.intp)
                neighb_dists = neighb.distances[i_loc]

                local_spacing = float(np.median(neighb_dists[row_immediate]))
                if local_spacing < 1e-12:
                    continue

                # Project neighbours into i's tangent frame (in lattice units)
                central_pos = self._lens_positions[i_glob]
                neighb_vectors = self._lens_positions[neighb_global] - central_pos
                neighb_u = neighb_vectors @ self._local_right[i_glob]
                neighb_v = neighb_vectors @ self._local_up[i_glob]
                neighb_uv = np.column_stack([neighb_u, neighb_v]) / local_spacing

                # Template (peripheral rhabs only) for this lens
                template_uv = np.column_stack([
                    template_dx_all[i_glob, periph_rhab],
                    template_dy_all[i_glob, periph_rhab],
                ])

                z_template = template_uv[:, 0] + 1j * template_uv[:, 1]
                z_neighb = neighb_uv[:, 0] + 1j * neighb_uv[:, 1]
                z_snap = z_neighb[row_immediate] if first_ring_only else z_neighb

                # Candidate similarity transforms w = scale * exp(i*theta)
                if pre_cull:
                    z_immediate = z_neighb[row_immediate]
                    diffs = np.abs(z_template[:, None] - z_immediate[None, :])
                    anchors = np.flatnonzero(np.min(diffs, axis=1) < assign_radius)
                else:
                    anchors = np.arange(P)

                valid_anchors = anchors[np.abs(z_template[anchors]) >= 1e-8]
                candidates_w = [1.0 + 0j]

                if valid_anchors.size > 0:
                    w_all = z_snap[None, :] / z_template[valid_anchors, None]
                    angle_ok = np.abs(np.angle(w_all)) <= angular_dev_rad
                    scale_ok = np.abs(np.abs(w_all) - 1.0) <= scale_dev
                    valid = angle_ok & scale_ok
                    if np.any(valid):
                        candidates_w.extend(w_all[valid].tolist())

                # Score candidates with snap priority Hungarian
                # Score = (-n_snaps, |angle(w)|, sum_of_matched_dists)
                # -> prefer more snaps, then near-identity rotations, then total snap quality
                best_score = (0, np.inf, np.inf)
                best_w = None

                for w in candidates_w:
                    z_placed = w * z_template
                    placed_uv = np.column_stack([z_placed.real, z_placed.imag])
                    cost = np.linalg.norm(placed_uv[:, None, :] - neighb_uv[None, :, :], axis=2)

                    # Penalise non-snap cells
                    cost_with_penalty = cost + np.where(
                        cost >= snap_radius, snap_priority_bonus, 0.0
                    )
                    row_idx, col_idx = linear_sum_assignment(cost_with_penalty)
                    matched_dists = cost[row_idx, col_idx]  # real distances for scoring
                    n_snaps = int(np.sum(matched_dists < snap_radius))
                    if n_snaps < min_required:
                        continue
                    score = (
                        -n_snaps,
                        float(np.abs(np.angle(w))),
                        float(np.sum(matched_dists)),
                    )
                    if score < best_score:
                        best_score = score
                        best_w = w

                if best_w is None:
                    continue

                # Final Hungarian assignment with the chosen candidate
                z_best = best_w * z_template
                corrected_uv = np.column_stack([z_best.real, z_best.imag])
                cost_final = np.linalg.norm(
                    corrected_uv[:, None, :] - neighb_uv[None, :, :], axis=2
                )
                rhab_idx, om_idx = linear_sum_assignment(
                    cost_final + np.where(cost_final >= snap_radius, snap_priority_bonus, 0.0))

                valid_mask = (cost_final[rhab_idx, om_idx] < assign_radius) & neighb.same_chirality[i_loc][om_idx]

                if np.any(valid_mask):
                    cartridge_map[i_glob, periph_rhab[rhab_idx[valid_mask]]] = neighb_global[om_idx[valid_mask]]

        self._cartridge_map = cartridge_map
        self.rcpt_static_data['cartridge_src'] = (cartridge_map * R + np.arange(R)).flatten().astype(np.uint32)
        self._cartridges_wired = True
        self._lens_dirty = True

        # Diagnostics
        is_periph = np.ones(R, dtype=bool)
        is_periph[center] = False
        recv_conf, don_conf = np.zeros(N, dtype=bool), np.zeros(N, dtype=bool)

        for r in range(R):
            if is_periph[r]:
                recv_conf |= (cartridge_map[:, r] == np.arange(N))
                don_conf |= (np.bincount(cartridge_map[:, r], minlength=N) != 1)

        self.receiving_conflicts = recv_conf
        self.donation_conflicts = don_conf
        self.have_conflicts = recv_conf | don_conf
        self._invalidate_spatial()  # ensure conflict-free trees are updated

    # Private helpers

    @staticmethod
    def _detect_eye_islands(
        positions: np.ndarray,
        k: int = 6,
        distance_multiplier: float = 2.5,
        ) -> np.ndarray:
        """
        Spatial connected-components clustering.

        Builds the kNN graph and keeps an edge (i, j) only if its length is
        below 'distance_multiplier * median_edge_length' over the full graph.
        Connected components of the resulting graph are the eye islands.

        Returns:
            (N,) uint32 array of island labels (0-indexed, in arbitrary order).
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        N = positions.shape[0]
        if N == 0:
            return np.empty(0, dtype=np.uint32)
        if N == 1:
            return np.zeros(1, dtype=np.uint32)

        actual_k = min(k, N - 1)
        tree = cKDTree(positions)
        dists, idx = tree.query(positions, k=actual_k + 1)
        dists = dists[:, 1:]   # drop self
        idx = idx[:, 1:]

        median_edge = float(np.median(dists))
        if median_edge <= 0.0:
            # Degenerate: all points coincident, return a single island
            return np.zeros(N, dtype=np.uint32)
        threshold = distance_multiplier * median_edge

        keep = dists < threshold
        rows = np.repeat(np.arange(N), actual_k)[keep.ravel()]
        cols = idx.ravel()[keep.ravel()]
        data = np.ones(rows.size, dtype=np.int8)
        adj = csr_matrix((data, (rows, cols)), shape=(N, N))

        _, labels = connected_components(adj, directed=False, return_labels=True)
        return labels.astype(np.uint32)

    @staticmethod
    def _resolve_eye_ids(
        positions: np.ndarray,
        eye_ids: Optional[ArrayLike],
        N: int,
        max_eyes: int = 8,
        ) -> np.ndarray:
        """
        Resolve per-lens eye membership.

        If 'eye_ids' is provided, use it directly.
        Otherwise, infer islands by spatial clustering on 'positions' and
        relabel them in order of ascending centroid x (so eye 0 is the
        leftmost island). Caps at 'max_eyes' due to the 3-bit metadata field.
        """

        if eye_ids is not None:
            arr = np.asarray(eye_ids, dtype=np.uint32).reshape(-1)
            if arr.size != N:
                raise ValueError(f"eye_ids size {arr.size} must equal N={N}")
            if int(arr.max()) >= max_eyes:
                raise ValueError(f"eye_ids exceed {max_eyes - 1} (3-bit field), got max={arr.max()}")
            return arr

        raw = ReceptorArray._detect_eye_islands(positions)
        unique = np.unique(raw)
        if unique.size > max_eyes:
            raise ValueError(
                f"Detected {unique.size} eye islands but the metadata field "
                f"supports at most {max_eyes}. Pass 'eye_ids=' explicitly or "
                f"check that the lens positions are not over-fragmented."
            )

        # Reorder: ascending centroid x. Stable tie-break on centroid y, z.
        centroids = np.stack([positions[raw == u].mean(axis=0) for u in unique])
        order = np.lexsort((centroids[:, 2], centroids[:, 1], centroids[:, 0]))

        # Build map raw -> new index
        remap = np.empty(unique.size, dtype=np.uint32)
        for new_id, raw_pos in enumerate(order):
            remap[unique[raw_pos]] = np.uint32(new_id)
        return remap[raw]

    def _build_eyes(self, midline_fraction: float = 0.1) -> None:
        """
        Build per-eye objects with a side label inferred from the eye centroid.

        Each eye gets side='left' / 'right' / 'midline'. The side is decided
        by comparing the eye's centroid_x to a threshold scaled to the *whole*
        receptor array, not the individual eye, so that small midline eyes
        (e.g. a median ocellus) get classified correctly even when paired eyes
        are present.

        With midline_fraction=0.1, an eye whose centroid is within 10% of the
        most-lateral lens x is considered on the midline.
        """
        self._eyes = []
        unique_ids = np.unique(self._lens_eye_ids)
        if self._lens_positions.shape[0] > 0:
            scale = float(np.max(np.abs(self._lens_positions[:, 0])))
        else:
            scale = 0.0
        threshold = midline_fraction * scale

        for eid in unique_ids:
            lens_idx = np.flatnonzero(self._lens_eye_ids == eid).astype(np.intp)
            centroid_x = float(self._lens_positions[lens_idx, 0].mean())
            if scale == 0.0:
                side = 'midline'
            elif abs(centroid_x) <= threshold:
                side = 'midline'
            else:
                side = 'left' if centroid_x < 0.0 else 'right'
            self._eyes.append(Eye(self, int(eid), lens_idx, side=side))

    def _invalidate_spatial(self) -> None:
        for eye in self._eyes:
            eye._invalidate()
        self._lens_dirty = True

    @staticmethod
    def _broadcast_ioa(
        value: ArrayLike,
        N: int,
        ) -> Tuple[np.ndarray, np.ndarray]:

        arr = np.asarray(value, dtype=np.float32)
        if arr.shape == (2,):
            axes = np.tile(arr, (N, 1))
        elif arr.shape == (N, 2):
            axes = arr.copy()
        else:
            raise ValueError(
                f"interommatidial_angles_rad must be shape (2,) or ({N}, 2), got {arr.shape}"
            )
        tilts = np.zeros(N, dtype=np.float32)
        return axes, tilts

    def _broadcast_acceptance(self,
        value: ArrayLike,
        N: int,
        R: int,
        ) -> np.ndarray:

        arr = np.asarray(value, dtype=np.float32)
        if arr.shape == (R,):
            # Same per-receptor for every lens, circular
            acc = np.tile(arr[:, None], (N, 1, 2)).reshape(N * R, 2)
        elif arr.shape == (N, R):
            acc = np.broadcast_to(arr[..., None], (N, R, 2)).reshape(N * R, 2).astype(np.float32)
        elif arr.shape == (N, R, 2):
            acc = arr.reshape(N * R, 2).astype(np.float32)
        else:
            raise ValueError(
                f"acceptance_angles_rad must be shape ({R},), ({N}, {R}), or ({N}, {R}, 2), got {arr.shape}"
            )
        return acc

    def _compute_acceptance_baseline(
        self,
        eye_parameter: Optional[Union[float, Tuple[float, float]]] = None,
        ) -> np.ndarray:
        """
        Per-receptor acceptance half-widths (rad), as (M, 2) (minor, major).

        Two cases:

        - Snyder optics (kernel.nodal_distance_um is not None):
              rho_geom = arctan(d_rhab / nodal_distance)
              rho_diff = lambda / lens_diameter
              rho = sqrt(rho_geom^2 + rho_diff^2)
          Then scaled by the eye parameter p:
              acc_minor = p_min * rho
              acc_major = p_maj * rho
          p = 1 -> pure Snyder, p > 1 -> wider RFs than Snyder predicts, p < 1 -> narrower
          Tuple p = (p_min, p_maj) gives anisotropic RFs (same Snyder baseline, different per-axis scaling).

          Note / TODO: per-receptor Snyder baseline is currently isotropic (single rhab_diameter and single lens_diameter). Anisotropy comes only from p_min vs p_maj.

        - No optics (kernel.nodal_distance_um is None):
              acc_minor = p_min * ioa_minor * (d_rhab / max(d_rhab))
              acc_major = p_maj * ioa_major * (d_rhab / max(d_rhab))
          Uses lattice spacing: RF width is the local IOA scaled
          by the eye parameter p and modulated by the per-receptor
          rhabdomere diameter (smaller rhabdomeres still get smaller RFs).

        Args:
            - eye_parameter: scalar 'p' applied to both axes, or a tuple '(p_min, p_maj)' for anisotropic scaling.
                None defaults to 1.0.
        """
        N, R = self.lens_count, self.receptors_per_lens
        kernel = self._kernel

        # Resolve eye parameter to (p_min, p_maj)
        if eye_parameter is None:
            p_min = p_maj = 1.0
        elif isinstance(eye_parameter, (int, float, np.number)):
            p_min = p_maj = float(eye_parameter)
        else:
            p = tuple(eye_parameter)
            if len(p) != 2:
                raise ValueError(
                    f"eye_parameter must be a scalar or 2-tuple, got length {len(p)}"
                )
            p_min, p_maj = float(p[0]), float(p[1])

        rhab = kernel.diameters_um
        wl_um = (kernel.wavelengths_nm * 1e-3).astype(np.float32)

        if kernel.nodal_distance_um is not None:
            # Snyder mode: physical baseline from diffraction + geometric optics
            nd = self.lens_static_data['nodal_distance_um'].astype(np.float32)
            ld = self.lens_static_data['lens_diameter_um'].astype(np.float32)

            rho_geom = np.arctan(rhab[None, :] / np.clip(nd[:, None], 1e-6, None))
            rho_diff = wl_um[None, :] / np.clip(ld[:, None], 1e-6, None)
            rho = np.sqrt(rho_geom ** 2 + rho_diff ** 2).astype(np.float32)

            acc_min = (p_min * rho).reshape(N * R)
            acc_maj = (p_maj * rho).reshape(N * R)
        else:
            # No optical model: lattice spacing mode
            ioa_axes = self.lens_static_data['ioa_axes'].astype(np.float32)
            ioa_minor = ioa_axes[:, 0]
            ioa_major = ioa_axes[:, 1]

            max_d = float(np.max(rhab)) if rhab.size > 0 else 1.0
            rel_d = (rhab / max_d).astype(np.float32) if max_d > 0 else np.ones(R, dtype=np.float32)

            acc_min = (np.repeat(p_min * ioa_minor, R)
                       * np.tile(rel_d, N)).astype(np.float32)
            acc_maj = (np.repeat(p_maj * ioa_major, R)
                       * np.tile(rel_d, N)).astype(np.float32)

        acc = np.column_stack([acc_min, acc_maj]).astype(np.float32)
        return acc

    def _compute_ioa_baseline(
            self,
            k_search: int = 8,
            neighbour_dist_factor: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Per-lens lattice properties from the local first-ring neighbour graph.

        Computes three things:

        - IOA (minor, major): angular separations in optical axis space.
            Sorted over first-ring neighbours.
            minor = mean of the 2 smallest,
            major = mean of the 2 largest
        - Tilt: lattice rotation in the lens's tangent frame, computed via the hexatic order
              Ψ6 over first-ring neighbour bearings.
              Magnitude of Ψ6: 1.0 = perfect hex, 0.0 = isotropic disorder
        - Lens spacing: median first-ring distance in world units

        Args:
            - k_search: number of neighbours to query per lens (~ a few rings).
            - neighbour_dist_factor: a neighbour is first-ring if its
                distance <= neighbour_dist_factor * closest.

        Returns:
            ioa_axes: (N, 2) (minor, major) in radians
            ioa_tilts: (N,) lattice tilt in radians, in [-π/6, +π/6]
            lens_spacing: (N,) median first-ring distance, world units

        Note: receptors get a separate per-receptor 'acc_tilt' written
        by the orientation pipeline (chi).
        The ioa_tilt here is the lattice's own rotation, the orientation pipeline does not overwrite this.
        """
        N = self.lens_count
        ioa_minor = np.zeros(N, dtype=np.float32)
        ioa_major = np.zeros(N, dtype=np.float32)
        ioa_tilts = np.zeros(N, dtype=np.float32)
        lens_spacing = np.zeros(N, dtype=np.float32)

        for eye in self._eyes:
            n = len(eye)
            if n < 2:
                continue
            k_eff = min(k_search, n - 1)

            neighbours = eye.neighbours(
                lens_indices=eye._lens_indices,
                k=k_eff,
                immediate_only=False,
                neighbour_dist_factor=neighbour_dist_factor,
            )
            if not neighbours:
                continue

            eye_idx = eye._lens_indices
            home_dirs = self._lens_directions[eye_idx]
            home_pos = self._lens_positions[eye_idx]
            home_right = self._local_right[eye_idx]
            home_up = self._local_up[eye_idx]

            neighb_global = neighbours.indices
            neighb_dirs = self._lens_directions[neighb_global]
            neighb_pos = self._lens_positions[neighb_global]
            is_immediate = neighbours.is_immediate
            n_imm = np.maximum(is_immediate.sum(axis=1), 1).astype(np.float64)

            # Optical IOA: angular separation in optical-axis space
            dots = np.clip(np.einsum('ik,ijk->ij', home_dirs, neighb_dirs), -1.0, 1.0)
            angular_sep = np.arccos(dots).astype(np.float32)

            # Bearings in home lens's tangent frame
            delta_pos = neighb_pos - home_pos[:, None, :]
            proj_x = np.einsum('ijk,ik->ij', delta_pos, home_right)
            proj_y = np.einsum('ijk,ik->ij', delta_pos, home_up)
            bearings = np.arctan2(proj_y, proj_x)

            # Hexatic order parameter Ψ6( over first ring only)
            phasors = np.where(is_immediate, np.exp(1j * 6 * bearings), 0.0 + 0.0j)
            z_avg = phasors.sum(axis=1) / n_imm
            e_tilts = (np.angle(z_avg) / 6.0).astype(np.float32)
            e_psi6_mag = np.abs(z_avg)

            # IOA: mean of 2 smallest / 2 largest first ring separations
            sep_for_min = np.where(is_immediate, angular_sep, np.inf)
            sep_for_max = np.where(is_immediate, -angular_sep, np.inf)

            min2 = np.sort(sep_for_min, axis=1)[:, :2]
            max2 = -np.sort(sep_for_max, axis=1)[:, :2]
            min2_valid = np.where(np.isfinite(min2), min2, np.nan)
            max2_valid = np.where(np.isfinite(max2), max2, np.nan)
            with np.errstate(all='ignore'):
                e_ioa_minor = np.nanmean(min2_valid, axis=1).astype(np.float32)
                e_ioa_major = np.nanmean(max2_valid, axis=1).astype(np.float32)

            # neighbourhood too sparse: fallback to simple mean over whatever is immediate, zero if nothing
            sparse_mask = is_immediate.sum(axis=1) < 2
            if np.any(sparse_mask):
                sep_imm = np.where(is_immediate, angular_sep, np.nan)
                with np.errstate(all='ignore'):
                    fallback = np.nanmean(sep_imm, axis=1)
                fallback = np.where(np.isfinite(fallback), fallback, 0.0).astype(np.float32)
                e_ioa_minor = np.where(sparse_mask, fallback, e_ioa_minor)
                e_ioa_major = np.where(sparse_mask, fallback, e_ioa_major)
                e_tilts = np.where(sparse_mask, 0.0, e_tilts).astype(np.float32)

            e_ioa_minor = np.where(np.isfinite(e_ioa_minor), e_ioa_minor, 0.0)
            e_ioa_major = np.where(np.isfinite(e_ioa_major), e_ioa_major, 0.0)

            # Lens spacing: median first-ring distance (in world units)
            nbr_dist = np.linalg.norm(delta_pos, axis=2)
            nbr_dist_masked = np.where(is_immediate, nbr_dist, np.nan)
            with np.errstate(all='ignore'):
                e_lens_spacing = np.nanmedian(nbr_dist_masked, axis=1).astype(np.float32)
            e_lens_spacing = np.where(np.isfinite(e_lens_spacing), e_lens_spacing, 0.0)

            ioa_minor[eye_idx] = e_ioa_minor
            ioa_major[eye_idx] = e_ioa_major
            ioa_tilts[eye_idx] = e_tilts
            lens_spacing[eye_idx] = e_lens_spacing

            logger.debug(
                f"Eye {eye.eye_id} lattice |Ψ6|: {float(np.mean(e_psi6_mag)):.3f} "
                f"(1.0 = perfect hex, 0.0 = isotropic disorder); "
                f"median lens_spacing: {float(np.median(e_lens_spacing)):.3f}"
            )

        ioa_axes = np.stack([ioa_minor, ioa_major], axis=-1).astype(np.float32)
        return ioa_axes, ioa_tilts, lens_spacing


##

if __name__ == '__main__':
    from insectvision.compound_eyes.kernel import drosophila_kernel

    ra = ReceptorArray.from_sphere(n=500)

    print(ra)
    print(f"  Total receptors: {ra.total_receptors}")
    print(f"  Eyes: {ra.eyes}")

    # Drosophila with flow direction
    droso = drosophila_kernel()

    ra2 = ReceptorArray.from_sphere(
        n=1600,
        kernel=droso,
        flow_direction=[1.0, 0.0, 0.0],  # Anterior flow
    )
    print(ra2)
    print(f"  Bundle orientations (first 5): {ra2.lenses[:5].bundle_orientations}")
    print(f"  Chiralities (first 5): {ra2.lenses[:5].chiralities}")
    print(f"  Cartridges wired: {ra2._cartridges_wired}")
    print(f"  Cartridge[0]: {ra2.cartridges[0]}")
