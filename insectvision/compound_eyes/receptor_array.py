import logging
from typing import List, Optional, Tuple, Union
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

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
    BundleOrientationField,
    OrientationResult,
    trivial_orientation,
    orientation_from_chi_chirality,
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
    Views (LensView, ReceptorView, Ommatidium, Cartridge, Eye) wrap subsets of these arrays for typed access.

    Args:
        - directions: (N, 3) array_like, Lens optical-axis (forward) directions. Will be normalised.
        - positions: (N, 3) array_like, Lens world positions.
        - kernel: RhabdomereKernel (optional), Per-species bundle model. Defaults to a single panchromatic receptor.
        - eye_ids: (N,) array_like of uint (optional), Eye membership for each lens (0-7).
            If None, splits in two eyes on x: x < 0 -> 0 (left), x ≥ 0 -> 1 (right).
        - lens_diameter_um: float or (N,) array_like, Lens aperture (μm). Default 16 μm.
        - interommatidial_angles_rad: (2,), (N, 2), or None, Per-lens (minor, major) IOA.
            If None, computed as the mean angular distance to lattice neighbours (isotropic baseline).
        - acceptance_angles_rad: (R,), (N, R), (N, R, 2), or None, Per-receptor acceptance half-widths.
            If None, computed via Snyder's combined geometric + diffraction formula.
        - bundle_orientations: (N,) array_like or None, Override chi from the orientation pipeline (rad).
        - chiralities: (N,) array_like or None, Override chirality from the orientation pipeline (±1).
        - orientation: BundleOrientationField or None, Explicit pipeline configuration.
            Required when R > 1 unless both 'bundle_orientations' and 'chiralities' are supplied.
        - flow_direction: (3,) array_like or None, Shorthand for `orientation=BundleOrientationField(flow_direction)`.
        - auto_wire_cartridges: bool, If True (default) and R > 1, calls `wire_cartridges()` after construction.
    """

    def __init__(self,
        directions: ArrayLike,
        positions: ArrayLike,
        kernel: Optional[RhabdomereKernel] = None,
        eye_ids: Optional[ArrayLike] = None,
        lens_diameter_um: Union[float, ArrayLike] = 16.0,
        interommatidial_angles_rad: Optional[ArrayLike] = None,
        acceptance_angles_rad: Optional[ArrayLike] = None,
        bundle_orientations: Optional[ArrayLike] = None,
        chiralities: Optional[ArrayLike] = None,
        orientation: Optional[BundleOrientationField] = None,
        flow_direction: Optional[ArrayLike] = None,
        auto_wire_cartridges: bool = True,
    ):

        # Validate geometry

        dirs = np.asarray(directions, dtype=np.float32).reshape(-1, 3)
        pos = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        if dirs.shape != pos.shape:
            raise ValueError(
                f"directions and positions must have the same shape; "
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

        # Eye membership
        self._lens_eye_ids = self._resolve_eye_ids(self._lens_positions, eye_ids, N)
        # TODO: Any n>2 eyes model should have left and right groups of eyes

        # Allocate structured arrays
        self.lens_static_data = np.zeros(N, dtype=LENS_STATIC_DTYPE)
        self.lens_dynamic_data = np.zeros(N, dtype=LENS_DYNAMIC_DTYPE)
        self.rcpt_static_data = np.zeros(N * R, dtype=RCPT_STATIC_DTYPE)
        self.rcpt_dynamic_data = np.zeros(N * R, dtype=RCPT_DYNAMIC_DTYPE)

        # Fill lens static data
        self.lens_static_data['right'] = self._local_right
        self.lens_static_data['up'] = self._local_up
        self.lens_static_data['forward'] = self._lens_directions

        nd_value = kernel.nodal_distance_um if kernel.nodal_distance_um is not None else 1.0
        self.lens_static_data['nodal_distance_um'] = nd_value

        ld = np.atleast_1d(np.asarray(lens_diameter_um, dtype=np.float32))
        if ld.size == 1:
            self.lens_static_data['lens_diameter_um'] = ld.item()
        elif ld.size == N:
            self.lens_static_data['lens_diameter_um'] = ld
        else:
            raise ValueError(f"lens_diameter_um size {ld.size} must be 1 or N={N}")

        # Broadcast kernel-level photomechanical biophysics to every lens
        self.lens_static_data['tau_rise'] = kernel.tau_rise
        self.lens_static_data['tau_relax'] = kernel.tau_relax
        self.lens_static_data['tau_fast'] = kernel.tau_fast
        self.lens_static_data['tau_adapt'] = kernel.tau_adapt
        self.lens_static_data['gain_lat_um'] = kernel.gain_lat_um
        self.lens_static_data['gain_ax_um'] = kernel.gain_ax_um

        # Build Eye objects (KDtrees & neighbour graphs lazy)
        self._eyes: List[Eye] = []
        self._build_eyes()

        # Interommatidial angles
        if interommatidial_angles_rad is not None:
            ioa_axes, ioa_tilts = self._broadcast_ioa(interommatidial_angles_rad, N)
        else:
            ioa_axes, ioa_tilts = self._compute_ioa_baseline()
        self.lens_static_data['ioa_axes'] = ioa_axes
        self.lens_static_data['ioa_tilt'] = ioa_tilts

        # Fill receptors static data
        self.rcpt_static_data['position'] = np.repeat(self._lens_positions, R, axis=0)
        self.rcpt_static_data['sensitivity'] = np.tile(kernel.sensitivity, (N, 1))
        self.rcpt_static_data['tau_membrane'] = kernel.tau_membrane
        self.rcpt_static_data['rhab_diameter_um'] = np.tile(kernel.diameters_um, N)
        self.rcpt_static_data['wavelength_um'] = np.tile(kernel.wavelengths_nm * 1e-3, N)
        # cartridge_src defaults to self (will be overwritten by wire_cartridges)
        self.rcpt_static_data['cartridge_src'] = np.arange(N * R, dtype=np.uint32)

        # Acceptance angles (rest + initial dynamic)
        if acceptance_angles_rad is not None:
            acc = self._broadcast_acceptance(acceptance_angles_rad, N, R)
        else:
            acc = self._compute_acceptance_baseline()

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
            result = orientation_from_chi_chirality(self, bundle_orientations, chiralities)
        else:
            # Pipeline needed
            if orientation is None:
                if flow_direction is None:
                    raise ValueError(
                        "Multi-receptor kernel (R={}) requires bundle orientation. "
                        "Supply one of:\n"
                        "  flow_direction=[ax, ay, az]\n"
                        "  orientation=BundleOrientationField(flow_direction=..., ...)\n"
                        "  both bundle_orientations=... and chiralities=..."
                        .format(R)
                    )
                orientation = BundleOrientationField(flow_direction)
            result = orientation.compute(
                self,
                override_chi=bundle_orientations,
                override_chirality=chiralities,
            )

        self._apply_orientation(result)

        # Book keeping
        self._cartridges_wired = False
        self._cartridge_members: dict = {}
        self._lens_dirty = True

        # Cartridges wiring
        if auto_wire_cartridges and R > 1:
            self.wire_cartridges()

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

            # Resolve eye ids: explicit field, else infer from is_left / is_right
            if 'eye_ids' not in kwargs:
                eye_ids = first_present(['eye_ids', 'eye_id'])
                if eye_ids is None:
                    is_left = first_present(['l', 'left', 'is_left'])
                    if is_left is not None:
                        # Convention: 0 = left, 1 = right (matches the default x-split)
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

    # Sizing and basic accessors

    def __len__(self) -> int:
        return int(self._lens_directions.shape[0])

    @property
    def lens_count(self) -> int:
        return int(self._lens_directions.shape[0])

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

    def __repr__(self) -> str:
        return (
            f"ReceptorArray(N={self.lens_count}, R={self.receptors_per_lens}, "
            f"eyes={len(self._eyes)}, kernel={self._kernel.name!r})"
        )

    # Animal-wide spatial queries (dispatch across eyes)

    def query_directions(self,
        directions: ArrayLike,
        k: int = 1,
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
            eye_k = min(k, len(eye))
            if eye_k == 0:
                continue
            idx, dist = eye.query_directions(dirs, k=eye_k)
            combined_idx = np.concatenate([best_idx, idx], axis=1)
            combined_dist = np.concatenate([best_dist, dist], axis=1)
            order = np.argsort(combined_dist, axis=1)[:, :k]
            rows = np.arange(Q)[:, None]
            best_idx = combined_idx[rows, order]
            best_dist = combined_dist[rows, order]

        return best_idx, best_dist

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
        Called by 'BundleOrientationField.apply()' and during construction.
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

    def wire_cartridges(self) -> None:
        """
        Wire each peripheral receptor to a cartridge anchored at the
        central R7/8 of the neighbouring ommatidium whose optical axis is
        closest to the peripheral's line of sight.
        """
        R = self.receptors_per_lens
        if R == 1:
            self._cartridges_wired = False
            return

        center = self._kernel.center_index
        cartridge_src = self.rcpt_static_data['cartridge_src'].copy()

        for eye in self._eyes:
            if len(eye) == 0:
                continue
            central_rcpt_indices = (eye._lens_indices * R + center).astype(np.intp)
            central_directions = self.rcpt_dynamic_data['direction'][central_rcpt_indices]
            tree = cKDTree(central_directions)

            for r in range(R):
                if r == center:
                    # Central receptor: cartridge_src is self (default identity).
                    rcpt_indices = (eye._lens_indices * R + r).astype(np.intp)
                    cartridge_src[rcpt_indices] = rcpt_indices.astype(np.uint32)
                    continue

                rcpt_indices = (eye._lens_indices * R + r).astype(np.intp)
                rcpt_directions = self.rcpt_dynamic_data['direction'][rcpt_indices]
                _, nearest_central_local = tree.query(rcpt_directions, k=1)
                cartridge_src[rcpt_indices] = central_rcpt_indices[nearest_central_local].astype(np.uint32)

        self.rcpt_static_data['cartridge_src'] = cartridge_src

        # Also build inverse mapping for fast Cartridge construction
        self._cartridge_members = {}
        for global_central in np.unique(cartridge_src):
            members = np.flatnonzero(cartridge_src == global_central).astype(np.intp)
            self._cartridge_members[int(global_central)] = members

        self._cartridges_wired = True
        self._lens_dirty = True

    # Private helpers

    @staticmethod
    def _resolve_eye_ids(
        positions: np.ndarray,
        eye_ids: Optional[ArrayLike],
        N: int,
    ) -> np.ndarray:

        if eye_ids is None:
            # Auto-split on x-coordinate. x < 0 -> eye 0 (left), x >= 0 -> eye 1 (right).
            return (positions[:, 0] >= 0).astype(np.uint32)

        arr = np.asarray(eye_ids, dtype=np.uint32).reshape(-1)
        if arr.size != N:
            raise ValueError(f"eye_ids size {arr.size} must equal N={N}")
        if int(arr.max()) > 7:
            raise ValueError(f"eye_ids exceed 3-bit range, max is 7, got {arr.max()}")

        return arr

    def _build_eyes(self) -> None:
        unique_ids = np.unique(self._lens_eye_ids)
        self._eyes = [
            Eye(self, int(eid), np.flatnonzero(self._lens_eye_ids == eid).astype(np.intp))
            for eid in unique_ids
        ]

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

    def _compute_acceptance_baseline(self) -> np.ndarray:
        """
        Snyder's combined acceptance half-width:
            rho² = rho_geom² + rho_diff²
            rho_geom = arctan(rhab_diameter / nodal_distance)
            rho_diff = wavelength / lens_diameter

        Isotropic per receptor (minor = major). For anisotropic ellipse aligned with the IOA,
        pass 'acceptance_angles_rad' of shape (N, R, 2) at construction.
        """
        # TODO: Should probably be anisotropic by default?
        
        N, R = self.lens_count, self.receptors_per_lens
        kernel = self._kernel

        nd = self.lens_static_data['nodal_distance_um'].astype(np.float32)
        ld = self.lens_static_data['lens_diameter_um'].astype(np.float32)
        rhab = kernel.diameters_um
        wl_um = (kernel.wavelengths_nm * 1e-3).astype(np.float32)

        rho_geom = np.arctan(rhab[None, :] / np.clip(nd[:, None], 1e-6, None))
        rho_diff = wl_um[None, :] / np.clip(ld[:, None], 1e-6, None)
        rho = np.sqrt(rho_geom ** 2 + rho_diff ** 2).astype(np.float32)

        acc = np.stack([rho, rho], axis=-1).reshape(N * R, 2)
        return acc

    def _compute_ioa_baseline(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-lens (minor, major) IOA from the local neighbour graph.

        Baseline: isotropic (minor == major) using the mean angular distance to lattice neighbours.
        For an anisotropic per-lens IOA, pass 'interommatidial_angles_rad' of shape (N, 2) at construction.
        """
        # TODO: Should be anisotropic by default? Info is there...
        
        N = self.lens_count
        ioa = np.zeros(N, dtype=np.float32)

        for eye in self._eyes:
            graph = eye._ensure_neighbour_graph(k=6)
            if graph.size == 0:
                continue

            local_dirs = self._lens_directions[eye._lens_indices]
            neigh_dirs = local_dirs[graph]
            dots = np.einsum('ik,ijk->ij', local_dirs, neigh_dirs)
            dots = np.clip(dots, -1.0, 1.0)
            angles = np.arccos(dots).astype(np.float32)
            ioa[eye._lens_indices] = angles.mean(axis=1)

        ioa_axes = np.stack([ioa, ioa], axis=-1).astype(np.float32)
        ioa_tilts = np.zeros(N, dtype=np.float32)
        return ioa_axes, ioa_tilts


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
