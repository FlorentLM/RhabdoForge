"""
The main user-facing CompoundEyeModel class that wraps the data buffers.
Does all the building, and exposes properties that the other views can re-expose.
"""
from pathlib import Path
from typing import Optional, Union, Tuple, List
import numpy as np
from numpy.typing import ArrayLike

from insectvision.utils.shared import norm_l2, broadcast_1d, broadcast_to_shape
from insectvision.engine.meshes import icosphere, fibonacci_sphere
from insectvision.engine.world_utils import WORLD_FORWARD
from insectvision.geometry.circular import resultant
from insectvision.geometry.hexatic import hexatic_axis_angle, hexatic_order
from insectvision.geometry.linalg import tangent_frames, local_to_world
from insectvision.geometry.neighbours import smooth_phasors, knn, smooth_field_partitioned
from insectvision.geometry.spherical import angle_to_chord
from insectvision.compound_eyes.buffers import Buffer
from insectvision.compound_eyes.rhabdomeres import RhabdomereBundle
from insectvision.compound_eyes.helpers.neural_superposition import (
    wire_neural_superposition, get_conflict_masks, get_noconflict_masks, refine_chi
)
from insectvision.compound_eyes.helpers.acceptance import (
    AcceptanceModel, SnyderAcceptance, SamplingAcceptance, LensOptics, RhabdomereOptics, ExplicitAcceptance
)
from insectvision.compound_eyes.helpers.ommatidia_lattice import voronoi_estimation
from insectvision.compound_eyes.helpers.alignment import BundlesAligner, AlignmentResult, apply_chirality, trivial_alignment
from insectvision.compound_eyes.views import SpatialQueries, BaseView, logger, OmmatidiumView, EyeView, RhabdomereView


class Model(SpatialQueries, BaseView):
    """
    A (set of) compound eyes specified as N ommatidia positions / directions
    and a bundle of R rhabdomeres per ommatidium.
    """

    # Fraction spacing that the lens diameter occupies (1.0 = fully touching, 0.9 leaves a small interommatidial cuticle gap, etc)
    HEX_PACKING_FACTOR = 0.9

    def __init__(self,
            directions: ArrayLike,
            positions: ArrayLike,
            bundle: Optional['RhabdomereBundle'] = None,
            eye_indices: Optional[ArrayLike] = None,
            aperture_um: Optional[Union[float, ArrayLike]] = None,
            interommatidial_angles_rad: Optional[ArrayLike] = None,
            acceptance: Optional['AcceptanceModel'] = None,
            bundle_orientations: Optional[ArrayLike] = None,
            chiralities: Optional[ArrayLike] = None,
            orientation: Optional['BundlesAligner'] = None,
            flow_direction: Optional[ArrayLike] = None,
            neural_superposition: bool = False
        ):

        self._spatial = {}

        # ============ Validate shapes ============

        pos = np.asarray(positions).reshape(-1, 3)
        fwd = np.asarray(directions).reshape(-1, 3)

        if fwd.shape != pos.shape:
            raise ValueError('Directions and Positions must have the same shape.')

        N = fwd.shape[0]
        if N == 0:
            raise ValueError('CompoundEyeModel needs at least 1 ommatidium')
        self._N = N

        # ============ Rhabdomere bundle ============

        if bundle is not None:
            self._bundle = bundle
        else:
            self._bundle = RhabdomereBundle()  # default R=1
        self._R = self._bundle.count

        # ============ All sizes now known -> allocate the buffer ============

        self._buf = Buffer(shape=(self._N, self._R))

        # ============ Apply geometry ============

        self._buf['position'] = pos
        self._buf['forward'] = norm_l2(fwd).astype(np.float32)
        self._buf['right'], self._buf['up'] = tangent_frames(self._buf['forward'])

        # ============ Fill bundle-originating stuff ============

        self._buf['tau_rise'] = self._bundle.tau_rise
        self._buf['tau_relax'] = self._bundle.tau_relax
        self._buf['tau_adapt_fast'] = self._bundle.tau_fast
        self._buf['tau_adapt_slow'] = self._bundle.tau_adapt
        self._buf['ampl_lateral'] = self._bundle.ampl_lat_um
        self._buf['ampl_axial'] = self._bundle.ampl_ax_um
        self._buf['sensitivity'] = self._bundle.sensitivity
        self._buf['tau_membrane'] = self._bundle.tau_membrane
        self._buf['diameter_um'] = self._bundle.diameters_um
        self._buf['wavelength_um'] = self._bundle.wavelengths_nm * 1e-3

        # ============ First need to know Eye membership ============

        self._spatial_version = 0

        if eye_indices is not None:
            arr = np.asarray(eye_indices, dtype=np.uint32).reshape(-1)

            max_eyes = self._buf.max_value('eye_id') # 3 bits, so 0-7 eyes
            if arr.size != self._N:
                raise ValueError(f"eye_indices size {arr.size} must equal N={self._N}")
            if int(arr.max()) >= max_eyes + 1:
                raise ValueError(f"eye_indices exceed {max_eyes} (3-bit field)")

            eye_membership = eye_indices
        else:
            eye_membership = self._identify_distinct_eyes()

        self._buf['eye_id'] = eye_membership

        self._eyes = self._instantiate_eyes()

        # ============ Ommatidia lattice properties ============

        # Lattice properties from the first-ring neighbour graph:
        # IOA (minor, major), tilt, hexatic order, and per-ommatidium spacing.
        ioa_angles, ioa_tilts, ioa_hexatic_order, ioa_spacing = self._compute_lattice_properties()

        self._hexatic_order = ioa_hexatic_order

        if interommatidial_angles_rad is not None:
            ioa_angles = broadcast_to_shape(
                interommatidial_angles_rad, shape=(N, 2),
                accepted=[((2,), (1,)), ((N, 2), (0, 1))],
                name='interommatidial_angles_rad',
            )

        self._buf['ioa_angles'] = ioa_angles
        self._buf['ioa_tilt'] = ioa_tilts

        # ============ Lens diameters (apertures) ============

        if aperture_um is not None:
            facet_apertures = broadcast_1d(aperture_um, self._N, 'aperture_um')
        else:
            facet_apertures = self._estimate_apertures(ioa_spacing)

        self._buf['aperture_um'] = facet_apertures

        # ============ Focal length ============

        # It scales with aperture (larger facets -> focus longer -> narrower RFs)
        #
        # Note: this feeds both the Snyder acceptance model and the microsaccade lever arm,
        # so it is computed whenever focal_um is available, no matter which acceptance model is used

        if self._bundle.focal_um is not None:

            # Scale bundle-provided focal length by the relative size of each lens
            median_aperture = np.median(self._buf['aperture_um'])
            f_number = self._bundle.focal_um / max(median_aperture, 1e-6)
            f_per_lens = f_number * self._buf['aperture_um']

            # Prevent absurdly small/zero focal lengths on degenerate edge lenses
            focal_um = np.where(f_per_lens > 1e-3, f_per_lens, self._bundle.focal_um)
        else:
            # Fallback: Assume diurnal apposition eye with F-number ~= 2.0,  f = D * F#
            logger.debug('Rhabdomere bundle focal_um is None. Assuming F-number = 2.0 for fallback focal lengths.')
            focal_um = self._buf['aperture_um'] * 2.0

        self._buf['focal_um'] = focal_um

        # ============ Rhabdomere bundle orientation ============

        use_aligner = (orientation is not None) or (flow_direction is not None) or (self._R > 1)

        if bundle_orientations is not None and chiralities is not None:
            result = apply_chirality(self, bundle_orientations, chiralities)

        elif use_aligner:
            if orientation is not None:
                aligner = orientation
            elif flow_direction is not None:
                aligner = BundlesAligner(flow_direction)
            else:
                aligner = BundlesAligner(-WORLD_FORWARD)

            result = aligner.compute(self, override_chi=bundle_orientations, override_chirality=chiralities)

        else:
            result = trivial_alignment(self._N)

            result.saccade_phasor = self._buf['up'].copy()
            # result.saccade_phasor = self._buf['right'].copy()  # debug thing

        self._bundle_orientation_backwrite(result)

        # ============ Acceptance angles ============

        if acceptance is not None:
            self._acceptance_model = acceptance
        else:
            self._acceptance_model = SnyderAcceptance() if self._bundle.focal_um is not None else SamplingAcceptance()

        acceptance_angles = self._compute_acceptance(self._acceptance_model)

        self._buf['rest_acc_angles'] = acceptance_angles
        self._buf['curr_acc_angles'] = acceptance_angles

        # ============ Fill other ommatidia neighbourhood related stuff ============

        self._buf['omm_id'] = self.omm_indices

        # First-ring neighbour count per ommatidium (Gabriel graph degree)
        neighbour_count = np.zeros(self._N, dtype=np.uint32)
        for eye in self._eyes:
            neighbour_count[eye.indices] = eye._get_first_ring_graph()['degree'].astype(np.uint32)

        self._buf['neighbour_count'] = neighbour_count

        # Edge / binocular classification
        self._buf['is_edge'] = self._identify_edge_ommatidia()
        self._buf['is_binocular'] = self._identify_binocular_ommatidia()

        # ============ Neural superposition ============

        if neural_superposition and self._R > 1:

            cartridge_src, unwired_mask = wire_neural_superposition(self)

            # Unwired peripheral slots fallback to self-wire
            own_src = (np.arange(self._N)[:, None] * self._R + np.arange(self._R)).astype(np.uint32)
            cartridge_src = np.where(unwired_mask, own_src, cartridge_src).astype(np.uint32)

            self._buf['cartridge_src'] = cartridge_src
            self._buf['is_wired'] = ~unwired_mask

            self._superposition_wired = True
            self._conflicts_cache = get_conflict_masks(self.cartridge_map, self._bundle.peripheral_indices)
            # Note: self._conflicts_cache must be set to None by any thing that changes the bundles orientation

        else:
            # No superposition: each receptor is its own source
            self._buf['cartridge_src'] = np.arange(self.size, dtype=np.uint32).reshape(self._N, self._R)
            self._buf['is_wired'] = True

            self._superposition_wired = False
            self._conflicts_cache = get_noconflict_masks(self._N, self._R)

    @classmethod
    def from_sphere(cls,
            n: int = 2000,
            eye_radius: float = 0.01,
            method: str = 'icosphere',
            force_isotropic: bool = False,
            **kwargs,
        ) -> 'Model':
        """
        Construct a uniform spherical compound eye.

        Args:
            - n: int, Approximate number of ommatidia.
            - eye_radius: float, Sphere radius in world units (default 0.01 m = 10 mm).
            - method: {'icosphere', 'fibonacci'}, Spherical sampling method.
            - **kwargs: Forwarded to __init__.
        """
        method = method.lower()
        dirs = icosphere(n) if method == 'icosphere' else fibonacci_sphere(n) if method == 'fibonacci' else None

        if dirs is None:
            raise ValueError("Method must be 'icosphere' or 'fibonacci'")

        if force_isotropic:
            # Theoretical IOA for N facets (lenses) tiled hexagonally on a sphere
            kwargs['interommatidial_angles_rad'] = [np.sqrt((4.0 * np.pi) / (n * np.sqrt(3.0) / 2.0))] * 2

        return cls(directions=dirs, positions=(dirs * float(eye_radius)).astype(np.float32), **kwargs)

    @classmethod
    def from_lenses(cls,
            directions: ArrayLike,
            positions: ArrayLike,
            **kwargs,
        ) -> 'Model':
        """
        Explicit facet (lens) placement. Forwards to __init__.
        """
        return cls(directions=directions, positions=positions, **kwargs)

    @classmethod
    def from_file(cls, path: Path | str, **kwargs) -> 'Model':
        """
        Load a species model from a .npz archive of raw geometry.

        This is a geometry-level loader: it reads lens positions and
        directions (plus a few optional fields) and re-runs the full
        construction pipeline.

        To save/load an already-built model (post-orientation, post-wiring),
        use Buffer.to_file() / Buffer.from_file().

        Required fields (any of these names accepted, in order of preference):
            - positions: (N, 3), lens world positions
                aliases: 'positions', 'pos', 'ommatidia_positions', 'lens_positions'
            - directions: (N, 3), lens optical axes
                aliases: 'directions', 'dirs', 'ommatidia_directions', 'lens_directions', 'forward'

        Optional fields (used when present, otherwise defaults apply):
            - eye_indices: 'eye_indices', 'eye_ids', 'eye_id', 'eye_idx', 'eye_index'
            - left / right: (N,) bool, fallback if no eye_indices
            - aperture_um: 'lens_aperture_um', 'lens_diameter', 'aperture_um'
            - interommatidial_angles_rad: 'interommatidial_angles_rad', 'ioa', 'ioa_angles', 'ioa_axes'
            - acceptance (rest half-widths): 'acceptance_angles', 'acceptance', 'rho'
            - bundle_orientations: 'bundle_orientations', 'chi'
            - chiralities: 'chiralities', 'chirality'

        Any keyword argument forwarded explicitly via **kwargs overrides what's found in the file.
        """

        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            available = [s.lower() for s in data.files]

            def fp(candidates):
                for name in candidates:
                    if name in available:
                        return np.asarray(data[name])
                return None

            positions = fp(['positions', 'pos', 'ommatidia_positions', 'lens_positions'])
            directions = fp(['directions', 'dirs', 'ommatidia_directions', 'lens_directions', 'forward'])

            if positions is None or directions is None:
                raise ValueError(
                    f"{path.name}: required 'positions' and 'directions' not found. "
                    f"Available keys: {available}"
                )

            # Resolve eye ids: explicit field, else infer from is_left / is_right.
            # If neither is present, falls back to island detection, and auto-assign sides from centroid x.
            if 'eye_indices' not in kwargs:
                eye_indices = fp(['eye_idx', 'eye_ids', 'eye_id', 'eye_index'])

                if eye_indices is None:
                    is_left = fp(['l', 'left', 'is_left'])
                    is_right = fp(['r', 'right', 'is_right'])

                    if is_left is not None:
                        # Convention: 0 = left, 1 = right
                        eye_indices = (~is_left.astype(bool)).astype(np.uint32)
                    elif is_right is not None:
                        eye_indices = is_right.astype(np.uint32)

                if eye_indices is not None:
                    kwargs['eye_indices'] = eye_indices

            aliases = {
                'aperture_um': ['lens_aperture_um', 'lens_aperture', 'lens_diameter_um', 'lens_diameter', 'aperture_um'],
                'interommatidial_angles_rad': ['interommatidial_angles_rad', 'interommatidial_angles', 'ioa', 'ioa_angles', 'ioa_axes'],
                'bundle_orientations': ['bundle_orientations', 'chi'],
                'chiralities': ['chiralities', 'chirality']
            }
            for param, al in aliases.items():
                if param not in kwargs and (v := fp(al)) is not None:
                    kwargs[param] = v

            if 'acceptance' not in kwargs and (rho := fp(['acceptance_angles_rad', 'acceptance_angles', 'acceptance', 'rho'])) is not None:
                kwargs['acceptance'] = ExplicitAcceptance(rho)

        logger.info(f"Loaded {positions.shape[0]} ommatidia from {path}")

        return cls(directions=directions, positions=positions, **kwargs)

    # Some basic properties

    def __repr__(self) -> str:
        return (
            f"CompoundEyeModel(N={self.shape[0]}, R={self.shape[1]}, "
            f"eyes={len(self._eyes)}, bundle={self._bundle.name!r})"
        )

    def __getitem__(self, key):
        return OmmatidiumView(self, self.omm_indices[key])

    @property
    def model(self) -> 'Model':
        return self

    @property
    def buffer(self) -> 'Buffer':
        return self._buf

    @property
    def omm_indices(self):
        return np.arange(self._N, dtype=np.intp)

    @property
    def rhab_indices(self):
        return np.arange(self._N * self._R, dtype=np.intp)

    @property
    def rhabdomeres(self):
        return RhabdomereView(self, self.rhab_indices)

    @property
    def ommatidia(self):
        return OmmatidiumView(self, self.omm_indices)

    # Private - Data management helpers

    def _bump_spatial_ver(self) -> None:
        self._spatial_version += 1

    # Private - Rhabdomere bundle orientation backwrite

    def _bundle_orientation_backwrite(self, orientation: AlignmentResult) -> None:
        """
        Backwrite an OrientationResult into the corresponding data fields.
        Called by 'BundlesAligner.apply()' and during construction.
        """

        chi = orientation.chi.astype(np.float32)
        chirality = orientation.chirality.astype(np.float32)

        sacc = orientation.saccade_phasor.astype(np.float32)

        self._buf['saccade_dxdy'][:, 0] = np.einsum('ij,ij->i', sacc, self._buf['right'])
        self._buf['saccade_dxdy'][:, 1] = np.einsum('ij,ij->i', sacc, self._buf['up'])

        # Write per-rhabdomere data

        rot_dx, rot_dy = self._bundle.rotated_offsets(chi, chirality)
        self._buf['rest_offset'] = np.stack([rot_dx, rot_dy], axis=-1).astype(np.float32)

        if self._R == 1:
            # Simple model (R=1): chi follows the hexagonal lattice tiling
            self._buf['chi'] = self._buf['ioa_tilt'].copy()    # TODO: is this really ideal?
        else:
            # Full model: actual rhabdomere bundle orientation
            self._buf['chi'] = chi

        self._buf['chirality_neg'] = (chirality < 0).astype(np.uint32)

        # Rhabdomeres rest directions: unit vector from the positioned rhabdomere tip through the lens nodal point
        nodal_dist = self._bundle.focal_um or np.median(self._buf['focal_um'])

        tip_local = np.stack([rot_dx, rot_dy, np.full(self.shape, -nodal_dist, dtype=np.float32)], axis=-1)

        tip_world = local_to_world(
            tip_local,
            self._buf['right'][:, None, :],
            self._buf['up'][:, None, :],
            self._buf['forward'][:, None, :]
        )

        # Initialise the actuated direction
        self._buf['curr_direction'] = norm_l2(-tip_world).astype(np.float32)

        self._conflicts_cache = None

    # Private - Derived properties assignment

    def _identify_binocular_ommatidia(self, angle_threshold: Optional[float] = None, degrees: bool = True) -> np.ndarray:
        """
        Identifies ommatidia: overlapping visual fields with contralateral eyes.
        """
        is_binocular = np.zeros(self._N, dtype=bool)

        for eye in self.eyes:
            other = np.setdiff1d(self.omm_indices, eye.indices) # all ommatidia *not* in this eye

            if other.size == 0:
                continue

            # If threshold not provided use 1.1 local IOA
            if angle_threshold is None:
                thresh_rad = np.mean(self._buf['ioa_angles'][eye.indices, 1]) * 1.1
            else:
                thresh_rad = np.deg2rad(angle_threshold) if degrees else angle_threshold

            other_view = OmmatidiumView(self, other)
            _, dist = other_view.query_directions(eye.directions, k=1)
            is_binocular[eye.indices] = dist[:, 0] <= angle_to_chord(thresh_rad)

        return is_binocular

    def _identify_edge_ommatidia(self) -> np.ndarray:
        """
        Edge ommatidia = incomplete first ring (< 6 Gabriel neighbours) with at least
        two first-ring neighbours that are themselves incomplete.
        """
        is_edge = np.zeros(self._N, dtype=bool)
        is_candidate = self._buf['neighbour_count'][:, 0] < 6

        for eye in self._eyes:
            adj = eye._get_first_ring_graph()['adjacency']
            eye_glob = eye.indices
            cand_local = is_candidate[eye_glob]
            for i_local in range(len(eye)):
                if cand_local[i_local]:
                    nb_local = adj[i_local]
                    if nb_local.size and int(np.sum(cand_local[nb_local])) >= 2:
                        is_edge[eye_glob[i_local]] = True
        return is_edge

    # Private - Facets (lenses) lattice geometry helpers

    def _compute_lattice_properties(self, k_search: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Per-ommatidium lattice properties from the local first-ring neighbour graph.

        Computes three things:

        - IOA (minor, major): angular separations in optical axis space.
            Sorted over first-ring neighbours.
            minor = mean of the 2 smallest,
            major = mean of the 2 largest
        - Tilt: lattice rotation in the ommatidium's tangent frame, computed via the hexatic order
              Ψ6 over first-ring neighbour bearings.
              Magnitude of Ψ6: 1.0 = perfect hex, 0.0 = isotropic disorder
        - Spacing: median first-ring distance in world units

        Args:
            - k_search: number of neighbours to query per ommatidium (~ a few rings).

        Returns:
            ioa_angles: (N, 2) (minor, major) in radians
            ioa_tilts: (N,) lattice tilt in radians, in [-π/6, +π/6]
            spacing: (N,) median first-ring distance, world units

        Note: receptors get a separate per-receptor 'acc_tilt' written by the orientation pipeline (chi).
        The ioa_tilt here is the lattice's own rotation, the orientation pipeline does not overwrite this.
        """

        ioa_minor = np.zeros(self._N, dtype=np.float32)
        ioa_major = np.zeros(self._N, dtype=np.float32)
        ioa_tilts = np.zeros(self._N, dtype=np.float32)
        ioa_order = np.zeros(self._N, dtype=np.float32)
        spacing = np.zeros(self._N, dtype=np.float32)

        for eye in self._eyes:
            n = len(eye)
            if n < 2:
                continue

            neighbours = eye.neighbours(
                query=eye.indices,
                k=min(k_search, n - 1),
                immediate_only=False
            )
            if not neighbours:
                continue

            home_dirs = self._buf['forward'][eye.indices]
            home_pos = self._buf['position'][eye.indices]
            home_right = self._buf['right'][eye.indices]
            home_up = self._buf['up'][eye.indices]

            neighb_dirs = self._buf['forward'][neighbours.indices]
            neighb_pos = self._buf['position'][neighbours.indices]
            is_immediate = neighbours.is_immediate

            # Optical IOA: angular separation in optical-axis space
            angular_sep = np.arccos(np.clip(np.einsum('ik,ijk->ij', home_dirs, neighb_dirs), -1.0, 1.0)).astype(np.float32)

            delta_pos = neighb_pos - home_pos[:, None, :]

            # Bearings in home ommatidium's tangent frame
            bearings = np.arctan2(np.einsum('ijk,ik->ij', delta_pos, home_up), np.einsum('ijk,ik->ij', delta_pos, home_right))

            # Hexatic order over the first ring
            z_avg = resultant(angles=bearings, weights=is_immediate, axis=1, fold=6)
            e_tilts = hexatic_axis_angle(z_avg).astype(np.float32)
            e_psi6_mag = hexatic_order(z_avg).astype(np.float32)

            # IOA: mean of 2 smallest / 2 largest first ring separations
            min2, max2 = np.sort(np.where(is_immediate, angular_sep, np.inf), axis=1)[:, :2], -np.sort(np.where(is_immediate, -angular_sep, np.inf), axis=1)[:, :2]
            with np.errstate(all='ignore'):
                e_ioa_minor = np.nanmean(np.where(np.isfinite(min2), min2, np.nan), axis=1).astype(np.float32)
                e_ioa_major = np.nanmean(np.where(np.isfinite(max2), max2, np.nan), axis=1).astype(np.float32)

            # neighbourhood too sparse: fallback to simple mean over whatever is immediate, zero if nothing
            if np.any(sparse_mask := is_immediate.sum(axis=1) < 2):
                with np.errstate(all='ignore'):
                    fallback = np.nanmean(np.where(is_immediate, angular_sep, np.nan), axis=1)
                fallback = np.where(np.isfinite(fallback), fallback, 0.0).astype(np.float32)
                e_ioa_minor = np.where(sparse_mask, fallback, e_ioa_minor)
                e_ioa_major = np.where(sparse_mask, fallback, e_ioa_major)
                e_tilts = np.where(sparse_mask, 0.0, e_tilts).astype(np.float32)

            # Ommatidia spacing: median first-ring distance (in world units)
            with np.errstate(all='ignore'):
                e_spacing = np.nanmedian(np.where(is_immediate, np.linalg.norm(delta_pos, axis=2), np.nan), axis=1).astype(np.float32)

            # Smooth lattice axis as a 6-fold phasor
            g2l = np.full(self._N, -1, dtype=np.intp)
            g2l[eye.indices] = np.arange(n)
            e_tilts_z6_smoothed = smooth_phasors(np.exp(6j * np.asarray(e_tilts, dtype=np.float64)), neighbours=np.where(is_immediate, g2l[neighbours.indices], -1), n_iter=2, weights=e_psi6_mag)
            e_tilts = hexatic_axis_angle(e_tilts_z6_smoothed)  # put angles back in (-pi/6, pi/6]

            ioa_minor[eye.indices] = np.where(np.isfinite(e_ioa_minor), e_ioa_minor, 0.0)
            ioa_major[eye.indices] = np.where(np.isfinite(e_ioa_major), e_ioa_major, 0.0)
            ioa_tilts[eye.indices] = e_tilts
            ioa_order[eye.indices] = e_psi6_mag
            spacing[eye.indices] = np.where(np.isfinite(e_spacing), e_spacing, 0.0)

            logger.debug(f"Eye {eye.eye_index} lattice |Ψ6|: {float(np.mean(e_psi6_mag)):.3f}, median spacing: {float(np.median(e_spacing)):.3f}")

        ioa_angles = np.stack([ioa_minor, ioa_major], axis=-1).astype(np.float32)

        return ioa_angles, ioa_tilts, ioa_order, spacing

    def _estimate_apertures(self, ioa_spacing: np.ndarray, fallback_aperture: float = 20.0) -> np.ndarray:
        """
        Estimate per-lens aperture (diameter, μm) when none is supplied.

        Starts from a lattice-spacing estimate (packing factor x first-ring spacing,
        with a fallback for degenerate spacing), refines it (per eye) with a Voronoi
        area estimator, then denoises the field once with an angular-metric median smooth.

        Args:
            - ioa_spacing: Interommatidial spacing
            - fallback_aperture: Last resort fallback value if can't estimate at all
        """

        # Base estimate from lattice spacing
        diameters = (self.HEX_PACKING_FACTOR * ioa_spacing).astype(np.float32)

        # Fallback for degenerate spacing: median of valid spacings
        valid_diameters = diameters[diameters > 0]
        fallback_val = np.median(valid_diameters) if valid_diameters.size > 0 else fallback_aperture
        diameters = np.where(diameters > 0, diameters, fallback_val)

        min_omm = 1 + 6 + 12 # central ommatidium + 6 first ring neighbours + 12 second ring neighbours

        for eye in self._eyes:
            if len(eye) > min_omm:
                try:
                    diameters[eye.indices] = voronoi_estimation(
                        self._buf['position'][eye.indices],
                        self._buf['forward'][eye.indices],
                        packing=self.HEX_PACKING_FACTOR
                    )
                except Exception:
                    # Voronoi estimator unavailable: keep the spacing estimate for this eye
                    logger.debug(f"Eye {eye.eye_index}: Voronoi estimation facet diameter failed, using spacing estimate.")

        # Denoise the single-edge readout across the whole field
        return self._smooth_aperture_field(diameters, k=6, n_iter=2, method='median', metric='angular')

    def _smooth_aperture_field(
            self,
            values: np.ndarray,
            k: int = 6,
            n_iter: int = 2,
            mask: Optional[np.ndarray] = None,
            method: str = 'mean',
            metric: str = 'angular',
    ) -> np.ndarray:
        """
        Smooth the per-lens diameter (aperture) scalar field (independently within each eye).
        (reuses each eye's cached KD-trees).
        """

        if n_iter <= 0:
            return values

        out = np.asarray(values).copy()

        g, n = [], []
        for eye in self._eyes:
            if len(eye) < 3:
                continue

            if metric == 'angular':
                _, neighb_indices = knn(
                    eye._get_tree('direction'),
                    self._buf['forward'][eye.indices],
                    k
                )
            else:
                neighb_graph = eye._get_neighbour_graph(k)
                neighb_indices = neighb_graph['neighbour_indices']

            g.append(eye.indices)
            n.append(neighb_indices)

        return smooth_field_partitioned(
            out,
            kind='scalar',
            groups=g,
            neighbours=n,
            n_iter=n_iter,
            mask=mask,
            method=method,
        ).astype(np.float32)

    # Private - Neural superposition cache helper

    @property
    def _get_conflicts(self):
        """Recompute (and cache) the conflict / completeness masks."""
        if self._conflicts_cache is None:
            if self._superposition_wired:
                self._conflicts_cache = get_conflict_masks(self.cartridge_map, self._bundle.peripheral_indices)
            else:
                self._conflicts_cache = get_noconflict_masks(self._N, self._R)

        return self._conflicts_cache


    # Private - Eye membership helpers (for construction)

    def _identify_distinct_eyes(self, k: int = 6, dist_factor: float = 2.5) -> np.ndarray:
        """
        Spatial connected-components clustering to detect eyes (as spatially separated groups of ommatidia).

        Args:
            - dist_factor: factor multiplying the median distance between all ommatidia. Can be generous.
        Returns:
            Array of island indices, (N,) uint32
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        dists, indices = knn(
            tree=None,      # will make a transient tree (since this runs before Eye objects exist)
            query_points=self._buf['position'],
            k=k
        )

        median_edge = np.median(dists)
        if median_edge <= 0.0:
            # Degenerate: all points coincide, return a single island
            return np.zeros(self._N, dtype=np.uint32)

        keep = dists < (dist_factor * median_edge)
        rows = np.repeat(self.omm_indices, indices.shape[1])[keep.ravel()]
        cols = indices.ravel()[keep.ravel()]
        data = np.ones(keep.sum(), dtype=np.int8)

        neighbours_matrix = csr_matrix((data, (rows, cols)), shape=(self._N, self._N))
        _, labels = connected_components(neighbours_matrix, directed=False, return_labels=True)

        labels_uniq = np.unique(labels)
        nb_eyes = len(labels_uniq)

        max_eyes = self._buf.max_value('eye_id')
        if nb_eyes > max_eyes:
            raise ValueError(f'Detected {nb_eyes} eyes but the metadata field supports at most {max_eyes}.')

        # Reorder: ascending centroid x
        centroids = np.stack([self._buf['position'][labels == e].mean(axis=0) for e in labels_uniq])
        order = np.lexsort((centroids[:, 2], centroids[:, 1], centroids[:, 0]))

        reordered = np.empty(nb_eyes, dtype=np.uint32)
        for new_id, raw_pos in enumerate(order):
            reordered[labels_uniq[raw_pos]] = np.uint32(new_id)

        return reordered

    def _instantiate_eyes(self, midline_buffer: float = 0.0) -> np.ndarray:
        """
        Instantiate Eye objects.

        Note: Each eye gets assigned a side ('left' / 'right' / 'midline').
        The side is decided by comparing the eye's centroid_x to an absolute
        world-coordinate threshold (midline_buffer).

        Args:
            - midline_buffer: Absolute threshold (in world position units) from the X=0 plane.
                Eyes whose centroid_x is within this distance are classified as 'midline' (e.g. ocelli).
        """

        eye_id = np.asarray(self._buf['eye_id']).reshape(self._N, self._R)[:, 0]
        nb_eyes = len(np.unique(eye_id))
        eye_instances = np.zeros(nb_eyes, dtype=object)

        for e in range(nb_eyes):
            omm_indices = np.flatnonzero(eye_id == e).astype(np.intp)
            centroid_x = self._buf['position'][omm_indices, 0].mean()

            if np.abs(centroid_x) <= midline_buffer:
                side = 'midline'
            elif centroid_x < 0.0:
                side = 'left'
            else:
                side = 'right'

            eye_instances[e] = EyeView(self, e, omm_indices, side=side)

        return eye_instances

    # Private - Acceptance angles helper

    def _compute_acceptance(self, acceptance_model: 'AcceptanceModel') -> np.ndarray:

        lens_optics = LensOptics(
            focal_um=self._buf['focal_um'],
            aperture_um=self._buf['aperture_um'],
            ioa_minor=self._buf['ioa_angles'][:, 0],
            ioa_major=self._buf['ioa_angles'][:, 1]
        )

        rhab_optics = RhabdomereOptics(
            diameter_um=np.atleast_1d(self._bundle.diameters_um).astype(np.float32),
            wavelength_um=np.atleast_1d(self._bundle.wavelengths_nm).astype(np.float32) * 1e-3,
        )

        acceptance_angles = acceptance_model(lens_optics, rhab_optics)
        if acceptance_angles.shape != (self._N, self._R, 2):
            raise ValueError(
                f'Acceptance model {type(acceptance_model).__name__} returned {acceptance_angles.shape}, expected {(self._N, self._R, 2)}')

        return acceptance_angles

    # Public - Bundle alignment refinement (post-superposition)

    def refine_superposition(self,
        min_donors: int = 2,
        smooth_iters: int = 0,
        relax: float = 0.5,
        max_nudge_deg: float = 20.0,
        adjust_scale: bool = True
    ) -> 'Model':
        """
        Refines the geometric layout of individual ommatidium bundles to best match
        the topological neural-superposition wiring. Nudges bundle rotation (chi)
        and scale to minimise angular disparity between ideal receptor lines of sight
        and actual target lenses.

        Args:
            min_donors: int, min wired donors needed to trust an ommatidium
            smooth_iters: int, passes to smooth the correction over the lattice
            relax: float, under-relaxation parameter (1.0 = strict geometry solve)
            max_nudge_deg: float, maximum rotation allowed (degrees)
            adjust_scale: bool, whether to also dynamically stretch/shrink bundles
        """
        refine_chi(self,
                   min_donors=min_donors,
                   smooth_iters=smooth_iters,
                   relax=relax,
                   max_nudge_deg=max_nudge_deg,
                   adjust_scale=adjust_scale
                   )
        return self

    # Disabling model-level neighbours queries

    # TODO: Should dispatch per-eye instead of raising

    def neighbours(self, *args, **kwargs):
        raise NotImplementedError('Neighbours queries must be done per-eye.')

    def directed_neighbours (self, *args, **kwargs):
        raise NotImplementedError('Neighbours queries must be done per-eye.')

    # Quick groups getters

    @property
    def eyes(self) -> List['EyeView']:
        return list(self._eyes)

    def eyes_by_side(self, side: str) -> List['EyeView']:
        """All eyes on a given side ('left', 'right', or 'midline')."""
        return [e for e in self._eyes if e.side == str(side)]

    @property
    def left_eyes(self) -> List['EyeView']:
        return self.eyes_by_side('left')

    @property
    def right_eyes(self) -> List['EyeView']:
        return self.eyes_by_side('right')

    @property
    def midline_eyes(self) -> List['EyeView']:
        return self.eyes_by_side('midline')

    def ommatidia_by_side(self, side: str) -> 'OmmatidiumView':
        """All ommatidia belonging to eyes on the given side."""
        eyes = self.eyes_by_side(side)
        if not eyes:
            return OmmatidiumView(self, np.empty(0, dtype=np.intp))
        return OmmatidiumView(self, np.concatenate([e.indices for e in eyes]))

    @property
    def left_ommatidia(self) -> 'OmmatidiumView':
        return self.ommatidia_by_side('left')

    @property
    def right_ommatidia(self) -> 'OmmatidiumView':
        return self.ommatidia_by_side('right')

    @property
    def midline_ommatidia(self) -> 'OmmatidiumView':
        return self.ommatidia_by_side('midline')


##

if __name__ == '__main__':

    from insectvision.compound_eyes.rhabdomeres import drosophila_bundle

    model = Model.from_sphere(n=500)

    print(model)
    print(f"  Total receptors: {model.size}")
    print(f"  Eyes: {model.eyes}")

    # Drosophila with flow direction
    droso = drosophila_bundle()

    model = Model.from_sphere(
        n=1600,
        bundle=droso,
        flow_direction=[1.0, 0.0, 0.0],  # Anterior flow
    )
    print(model)

    print(f"  Bundle orientations (first 5): {model[:5].bundle_orientation}")
    print(f"  Chiralities (first 5): {model[:5].chirality}")
    print(f"  Neural superposition: {model.neural_superposition}")
    print(f"  Cartridges[0] receptors: {model.cartridges[0].rhabdomeres}")