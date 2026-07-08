"""
The main user-facing CompoundEyeModel class that wraps the data buffers.
Does all the building, and exposes properties that the other views can re-expose.
"""
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, Tuple, List
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import Delaunay, cKDTree

from insectvision.engine.meshes import icosphere, fibonacci_sphere
from insectvision.geometry.ghosting import combine_clouds
from insectvision.geometry.lattice import lattice_confidence
from insectvision.utils import WORLD_FORWARD, norm_l2, broadcast_to_shape, broadcast_1d

from insectvision.geometry.circular import resultant
from insectvision.geometry.hexatic import hexatic_axis_angle, hexatic_order
from insectvision.geometry.linalg import tangent_frames, local_to_world, tangent_bearing
from insectvision.geometry.neighbours import knn, topological_spacing, metric_spacing, delaunay_neighbours, \
    padded_neighbours, identify_boundary_points
from insectvision.geometry.fields import smooth_phasors, smooth_field_partitioned
from insectvision.geometry.spherical import angle_to_chord, sphere_to_stereo
from insectvision.geometry.polygons import triangle_areas

from insectvision.compound_eyes.buffers import Buffer, _BIT_LAYOUT
from insectvision.compound_eyes.rhabdomeres import RhabdomereBundle
from insectvision.compound_eyes.helpers.neural_superposition import (
    wire_neural_superposition, get_conflict_masks, get_noconflict_masks, refine_chi
)
from insectvision.compound_eyes.helpers.acceptance import (
    SnyderAcceptance, SamplingAcceptance, LensOptics, RhabdomereOptics, ExplicitAcceptance
)
from insectvision.compound_eyes.helpers.alignment import BundlesAligner
from insectvision.compound_eyes.views import SpatialQueries, BaseView, OmmatidiumView, EyeView, RhabdomereView

if TYPE_CHECKING:
    from insectvision.compound_eyes.views import NeighbourResult
    from insectvision.compound_eyes.helpers.acceptance import AcceptanceModel
    from insectvision.compound_eyes.helpers.alignment import AlignmentResult


# TODO Should store the rotational alignment of the anisotropy to the lattice somewhere??

logger = logging.getLogger(__name__)


class Model(SpatialQueries, BaseView):
    """
    A (set of) compound eyes specified as N ommatidia positions / directions
    and a bundle of R rhabdomeres per ommatidium.
    """

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
            neural_superposition: bool = False,
            lens_packing: float = 0.9,
            lattice_beta: float = 1.0
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

            _, eye_id_bits = _BIT_LAYOUT['eye_id']
            max_eyes = self._buf.max_value('eye_id')

            if arr.size != self._N:
                raise ValueError(f'eye_indices size {arr.size} must equal N={self._N}')
            if int(arr.max()) >= max_eyes + 1:
                raise ValueError(f'eye_indices exceed {max_eyes} ({eye_id_bits}-bit field)')

            eye_membership = eye_indices
        else:
            eye_membership = self._identify_distinct_eyes()

        self._buf['eye_id'] = eye_membership

        self._apply_canonical_order()

        self._eyes = self._instantiate_eyes()

        # ============ Ommatidia lattice properties ============

        # Fraction spacing that the lens diameter occupies: 1.0 = fully touching, 0.9 leaves a small interommatidial cuticle gap, etc
        self._lens_packing: float = float(max(0.0, lens_packing))

        # β parameter for the β-skeleton graph: beta slightly <1 keeps edges that shear pushes past 90°
        self._lattice_beta: float = float(max(0.0, lattice_beta))

        # Build IOA / tilt / order / is_edge / trust, and the spacing used for apertures
        ioa_spacing = self._compute_lattice_properties(ioa_override=interommatidial_angles_rad)

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
            result = BundlesAligner.apply_chirality(self, bundle_orientations, chiralities)

        elif use_aligner:
            if orientation is not None:
                aligner = orientation
            elif flow_direction is not None:
                aligner = BundlesAligner(flow_direction)
            else:
                aligner = BundlesAligner(-WORLD_FORWARD)

            result = aligner.compute(self, override_chi=bundle_orientations, override_chirality=chiralities)

        else:
            result = BundlesAligner.trivial_alignment(self._N)

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
            neighbour_count[eye.indices] = eye._get_first_ring_graph()['degree']

        self._buf['neighbour_count'] = neighbour_count
        self._buf['is_binocular'] = self._identify_binocular_ommatidia()

        # ============ Neural superposition ============
        self._wiring_trace = None

        if neural_superposition and self._R > 1:
            wire_neural_superposition(self, apply=True)
            # Note: self._conflicts_cache is initialised with apply=True,
            #  but it must be set to None by any thing that changes the bundles orientation

        else:
            # No superposition: each receptor is its own source
            self._buf['cartridge_src'] = np.arange(self.size, dtype=np.uint32).reshape(self._N, self._R)
            self._buf['is_wired'] = True

            self._superposition_wired = False
            self._conflicts_cache = get_noconflict_masks(self._N, self._R)

    @classmethod
    def from_sphere(cls,
            n: int = 2000,
            eye_radius: float = 0.005,
            method: str = 'icosphere',
            force_isotropic: bool = False,
            separation: Optional[float] = 0.0025,
            **kwargs,
        ) -> 'Model':
        """
        Construct a spherical compound eye.

        Args:
            - n: int, Approximate number of ommatidia.
            - eye_radius: float, Sphere radius in world units (default 0.005 m = 5 mm).
            - method: {'icosphere', 'fibonacci'}, Spherical sampling method.
            - separation: float, optional. Lateral gap between the two hemispheres in world units (default 0.0025 m = 2.5 mm).
                None or <= 0 disables the split and generates a full sphere.
                >= 0 separates the sphere in two eyes and translates each half outward along x by separation/2.
            - **kwargs: Forwarded to __init__.
        """

        method = str(method).lower()
        dirs = icosphere(n) if 'ico' in method else fibonacci_sphere(n) if 'fibo' in method else None

        if dirs is None:
            raise ValueError("Method must be 'icosphere' or 'fibonacci'")

        if force_isotropic:
            # Theoretical IOA for N facets (lenses) tiled hexagonally on a sphere
            kwargs['interommatidial_angles_rad'] = [np.sqrt((4.0 * np.pi) / (n * np.sqrt(3.0) / 2.0))] * 2

        positions = (dirs * float(eye_radius)).astype(np.float32)

        split_eyes = separation is not None and separation >= 0.0
        if split_eyes:
            right = dirs[:, 0] >= 0.0
            kwargs['eye_indices'] = right.astype(np.uint32)  # 0 = left (-x), 1 = right (+x)

            shift = 0.5 * float(separation)
            positions[right, 0] += shift
            positions[~right, 0] -= shift

        return cls(directions=dirs, positions=positions, **kwargs)

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

    def _apply_canonical_order(self) -> None:
        """
        Permute all buffers in place into a deterministic canonical order.

        Sorted by eye, then a serpentine latitude raster within each eye.
        Stable lexsort + using original index as tiebreaker -> identical inputs give bitwise-identical order

        Bands are anchored at the projected eye centre (function of the eye's mean direction), so
        only ommatidia within ~1 spacing of a band edge can move in the ordering (and they move by at most ~1 row)
        """

        eye_id = np.asarray(self._buf['eye_id']).reshape(self._N, self._R)[:, 0]
        directions = np.asarray(self._buf['forward']).reshape(self._N, 3)

        blocks = []
        for e in np.unique(eye_id):
            loc = np.flatnonzero(eye_id == e)  # Global indices, ascending
            if loc.size <= 2:
                blocks.append(loc)
                continue

            # Stereographic projection about the eye's mean direction
            pts2d, *_ = sphere_to_stereo(directions[loc])
            py = pts2d[:, 1]

            # Band width ~1 ommatidial spacing (median nearest-neighbour gap)
            w = np.nanmedian(metric_spacing(query_points=pts2d, k=1))
            if not np.isfinite(w) or w <= 0.0:
                w = 1.0

            band  = np.floor(py / w).astype(np.int64)  # latitude bands anchored at 0
            sweep = np.where(band % 2 == 0, pts2d[:, 0], -pts2d[:, 0])  # serpent: alternate sweep

            order = np.lexsort((loc, sweep, band))  # band > sweep > original idx
            blocks.append(loc[order])

        self._buf.reorder(np.concatenate(blocks).astype(np.intp))

    # Private - Rhabdomere bundle orientation backwrite

    def _bundle_orientation_backwrite(self, orientation: 'AlignmentResult') -> None:
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
            self._buf['chi'] = self._buf['ioa_tilt'].copy()
        else:
            # Full model: actual rhabdomere bundle orientation
            self._buf['chi'] = chi

        self._buf['chirality_neg'] = (chirality < 0).astype(np.uint32)

        # Rhabdomeres rest directions: unit vector from the positioned rhabdomere tip through the lens nodal point
        nodal_dist = self._bundle.focal_um or np.median(self._buf['focal_um'])

        tip_local = np.stack([rot_dx, rot_dy, np.full(self.shape, -nodal_dist, dtype=np.float32)], axis=-1)
        tip_world = local_to_world(tip_local, self._buf['right'], self._buf['up'], self._buf['forward'])

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

    # Private - Facets (lenses) lattice geometry helpers
    #
    # TODO: This is now a lot of logic for lattice properties, should probably move to a file in helpers submodule

    def _lattice_props_pass1(self, k_search: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        First pass (before is_edge/IOA exist).

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

            indices = eye.indices

            neighbours = eye.neighbours(query=indices, k=k_search, immediate_only=False)
            if not neighbours:
                continue

            neighb_indices = neighbours.indices

            positions = self._buf['position', indices]
            directions = self._buf['forward', indices]
            rights = self._buf['right', indices]
            ups = self._buf['up', indices]

            neighb_dirs = self._buf['forward', neighb_indices]
            neighb_pos = self._buf['position', neighb_indices]
            is_immediate = neighbours.immediate

            # Optical IOA: angular separation in optical-axis space
            angular_sep = np.arccos(
                np.clip(np.einsum('ik,ijk->ij', directions, neighb_dirs), -1.0, 1.0)
            )

            bearings = tangent_bearing(neighb_pos, positions, rights, ups)

            # Hexatic order over the first ring
            z_avg = resultant(angles=bearings, weights=is_immediate, axis=1, fold=6)

            e_tilts = hexatic_axis_angle(z_avg)
            e_psi6_mag = hexatic_order(z_avg)

            # IOA: mean of 2 smallest / 2 largest first ring separations
            min2 = np.sort(np.where(is_immediate, angular_sep, np.inf), axis=1)[:, :2]
            max2 = -np.sort(np.where(is_immediate, -angular_sep, np.inf), axis=1)[:, :2]

            with np.errstate(all='ignore'):
                e_ioa_minor = np.nanmean(np.where(np.isfinite(min2), min2, np.nan), axis=1)
                e_ioa_major = np.nanmean(np.where(np.isfinite(max2), max2, np.nan), axis=1)

            # Neighbourhood too sparse: fallback to simple mean over whatever is immediate, zero if nothing
            if np.any(sparse_mask := is_immediate.sum(axis=1) < 2):

                with np.errstate(all='ignore'):
                    fallback = np.nanmean(np.where(is_immediate, angular_sep, np.nan), axis=1)

                fallback = np.where(np.isfinite(fallback), fallback, 0.0)

                e_ioa_minor = np.where(sparse_mask, fallback, e_ioa_minor)
                e_ioa_major = np.where(sparse_mask, fallback, e_ioa_major)
                e_tilts = np.where(sparse_mask, 0.0, e_tilts)

            # Ommatidia spacing
            adj = eye._get_first_ring_graph()['adjacency']
            e_spacing = topological_spacing(points=positions, neighbours=adj, reduce=np.median)

            # Smooth lattice axis as a 6-fold phasor
            g2l = np.full(self._N, -1, dtype=np.intp)
            g2l[indices] = np.arange(n)

            e_tilts_z6_smoothed = smooth_phasors(
                values=np.exp(6j * np.asarray(e_tilts, dtype=np.float64)),
                neighbours=np.where(is_immediate, g2l[neighb_indices], -1),
                weights=e_psi6_mag,
                n_iter=2
            )
            e_tilts = hexatic_axis_angle(e_tilts_z6_smoothed)  # put angles back in (-pi/6, pi/6]

            ioa_minor[indices] = np.where(np.isfinite(e_ioa_minor), e_ioa_minor, 0.0)
            ioa_major[indices] = np.where(np.isfinite(e_ioa_major), e_ioa_major, 0.0)
            ioa_tilts[indices] = e_tilts
            ioa_order[indices] = e_psi6_mag
            spacing[indices] = e_spacing

            logger.debug(f"Eye {eye.eye_index} lattice |Ψ6|: {float(np.mean(e_psi6_mag)):.3f}, "
                         f"median spacing: {float(np.median(e_spacing)):.3f}")
                        # TODO: Move this log in _lattice_properties, after the second pass

        ioa_angles = np.stack([ioa_minor, ioa_major], axis=-1)

        return ioa_angles.astype(np.float32), ioa_tilts.astype(np.float32), ioa_order.astype(np.float32), spacing.astype(np.float32)

    def _lattice_props_pass2(self):
        """
        Second pass (for once is_edge/IOA exist).

        Closes each boundary ring with the ghosts and re-measures IOA / tilt / |Psi6| / spacing
        over the closed ring, blending each into the provisional field by trust
        (so interior is almost untouched, and boundary is taken from the closed ring)
        """

        minor = self.interommatidial_angles[:, 0].copy()
        major = self.interommatidial_angles[:, 1].copy()

        tilt = self.interommatidial_tilt.copy()
        h_order = self._hexatic_order.copy()

        spacing = np.zeros(self._N, dtype=np.float32)

        for eye in self._eyes:
            n = len(eye)

            indices = eye.indices
            adj = eye._get_first_ring_graph()['adjacency']
            positions = self._buf['position', indices]

            if n < 3:
                if n >= 2:
                    spacing[indices] = topological_spacing(positions, adj, reduce=np.median)
                continue

            directions = self._buf['forward', indices]
            rights = self._buf['right', indices]
            ups = self._buf['up', indices]

            g_pos, g_dirs = eye.ghost_positions, eye.ghost_directions

            combined_pos, combined_dirs, _ = combine_clouds(
                real_pos=positions, ghost_pos=g_pos,
                real_dirs=directions, ghost_dirs=g_dirs
            )

            combined_pos_tree = cKDTree(combined_pos)       # TODO: this should be cached per eye

            # Exact topological first ring on the sphere
            neighb_ragged = delaunay_neighbours(combined_dirs, max_length_factor=1.8)

            # Take just the real points' queries, pad them, and get the valid topological mask
            neighbours, valid_neighbours = padded_neighbours(neighbours=neighb_ragged[:n], masked=True)

            neighb_dirs = combined_dirs[neighbours]
            neighb_pos = combined_pos[neighbours]

            angular_sep = np.arccos(
                np.clip(np.einsum('ik,ijk->ij', directions, neighb_dirs), -1.0, 1.0)
            )

            delta = neighb_pos - positions[:, None, :]

            u = np.einsum('ijk,ik->ij', delta, rights)
            v = np.einsum('ijk,ik->ij', delta, ups)
            bearings = np.arctan2(v, u)

            z = resultant(angles=bearings, weights=valid_neighbours, axis=1, fold=6)

            e_tilt = hexatic_axis_angle(z)
            e_order = hexatic_order(z)

            # Mask out invalid neighbours so they don't corrupt the min/max separation sorts
            sep_asc = np.sort(np.where(valid_neighbours, angular_sep, np.inf), axis=1)[:, :2]
            sep_desc = np.sort(np.where(valid_neighbours, angular_sep, -np.inf), axis=1)[:, -2:]

            with np.errstate(all='ignore'):
                e_minor = np.nanmean(np.where(np.isfinite(sep_asc), sep_asc, np.nan), axis=1)
                e_major = np.nanmean(np.where(np.isfinite(sep_desc), sep_desc, np.nan), axis=1)

            e_minor = np.where(np.isfinite(e_minor), e_minor, minor[indices])
            e_major = np.where(np.isfinite(e_major), e_major, e_minor)

            # Smooth the refined tilt over the real first ring
            e_tilt = hexatic_axis_angle(
                smooth_phasors(values=np.exp(6j * e_tilt), neighbours=adj, weights=e_order, n_iter=2)
            )

            s_ghost = metric_spacing(tree=combined_pos_tree, query_points=positions, k=6, reduce=np.median)
            s_graph = topological_spacing(points=positions, neighbours=adj, reduce=np.median)

            c = self._trust[indices]  # 1 = trust provisional, 0 = take refined
            cz = c.astype(np.float64)

            minor[indices] = (c * minor[indices] + (1.0 - c) * e_minor)
            major[indices] = (c * major[indices] + (1.0 - c) * e_major)
            h_order[indices] = (c * h_order[indices] + (1.0 - c) * e_order)

            # Tilt blends as a 6-fold phasor
            z_blend = cz * np.exp(6j * tilt[indices]) + (1.0 - cz) * np.exp(6j * e_tilt)
            tilt[indices] = hexatic_axis_angle(z_blend)

            s = c * s_graph + (1.0 - c) * s_ghost
            bad = ~(np.isfinite(s) & (s > 0))
            s[bad] = s_ghost[bad]
            spacing[indices] = s

        ioa_angles = np.stack([minor, major], axis=-1)

        return ioa_angles.astype(np.float32), tilt.astype(np.float32), h_order.astype(np.float32), spacing.astype(np.float32)

    def _compute_lattice_properties(self, ioa_override: Optional[ArrayLike] = None, n_rings: int = 1) -> np.ndarray:
        """
        Build every first-ring lattice property

            1. provisional IOA / tilt / order   (_lattice_props_pass1, no ghosts)
            2. edge classification              (_identify_edge_ommatidia)
            3. trust field                      (_lattice_trust_field)
            4. ghost-refined IOA / tilt / order / spacing, blended by trust (_lattice_props_pass2)

        Writes ioa_angles, ioa_tilt, _hexatic_order, is_edge and _trust,
        and returns the (N,) spacing that feeds aperture estimation

        Args:
            ioa_override: optional user-supplied IOA. Ghosts still use it,
                but the refined IOA is not written back over it.
        """
        ioa_prov, tilt_prov, order_prov, _ = self._lattice_props_pass1()

        self._hexatic_order = order_prov

        if ioa_override is not None:
            ioa_prov = broadcast_to_shape(
                ioa_override, shape=(self._N, 2),
                accepted=[((2,), (1,)), ((self._N, 2), (0, 1))],
                name='interommatidial_angles_rad',
            )

        self._buf['ioa_angles'] = ioa_prov
        self._buf['ioa_tilt'] = tilt_prov

        is_edge = np.zeros(self._N, dtype=bool)
        trust = np.zeros(self._N, dtype=np.float32)

        for eye in self._eyes:
            indices = eye.indices

            if len(eye) < 3:
                is_edge[indices] = True
                continue

            graph = eye._get_first_ring_graph()

            if n_rings <= 1:
                is_edge[indices] = graph['is_boundary']
            else:
                # Only propagate if requested
                is_edge[indices] = identify_boundary_points(
                    points2d=graph['points2d'],
                    neighbours=graph['adjacency'],
                    simplices=graph['simplices'],       # only if method = 'alpha'
                    method='gap',
                    n_rings=n_rings
                )

            trust[indices] = lattice_confidence(
                hex_order=self._hexatic_order[indices],
                is_boundary=self._buf['is_edge', indices]
            )

        self._buf['is_edge'] = is_edge
        self._trust = trust

        # Refined pass uses the closed boundary rings (ghosts)
        r_ioa, r_tilt, r_order, spacing = self._lattice_props_pass2()

        if ioa_override is None:
            self._buf['ioa_angles'] = r_ioa
            self._buf['ioa_tilt'] = r_tilt
            self._hexatic_order = r_order

        return spacing

    def _estimate_apertures(self,
            ioa_spacing: np.ndarray,
            fallback_aperture: float = 20.0
        ) -> np.ndarray:
        """
        Estimate per-lens aperture (diameter, μm) using the β-skeleton area dual.
        """

        # Base fallback from median spacing
        diameters = (self._lens_packing * ioa_spacing).astype(np.float32)
        valid_diameters = diameters[diameters > 0]
        fallback_val = np.median(valid_diameters) if valid_diameters.size > 0 else fallback_aperture
        diameters = np.where(diameters > 0, diameters, fallback_val)

        for eye in self._eyes:
            n = len(eye)
            if n < 3:
                continue

            indices = eye.indices
            graph = eye._get_first_ring_graph()

            # Only keep triangles whose 3 edges exist in the β-skeleton
            simplices = graph['simplices']
            big = graph['big']
            keys = graph['pair_keys']

            # Map local simplex indices to global indices
            global_tri = eye.omm_indices[simplices]

            # Key the 3 edges of every triangle
            def key_edges(i, j):
                lo = np.minimum(global_tri[:, i], global_tri[:, j])
                hi = np.maximum(global_tri[:, i], global_tri[:, j])
                return lo * big + hi

            k1, k2, k3 = key_edges(0, 1), key_edges(1, 2), key_edges(2, 0)

            pair_keys_arr = np.array(list(keys), dtype=np.int64)
            mask_v = np.isin(k1, pair_keys_arr) & np.isin(k2, pair_keys_arr) & np.isin(k3, pair_keys_arr)
            valid_simplices = simplices[mask_v]

            if len(valid_simplices) == 0:
                continue

            pts3d = self._buf['position', indices]

            # 3D area of each triangle
            tri_areas = triangle_areas(
                pts3d[valid_simplices[:, 0]],
                pts3d[valid_simplices[:, 1]],
                pts3d[valid_simplices[:, 2]]
            )

            # Accumulate 1/3 of the triangle's area to each of its 3 vertices
            point_areas = np.zeros(n)
            np.add.at(point_areas, valid_simplices[:, 0], tri_areas / 3.0)
            np.add.at(point_areas, valid_simplices[:, 1], tri_areas / 3.0)
            np.add.at(point_areas, valid_simplices[:, 2], tri_areas / 3.0)

            # Convert area back to hex flat-to-flat diameter
            mask_has_area = point_areas > 0
            calc_diameters = self._lens_packing * np.sqrt(2.0 * point_areas[mask_has_area] / np.sqrt(3.0))

            # Apply only to interior points (boundary points just keep their ioa_spacing estimate)
            update_mask = mask_has_area & eye.is_interior

            # Map local 'update_mask' back to global 'indices'
            diameters[indices[update_mask]] = calc_diameters[update_mask[mask_has_area]]

        # Denoise
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

            indices = eye.indices

            if metric == 'angular':
                _, neighb_indices = knn(
                    eye._get_tree('direction'),
                    self._buf['forward', indices],
                    k
                )
            else:
                neighb_graph = eye._get_neighbour_graph(k)
                neighb_indices = neighb_graph['neighbour_indices']

            g.append(indices)
            n.append(neighb_indices)

        return smooth_field_partitioned(
            values=out,
            neighbours=n,
            kind='scalar',
            groups=g,
            mask=mask,
            method=method,
            n_iter=n_iter,
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

    def neighbours(self,
            query: Optional[ArrayLike] = None,
            positions: Optional[ArrayLike] = None,
            k: int = 6,
            **kwargs
        ) -> 'NeighbourResult':
        """
        Global k-NN query across all eyes.
        """
        return super().neighbours(query=query, positions=positions, k=k, **kwargs)

    def directed_neighbours(self, direction: ArrayLike, query: Optional[ArrayLike] = None, **kwargs) -> np.ndarray:
        """
        Global directed search across all eyes.
        """
        return super().directed_neighbours(direction=direction, query=query, **kwargs)

    # Advanced properties

    @property
    def lens_packing(self) -> float:
        return self._lens_packing

    @lens_packing.setter
    def lens_packing(self, value: float) -> None:

        val = float(max(0.0, value))
        if self._lens_packing == val:
            return

        # Apertures depend on packing (rebuild the lattice then re-estimate)
        ioa_spacing = self._compute_lattice_properties()
        self._buf['aperture_um'] = self._estimate_apertures(ioa_spacing)

        # Maintain F-number
        f = self._buf['focal_um']
        d = self._buf['aperture_um']
        f_number = np.median(f / np.clip(d, 1e-6, None))

        self._buf['focal_um'] = self._buf['aperture_um'] * f_number

        # Recompute optics-dependent fields
        self._buf['rest_acc_angles'] = self._compute_acceptance(self._acceptance_model)
        self._buf['curr_acc_angles'] = self._buf['rest_acc_angles'].copy()

        self._bump_spatial_ver()

    @property
    def lattice_beta(self) -> float:
        return self._lattice_beta

    @lattice_beta.setter
    def lattice_beta(self, value: float) -> None:

        value = float(max(0.0, value))
        if self._lattice_beta == value:
            return

        # Beta changes the topology of the first-ring graph
        self._bump_spatial_ver()
        ioa_spacing = self._compute_lattice_properties()
        self._buf['aperture_um'] = self._estimate_apertures(ioa_spacing)

    @property
    def wiring_trace(self) -> Optional[dict]:
        return self._wiring_trace

    # High-level physiological and architectural properties

    @property
    def fused_rhabdoms(self) -> bool:
        """Whether the photoreceptors share a single fused central waveguide."""
        return self._bundle.fused_rhabdoms

    apposition = is_apposition = fused_rhabdoms

    @property
    def equatorial_discontinuity(self) -> bool:
        """
        Whether the eye features a dorsal/ventral anatomical equator.
        """
        for eye in self.eyes:
            chir = eye.chirality
            if np.any(chir > 0) and np.any(chir < 0):
                return True
        return False

    @property
    def has_microsaccades(self) -> bool:
        """
        Whether the ommatidia have non-zero photomechanical actuation capabilities.
        """
        max_lat = float(np.max(np.abs(self._buf['ampl_lateral'])))
        max_ax = float(np.max(np.abs(self._buf['ampl_axial'])))
        return max_lat > 1e-6 or max_ax > 1e-6

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