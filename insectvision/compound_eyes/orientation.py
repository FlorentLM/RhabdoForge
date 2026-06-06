from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

from insectvision.utils.math import norm_l2, tangent_frames, rotate_in_tangent_plane, broadcast_1d, local_to_world
from insectvision.utils.knns import knn
from insectvision.utils.circular import wrap_angle
from insectvision.utils.fields import smooth_nematic_vectors

if TYPE_CHECKING:
    from insectvision.compound_eyes import CompoundEyeModel



@dataclass
class OrientationResult:
    """
    Output of BundlesAligner.compute().
    """

    chi: np.ndarray                 # Bundle yaw (rad)
    chirality: np.ndarray           # +/- 1
    saccade_phasor: np.ndarray      # World-space unit vector

    # Bookkeeping / diagnostic stuff
    alignment_phasor: Optional[np.ndarray] = None
    major_axis: Optional[np.ndarray] = None
    eye_sign: Optional[np.ndarray] = None
    hemisphere_sign: Optional[np.ndarray] = None
    flow_frame: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None


def _alignment_phasor_field(
    lens_directions: np.ndarray,
    e_x: np.ndarray,
    e_y: np.ndarray,
    e_z: np.ndarray,
    eye_sign: np.ndarray,
    hemisphere_sign: np.ndarray,
    strength: float = 1.0,
    falloff: float = 0.7,
    diagonal_strength: float = 1.0,
    diagonal_angle_deg: float = 45.0,
) -> np.ndarray:
    """
    Four-zone combed-flow alignment phasor field.

    Args:
        - 'eye_sign': mirrors e_y across the body midline
        - 'hemisphere_sign': mirrors e_z across the flow-frame equator.
        - 'diagonal_strength': magnitude of the diagonal pull relative
            to the flow target (the combed target is renormalised against
            |proj_S| downstream, so this is effectively a ratio).
        - 'diagonal_angle_deg': angle of the diagonal pull in the focus
            tangent plane (perpendicular to e_x at the point of expansion).
            Decomposes the pull into lateral (e_y) and vertical (e_z) parts:
                  0° -> purely lateral  (collapses chirality / dorsoventral split)
                 45° -> equal lateral and vertical (default)
                 90° -> purely vertical (collapses the L/R distinction)
    """
    proj_S = e_x[None, :] - (lens_directions @ e_x)[:, None] * lens_directions

    a = float(np.radians(diagonal_angle_deg))

    # sqrt(2) normalisation keeps the total diagonal magnitude constant across the sweep
    sqrt2 = float(np.sqrt(2.0))
    lateral_scale = diagonal_strength * np.cos(a) * sqrt2
    vertical_scale = diagonal_strength * np.sin(a) * sqrt2

    target_world = (
            -e_x[None, :]
            + lateral_scale * (-eye_sign[:, None]) * e_y[None, :]
            + vertical_scale * hemisphere_sign[:, None] * e_z[None, :]
    ).astype(np.float32)

    target_proj = target_world - np.sum(target_world * lens_directions,
                                        axis=1, keepdims=True) * lens_directions
    target_norm = np.linalg.norm(target_proj, axis=1, keepdims=True)
    proj_S_mag = np.linalg.norm(proj_S, axis=1, keepdims=True)
    target_proj = np.where(
        target_norm > 1e-6,
        target_proj / target_norm.clip(min=1e-8) * proj_S_mag,
        proj_S,
    )

    # Falloff: chord distance from the point of expansion on the unit sphere
    source = -e_x
    dist = np.linalg.norm(lens_directions - source[None, :], axis=1)
    w = np.clip(1.0 - dist * falloff, 0.0, None) * strength

    combed = (1.0 - w[:, None]) * proj_S + w[:, None] * target_proj
    norms = np.linalg.norm(combed, axis=1, keepdims=True).clip(min=1e-8)
    return (combed / norms).astype(np.float32)


def _major_axis_field(
        alignment_phasor: np.ndarray,
        lens_directions: np.ndarray,
        base_rotation_rad: float,
        eye_sign: np.ndarray,
        hemisphere_sign: np.ndarray,
) -> np.ndarray:
    """
    Bundle main axis: per-eye rotation of the alignment phasor in each
    tangent plane ('base_rotation_rad * eye_sign'), then hemisphere-aware
    disambiguation.
    """
    angles = (base_rotation_rad * eye_sign).astype(np.float32)
    rotated = rotate_in_tangent_plane(
        alignment_phasor, lens_directions, angles, normalize=False
    )

    # Equatorial flip: bundle point up dorsally, and down ventrally
    ref = -hemisphere_sign[:, None] * alignment_phasor
    dot_check = np.einsum('ij,ij->i', rotated, ref)
    rotated[dot_check < 0] *= -1.0

    return norm_l2(rotated).astype(np.float32)


def _saccade_phasor_field(
    major_axis: np.ndarray,
    lens_directions: np.ndarray,
    base_rotation_rad: float,
    chirality: np.ndarray,
) -> np.ndarray:
    """
    Saccade phasor: rotate the major axis by 'base_rotation_rad * chirality' in each tangent plane.
    The result is nematic (direction locally ambiguous).
    Smoothing, then polarisation, then give global coherence.
    """
    angles = (base_rotation_rad * chirality).astype(np.float32)
    sacc = rotate_in_tangent_plane(major_axis, lens_directions, angles, normalize=True)
    return sacc.astype(np.float32)


def _smooth_phasor_field(
    field: np.ndarray,
    partition: np.ndarray,
    positions: np.ndarray,
    n_neighbours: int = 8,
    iterations: int = 10,
) -> np.ndarray:
    """
    Per-zone nematic phasor smoothing: each neighbour is flipped to align with the centre vector before averaging.

    Args:
        - 'partition' per-lens label (any hashable).
            Lenses with same label smooth together, lenses with different labels don't.
            For "smooth within each eye": pass eye_id-per-lens
            For "smooth within each (eye, hemisphere) quadrant": pass a 4-value label
    """
    if iterations <= 0:
        return field

    out = field.copy()
    for label in np.unique(partition):
        mask = np.flatnonzero(partition == label)
        n = mask.size
        if n < 2:
            continue

        positions_zone = positions[mask]
        k = min(n_neighbours, n - 1)
        tree = cKDTree(positions_zone)
        _, nidx = knn(tree, positions_zone, k)  # transient tree

        smoothed = smooth_nematic_vectors(out[mask], nidx, iterations=iterations)
        out[mask] = smoothed.astype(np.float32)
    return out


class BundlesAligner:
    """
    Computes per-lens bundle orientation (chi, chirality, saccade phasor)
    from a global optic flow direction and the head frame.

    Args:
        - flow_direction: (3,) array_like, Anterior flow direction in head coordinates.
            The point of expansion is at -flow_direction on the unit sphere.
        - diagonal_strength: float, Ratio of diagonal pull to pure flow target in the alignment phasor.
            1.0 equal pull (default), <1.0 makes the field more aligned with the flow, >1.0 pushes it off-axis.
        - diagonal_angle_deg: float, Angle of the diagonal pull in the focus tangent plane (default 45°).
            0° -> purely lateral, 45° -> balanced, 90° -> purely vertical.
        - alignment_smoothing_iterations: int, Per-zone nematic smoothing passes on the alignment phasor field.
        - saccade_smoothing_iterations: int, Per-eye nematic smoothing passes on the saccade phasor field.
        - falloff: float, Spatial falloff of the diagonal target (away from the point of expansion).
        - strength: float, Overall weighting of the combed target vs. raw flow projection.
    """
    def __init__(self,
        flow_direction: ArrayLike,
        diagonal_strength: float = 1.0,
        diagonal_angle_deg: float = 45.0,
        alignment_smoothing_iterations: int = 5,
        saccade_smoothing_iterations: int = 5,
        falloff: float = 0.7,
        strength: float = 1.0,
        ):

        S = np.asarray(flow_direction, dtype=np.float32).reshape(-1)

        if S.shape != (3,):
            raise ValueError(f"flow_direction must be a 3-vector, got shape {S.shape}")
        if float(np.linalg.norm(S)) < 1e-8:
            raise ValueError("flow_direction has zero magnitude")

        self.flow_direction = S / float(np.linalg.norm(S))

        self.diagonal_strength = float(diagonal_strength)
        self.diagonal_angle_deg = float(diagonal_angle_deg)

        self.alignment_smoothing_iterations = int(alignment_smoothing_iterations)
        self.saccade_smoothing_iterations = int(saccade_smoothing_iterations)

        self.falloff = float(falloff)
        self.strength = float(strength)

    def compute(self,
                model: 'CompoundEyeModel',
                override_chi: Optional[ArrayLike] = None,
                override_chirality: Optional[ArrayLike] = None,
                ) -> OrientationResult:
        """
        Compute the orientation field for the CompoundEyeModel's lens geometry.
        'override_chi' or 'override_chirality' can be supplied to bypass the corresponding pipeline step.
        The bypassed quantity is passed through to derive whatever depends on it.
        """

        lens_directions = model._lens_directions
        lens_positions = model._lens_positions
        bundle = model._bundle
        N = lens_directions.shape[0]

        e_x = self.flow_direction
        _rgt, _up = tangent_frames(e_x)
        e_y, e_z = -_rgt, _up

        # Eye / hemisphere signs (geometric)
        side_map = {e.eye_index: e.side_sign for e in model.eyes}
        eye_sign = np.array([side_map[eid] for eid in model._lens_eye_index], dtype=np.float32)
        eye_sign[eye_sign == 0] = 1.0   # midline -> left

        hemisphere_sign = np.sign(lens_positions @ e_z).astype(np.float32)
        hemisphere_sign[hemisphere_sign == 0] = 1.0   # equator -> dorsal

        # Resolve chirality
        if override_chirality is not None:
            chirality = _prepare_per_lens(override_chirality, N, 'chirality')
        else:
            chirality = (eye_sign * hemisphere_sign).astype(np.float32)

        # Alignment phasor field (with optional zoned smoothing)
        alignment = _alignment_phasor_field(
            lens_directions=lens_directions,
            e_x=e_x, e_y=e_y, e_z=e_z,
            eye_sign=eye_sign, hemisphere_sign=hemisphere_sign,
            strength=self.strength,
            falloff=self.falloff,
            diagonal_strength=self.diagonal_strength,
            diagonal_angle_deg=self.diagonal_angle_deg,
        )
        if self.alignment_smoothing_iterations > 0:
            zone_labels = (
                (eye_sign > 0).astype(np.int32) * 2 + (hemisphere_sign > 0).astype(np.int32)
            )
            alignment = _smooth_phasor_field(
                field=alignment,
                partition=zone_labels,
                positions=lens_positions,
                n_neighbours=8,
                iterations=self.alignment_smoothing_iterations,
            )

        # Major axis: alignment rotated by +/- flow_axis_deg, per eye
        base_flow_rot = float(np.radians(bundle.flow_axis_deg))
        major = _major_axis_field(
            alignment_phasor=alignment,
            lens_directions=lens_directions,
            base_rotation_rad=base_flow_rot,
            eye_sign=eye_sign,
            hemisphere_sign=hemisphere_sign,
        )

        # Bundle yaw chi from the major axis direction in each tangent frame
        if override_chi is not None:
            chi = _prepare_per_lens(override_chi, N, 'chi').astype(np.float32)
        else:
            major_angle = np.arctan2(
                np.sum(major * model._local_up, axis=1),
                np.sum(major * model._local_right, axis=1),
            )
            effective_main = np.where(
                chirality > 0,
                float(bundle.main_axis_rad) + np.pi,
                -float(bundle.main_axis_rad),
            ).astype(np.float32)
            chi = (major_angle - effective_main).astype(np.float32)
            chi = wrap_angle(chi)

        # Saccade phasor: major axis rotated by base_sacc * chirality (4 zones), smoothed, and polarised
        base_sacc_rot = -float(np.radians(bundle.saccade_offset_deg))
        sacc = _saccade_phasor_field(
            major_axis=major,
            lens_directions=lens_directions,
            base_rotation_rad=base_sacc_rot,
            chirality=chirality,
        )
        if self.saccade_smoothing_iterations > 0:
            # Per-eye smoothing: partition by eye_id
            sacc = _smooth_phasor_field(
                field=sacc,
                partition=model._lens_eye_index,
                positions=lens_positions,
                n_neighbours=8,
                iterations=self.saccade_smoothing_iterations,
            )
        # Polarise: saccade phasor consistently 'up' in the flow frame
        sacc[sacc @ e_z < 0] *= -1.0

        return OrientationResult(
            chi=chi.astype(np.float32),
            chirality=chirality.astype(np.float32),
            saccade_phasor=sacc.astype(np.float32),
            alignment_phasor=alignment,
            major_axis=major,
            eye_sign=eye_sign,
            hemisphere_sign=hemisphere_sign,
            flow_frame=(e_x, e_y, e_z),
        )

    def apply(self,
              model: 'CompoundEyeModel',
              override_chi: Optional[ArrayLike] = None,
              override_chirality: Optional[ArrayLike] = None,
              ) -> OrientationResult:
        """
        Compute and write the result into the CompoundEyeModel (also return it).
        """
        result = self.compute(model, override_chi=override_chi, override_chirality=override_chirality)
        model._apply_orientation(result)
        return result


##
# Helpers for callers that only need simpler pipeline


def trivial_orientation(N: int) -> OrientationResult:
    """
    For R=1 bundles or any case with no bundle to orient.
    """
    return OrientationResult(
        chi=np.zeros(N, dtype=np.float32),
        chirality=np.ones(N, dtype=np.float32),
        saccade_phasor=np.zeros((N, 3), dtype=np.float32),
    )


def _prepare_per_lens(value: ArrayLike, N: int, name: str) -> np.ndarray:
    return broadcast_1d(value, N, name)


def apply_chirality(
    model: 'CompoundEyeModel',
    chi: ArrayLike,
    chirality: ArrayLike,
) -> OrientationResult:
    """
    Derive a OrientationResult when the user supplies chi and chirality directly
    (no flow direction available).

    Major axis is bundle.main_axis_rad + chi + chirality
    saccade phasor is major + bundle.saccade_offset_deg + chirality

    No smoothing (it is presumed the user knows what they want).
    """

    N = model._lens_directions.shape[0]
    chi = _prepare_per_lens(chi, N, 'chi').astype(np.float32)
    chirality = _prepare_per_lens(chirality, N, 'chirality').astype(np.float32)

    main_rad = float(model._bundle.main_axis_rad)
    effective_main = np.where(chirality > 0, main_rad, np.pi - main_rad).astype(np.float32)
    major_angle = chi + effective_main
    major = local_to_world(
        np.stack([np.cos(major_angle), np.sin(major_angle)], axis=-1),
        model._local_right, model._local_up,
    )

    base_sacc_rot = -float(np.radians(model._bundle.saccade_offset_deg))
    angles = (base_sacc_rot * chirality).astype(np.float32)
    sacc = rotate_in_tangent_plane(major, model._lens_directions, angles, normalize=True)
    sacc = sacc.astype(np.float32)

    return OrientationResult(chi=chi, chirality=chirality, saccade_phasor=sacc, major_axis=major)