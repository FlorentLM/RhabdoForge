from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
from numpy.typing import ArrayLike

from insectvision.utils import norm_l2, broadcast_1d
from insectvision.geometry.linalg import tangent_frames, rotate_in_tangent_plane, local_to_world
from insectvision.geometry.neighbours import smooth_field_partitioned
from insectvision.geometry.circular import wrap_angle

if TYPE_CHECKING:
    from insectvision.compound_eyes import Model



@dataclass
class AlignmentResult:
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



# TODO: Maybe this helper can be absorbed by more generic pure functions

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
            equatorial_discontinuity: bool = True,
            falloff: float = 0.7,
            strength: float = 1.0,
        ):

        S = np.asarray(flow_direction, dtype=np.float32).reshape(-1)

        if S.shape != (3,):
            raise ValueError(f"flow_direction must be a 3-vector, got shape {S.shape}")
        if float(np.linalg.norm(S)) < 1e-8:
            raise ValueError("flow_direction has zero magnitude")

        self.equatorial_discontinuity = bool(equatorial_discontinuity)

        self.flow_direction = S / float(np.linalg.norm(S))

        self.diagonal_strength = float(diagonal_strength)
        self.diagonal_angle_deg = float(diagonal_angle_deg)

        self.alignment_smoothing_iterations = int(alignment_smoothing_iterations)
        self.saccade_smoothing_iterations = int(saccade_smoothing_iterations)

        self.falloff = float(falloff)
        self.strength = float(strength)

    def compute(self,
                model: 'Model',
                override_chi: Optional[ArrayLike] = None,
                override_chirality: Optional[ArrayLike] = None,
                ) -> AlignmentResult:
        """
        Compute the orientation field for the CompoundEyeModel's lens geometry.
        'override_chi' or 'override_chirality' can be supplied to bypass the corresponding pipeline step.
        The bypassed quantity is passed through to derive whatever depends on it.
        """

        e_x = self.flow_direction
        _rgt, _up = tangent_frames(e_x)
        e_y, e_z = -_rgt, _up

        # Eye / hemisphere signs (geometric)
        side_map = {e.eye_index: e.side_sign for e in model.eyes}
        bilateral_sign = np.array([side_map[e] for e in model.eye_index], dtype=np.float32)
        bilateral_sign[bilateral_sign == 0] = 1.0   # midline -> left

        equatorial_sign = np.sign(model.positions @ e_z).astype(np.float32)
        equatorial_sign[equatorial_sign == 0] = 1.0   # equator -> dorsal

        if not self.equatorial_discontinuity:
            equatorial_sign.fill(1.0)

        alignment = _alignment_phasor_field(
            lens_directions=model.directions,
            e_x=e_x, e_y=e_y, e_z=e_z,
            eye_sign=bilateral_sign,
            hemisphere_sign=equatorial_sign,
            strength=self.strength,
            falloff=self.falloff,
            diagonal_strength=self.diagonal_strength,
            diagonal_angle_deg=self.diagonal_angle_deg,
        )

        if self.alignment_smoothing_iterations > 0:
            zone_labels = (
                    (bilateral_sign > 0).astype(np.int32) * 2 + (equatorial_sign > 0).astype(np.int32)
            )
            alignment = smooth_field_partitioned(
                values=alignment,
                kind='nematic',
                partition=zone_labels,
                positions=model.positions,
                k=8,
                n_iter=self.alignment_smoothing_iterations,
            ).astype(np.float32)

        # Major axis: alignment rotated by +/- flow_axis_deg, per eye
        rotated = rotate_in_tangent_plane(
            vectors=alignment,
            normals=model.directions,
            angles=np.deg2rad(model.bundle.flow_axis_deg) * bilateral_sign,
            normalize=False
        )

        # Resolve chirality (most of it is no-op if equatorial_discontinuity is False, but kept for robustness)

        if override_chirality is not None:
            chirality = broadcast_1d(override_chirality, model.shape[0], 'chirality')
        else:
            chirality = (bilateral_sign * equatorial_sign).astype(np.float32)

        # Equatorial flip: bundle point up dorsally, and down ventrally
        ref = -equatorial_sign[:, None] * alignment
        dot_check = np.einsum('ij,ij->i', rotated, ref)
        rotated[dot_check < 0] *= -1.0

        major_axis = norm_l2(rotated).astype(np.float32)

        # Bundle yaw (chi) from the major axis direction in each tangent frame
        if override_chi is not None:
            chi = broadcast_1d(override_chi, model.shape[0], 'chi')
        else:
            major_angle = np.arctan2(
                np.sum(major_axis * model.up, axis=1),
                np.sum(major_axis * model.right, axis=1),
            )
            effective_main = np.where(
                chirality > 0,
                model.bundle.main_axis_rad + np.pi,
                -model.bundle.main_axis_rad,
            ).astype(np.float32)

            chi = wrap_angle(major_angle - effective_main).astype(np.float32)

        # Saccade phasor: major axis rotated by base saccade offset * chirality (4 zones), smoothed, and polarised
        sacc = rotate_in_tangent_plane(
            vectors=major_axis,
            normals=model.directions,
            angles=-np.deg2rad(model.bundle.saccade_offset_deg) * chirality,
            normalize=True
        )

        if self.saccade_smoothing_iterations > 0:
            sacc = smooth_field_partitioned(
                values=sacc,
                kind='nematic',
                partition=model.eye_index,
                positions=model.positions,
                k=8,
                n_iter=self.saccade_smoothing_iterations,
            ).astype(np.float32)

        # Polarise: saccade phasor consistently 'up' in the flow frame
        sacc[equatorial_sign < 0] *= -1.0

        # Make sure the polarised field really points up globally
        for eye in model.eyes:
            if np.sum(sacc[eye.indices] @ e_z) < 0:
                sacc[eye.indices] *= -1.0

        return AlignmentResult(
            chi=chi.astype(np.float32),
            chirality=chirality.astype(np.float32),
            saccade_phasor=sacc.astype(np.float32),
            alignment_phasor=alignment,
            major_axis=major_axis,
            eye_sign=bilateral_sign,
            hemisphere_sign=equatorial_sign,
            flow_frame=(e_x, e_y, e_z),
        )

    def apply(self,
              model: 'Model',
              override_chi: Optional[ArrayLike] = None,
              override_chirality: Optional[ArrayLike] = None,
              ) -> AlignmentResult:
        """
        Compute and write the result into the CompoundEyeModel (also return it).
        """
        result = self.compute(model, override_chi=override_chi, override_chirality=override_chirality)
        model._bundle_orientation_backwrite(result)
        return result


##
# Helpers for callers that only need simpler pipeline

# TODO: Move these as class methods in BundlesAligner

def trivial_alignment(N: int) -> AlignmentResult:
    """
    For R=1 bundles or any case with no bundle to orient.
    """
    return AlignmentResult(
        chi=np.zeros(N, dtype=np.float32),
        chirality=np.ones(N, dtype=np.float32),
        saccade_phasor=np.zeros((N, 3), dtype=np.float32),
    )


def apply_chirality(
    model: 'Model',
    chi: ArrayLike,
    chirality: ArrayLike,
) -> AlignmentResult:
    """
    Derive a OrientationResult when the user supplies chi and chirality directly
    (no flow direction available).

    Major axis is bundle.main_axis_rad + chi + chirality
    saccade phasor is major + bundle.saccade_offset_deg + chirality

    No smoothing (it is presumed the user knows what they want).
    """

    N = model.directions.shape[0]
    chi = broadcast_1d(chi, N, 'chi')
    chirality = broadcast_1d(chirality, N, 'chirality')

    main_rad = float(model.bundle.main_axis_rad)
    effective_main = np.where(chirality > 0, main_rad, np.pi - main_rad).astype(np.float32)
    major_angle = chi + effective_main
    major = local_to_world(
        np.stack([np.cos(major_angle), np.sin(major_angle)], axis=-1),
        model.right, model.up,
    )

    sacc = rotate_in_tangent_plane(
        vectors=major,
        normals=model.directions,
        angles=-np.radians(model.bundle.saccade_offset_deg) * chirality,
        normalize=True
    )

    return AlignmentResult(chi=chi, chirality=chirality, saccade_phasor=sacc, major_axis=major)