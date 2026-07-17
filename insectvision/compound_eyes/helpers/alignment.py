from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, TYPE_CHECKING
import numpy as np
from numpy.typing import ArrayLike

from insectvision.utils import broadcast_1d, norm_l2
from insectvision.geometry.linalg import tangent_frames, project_to_tangent, local_to_world, projected_bearing
from insectvision.geometry.circular import wrap_angle, fold_angle
from insectvision.geometry.fields import smooth_field_partitioned

if TYPE_CHECKING:
    from insectvision.compound_eyes import Model


@dataclass
class AlignmentResult:
    """
    Output of BundlesAligner.compute().

    chi and chirality are the only quantities the model needs,
    rest are diagnostics for plotting / debugging the field.
    """

    chi: np.ndarray                 # (N,) main-axis bearing in each tangent frame (rad)
    chirality: np.ndarray           # (N,) +/- 1

    saccade_phasor: np.ndarray      # (N, 3) world-space saccade axis (unit)

    major_axis: Optional[np.ndarray] = None         # (N, 3) world-space main axis (unit)
    flow_line: Optional[np.ndarray] = None          # (N, 3) world-space combed flow reference (unit)
    side_sign: Optional[np.ndarray] = None          # (N,) +1 left/midline, -1 right
    equator_sign: Optional[np.ndarray] = None       # (N,) +1 dorsal, -1 ventral
    flow_frame: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    diagnostics: Dict[str, float] = field(default_factory=dict)


def _comb_field(
    directions: np.ndarray,
    e_x: np.ndarray,
    e_y: np.ndarray,
    e_z: np.ndarray,
    side_sign: np.ndarray,
    equator_sign: np.ndarray,
    falloff: float = 0.7,
    strength: float = 1.0,
    angle_deg: float = 45.0,
) -> np.ndarray:
    """
    De-singularised optic-flow reference direction, per ommatidium (unit, world).

    Away from the point of expansion this is just the flow projected into each
    tangent plane. That projection vanishes at the point of expansion (-e_x),
    so there the field is combed toward a four-zone diagonal target instead,
    blended in by a weight that peaks at -e_x and falls off with chord distance.
    The target is mirror-symmetric by construction, so the combed field
    (and everything derived from it) keeps the 4-zone symmetry.
    """

    proj = project_to_tangent(e_x, directions)          # Radial flow (singularity at +/-e_x)

    a = np.radians(angle_deg)
    lateral = np.cos(a)
    vertical = np.sin(a)

    local_coords = np.stack([
        np.ones(len(side_sign)),
        -side_sign * lateral,
        equator_sign * vertical
    ], axis=-1)

    target_world = local_to_world(local_coords, e_x, e_y, e_z)
    pnorm = np.linalg.norm(proj, axis=1, keepdims=True)

    target = project_to_tangent(target_world, directions)
    target = norm_l2(target) * pnorm

    dist = np.linalg.norm(directions - (-e_x)[None, :], axis=1)
    w = np.clip(1.0 - dist * falloff, 0.0, None) * strength

    combed = (1.0 - w[:, None]) * proj + w[:, None] * target

    return norm_l2(combed)


class BundlesAligner:
    """
    Per-ommatidium bundle orientation from a single external reference direction.

    The main axis is placed at a fixed offset (`bundle.alignment_offset`) from the
    locally projected optic flow, signed by chirality = side_sign * equator_sign, so
    the field is mirror-symmetric across the sagittal midline and the dorso-ventral
    equator.

    The flow reference is de-singularised by combing near the point of expansion.

    The line fields are then nematically optionally smoothed (mod 180°):
        - alignment: the main-axis field, per chirality zone within each eye,
            i.e. partitioned by (eye, equator_sign).
        - saccade: the saccade-axis field, per whole eye.

    Chirality mirror-flips the bundle across the main axis (doesn't rotate it).
    The saccade axis, an offset from the main axis, follows the flip: chi + chirality * saccade_offset

    Args:
        - ref_direction: (3,) array_like. External reference (flow) direction in head coords,
            the point of expansion is at -ref_direction on the unit sphere.
        - combing_strength / combing_angle_deg / combing_falloff: combing knobs.
        - alignment_smoothing_iter: nematic passes on the main-axis field (per zone).
        - saccade_smoothing_iter: nematic passes on the saccade field (per eye).
        - smoothing_k: neighbours per point for the smoothing passes.
        - flip_polarity: global 180° flip of the saccade axes.
        - flip_saccade_polarity: global 180° flip of the saccade axes.
        - equatorial_discontinuity: if False, the dorso-ventral chirality split is disabled
            (equator_sign == +1 everywhere -> one alignment zone per eye).
    """

    def __init__(self,
            ref_direction: ArrayLike,
            combing_strength: float = 1.0,
            combing_angle_deg: float = 45.0,
            combing_falloff: float = 0.7,
            alignment_smoothing_iter: int = 5,
            saccade_smoothing_iter: int = 5,
            smoothing_k: int = 8,
            flip_polarity: bool = False,
            flip_saccade_polarity: bool = False,
            equatorial_discontinuity: bool = True
        ):

        S = np.asarray(ref_direction, dtype=np.float32).reshape(-1)

        if S.shape != (3,):
            raise ValueError(f'Reference direction must be a 3-vector, got shape {S.shape}')
        n = np.linalg.norm(S)
        if n < 1e-8:
            raise ValueError('Reference direction has zero magnitude')

        self.ref_direction = (S / n).astype(np.float32)
        self.combing_strength = float(combing_strength)
        self.combing_angle_deg = float(combing_angle_deg)
        self.combing_falloff = float(combing_falloff)
        self.alignment_smoothing_iter = int(alignment_smoothing_iter)
        self.saccade_smoothing_iter = int(saccade_smoothing_iter)
        self.smoothing_k = int(smoothing_k)
        self.flip_polarity = bool(flip_polarity)
        self.flip_saccade_polarity = bool(flip_saccade_polarity)
        self.equatorial_discontinuity = bool(equatorial_discontinuity)

    def _zone_signs(self, model: 'Model', e_z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Per-ommatidium (side_sign, equator_sign)."""
        N = model.shape[0]

        side_map = {e.eye_index: e.side_sign for e in model.eyes}
        side = np.array([side_map[e] for e in model.eye_index], dtype=np.float32)
        side[side == 0] = 1.0   # midline -> left

        if self.equatorial_discontinuity:
            equator = np.sign(model.positions @ e_z).astype(np.float32)
            equator[equator == 0] = 1.0   # on the equator -> dorsal
        else:
            equator = np.ones(N, dtype=np.float32)

        return side, equator

    def compute(self,
            model: 'Model',
            override_chi: Optional[ArrayLike] = None,
            override_chirality: Optional[ArrayLike] = None,
            verbose: bool = True,
        ) -> AlignmentResult:
        """
        Compute chi / chirality / saccade axis for the model's lens geometry, with optiojnal
        nematic per-zone smoothing

        Args:
         - override_chi / override_chirality: bypass the corresponding step
        """

        N = model.shape[0]
        right, up = model.right, model.up

        # Reference external flow frame
        e_x = self.ref_direction
        rgt, up_f = tangent_frames(e_x)
        e_y, e_z = -rgt, up_f

        side, equator = self._zone_signs(model, e_z)

        ref_flow_line = _comb_field(
            directions=model.directions,
            e_x=e_x, e_y=e_y, e_z=e_z,
            side_sign=side, equator_sign=equator,
            falloff=self.combing_falloff,
            strength=self.combing_strength,
            angle_deg=self.combing_angle_deg,
        )
        ref_flow_bearing = projected_bearing(ref_flow_line, right, up)

        # Chirality (four zones)
        if override_chirality is not None:
            chirality = broadcast_1d(override_chirality, N, 'chirality')
        else:
            chirality = side * equator

        # Main axis: fixed offset from flow, signed by chirality, then smoothed per zone
        offset = model.bundle.alignment_offset_rad
        if override_chi is not None:
            chi = broadcast_1d(override_chi, N, 'chi')
            major = local_to_world(np.stack([np.cos(chi), np.sin(chi)], -1), right, up)

        else:
            chi0 = wrap_angle(ref_flow_bearing + chirality * offset)
            major = local_to_world(np.stack([np.cos(chi0), np.sin(chi0)], -1), right, up)

            if self.alignment_smoothing_iter > 0:
                align_partition = (np.asarray(model.eye_index, dtype=np.int64) * 2
                                   + (equator > 0).astype(np.int64))
                major = smooth_field_partitioned(
                    values=major, kind='nematic', partition=align_partition,
                    positions=model.positions, k=self.smoothing_k,
                    n_iter=self.alignment_smoothing_iter,
                )
                major = norm_l2(major)

            chi = wrap_angle(projected_bearing(major, right, up))

            # Polarity: the residual 180 deg DOF
            if self.flip_polarity:
                chi = wrap_angle(chi + np.pi)
                major = local_to_world(np.stack([np.cos(chi), np.sin(chi)], -1), right, up)

        # Saccade axis: derived from the (smoothed) main axis, then smoothed per eye
        saccade = self._saccade_from_major(model, major, chirality, equator, skip_smooth=False)

        result = AlignmentResult(
            chi=chi,
            chirality=chirality,
            saccade_phasor=saccade,
            major_axis=major,
            flow_line=ref_flow_line,
            side_sign=side,
            equator_sign=equator,
            flow_frame=(e_x, e_y, e_z),
        )

        diag = self._diagnostics(model, result)

        if verbose:
            if 'align_mean' in diag:
                print(f"[ align ] offset from flow: mean={diag['align_mean']:5.1f}deg, "
                      f"std={diag['align_std']:4.1f}deg, target={diag['align_target']:.1f}deg, "
                      f"within+-{diag['tol_deg']:g}deg: {100 * diag['align_frac']:.0f}%, "
                      f"resulting mean error {diag['align_error_pct']:.1f}%")

            if 'saccade_mean' in diag:
                print(f"[saccade] offset from main: mean={diag['saccade_mean']:5.1f}deg, "
                      f"std={diag['saccade_std']:4.1f}deg, target={diag['saccade_target']:.1f}deg, "
                      f"within+-{diag['tol_deg']:g}deg: {100 * diag['saccade_frac']:.0f}%, "
                      f"resulting mean error {diag['saccade_error_pct']:.1f}%")

        result.diagnostics = diag

        return result

    def _diagnostics(self, model: 'Model', result: AlignmentResult, tol_deg: float = 2.0) -> Dict[str, float]:
        """
        Line-angle (mod 180, folded to [0, 90]) between:
          - the main axis and the flow reference -> should average alignment_offset
          - the saccade axis and the main axis -> should average saccade_offset

        Reports mean / std / fraction within tol_deg of the (folded) target.
        """

        right, up = model.right, model.up

        b_major = projected_bearing(result.major_axis, right, up, degrees=True)
        b_flow = projected_bearing(result.flow_line, right, up, degrees=True)
        b_sacc = projected_bearing(result.saccade_phasor, right, up, degrees=True)

        valid = np.linalg.norm(result.flow_line, axis=1) > 1e-3   # Skip the un-combed antipode

        # Differences (mod 180)
        da = (b_major - b_flow) % 180.0
        align_off = np.minimum(da, 180.0 - da)[valid]

        ds = (b_sacc - b_major) % 180.0
        sacc_off = np.minimum(ds, 180.0 - ds)

        a_t = fold_angle(model.bundle.alignment_offset_deg, degrees=True)
        s_t = fold_angle(model.bundle.saccade_offset_deg, degrees=True)

        d = {'tol_deg': float(tol_deg), 'align_target': float(a_t), 'saccade_target': float(s_t)}

        if align_off.size:
            mae = np.mean(np.abs(align_off - a_t))
            d.update(
                align_mean=float(align_off.mean()),
                align_std=float(align_off.std()),
                align_frac=float(np.mean(np.abs(align_off - a_t) <= tol_deg)),
                align_error_pct=float((mae / a_t * 100) if a_t > 0 else 0.0)
            )

        if sacc_off.size:
            mae = np.mean(np.abs(sacc_off - s_t))
            d.update(
                saccade_mean=float(sacc_off.mean()),
                saccade_std=float(sacc_off.std()),
                saccade_frac=float(np.mean(np.abs(sacc_off - s_t) <= tol_deg)),
                saccade_error_pct=float((mae / s_t * 100) if s_t > 0 else 0.0)
            )

        return d

    def _saccade_from_major(self,
            model: 'Model',
            major: np.ndarray,
            chirality: Optional[np.ndarray] = None,
            equator_sign: Optional[np.ndarray] = None,
            skip_smooth: bool = False
        ) -> np.ndarray:
        """
        Saccade-axis world field from the main-axis field: offset each main axis by
        chirality * saccade_offset, resolve the per-eye polarity, then (optionally)
        nematically smooth per eye.
        """
        if chirality is None or equator_sign is None:
            _, e_z = tangent_frames(self.ref_direction)
            side, equator = self._zone_signs(model, e_z)
            if chirality is None:
                chirality = side * equator
            if equator_sign is None:
                equator_sign = equator

        chi = projected_bearing(major, model.right, model.up)
        sacc_bearing = chi + chirality * model.bundle.saccade_offset_rad

        # Per-eye polarity
        sacc_bearing = sacc_bearing + np.pi * (equator_sign < 0).astype(np.float32)
        if self.flip_saccade_polarity:
            sacc_bearing = sacc_bearing + np.pi

        saccade = local_to_world(
            np.stack([np.cos(sacc_bearing), np.sin(sacc_bearing)], axis=-1),
            model.right, model.up,
        )

        if self.saccade_smoothing_iter > 0 and not skip_smooth:
            saccade = smooth_field_partitioned(
                values=saccade, kind='nematic', partition=np.asarray(model.eye_index),
                positions=model.positions, k=self.smoothing_k,
                n_iter=self.saccade_smoothing_iter,
            )
            saccade = norm_l2(saccade)

        return saccade

    def apply(self,
            model: 'Model',
            override_chi: Optional[ArrayLike] = None,
            override_chirality: Optional[ArrayLike] = None,
        ) -> AlignmentResult:
        """Compute and write the result into the model (also returns it)."""
        result = self.compute(model, override_chi=override_chi, override_chirality=override_chirality)
        model._bundle_orientation_backwrite(result)
        return result

    # Simplified constructors for callers with no flow pipeline

    @staticmethod
    def trivial_alignment(N: int) -> AlignmentResult:
        """For R=1 bundles or any case with no bundle to orient."""
        return AlignmentResult(
            chi=np.zeros(N, dtype=np.float32),
            chirality=np.ones(N, dtype=np.float32),
            saccade_phasor=np.zeros((N, 3), dtype=np.float32),
        )

    @staticmethod
    def explicit_alignment(model: 'Model', chi: ArrayLike, chirality: ArrayLike) -> AlignmentResult:
        """
        Build an AlignmentResult from directly supplied chi / chirality
        (no flow direction, no smoothing). chi is the main-axis bearing,
        the saccade axis is chi + chirality * saccade_offset.
        """
        N = model.directions.shape[0]
        chi = broadcast_1d(chi, N, 'chi')
        chirality = broadcast_1d(chirality, N, 'chirality')

        sacc_bearing = chi + chirality * model.bundle.saccade_offset_rad
        major = local_to_world(np.stack([np.cos(chi), np.sin(chi)], -1), model.right, model.up)
        saccade = local_to_world(np.stack([np.cos(sacc_bearing), np.sin(sacc_bearing)], -1), model.right, model.up)

        return AlignmentResult(chi=chi, chirality=chirality, saccade_phasor=saccade, major_axis=major)