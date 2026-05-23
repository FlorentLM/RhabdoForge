import warnings
from pathlib import Path
from typing import Optional, Union, Tuple, Sequence, List
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from insectvision.engine.world_utils import WORLD_UP, WORLD_RIGHT, WORLD_FORWARD
from insectvision.utils.math import normalise_vectors, tangent_frames, fibonacci_sphere, icosphere
from insectvision.compound_eyes.datatypes import (
    LENS_STATIC_DTYPE, LENS_DYNAMIC_DTYPE, RCPT_STATIC_DTYPE, RCPT_DYNAMIC_DTYPE
)
from insectvision.compound_eyes.kernel import RhabdomereKernel
from insectvision.compound_eyes.proxies import Eye, Ommatidium, Cartridge, LensView, ReceptorView


## Build helpers


def _flow_aligned_frame(flow_direction: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Orthonormal frame aligned with optic flow.
      e_x : flow direction (anterior)
      e_y : lateral axis (cross of WORLD_UP with e_x)
      e_z : dorsal axis (perpendicular to both). Coincides with WORLD_UP when flow is purely sagittal
    """
    S = np.asarray(flow_direction, dtype=np.float32)
    S = S / np.linalg.norm(S)

    up = np.asarray(WORLD_UP, dtype=np.float32)
    if abs(float(S @ up)) > 0.999:
        ref = np.asarray(WORLD_FORWARD, dtype=np.float32)
        e_y = np.cross(ref, S)
    else:
        e_y = np.cross(up, S)
    e_y = e_y / np.linalg.norm(e_y)
    e_z = np.cross(S, e_y)
    return S.astype(np.float32), e_y.astype(np.float32), e_z.astype(np.float32)


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
) -> np.ndarray:
    """
    Combed-flow alignment phasor field.

    For each lens we blend the projected optic flow with a diagonal target
    that depends on which eye and which D/V hemisphere the lens is on.

    Args:
        - eye_sign: +1 on one eye and -1 on the other. Mirrors the e_y component across the body midline.
        - hemisphere_sign: +1 dorsal / -1 ventral. Mirrors the e_z component across the flow-frame equator.

        Together they give the 4-quadrants pattern, with only the equator discontinuous (no radial expansion at the FoC)

        - diagonal_strength: Controls how strongly the target leans into the diagonal vs. the pure anti-flow direction
        1.0: equal weighting (default), <1: make the combed field more "backwards", >1: push it more strongly off-axis
    """
    source = -e_x

    # projection of flow onto each tangent plane
    proj_S = e_x[None, :] - (lens_directions @ e_x)[:, None] * lens_directions

    # Bilateral + hemisphere target
    target_world = (
        -e_x[None, :]
        + diagonal_strength * (
            - eye_sign[:, None]        * e_y[None, :]
            + hemisphere_sign[:, None] * e_z[None, :]
        )
    ).astype(np.float32)

    # Project target in each tangent plane
    target_proj = target_world - np.sum(target_world * lens_directions,
                                        axis=1, keepdims=True) * lens_directions
    target_norm = np.linalg.norm(target_proj, axis=1, keepdims=True)
    proj_S_mag  = np.linalg.norm(proj_S,      axis=1, keepdims=True)
    target_proj = np.where(target_norm > 1e-6,
                           target_proj / target_norm.clip(min=1e-8) * proj_S_mag,
                           proj_S)

    # Falloff: chord distance from the FoC on the unit sphere of lens directions
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
    Bundle main axis (R3-R6 in Drosophila): per-eye rotation of the alignment
    phasor in each tangent plane, then hemisphere-aware disambiguation.

    - Rotation is `base_rotation_rad * eye_sign`. For Drosophila this is -81°
    on one eye and +81° on the other, so the bundle's main axis ends up bilaterally symmetric.
    - Disambiguation reference is `-hemisphere_sign * alignment_phasor`, not the binormal `n × alignment_phasor`.
    This produces the strong equatorial flip: major axis points up in the dorsal hemisphere and down in the ventral.
    """
    angles = (base_rotation_rad * eye_sign).astype(np.float32)
    cos_a = np.cos(angles)[:, None]
    sin_a = np.sin(angles)[:, None]

    n_cross_v = np.cross(lens_directions, alignment_phasor)
    rotated = alignment_phasor * cos_a + n_cross_v * sin_a

    ref = -hemisphere_sign[:, None] * alignment_phasor
    dot_check = np.einsum('ij,ij->i', rotated, ref)
    rotated[dot_check < 0] *= -1.0

    norms = np.linalg.norm(rotated, axis=1, keepdims=True).clip(min=1e-8)
    return (rotated / norms).astype(np.float32)


def _saccade_phasor_field(
        major_axis: np.ndarray,
        lens_directions: np.ndarray,
        base_rotation_rad: float,
        chirality: np.ndarray,
) -> np.ndarray:
    """
    Saccade phasor field: rotate the major axis by `base_rotation_rad * chirality` in each tangent plane.

    Four-zone pattern (chirality = eye_sign * hemisphere_sign), so that the rotation
    sign flips both between eyes and across the equator.

    Returns a nematic field (direction is locally ambiguous, the downstream smoothing step is gives global coherence).
    """
    angles = (base_rotation_rad * chirality).astype(np.float32)
    cos_a = np.cos(angles)[:, None]
    sin_a = np.sin(angles)[:, None]

    n_cross_v = np.cross(lens_directions, major_axis)
    sacc = major_axis * cos_a + n_cross_v * sin_a

    norms = np.linalg.norm(sacc, axis=1, keepdims=True).clip(min=1e-8)
    return (sacc / norms).astype(np.float32)


def _smooth_phasor_field(
        field: np.ndarray,
        eyes: List['Eye'],
        lens_positions: np.ndarray,
        n_neighbours: int = 8,
        iterations: int = 10,
) -> np.ndarray:
    """
    Smooths a phasor field (runs per-eye).
    Nematic averaging: each neighbour is flipped to align with the centre vector before averaging.
    """
    out = field.copy()

    for eye in eyes:
        neighb = eye.neighbours(
            points=lens_positions,
            k=n_neighbours,
            immediate_only=False,
        )
        if neighb is None:
            continue

        mask = neighb.mask
        nidx_local = neighb.indices
        eye_field = out[mask].copy()

        for _ in range(iterations):
            base = eye_field
            neigh = eye_field[nidx_local]

            dots = np.einsum('ik,ijk->ij', base, neigh)
            neigh = np.where(dots[..., None] < 0, -neigh, neigh)

            stacked = np.concatenate([base[:, None, :], neigh], axis=1)
            avg = stacked.mean(axis=1)
            norms = np.linalg.norm(avg, axis=1, keepdims=True)
            eye_field = np.where(norms > 1e-8, avg / norms.clip(min=1e-8), base)

        out[mask] = eye_field.astype(np.float32)

    return out


def _smooth_phasor_field_zoned(
        field: np.ndarray,
        partition: np.ndarray,
        lens_positions: np.ndarray,
        n_neighbours: int = 8,
        iterations: int = 10,
) -> np.ndarray:
    """
    Per-zone nematic phasor smoothing: like `_smooth_phasor_field`, but the partition is an arbitrary per-lens label.
    Lenses sharing a label smooth together.
    Singletons / empty groups are passed through unchanged.
    """
    if iterations <= 0:
        return field

    out = field.copy()
    for label in np.unique(partition):
        idx = np.flatnonzero(partition == label)
        n_zone = idx.size
        if n_zone < 2:
            continue

        positions_zone = lens_positions[idx]
        k = min(n_neighbours, n_zone - 1)
        tree = cKDTree(positions_zone)
        _, nidx_local = tree.query(positions_zone, k=k + 1)
        nidx_local = nidx_local[:, 1:]

        zone_field = out[idx].copy()
        for _ in range(iterations):
            base = zone_field
            neigh = zone_field[nidx_local]

            dots = np.einsum('ik,ijk->ij', base, neigh)
            neigh = np.where(dots[..., None] < 0, -neigh, neigh)

            stacked = np.concatenate([base[:, None, :], neigh], axis=1)
            avg = stacked.mean(axis=1)
            norms = np.linalg.norm(avg, axis=1, keepdims=True)
            zone_field = np.where(norms > 1e-8, avg / norms.clip(min=1e-8), base)

        out[idx] = zone_field.astype(np.float32)

    return out


def _lens_geometry(
        directions: Optional[ArrayLike],
        positions: Optional[ArrayLike],
        ommatidia_count: Optional[int],
        eye_radius: float,
        icosphere_method: bool
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Generates (or normalises) the base 3D lens positions and directions.
    """

    if directions is None and ommatidia_count is None:
        raise ValueError("Requires either 'directions' or 'ommatidia_count'.")

    if directions is not None:
        dirs = np.asarray(directions)
    else:
        dirs = icosphere(ommatidia_count) if icosphere_method else fibonacci_sphere(ommatidia_count)
    dirs = normalise_vectors(dirs).astype(np.float32)
    N = len(dirs)

    if positions is not None:
        pos = np.asarray(positions, dtype=np.float32)

        if pos.ndim == 1 and pos.shape[0] == 3:
            pos = np.tile(pos, (N, 1))

        elif pos.shape != (N, 3):
            raise ValueError(f"Invalid 'positions' shape {pos.shape}. Expected ({N}, 3) or (3,).")
    else:
        pos = dirs * eye_radius

    return dirs, pos, N


def _receptors_geometry(
        lens_directions: np.ndarray,
        lens_positions: np.ndarray,
        local_right: np.ndarray,
        local_up: np.ndarray,
        kernel: 'RhabdomereKernel',
        bundle_orientation: np.ndarray,
        chirality: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates 3D world directions and positions for every rhabdomere behind each lens.

    The tip offset vector (lateral offsets + nodal distance, in μm) defines each receptor's viewing direction.

    Receptor positions are placed at the lens surface (μm-scale sub-lens geometry is probably
    negligible in world coordinates, it only really matters for the angular offset).
    """

    N = len(lens_directions)
    R = kernel.count

    nodal_dist = kernel.nodal_distance_um
    if nodal_dist is None:
        if R > 1:
            raise ValueError(
                "nodal_distance_um must be set for multi-receptor kernels (R > 1). "
                "It defines the lever arm that converts lateral offsets to angular shifts."
            )
        nodal_dist = 1.0  # R=1 with zero offsets: any positive value gives the lens axis

    rot_dx, rot_dy = kernel.rotated_offsets(bundle_orientation, chirality)

    # Tip offset in local frame (μm), only used for direction
    local_tip = np.stack([
        rot_dx,
        rot_dy,
        np.full((N, R), -nodal_dist, dtype=np.float32)
    ], axis=-1)

    world_tip = (
            local_tip[..., 0:1] * local_right[:, None, :] +
            local_tip[..., 1:2] * local_up[:, None, :] +
            local_tip[..., 2:3] * lens_directions[:, None, :]
    ).reshape(N * R, 3)

    # Receptor position: at the lens surface (tip offset only used for direction computation)
    rec_pos = np.repeat(lens_positions, R, axis=0)

    # Receptor direction: from tip through lens centre = -world_tip (normalised)
    rec_dirs = normalise_vectors(-world_tip.copy())

    return rec_dirs, rec_pos


def _acceptance_angles(
        lens_count: int,
        kernel: 'RhabdomereKernel',
        ioa_minor: np.ndarray,
        ioa_major: np.ndarray,
        eye_parameter: Optional[Union[float, Tuple]],
        explicit_angles_rad: Optional[ArrayLike],
        lens_diameters_um: np.ndarray,
) -> np.ndarray:
    """
    Computes the acceptance angles for all receptors.
    """

    N = lens_count
    R = kernel.count
    M = N * R

    if explicit_angles_rad is not None:
        angles_arr = np.asarray(explicit_angles_rad, dtype=np.float32)

        if angles_arr.shape == (M, 2):
            return angles_arr

        elif angles_arr.shape == (M,):
            return np.column_stack([angles_arr, angles_arr])

        elif angles_arr.shape == (N, 2) or angles_arr.shape == (N,):
            # Broadcast lens-level explicit to receptor-level
            if angles_arr.ndim == 1:
                angles_arr = np.column_stack([angles_arr, angles_arr])
            return np.repeat(angles_arr, R, axis=0)

        raise ValueError(f"Invalid explicit_angles_rad shape: {angles_arr.shape}")

    # Resolve eye parameter p to a (p_min, p_maj) pair. Defaults to 1.0 (pure Snyder / pure IOA)
    p = eye_parameter if eye_parameter is not None else 1.0
    p_min, p_maj = (p, p) if isinstance(p, (int, float, np.number)) else p

    if kernel.nodal_distance_um is not None:
        # Optics available: Snyder is the physical baseline and eye_parameter acts as a multiplicative scale
        #   p = 1.0 -> pure Snyder optics (diffraction-limited)
        #   p > 1.0 -> RFs wider than Snyder
        #   p < 1.0 -> RFs narrower than Snyder
        wavelengths_um = kernel.wavelengths_nm * 1e-3
        diffraction = wavelengths_um[None, :] / lens_diameters_um[:, None]
        diffraction = diffraction.ravel()
        geometric = np.tile(kernel.diameters_um / kernel.nodal_distance_um, N)
        snyder_acc = np.sqrt(diffraction ** 2 + geometric ** 2)
        acc_min = p_min * snyder_acc
        acc_maj = p_maj * snyder_acc
        # TODO: anisotropic Snyder (different λ/D in two axes for non-circular lenses)
        return np.column_stack([acc_min, acc_maj])

    # No optical model: fallback to the eye parameter / lattice convention
    # (also scaled by relative rhabdomere diameter so smaller rhabdomeres still get smaller RFs)
    max_d = np.max(kernel.diameters_um)
    rel_d = kernel.diameters_um / max_d if max_d > 0 else np.ones(R)
    acc_min = np.repeat(p_min * ioa_minor, R) * np.tile(rel_d, N)
    acc_maj = np.repeat(p_maj * ioa_major, R) * np.tile(rel_d, N)
    return np.column_stack([acc_min, acc_maj])


def _lattice_properties(
        directions: np.ndarray,
        positions: np.ndarray,
        eyes: list,
        k: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate local lattice properties by separating structural topology (positions)
    from optical axes (directions).

    Returns:
        ioa_minor, ioa_major: Per-lens interommatidial angles (rad)
        tilts: Per-lens hexatic lattice orientation (rad)
        neighb_counts: Per-lens immediate-neighbour count
        lens_spacing: Per-lens median distance to immediate neighbours (same units as positions)
                      Used as a default for lens_diameter_um (close-packing)
    """

    N = len(directions)
    ioa_minor = np.zeros(N, dtype=np.float32)
    ioa_major = np.zeros(N, dtype=np.float32)
    tilts = np.zeros(N, dtype=np.float32)
    neighb_counts = np.zeros(N, dtype=np.uint32)
    lens_spacing = np.zeros(N, dtype=np.float32)

    for eye in eyes:
        neighb = eye.neighbours(points=positions, k=k, immediate_only=True)
        if neighb is None:
            continue

        mask = neighb.mask
        e_dirs = directions[mask]
        e_pos  = positions[mask]

        nb_immediate_neighb = np.maximum(np.sum(neighb.is_immediate, axis=1), 1, dtype=np.float64)
        neighb_counts[mask] = nb_immediate_neighb

        # Optical IOA
        dots = np.sum(e_dirs[:, None, :] * e_dirs[neighb.indices], axis=2)
        angular_sep = np.arccos(np.clip(dots, -1.0, 1.0))

        # Tangent-plane projections for lattice tilt
        local_x, local_y = tangent_frames(e_dirs, world_up=WORLD_UP, world_right=WORLD_RIGHT)
        delta_pos = e_pos[neighb.indices] - e_pos[:, None, :]
        proj_x = np.sum(delta_pos * local_x[:, None, :], axis=2)
        proj_y = np.sum(delta_pos * local_y[:, None, :], axis=2)

        # Physical lattice spacing in world units (median over immediate neighbours)
        nbr_dist = np.linalg.norm(delta_pos, axis=2)
        nbr_dist_masked = np.where(neighb.is_immediate, nbr_dist, np.nan)
        with np.errstate(all='ignore'):
            e_lens_spacing = np.nanmedian(nbr_dist_masked, axis=1).astype(np.float32)
        e_lens_spacing = np.where(np.isfinite(e_lens_spacing), e_lens_spacing, 0.0)

        # Hexatic order parameter (Ψ6)
        angles  = np.arctan2(proj_y, proj_x)
        phasors = np.exp(1j * 6 * angles)
        phasors[~neighb.is_immediate] = 0.0
        z_avg   = np.sum(phasors, axis=1) / nb_immediate_neighb
        e_psi6  = np.abs(z_avg)
        e_tilts = np.angle(z_avg) / 6.0

        # IOA from sorted angular separations
        sep_masked = np.where(neighb.is_immediate, angular_sep, np.inf)
        sep_sorted = np.sort(sep_masked, axis=1)
        e_ioa_minor = np.mean(sep_sorted[:, :2], axis=1).astype(np.float32)
        sep_rev = np.where(neighb.is_immediate, -angular_sep, np.inf)
        top2_rev = np.sort(sep_rev, axis=1)[:, :2]
        e_ioa_major = np.mean(-top2_rev, axis=1).astype(np.float32)

        # Sparse fallback
        sparse = nb_immediate_neighb < 2
        if np.any(sparse):
            has_any = nb_immediate_neighb[sparse] > 0
            fallback = np.where(has_any, np.mean(sep_masked[sparse], axis=1, where=neighb.is_immediate[sparse]), 0.0)
            e_ioa_minor[sparse] = fallback
            e_ioa_major[sparse] = fallback
            e_tilts[sparse] = 0.0

        tilts[mask] = e_tilts
        ioa_minor[mask] = e_ioa_minor
        ioa_major[mask] = e_ioa_major
        lens_spacing[mask] = e_lens_spacing

        print(f"Eye {eye.eye_id} lattice hexatic quality (Ψ6): {np.mean(e_psi6):.3f}")

    return ioa_minor, ioa_major, tilts, neighb_counts, lens_spacing


def _pack_metadata(
        N: int,
        R: int,
        eye_ids: np.ndarray,
        receptor_types: Optional[np.ndarray],
        nb_counts: np.ndarray,
        chirality: np.ndarray
) -> np.ndarray:
    """
    Packs IDs, types, counts, and chirality flag into the uint32 bitfield.
    """

    eid = np.repeat(eye_ids, R).astype(np.uint32)
    lindex = np.repeat(np.arange(N, dtype=np.uint32), R)
    nb_rep = np.repeat(nb_counts, R).astype(np.uint32)

    is_mirrored = (np.repeat(chirality, R) < 0).astype(np.uint32)

    if receptor_types is not None and R == 1:
        # Override for R=1 where user specified the exact receptor type
        # TODO: Not sure this is worth keeping
        rtypes = np.asarray(receptor_types, dtype=np.uint32)
    else:
        # standard kernel layout
        rtypes = np.tile(np.arange(R, dtype=np.uint32), N)

    return ((eid & 0x07) |
            ((rtypes & 0x0F) << 3) |
            ((nb_rep & 0x0F) << 7) |
            ((lindex & 0xFFFF) << 11) |
            ((is_mirrored & 0x01) << 27))


##


class ReceptorArray:
    """
    Flat (GPU-friendly) structured array of receptors for the renderer.
    """

    def __init__(self,
                 directions: Optional[ArrayLike] = None,
                 positions: Optional[ArrayLike] = None,
                 ommatidia_count: Optional[int] = None,
                 kernel: Optional['RhabdomereKernel'] = None,
                 bundle_orientation: Optional[ArrayLike] = None,
                 chirality: Optional[ArrayLike] = None,
                 eye_ids: Optional[Union[int, ArrayLike]] = None,
                 receptor_types: Optional[Union[int, ArrayLike]] = None,
                 eye_parameter: Optional[Union[float, Tuple]] = None,
                 lens_diameter_um: Optional[Union[float, ArrayLike]] = None,
                 interommatidial_angles_rad: Optional[Union[ArrayLike, Tuple, float]] = None,
                 acceptance_angles_rad: Optional[Union[ArrayLike, Tuple, float]] = None,
                 eye_radius: float = 0.01,
                 icosphere_method: bool = True,
                 flow_direction: Optional[ArrayLike] = None,
                 diagonal_strength: float = 1.0,
                 alignment_smoothing_iterations: int = 0,
                 saccade_smoothing_iterations: int = 10,
                 ):
        """
        Construct a full receptor array.

        Each of the *N* lenses contains *R* receptors whose world-space directions are
        determined by the kernel offsets and rotated by the per-lens bundle orientation (chi).

        Acceptance angles are computed from the full optical model (Snyder 1979):

            Δρ = sqrt( (λ/D)² + (d_rhab/f)² )

        where λ = kernel.wavelength_nm, D = kernel.lens_diameter_um, d_rhab = kernel.diameters_um, f = kernel.nodal_distance_um

        This can be overridden with eye_parameter: p = delta_rho / delta_phi

        Args:
            directions: (N, 3) lens optical axes
            positions: (N, 3) lens positions in head space
            ommatidia_count: Number of ommatidia to build a uniform eye for (when positions are not specified)
            kernel: Species-level rhabdomere geometry
            bundle_orientation: (N,) chi per lens (radians in tangent plane)
            chirality:
            eye_ids: (N,) integer eye id per lens, 0-7
            eye_parameter: Optional scale factor on acceptance angles
                When the kernel provides nodal_distance_um (optical model available),
                p multiplies the Snyder acceptance angles uniformly
                    Δρ_actual = p × √((λ/D)² + (d_rhab/f)²)

                When no nodal distance is available (simplified eye model), p falls
                back to the classical meaning as acceptance-to-interommatidial ratio:
                    Δρ_actual = p × Δφ × (d_rhab / max_d_rhab)
            interommatidial_angles_rad: (N,) or (N,2) if known, otherwise estimated
        """
        # TODO: Add missing args in docstring

        # Position lenses
        lens_directions, lens_positions, lens_count = _lens_geometry(
            directions=directions,
            positions=positions,
            ommatidia_count=ommatidia_count,
            eye_radius=eye_radius,
            icosphere_method=icosphere_method
        )
        self._lens_directions = lens_directions
        self._lens_positions = lens_positions
        self._lens_count = N = lens_count

        local_right, local_up = tangent_frames(lens_directions)
        self._local_right = local_right
        self._local_up = local_up

        self._kernel = kernel if kernel is not None else RhabdomereKernel()
        R = self._kernel.count

        if eye_ids is not None:
            # User-provided values
            e_ids = self._prepare_param(eye_ids, 'eye_ids', N).astype(np.uint32)

        else:
            # Auto-split into left (0) and right (1) based on lateral position
            lateral = np.dot(self._lens_positions, np.asarray(WORLD_RIGHT, dtype=np.float32))
            e_ids = np.where(lateral >= 0, 1, 0).astype(np.uint32)  # TODO: This is dumb. What about more than 2 eyes?

        self._lens_eye_ids = e_ids

        self._eye_cache = {}
        for eid in np.unique(e_ids):
            mask = np.where(e_ids == eid)[0]
            self._eye_cache[int(eid)] = Eye(self, int(eid), mask)

        # Interommatidial angles
        if interommatidial_angles_rad is not None:
            # User-provided values
            lattice_tilts = np.zeros(N, dtype=np.float32)
            nb_counts = np.zeros(N, dtype=np.uint32)
            lens_spacing = np.zeros(N, dtype=np.float32)

            ioa_arr = self._prepare_param(interommatidial_angles_rad, 'interommatidial_angles', N, allow_2d=True)

            ioa_minor = ioa_arr[:, 0] if ioa_arr.ndim == 2 else ioa_arr
            ioa_major = ioa_arr[:, 1] if ioa_arr.ndim == 2 else ioa_arr

        elif N > 1 and np.ptp(self._lens_positions, axis=0).max() > 1e-6:
            # Positions have meaningful spatial extent: estimate from lattice
            ioa_minor, ioa_major, lattice_tilts, nb_counts, lens_spacing = _lattice_properties(
                directions=self._lens_directions,
                positions=self._lens_positions,
                eyes=self.eyes,
            )

        else:
            # Positions are degenerate (single lens, or all coincident): can't estimate IOA
            ioa_minor = np.zeros(N, dtype=np.float32)
            ioa_major = np.zeros(N, dtype=np.float32)
            lattice_tilts = np.zeros(N, dtype=np.float32)
            nb_counts = np.zeros(N, dtype=np.uint32)
            lens_spacing = np.zeros(N, dtype=np.float32)

        # Resolve lenses diameters
        # Priority: explicit override > lattice-derived (close-packed) > zero
        HEX_PACKING_FACTOR = 1.0  # 1.0 = fully touching lenses, 0.9 for small intercommatidial cuticle gaps, etc

        if lens_diameter_um is not None:
            # Explicit override
            diam = np.asarray(lens_diameter_um, dtype=np.float32)
            if diam.ndim == 0:
                lens_diameters_um = np.full(N, diam.item(), dtype=np.float32)
            elif diam.shape == (N,):
                lens_diameters_um = diam
            else:
                raise ValueError(f"lens_diameter_um must be scalar or shape ({N},); got {diam.shape}")

        elif np.any(lens_spacing > 0):
            # Lattice-derived, sanity check
            lens_diameters_um = HEX_PACKING_FACTOR * lens_spacing
            med = float(np.median(lens_diameters_um[lens_diameters_um > 0]))
            if not (0.5 <= med <= 5000.0):
                warnings.warn(
                    f"Lattice-derived lens diameter has median {med:.4g}. "
                    "This is outside the typical 1-1000 μm range: your lens positions "
                    "may not be in micrometres."
                )

        else:
            lens_diameters_um = np.zeros(N, dtype=np.float32)
            if interommatidial_angles_rad is not None:
                warnings.warn(
                    "lens_diameter_um was not provided, and interommatidial_angles_rad was given "
                    "instead of an extensive lens-position array, so no physical lattice spacing "
                    "could be derived. Lens diameter is independent of IOA in general. Pass "
                    "lens_diameter_um explicitly. Diffraction term in the Snyder formula will be invalid."
                )
            else:
                warnings.warn(
                    "lens_diameter_um was not provided and lens positions are degenerate "
                    "(single lens or all coincident). Diffraction term in the Snyder formula "
                    "will be invalid. Pass lens_diameter_um explicitly."
                )

        e_x_flow, e_y_flow, e_z_flow = _flow_aligned_frame(flow_direction)
        self._flow_frame = (e_x_flow, e_y_flow, e_z_flow)

        right_axis = np.asarray(WORLD_RIGHT, dtype=np.float32)
        eye_sign = -np.sign(self._lens_positions @ right_axis).astype(np.float32)
        eye_sign[eye_sign == 0] = 1.0           # midline -> left

        hemisphere_sign = np.sign(self._lens_positions @ e_z_flow).astype(np.float32)
        hemisphere_sign[hemisphere_sign == 0] = 1.0   # equator -> dorsal

        # Chirality: user-supplied wins. Default is eye_sign * hemisphere_sign.
        if chirality is not None:
            chirality_arr = self._prepare_param(chirality, 'chirality', N)
        else:
            chirality_arr = (eye_sign * hemisphere_sign).astype(np.float32)

        self._chirality_arr = chirality_arr

        # Alignment phasor (4-zone diagonal target)
        alignment = _alignment_phasor_field(
            lens_directions=self._lens_directions,
            e_x=e_x_flow,
            e_y=e_y_flow,
            e_z=e_z_flow,
            eye_sign=eye_sign,
            hemisphere_sign=hemisphere_sign,
            strength=1.0,
            falloff=0.7,
            diagonal_strength=diagonal_strength,
        )

        if alignment_smoothing_iterations > 0:
            zone_labels = (
                (eye_sign > 0).astype(np.int32) * 2
                + (hemisphere_sign > 0).astype(np.int32)
            )
            alignment = _smooth_phasor_field_zoned(
                field=alignment,
                partition=zone_labels,
                lens_positions=self._lens_positions,
                n_neighbours=8,
                iterations=alignment_smoothing_iterations,
            )
        self._alignment_phasor = alignment

        # Major axis: alignment rotated by +/- radians(flow_axis_deg) per eye
        base_flow_rot = float(np.radians(self._kernel.flow_axis_deg))
        major = _major_axis_field(
            alignment_phasor=alignment,
            lens_directions=self._lens_directions,
            base_rotation_rad=base_flow_rot,
            eye_sign=eye_sign,
            hemisphere_sign=hemisphere_sign,
        )
        self._major_axis = major

        # Bundle yaw chi from the major axis direction
        if bundle_orientation is not None:
            chi = self._prepare_param(bundle_orientation, 'bundle_orientation', N)
        else:
            major_angle = np.arctan2(
                np.sum(major * self._local_up, axis=1),
                np.sum(major * self._local_right, axis=1),
            )
            effective_main = np.where(
                chirality_arr > 0,
                float(self._kernel.main_axis_rad),
                np.pi - float(self._kernel.main_axis_rad),
            ).astype(np.float32)
            chi = (major_angle - effective_main).astype(np.float32)
            chi = (chi + np.pi) % (2.0 * np.pi) - np.pi
        self._bundle_orientation = chi

        # Saccade phasor: major axis rotated by base * chirality (4-zone)
        base_sacc_rot = -float(np.radians(self._kernel.saccade_offset_deg))
        sacc_raw = _saccade_phasor_field(
            major_axis=major,
            lens_directions=self._lens_directions,
            base_rotation_rad=base_sacc_rot,
            chirality=chirality_arr,
        )
        sacc = _smooth_phasor_field(
            field=sacc_raw,
            eyes=self.eyes,
            lens_positions=self._lens_positions,
            n_neighbours=8,
            iterations=saccade_smoothing_iterations,
        )

        # Polarise the smoothed (nematic) saccade field so it points consistently "up" in the head frame
        dot_z = sacc @ e_z_flow
        sacc[dot_z < 0] *= -1.0
        self._saccade_cache = sacc.astype(np.float32)

        # Receptors geometry
        receptor_dirs, receptor_pos = _receptors_geometry(
            kernel=self._kernel,
            lens_positions=self._lens_positions,
            lens_directions=self._lens_directions,
            local_right=self._local_right,
            local_up=self._local_up,
            bundle_orientation=self._bundle_orientation,
            chirality=chirality_arr,
        )

        # Acceptance angles
        acc_axes = _acceptance_angles(
            lens_count=N,
            kernel=self._kernel,
            ioa_minor=ioa_minor,
            ioa_major=ioa_major,
            eye_parameter=eye_parameter,
            explicit_angles_rad=acceptance_angles_rad,
            lens_diameters_um=lens_diameters_um,
        )

        # Fill main data structures

        # Lens static
        self.lens_static_data = np.zeros(N, dtype=LENS_STATIC_DTYPE)
        self.lens_static_data['right'] = local_right
        self.lens_static_data['up'] = local_up
        self.lens_static_data['sacc_x'] = np.sum(sacc * local_right, axis=1)
        self.lens_static_data['sacc_y'] = np.sum(sacc * local_up, axis=1)
        self.lens_static_data['forward'] = lens_directions
        self.lens_static_data['ioa_tilt'] = lattice_tilts
        self.lens_static_data['ioa_axes'][:, 0] = ioa_minor
        self.lens_static_data['ioa_axes'][:, 1] = ioa_major
        self.lens_static_data['nodal_distance_um'] = self._kernel.nodal_distance_um or 0.0  # TODO: This broadcasts to every lens, there should be a way to do it per-lens
        self.lens_static_data['lens_diameter_um']  = lens_diameters_um

        # Lens dynamic
        self.lens_dynamic_data = np.zeros(N, dtype=LENS_DYNAMIC_DTYPE)

        # Receptor static
        self.rcpt_static_data = np.zeros(N * R, dtype=RCPT_STATIC_DTYPE)
        self.rcpt_static_data['position'] = receptor_pos
        self.rcpt_static_data['metadata'] = _pack_metadata(N, R, e_ids, receptor_types, nb_counts, chirality_arr)
        self.rcpt_static_data['rest_acc'] = acc_axes

        rot_dx, rot_dy = self._kernel.rotated_offsets(chi, chirality_arr)
        self.rcpt_static_data['rot_offset'][:, 0] = rot_dx.ravel()
        self.rcpt_static_data['rot_offset'][:, 1] = rot_dy.ravel()
        self.rcpt_static_data['acc_tilt'] = np.repeat(chi, R)
        self.rcpt_static_data['tau_membrane'] = self._kernel.tau_membrane
        self.rcpt_static_data['sensitivity'] = np.tile(self._kernel.sensitivity, (N, 1))

        # Default cartridge wiring: straight down
        self.rcpt_static_data['cartridge_src'] = np.repeat(np.arange(N), R) * R + np.tile(np.arange(R), N)

        self.rcpt_static_data['rhab_diameter_um'] = np.tile(self._kernel.diameters_um, N)
        self.rcpt_static_data['wavelength_um'] = np.tile(self._kernel.wavelengths_nm * 1e-3, N)

        # Receptor dynamic
        self.rcpt_dynamic_data = np.zeros(N * R, dtype=RCPT_DYNAMIC_DTYPE)
        self.rcpt_dynamic_data['direction'] = receptor_dirs
        self.rcpt_dynamic_data['acc_axes'] = acc_axes
        self.rcpt_dynamic_data['adaptation_state'] = 1.0

        self.dirty_mask = np.zeros(N * R, dtype=bool)
        self.lens_dirty = False

        # Cached namespace views (invalidated by geometry changes)
        self._lenses_view = None
        self._receptors_view = None

        if R > 1:
            self.wire_cartridges()

    @classmethod
    def from_file(cls, file_path: Union[str, Path], **kwargs):
        """
        Load from .npz archive.

        The file may contain: directions, positions, eye_id, chirality,
        bundle_orientation, acceptance_angles_rad, interommatidial_angles_rad,
        receptor_types.

        - If the file does not contain 'bundle_orientation', tissue mechanics
        runs automatically. Pass optic_flow_direction, weight_flow, etc.
        via kwargs to control it.

        - If the file contains 'bundle_orientation', tissue mechanics is
        skipped. Call recompute_bundle_orientation() afterwards if needed.
        """

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Cannot find: {path}")

        data = np.load(path)
        if 'directions' not in data:
            raise ValueError(f"'{path}' missing required 'directions' array.")

        args = {
            'directions': data['directions'],
            'positions': data.get('positions'),
            'acceptance_angles_rad': data.get('acceptance_angles_rad'),
            'interommatidial_angles_rad': data.get('interommatidial_angles_rad'),
            'receptor_types': data.get('receptor_types'),
            'eye_ids': data.get('eye_id'),
            'bundle_orientation': data.get('bundle_orientation'),
            'chirality': data.get('chirality'),
        }
        args.update(kwargs)
        instance = cls(**args)

        return instance

    # Overrides and internal helpers

    def __len__(self):
        return len(self.rcpt_static_data)

    def __repr__(self):
        return f"<ReceptorArray(lenses={self.lens_count}, R={self.receptors_per_lens}, total={len(self.rcpt_static_data)})>"

    @property
    def lenses(self) -> 'LensView':
        """Lens-level data (animal-level)."""

        if self._lenses_view is None:
            self._lenses_view = LensView(
                ra=self,
                lens_indices=np.arange(self._lens_count),
                single_eye=False,
            )

        return self._lenses_view

    @property
    def receptors(self) -> 'ReceptorView':
        """Receptor-level data (animal-level)."""

        if self._receptors_view is None:
            self._receptors_view = ReceptorView(
                ra=self,
                receptor_indices=np.arange(len(self))
            )

        return self._receptors_view

    def _prepare_param(self, param, name="param", expected_len=None, allow_2d=False):
        arr = np.asarray(param, dtype=np.float32)

        if expected_len is None:
            expected_len = self.lens_count

        if arr.ndim == 0:
            return np.full(expected_len, arr.item())
        if arr.ndim == 1 and len(arr) == expected_len:
            return arr
        if allow_2d and arr.ndim == 2 and arr.shape == (expected_len, 2):
            return arr

        raise ValueError(f"'{name}' shape invalid. Need scalar or length-{expected_len}.")

    def _unpack(self, param, name="param"):
        if isinstance(param, Sequence):
            return self._prepare_param(param[0], f"{name}_min"), self._prepare_param(param[1], f"{name}_maj")
        p = self._prepare_param(param, name)
        return p, p

    # Eye / Ommatidium / Cartridge access

    def eye(self, eye_id: int) -> 'Eye':
        """Eye view for eye_id (0-7)."""

        if eye_id not in self._eye_cache:
            raise KeyError(f'No eye with id {eye_id}. Available: {sorted(self._eye_cache)}')

        return self._eye_cache[eye_id]

    @property
    def eyes(self) -> List['Eye']:
        """List of Eye views for all eye_ids present."""

        return [self._eye_cache[eid] for eid in sorted(self._eye_cache)]

    @property
    def eye_ids(self) -> np.ndarray:

        return np.array(sorted(self._eye_cache), dtype=np.uint32)

    def _eye_cache_by_lens(self, lens_index: int) -> 'Eye':
        """Lookup which Eye owns a given lens (animal-level)."""

        eid = int(self._lens_eye_ids[lens_index])
        return self._eye_cache[eid]

    def ommatidium(self, lens_index: int) -> 'Ommatidium':
        """Animal-level lens index -> Ommatidium group view."""

        return Ommatidium(self, lens_index)

    def cartridge(self, lens_index: int) -> 'Cartridge':
        """Animal-level lens index -> Cartridge."""

        return Cartridge(self, lens_index)

    # Properties

    @property
    def total_receptors(self) -> int:
        return len(self)

    @property
    def lens_count(self) -> int:
        return self._lens_count

    @property
    def receptors_per_lens(self) -> int:
        return self._kernel.count

    @property
    def max_gap(self) -> float:
        """Largest angular gap between any lens and its nearest neighbour."""

        return max(eye.max_gap() for eye in self.eyes)

    @property
    def kernel(self) -> Optional['RhabdomereKernel']:
        return self._kernel

    @property
    def bundle_orientation(self) -> np.ndarray:
        """Bundle orientation (chi), per lens. Alias for lenses.bundle_orientations."""

        return self._bundle_orientation

    def saccade_field(self) -> np.ndarray:
        """
        Compute (and cache) the saccades vector field.
        """

        return self._saccade_cache

    @property
    def chirality(self) -> np.ndarray:
        """Per-lens chirality. Alias for lenses.chirality."""
        return self.lenses.chirality

    @property
    def interommatidial_angles_rad(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per-lens IOA. Alias for lenses.interommatidial_angles."""
        return self.lenses.interommatidial_angles

    @property
    def cartridge_indices(self) -> np.ndarray:
        """
        Returns (N, R) array of global receptor indices (for neural superposition).
        """
        return self.rcpt_static_data['cartridge_src'].reshape(self.lens_count, self.receptors_per_lens)

    # Animal-level spatial queries

    def query_directions(self, directions: ArrayLike, k: int = 1, return_distances: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Find lenses whose optical axis best matches query directions.
        Returns animal-level lens indices (best match across all eyes).
        """

        is_single = np.asarray(directions).ndim == 1
        q = np.atleast_2d(np.asarray(directions, dtype=np.float32))
        q = normalise_vectors(q)
        Q = len(q)

        best_dist = np.full(Q, np.inf)
        best_idx = np.zeros(Q, dtype=np.intp)

        for eye in self.eyes:
            dist, local_idx = eye._directions_tree.query(q, k=k)
            dist, local_idx = dist.ravel(), local_idx.ravel()
            animal_idx = eye.lens_indices[local_idx]
            better = dist < best_dist
            best_dist[better] = dist[better]
            best_idx[better] = animal_idx[better]

        if is_single:
            return int(best_idx[0])

        if return_distances:
            return best_dist, best_idx
        return best_idx

    def query_position(self, positions: ArrayLike, k: int = 1, return_distances: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Find lenses closest to query positions.
        Returns animal-level lens indices (best match across all eyes).
        """

        is_single = np.asarray(positions).ndim == 1
        q = np.atleast_2d(np.asarray(positions, dtype=np.float32))
        Q = len(q)

        best_dist = np.full(Q, np.inf)
        best_idx = np.zeros(Q, dtype=np.intp)

        for eye in self.eyes:
            dist, local_idx = eye._positions_tree.query(q, k=k)
            dist, local_idx = dist.ravel(), local_idx.ravel()
            animal_idx = eye.lens_indices[local_idx]
            better = dist < best_dist
            best_dist[better] = dist[better]
            best_idx[better] = animal_idx[better]

        if is_single:
            return int(best_idx[0])

        if return_distances:
            return best_dist, best_idx
        return best_idx

    def query_lookat(self, targets: ArrayLike, k: int = 1) -> np.ndarray:
        """
        Find lenses looking at world-space target points.
        Returns animal-level lens indices (best match across all eyes).
        """

        if k < 1:
            raise ValueError("k must be >= 1")

        is_single = np.asarray(targets).ndim == 1
        q = np.atleast_2d(np.asarray(targets, dtype=np.float32))

        pos = self._lens_positions
        dirs = self._lens_directions

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
        """
        Find all lenses within angle of a centre direction.
        Returns animal-level lens indices (union across all eyes).
        """

        c = np.asarray(center_direction, dtype=np.float32)
        c /= np.linalg.norm(c)
        a = np.deg2rad(angle) if degrees else angle
        r = 2.0 * np.sin(a / 2.0)

        result = []
        for eye in self.eyes:
            local_hits = eye._directions_tree.query_ball_point(c, r=r)
            hits = np.atleast_1d(np.asarray(local_hits, dtype=np.intp))
            if len(hits) > 0:
                result.extend(eye.lens_indices[hits].tolist())

        return np.array(result, dtype=np.intp)

    def query_ball(self, center_position: ArrayLike, radius: float) -> np.ndarray:
        """
        Find all lenses within radius of a centre position.
        Returns animal-level lens indices (union across all eyes).
        """

        c = np.asarray(center_position, dtype=np.float32)

        result = []
        for eye in self.eyes:
            local_hits = eye._positions_tree.query_ball_point(c, r=radius)
            hits = np.atleast_1d(np.asarray(local_hits, dtype=np.intp))
            if len(hits) > 0:
                result.extend(eye.lens_indices[hits].tolist())

        return np.array(result, dtype=np.intp)

    # Neural superposition (re)wiring

    def wire_cartridges(self,
            snap_radius: float = 0.2,
            assign_radius: float = 1.0,
            angular_dev: float = 25.0,
            scale_dev: float = 0.3,
            pre_cull: bool = False,
            first_ring_only: bool = False,
            neighbour_dist_factor: float = 1.3,
            min_snap_matches: int = 2,
        ):

        N = self.lens_count
        R = self.receptors_per_lens

        centre_rhab = self._kernel.center_index
        periph_rhab = np.array([i for i in range(R) if i != centre_rhab])
        P = len(periph_rhab)

        # Default: everything projects to its own lens
        cartridge = np.tile(np.arange(N)[:, None], (1, R))

        if P == 0:
            self.rcpt_static_data['cartridge_src'] = np.arange(len(self), dtype=np.uint32)
            return

        # Centre and normalise rhabdomere kernel to match local lattice spacing
        rel_rhab_offsets = self._kernel.offsets_um - self._kernel.offsets_um[centre_rhab]
        norm_rhab_offsets = rel_rhab_offsets / np.mean(np.linalg.norm(rel_rhab_offsets[periph_rhab], axis=1))

        min_required = max(1, min(min_snap_matches, P))

        k_search = 30  # up to ~3 rings away

        for eye in self.eyes:
            neighb = eye.neighbours(
                points=self._lens_positions,
                k=k_search,
                immediate_only=True,
                neighbour_dist_factor=neighbour_dist_factor,
            )
            if neighb is None:
                continue

            gidx = neighb.mask

            for i_loc in range(len(gidx)):
                i_glob = gidx[i_loc]

                central_lens_pos = self._lens_positions[i_glob]
                central_lens_chirality = self.chirality[i_glob]

                # Map eye-local neighbour indices back to global
                neighb_indices = gidx[neighb.indices[i_loc]]
                neighb_vectors = self._lens_positions[neighb_indices] - central_lens_pos
                neighb_distances = neighb.distances[i_loc]

                if len(neighb_indices) < 1:
                    continue

                # To tangent plane
                neighb_u = neighb_vectors @ self._local_right[i_glob]
                neighb_v = neighb_vectors @ self._local_up[i_glob]

                is_first_ring = neighb.is_immediate[i_loc]
                if not np.any(is_first_ring):
                    continue

                local_spacing = np.median(neighb_distances[is_first_ring])
                if local_spacing < 1e-6:
                    continue

                neighb_uv = np.column_stack([neighb_u, neighb_v]) / local_spacing

                # Rotated and scaled template in 'lattice units'
                effective_chi = self._bundle_orientation[i_glob]
                cos_chi, sin_chi = np.cos(effective_chi), np.sin(effective_chi)

                periph_offsets = norm_rhab_offsets[periph_rhab]
                tx = periph_offsets[:, 0] * central_lens_chirality
                ty = periph_offsets[:, 1]

                template_uv = np.column_stack([
                    tx * cos_chi - ty * sin_chi,
                    tx * sin_chi + ty * cos_chi
                ])

                z_template = template_uv[:, 0] + 1j * template_uv[:, 1]
                z_nb = neighb_uv[:, 0] + 1j * neighb_uv[:, 1]
                # Snap targets = either only first ring, or all neighbours
                z_snap = z_nb[is_first_ring] if first_ring_only else z_nb

                # Determine neighbour lenses to use as anchors
                if pre_cull:
                    diff_initial = template_uv[:, None, :] - neighb_uv[is_first_ring][None, :, :]
                    anchors = np.where(np.min(np.linalg.norm(diff_initial, axis=2), axis=1) < assign_radius)[0]
                else:
                    anchors = np.arange(P)

                valid_anchors = anchors[np.abs(z_template[anchors]) >= 1e-8]
                candidates_w = [1.0 + 0j]
                if len(valid_anchors) > 0:
                    w_all = z_snap[None, :] / z_template[valid_anchors, None]
                    angle_ok = np.abs(np.angle(w_all)) <= np.radians(angular_dev)
                    scale_ok = np.abs(np.abs(w_all) - 1.0) <= scale_dev
                    valid = angle_ok & scale_ok
                    candidates_w.extend(w_all[valid].tolist())

                # Evaluate candidate snaps: angular proximity 1st, total snap quality 2nd
                best_score = (np.inf, np.inf)
                best_w = None
                for w in candidates_w:
                    z_placed = w * z_template
                    placed_uv = np.column_stack([z_placed.real, z_placed.imag])

                    cost_matrix = np.linalg.norm(
                        placed_uv[:, None, :] - neighb_uv[None, :, :], axis=2)
                    row_idx, col_idx = linear_sum_assignment(cost_matrix)
                    matched_dists = cost_matrix[row_idx, col_idx]

                    if int(np.sum(matched_dists < snap_radius)) < min_required:
                        continue

                    score_tuple = (np.abs(np.angle(w)), float(np.sum(matched_dists)))
                    if score_tuple < best_score:
                        best_score = score_tuple
                        best_w = w

                if best_w is None:
                    continue

                # Final assignment: wire to the closest neighbour in selected snap
                z_best = best_w * z_template
                corrected_uv = np.column_stack([z_best.real, z_best.imag])

                cost_matrix = np.linalg.norm(corrected_uv[:, None, :] - neighb_uv[None, :, :], axis=2)

                rhab_indices, om_indices = linear_sum_assignment(cost_matrix ** 2)

                # same chirality only
                same_chirality = self.chirality[neighb_indices] == central_lens_chirality

                valid_mask = (cost_matrix[rhab_indices, om_indices] < assign_radius) & same_chirality[om_indices]
                cartridge[i_glob, periph_rhab[rhab_indices[valid_mask]]] = neighb_indices[om_indices[valid_mask]]

        flat_rcpt_indices = (cartridge * R + np.arange(R)).flatten()
        self.rcpt_static_data['cartridge_src'] = flat_rcpt_indices.astype(np.uint32)

    # Spatial structures

    def _invalidate_spatial(self):
        """Invalidate caches that depend on lens positions or eye spatial structure."""

        for eye in self._eye_cache.values():
            eye._invalidate()

        self._lenses_view = None
        self._receptors_view = None
        self._saccade_cache = None

    # Whole-array transforms (initial unit scaling, agent setup, etc)

    def scale(self, factor: float):
        """Scale all receptor and lens positions by a factor."""

        self.rcpt_static_data['position'] *= factor
        self._lens_positions *= factor
        self._invalidate_spatial()

        return self

    def translate(self, vector: ArrayLike):
        """Translate all receptor and lens positions by a vector."""

        vector = np.asarray(vector, dtype=np.float32)

        self.rcpt_static_data['position'] += vector
        self._lens_positions += vector
        self._invalidate_spatial()

        return self

    @property
    def cartridge_map(self) -> np.ndarray:
        return self.rcpt_static_data['cartridge_src'].reshape(self.lens_count, self.receptors_per_lens) // self.receptors_per_lens



## Examples:
#
# # Minimal, uniform sphere model
#
# # Generates ~2000 ommatidia on a sphere
# # Defaults to R=1 (point kernel), p=1.0 (acceptance = IOA)
# ra = ReceptorArray(ommatidia_count=2000)
#
# # or with a custom eye parameter or radius
# ra = ReceptorArray(
#     ommatidia_count=2000,
#     eye_radius=0.05,
#     eye_parameter=1.2 # slightly overlapping fields of view
# )
#
# # _____________________________________________________________________________
#
# # Intermediate, with ommatidia positions
#
# # dirs and pos are (N, 3) arrays
# ra = ReceptorArray(
#     directions=dirs,
#     positions=pos,
#     # Overriding default kernel to set specific lens properties for the R=1 model
#     kernel=RhabdomereKernel(lens_diameter_um=16.0, diameters_um=2.0)
# )
#
# # _____________________________________________________________________________
#
# DROSOPHILA_KERNEL = RhabdomereKernel(...)
#
# # Full model with rhabdomere data
# ra = ReceptorArray(
#     directions=dirs,
#     positions=pos,
#     kernel=DROSOPHILA_KERNEL,       # R=7 defined kernel
#     bundle_orientation=chi_array,   # (N,) array of bundle rotations
#     chirality=chirality_array,      # (N,) array of +1 or -1 for the equator flip
#     eye_ids=eye_id_array            # (N,) mapping left/right eyes
# )