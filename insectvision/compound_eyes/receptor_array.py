from pathlib import Path
from typing import Optional, Union, Tuple, Sequence
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment

from insectvision.engine.world_utils import WORLD_UP, WORLD_RIGHT, WORLD_FORWARD
from insectvision.utils.math import (
    normalise_vectors, tangent_frames, fibonacci_sphere, icosphere,
    project_to_tangent,
)

from .datatypes import LENS_STATIC_DTYPE, LENS_DYNAMIC_DTYPE, RCPT_STATIC_DTYPE, RCPT_DYNAMIC_DTYPE
from .kernel import RhabdomereKernel
from .proxies import Eye, Ommatidium, Cartridge, LensView, ReceptorView


## Build helpers


def _consensus_yaws(
        yaws,
        chirality,
        eyes,
        lens_positions,
        local_right,
        local_up,
        k=15,
        max_passes=5,
) -> np.ndarray:
    """
    Resolve mod-pi ambiguity in a yaw field using 3D vector majority vote.
    """
    out = yaws.copy()

    for eye in eyes:
        neighb = eye.neighbours(
            points=lens_positions, k=k, chirality=chirality, include_self=True
        )
        if neighb is None: continue

        for _pass in range(max_passes):
            g_yaws = out[neighb.mask]

            V = np.cos(g_yaws)[:, None] * local_right[neighb.mask] + np.sin(g_yaws)[:, None] * local_up[neighb.mask]

            V_nb = V[neighb.indices]
            V_mean = np.sum(V_nb * neighb.same_chirality[:, :, None], axis=1)

            agreement = np.sum(V * V_mean, axis=1)

            flip = agreement < 0
            if not np.any(flip):
                break
            out[neighb.mask[flip]] += np.pi

    return (out + np.pi) % (2 * np.pi) - np.pi


def _floodfill_yaws(yaws, chirality, eyes, lens_positions, local_right, local_up, k=15):
    """Flood fill propagation from most locally consistent 3D seed."""

    out = yaws.copy()

    for eye in eyes:
        neighb = eye.neighbours(
            points=lens_positions, k=k, chirality=chirality, include_self=True
        )
        if neighb is None: continue

        g_chirality = chirality[neighb.mask]
        g_yaws = out[neighb.mask]

        V = np.cos(g_yaws)[:, None] * local_right[neighb.mask] + np.sin(g_yaws)[:, None] * local_up[neighb.mask]

        V_nb = V[neighb.indices]
        agreement = np.sum(V[:, None, :] * V_nb, axis=2) * neighb.same_chirality
        confidence = np.sum(agreement, axis=1)

        n_g = len(neighb.mask)
        seed = np.argmax(confidence)
        visited = np.zeros(n_g, dtype=bool)
        visited[seed] = True
        queue = [seed]

        while queue:
            current = queue.pop(0)
            for j in neighb.indices[current]:
                if visited[j]: continue
                visited[j] = True

                dot = np.sum(V[current] * V[j])
                if dot < 0 and g_chirality[current] == g_chirality[j]:
                    out[neighb.mask[j]] += np.pi
                    V[j] *= -1.0  # keep the vector updated for the next iters

                queue.append(j)

    return (out + np.pi) % (2 * np.pi) - np.pi


def _orient_rhabdomeres(
        kernel: RhabdomereKernel,
        lens_positions: np.ndarray,
        lens_directions: np.ndarray,
        local_right: np.ndarray,
        local_up: np.ndarray,
        chirality: np.ndarray,
        flow_direction: np.ndarray,
        weight_tissue: float,
        weight_flow: float,
        eyes: list,
        k: int = 15,
) -> np.ndarray:
    """
    Compute per-ommatidium rhabdomere bundle yaw via a orientation compromise.
    """

    # Tangent plane projection of external flow
    flow_vec = np.asarray(flow_direction, dtype=np.float32)
    v_proj = project_to_tangent(flow_vec, lens_directions)
    flow_mags = np.linalg.norm(v_proj, axis=1)

    angle_flow_external = np.arctan2(
        np.sum(v_proj * local_up, axis=1),
        np.sum(v_proj * local_right, axis=1)
    )

    # Intrinsic target: kernel flow axis (after chirality x-flip only: the D/V y-flip is handled
    # separately via a dorsal pi-offset applied at placement time, invisible in the doubled-angle line-field space used here.
    flow_angle_kernel = kernel.flow_axis_rad
    intrinsic_target = np.arctan2(
        np.sin(flow_angle_kernel),
        np.cos(flow_angle_kernel) * chirality
    )

    yaw_flow = angle_flow_external - intrinsic_target

    # Confidence weighting
    max_f = np.max(flow_mags)
    flow_conf = flow_mags / max_f if max_f > 1e-12 else np.zeros_like(flow_mags)
    uncertainty = 1.0 - flow_conf

    # Represent in 2-theta space (line field)
    # (magnitude is the strength of opinion)
    Z_flow = (weight_flow * flow_conf).astype(np.complex64) * np.exp(2j * yaw_flow)

    # Diffusion passes
    Z_final = Z_flow.copy()

    n_passes = 5 if weight_tissue > 0 else 0
    for _pass in range(n_passes):
        Z_next = Z_final.copy()

        for eye in eyes:
            neighb = eye.neighbours(
                points=lens_positions,
                k=k,
                chirality=chirality,
                include_self=True
            )
            if neighb is None:
                continue

            g_indices = neighb.mask
            Z_nb = Z_final[g_indices][neighb.indices]

            # Distance weights (Gaussian)
            sigma = neighb.distances[:, -1] / 2.0
            sigma = np.where(sigma < 1e-6, 1.0, sigma)
            w_dist = np.exp(-neighb.distances ** 2 / (2.0 * sigma[:, None] ** 2))

            w_total = w_dist * neighb.same_chirality

            sum_w = np.sum(w_total, axis=1)
            Z_smooth = np.sum(Z_nb * w_total, axis=1) / np.where(sum_w > 0, sum_w, 1.0)

            # Compromise: flow + (tissue weight * neighbour opinion)
            w_t = weight_tissue * (1.0 + uncertainty[g_indices])
            Z_next[g_indices] = Z_flow[g_indices] + w_t.astype(np.complex64) * Z_smooth

        Z_final = Z_next

    final_yaws = 0.5 * np.angle(Z_final)

    # Disambiguation (resolution of the mod-pi line field into a vector field)
    final_yaws = _consensus_yaws(
        yaws=final_yaws,
        chirality=chirality,
        eyes=eyes,
        lens_positions=lens_positions,
        local_right=local_right,
        local_up=local_up,
        k=k,
        max_passes=15
    )

    final_yaws = _floodfill_yaws(
        yaws=final_yaws,
        chirality=chirality,
        eyes=eyes,
        lens_positions=lens_positions,
        local_right=local_right,
        local_up=local_up,
        k=k
    )

    return final_yaws.astype(np.float32)


def _saccades_field(
        yaws: np.ndarray,
        kernel: RhabdomereKernel,
        chirality: np.ndarray,
        dorsal_sign: np.ndarray,
        local_right: np.ndarray,
        local_up: np.ndarray,
        positions: np.ndarray,
        eyes: list,
        k: int = 8,
        max_cleanup_passes: int = 5
) -> np.ndarray:
    """
    Compute a consistent saccade vector field and per-ommatidium actuation signs.

    Args:
        yaws:           (N,) bundle rotation angles from tissue mechanics
        kernel:         Rhabdomere kernel (for saccade_axis_deg, axis_indices)
        chirality:      (N,) +1/-1 per ommatidium (combined L/R × D/V)
        dorsal_sign:    (N,) +1 ventral, -1 dorsal
        local_right:    (N, 3) local tangent frame right vectors
        local_up:       (N, 3) local tangent frame up vectors
        positions:      (N, 3) lens positions (for neighbour lookup)
        eyes:           List of Eye objects for per-eye neighbour iteration
        k:              Neighbour count for majority vote cleanup
        max_cleanup_passes: Maximum iterations for majority vote convergence
    """

    # Dorsal ommatidia need a pi offset to convert the x-flip (combined
    # chirality) into the correct diag(chi_lr, dorsal_sign) transform
    dorsal_offset = np.where(dorsal_sign < 0, np.pi, 0.0)
    effective_yaw = yaws + dorsal_offset

    # Saccade vectors
    sx = np.cos(kernel.saccade_axis_rad) * chirality
    sy = np.sin(kernel.saccade_axis_rad)
    cos_y = np.cos(effective_yaw)
    sin_y = np.sin(effective_yaw)

    bx = sx * cos_y - sy * sin_y
    by = sx * sin_y + sy * cos_y
    vectors = bx[:, None] * local_right + by[:, None] * local_up
    vectors = normalise_vectors(vectors)

    vectors[dorsal_sign < 0] *= -1.0

    for eye in eyes:
        neighb = eye.neighbours(
            points=positions,
            k=k,
            chirality=chirality
        )
        if neighb is None:
            continue

        for _pass in range(max_cleanup_passes):
            g_vectors = vectors[neighb.mask]

            neighbour_vectors = g_vectors[neighb.indices]
            dots = np.sum(g_vectors[:, None, :] * neighbour_vectors, axis=2)

            agreement = np.sum(dots * neighb.same_chirality, axis=1)
            flip = agreement < 0
            if not np.any(flip):
                break

            vectors[neighb.mask[flip]] *= -1.0

    return vectors


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
        kernel: RhabdomereKernel,
        bundle_orientation: np.ndarray,
        chirality: np.ndarray,
        dorsal_sign: np.ndarray
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

    # Dorsal pi-offset converts the chirality x-flip into the full
    # diag(chi_lr, dorsal_sign) mirror before rotation.
    dorsal_offset = np.where(dorsal_sign < 0, np.pi, 0.0).astype(np.float32)
    effective_yaw = bundle_orientation + dorsal_offset

    rot_dx, rot_dy = kernel.rotated_offsets(effective_yaw, chirality)

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
        kernel: RhabdomereKernel,
        ioa_minor: np.ndarray,
        ioa_major: np.ndarray,
        wavelength_nm: float,
        eye_parameter: Optional[Union[float, Tuple]],
        explicit_angles_rad: Optional[ArrayLike]
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

    # If nodal distance is unset (simplified model) or eye_parameter explicitly given, use p * IOA
    if eye_parameter is not None or kernel.nodal_distance_um is None:
        p = eye_parameter if eye_parameter is not None else 1.0
        p_min, p_maj = (p, p) if isinstance(p, (int, float, np.number)) else p

        # Scale acceptance by relative receptor diameters
        max_d = np.max(kernel.diameters_um)
        rel_d = kernel.diameters_um / max_d if max_d > 0 else np.ones(R)

        acc_min = np.repeat(p_min * ioa_minor, R) * np.tile(rel_d, N)
        acc_maj = np.repeat(p_maj * ioa_major, R) * np.tile(rel_d, N)

        return np.column_stack([acc_min, acc_maj])

    # Acceptance angles computed from the Snyder optical model (Snyder 1979):
    # Δρ = sqrt( (λ/D)² + (d_rhab/f)² )
    wavelength_um = wavelength_nm * 1e-3
    diffraction = wavelength_um / kernel.lens_diameter_um
    geometric = kernel.diameters_um / kernel.nodal_distance_um
    full_acceptance = np.sqrt(diffraction ** 2 + geometric ** 2)

    acc_1d = np.tile(full_acceptance, N)
    # TODO: Elliptical acceptance angles here too?
    return np.column_stack([acc_1d, acc_1d])


def _lattice_properties(
        directions: np.ndarray,
        positions: np.ndarray,
        eyes: list,
        k: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate local lattice properties by separating structural topology (positions)
    from optical axes (directions).
    """

    N = len(directions)
    ioa_minor = np.zeros(N, dtype=np.float32)
    ioa_major = np.zeros(N, dtype=np.float32)
    tilts = np.zeros(N, dtype=np.float32)
    neighb_counts = np.zeros(N, dtype=np.uint32)

    for eye in eyes:
        neighb = eye.neighbours(
            points=positions,
            k=k,
            immediate_only=True
        )
        if neighb is None:
            continue

        mask = neighb.mask
        e_dirs = directions[mask]
        e_pos = positions[mask]

        nb_immediate_neighb = np.maximum(np.sum(neighb.is_immediate, axis=1), 1, dtype=np.float64)
        neighb_counts[mask] = nb_immediate_neighb

        # Optical IOA: angular separation to each neighbour
        dots = np.sum(e_dirs[:, None, :] * e_dirs[neighb.indices], axis=2)
        angular_sep = np.arccos(np.clip(dots, -1.0, 1.0))

        # Tangent-plane projections for lattice tilt
        local_x, local_y = tangent_frames(e_dirs, world_up=WORLD_UP, world_right=WORLD_RIGHT)
        delta_pos = e_pos[neighb.indices] - e_pos[:, None, :]
        proj_x = np.sum(delta_pos * local_x[:, None, :], axis=2)
        proj_y = np.sum(delta_pos * local_y[:, None, :], axis=2)

        # Hexatic order parameter (Ψ6)
        angles = np.arctan2(proj_y, proj_x)
        phasors = np.exp(1j * 6 * angles)
        phasors[~neighb.is_immediate] = 0.0

        z_avg = np.sum(phasors, axis=1) / nb_immediate_neighb
        e_psi6 = np.abs(z_avg)
        e_tilts = np.angle(z_avg) / 6.0

        # IOA from sorted angular separations (mask non-immediate to inf)
        sep_masked = np.where(neighb.is_immediate, angular_sep, np.inf)
        sep_sorted = np.sort(sep_masked, axis=1)

        # Minor = mean of 2 smallest, major = mean of 2 largest immediate neighb
        e_ioa_minor = np.mean(sep_sorted[:, :2], axis=1).astype(np.float32)

        sep_rev = np.where(neighb.is_immediate, -angular_sep, np.inf)
        top2_rev = np.sort(sep_rev, axis=1)[:, :2]
        e_ioa_major = np.mean(-top2_rev, axis=1).astype(np.float32)

        # Fallback to simple mean if < 2 immediate neighbours
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

        print(f"Eye {eye.eye_id} lattice hexatic quality (Ψ6): {np.mean(e_psi6):.3f}")

    return ioa_minor, ioa_major, tilts, neighb_counts


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
    Every element is one rhabdomere.
    The GPU traces rays for len(data) rhabdomeres.
    """

    def __init__(self,
                 directions: Optional[ArrayLike] = None,
                 positions: Optional[ArrayLike] = None,
                 ommatidia_count: Optional[int] = None,
                 kernel: Optional[RhabdomereKernel] = None,
                 bundle_orientation: Optional[ArrayLike] = None,
                 chirality: Optional[ArrayLike] = None,
                 eye_ids: Optional[Union[int, ArrayLike]] = None,
                 receptor_types: Optional[Union[int, ArrayLike]] = None,
                 eye_parameter: Optional[Union[float, Tuple]] = None,
                 interommatidial_angles_rad: Optional[Union[ArrayLike, Tuple, float]] = None,
                 acceptance_angles_rad: Optional[Union[ArrayLike, Tuple, float]] = None,
                 wavelength_nm: float = 500.0,
                 eye_radius: float = 0.01,
                 icosphere_method: bool = True,
                 flow_direction: Optional[ArrayLike] = None,
                 weight_flow: float = 1.0,
                 weight_tissue: float = 0.6,
                 ):
        """
        Construct a full receptor array.

        Each of the *N* lenses contains *R* receptors whose world-space directions are
        determined by the kernel offsets and rotated by the per-lens bundle orientation (chi).

        Acceptance angles are computed from the full optical model (Snyder 1979):

            Δρ = sqrt( (λ/D)² + (d_rhab/f)² )

        where λ = wavelength_nm, D = kernel.lens_diameter_um, d_rhab = kernel.diameters_um, f = kernel.nodal_distance_um

        This can be overridden with eye_parameter: p = delta_rho / delta_phi

        Args:
            directions: (N, 3) lens optical axes
            positions: (N, 3) lens positions in head space
            ommatidia_count: Number of ommatidia to build a uniform eye for (when positions are not specified)
            kernel: Species-level rhabdomere geometry
            bundle_orientation: (N,) chi per lens (radians in tangent plane)
            chirality:
            eye_ids: (N,) integer eye id per lens, 0-7
            eye_parameter: Optional p = delta_rho / delta_phi override
                Bypasses the optical formula and computes acceptance as p * IOA (as in the simplified path)
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

        self._wavelength_nm = wavelength_nm

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

            ioa_arr = self._prepare_param(interommatidial_angles_rad, 'interommatidial_angles', N, allow_2d=True)

            ioa_minor = ioa_arr[:, 0] if ioa_arr.ndim == 2 else ioa_arr
            ioa_major = ioa_arr[:, 1] if ioa_arr.ndim == 2 else ioa_arr

        elif N > 1 and np.ptp(self._lens_positions, axis=0).max() > 1e-6:
            # Positions have meaningful spatial extent: estimate from lattice
            ioa_minor, ioa_major, lattice_tilts, nb_counts = _lattice_properties(
                directions=self._lens_directions,
                positions=self._lens_positions,
                eyes=self.eyes,
                k=8
            )

        else:
            # Positions are degenerate (single lens, or all coincident): can't estimate IOA
            ioa_minor = np.zeros(N, dtype=np.float32)
            ioa_major = np.zeros(N, dtype=np.float32)
            lattice_tilts = np.zeros(N, dtype=np.float32)
            nb_counts = np.zeros(N, dtype=np.uint32)

        # Chirality
        if chirality is not None:
            # User-provided values
            chirality_arr = self._prepare_param(chirality, 'chirality', N)

        else:
            right_side = np.dot(self._lens_positions, np.asarray(WORLD_RIGHT)) >= 0
            dorsal = np.dot(self._lens_positions, np.asarray(WORLD_UP)) >= 0

            chirality_arr = np.where(right_side == dorsal, -1.0, 1.0).astype(np.float32)

        # Dorsal sign: independent of chirality, but needed so that the kernel placement can apply the D/V y-flip
        is_dorsal = np.dot(self._lens_positions, np.asarray(WORLD_UP)) >= 0
        dorsal_sign = np.where(is_dorsal, -1.0, 1.0).astype(np.float32)
        self._dorsal_sign = dorsal_sign

        # Rhabdomere kernel orientation

        if bundle_orientation is not None:
            # User-provided values
            chi = self._prepare_param(bundle_orientation, 'bundle_orientation', N)

        else:
            flow_vec = np.asarray(flow_direction if flow_direction is not None else WORLD_FORWARD, dtype=np.float32)

            chi = _orient_rhabdomeres(
                kernel=self._kernel,
                lens_positions=self._lens_positions,
                lens_directions=self._lens_directions,
                local_right=self._local_right,
                local_up=self._local_up,
                chirality=chirality_arr,
                flow_direction=flow_vec,
                weight_tissue=weight_tissue,
                weight_flow=weight_flow,
                eyes=self.eyes,
            )
        self._bundle_orientation = chi

        # Receptors geometry
        receptor_dirs, receptor_pos = _receptors_geometry(
            kernel=self._kernel,
            lens_positions=self._lens_positions,
            lens_directions=self._lens_directions,
            local_right=self._local_right,
            local_up=self._local_up,
            bundle_orientation=self._bundle_orientation,
            chirality=chirality_arr,
            dorsal_sign=dorsal_sign
        )

        # Acceptance angles
        acc_axes = _acceptance_angles(
            lens_count=N,
            kernel=self._kernel,
            ioa_minor=ioa_minor,
            ioa_major=ioa_major,
            wavelength_nm=wavelength_nm,
            eye_parameter=eye_parameter,
            explicit_angles_rad=acceptance_angles_rad
        )

        # Fill main data structures

        sacc = _saccades_field(chi, self._kernel, chirality_arr, dorsal_sign, local_right, local_up, lens_positions,
                               self.eyes)

        # Lens static
        self.lens_static_data = np.zeros(N, dtype=LENS_STATIC_DTYPE)
        self.lens_static_data['right'] = local_right
        self.lens_static_data['sacc_x'] = np.sum(sacc * local_right, axis=1)
        self.lens_static_data['up'] = local_up
        self.lens_static_data['sacc_y'] = np.sum(sacc * local_up, axis=1)
        self.lens_static_data['forward'] = lens_directions
        self.lens_static_data['ioa_tilt'] = lattice_tilts
        self.lens_static_data['ioa_axes'][:, 0] = ioa_minor
        self.lens_static_data['ioa_axes'][:, 1] = ioa_major

        # Lens dynamic
        self.lens_dynamic_data = np.zeros(N, dtype=LENS_DYNAMIC_DTYPE)

        # Receptor static
        self.rcpt_static_data = np.zeros(N * R, dtype=RCPT_STATIC_DTYPE)
        self.rcpt_static_data['position'] = receptor_pos
        self.rcpt_static_data['metadata'] = _pack_metadata(N, R, e_ids, receptor_types, nb_counts, chirality_arr)
        self.rcpt_static_data['rest_acc'] = acc_axes

        dorsal_offset = np.where(dorsal_sign < 0, np.pi, 0.0).astype(np.float32)
        rot_dx, rot_dy = self._kernel.rotated_offsets(chi + dorsal_offset, chirality_arr)
        self.rcpt_static_data['rot_offset'][:, 0] = rot_dx.ravel()
        self.rcpt_static_data['rot_offset'][:, 1] = rot_dy.ravel()

        self.rcpt_static_data['acc_tilt'] = np.repeat(chi + dorsal_offset, R)
        self.rcpt_static_data['sensitivity'] = self._kernel.sensitivity
        self.rcpt_static_data['tau'] = self._kernel.tau_s

        # Default cartridge wiring: straight down
        self.rcpt_static_data['cartridge_src'] = np.repeat(np.arange(N), R) * R + np.tile(np.arange(R), N)

        # Receptor dynamic
        self.rcpt_dynamic_data = np.zeros(N * R, dtype=RCPT_DYNAMIC_DTYPE)
        self.rcpt_dynamic_data['direction'] = receptor_dirs
        self.rcpt_dynamic_data['acc_axes'] = acc_axes

        self.dirty_mask = np.zeros(N * R, dtype=bool)
        self.lens_dirty = False

        # Cached namespace views (invalidated by geometry changes)
        self._lenses_view = None
        self._receptors_view = None
        self._saccade_cache = None

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
        return f"<ReceptorArray(lenses={self.lens_count}, R={self.receptor_count}, total={len(self.rcpt_static_data)})>"

    @property
    def lenses(self) -> LensView:
        """Lens-level data (animal-level)."""

        if self._lenses_view is None:
            self._lenses_view = LensView(
                ra=self,
                lens_indices=np.arange(self._lens_count),
                single_eye=False,
            )

        return self._lenses_view

    @property
    def receptors(self) -> ReceptorView:
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

    def eye(self, eye_id: int) -> Eye:
        """Eye view for eye_id (0-7)."""

        if eye_id not in self._eye_cache:
            raise KeyError(f'No eye with id {eye_id}. Available: {sorted(self._eye_cache)}')

        return self._eye_cache[eye_id]

    @property
    def eyes(self) -> list:
        """List of Eye views for all eye_ids present."""

        return [self._eye_cache[eid] for eid in sorted(self._eye_cache)]

    @property
    def eye_ids(self) -> np.ndarray:

        return np.array(sorted(self._eye_cache), dtype=np.uint32)

    def _eye_cache_by_lens(self, lens_index: int) -> Eye:
        """Lookup which Eye owns a given lens (animal-level)."""

        eid = int(self._lens_eye_ids[lens_index])
        return self._eye_cache[eid]

    def ommatidium(self, lens_index: int) -> Ommatidium:
        """Animal-level lens index -> Ommatidium group view."""

        return Ommatidium(self, lens_index)

    def cartridge(self, lens_index: int) -> Cartridge:
        """Animal-level lens index -> Cartridge."""

        return Cartridge(self, lens_index)

    # Properties

    @property
    def lens_count(self) -> int:
        return self._lens_count

    @property
    def receptor_count(self) -> int:
        return self._kernel.count

    @property
    def max_gap(self) -> float:
        """Largest angular gap between any lens and its nearest neighbour."""

        return max(eye.max_gap() for eye in self.eyes)

    @property
    def kernel(self) -> Optional[RhabdomereKernel]:
        return self._kernel

    @property
    def bundle_orientation(self) -> np.ndarray:
        """Bundle orientation (chi), per lens. Alias for lenses.bundle_orientations."""

        return self._bundle_orientation

    def saccade_field(self) -> np.ndarray:
        """
        Compute (and cache) the saccades vector field.
        """
        if self._saccade_cache is not None:
            return self._saccade_cache

        result = _saccades_field(
            yaws=self._bundle_orientation,
            kernel=self._kernel,
            chirality=self.lenses.chirality.astype(np.float32),
            dorsal_sign=self._dorsal_sign,
            local_right=self._local_right,
            local_up=self._local_up,
            positions=self._lens_positions,
            eyes=self.eyes
        )
        self._saccade_cache = result
        return result

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
        return self.rcpt_static_data['cartridge_src'].reshape(self.lens_count, self.receptor_count)

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
        R = self.receptor_count

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
                dorsal_offset = np.pi if self._dorsal_sign[i_glob] < 0 else 0.0
                effective_chi = self._bundle_orientation[i_glob] + dorsal_offset
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

    # Rhabdomere actuation

    def actuate(self,
                lateral_um: Union[float, ArrayLike] = 0.0,
                axial_um: Union[float, ArrayLike] = 0.0,
                to_actuate: Optional[ArrayLike] = None
                ):
        """
        Displace rhabdomeres manually from the CPU.
        """

        R = self.receptor_count

        d_rest = self._kernel.nodal_distance_um
        if d_rest is None:
            raise ValueError("Can't actuate: kernel nodal distance is not set.")

        to_actuate = np.asarray(to_actuate) if to_actuate is not None else np.arange(self.lens_count)
        nb_actuated = len(to_actuate)

        lateral_disp = np.broadcast_to(lateral_um, nb_actuated).astype(np.float32)
        axial_displ = np.broadcast_to(axial_um, nb_actuated).astype(np.float32)

        # Update lens dynamic state
        self.lens_dynamic_data['lateral_um'][to_actuate] = lateral_disp
        self.lens_dynamic_data['axial_um'][to_actuate] = axial_displ
        self.lens_dirty = True  # Flag to tell renderer to upload lens states

        # Effective nodal distance
        d_eff = np.maximum(d_rest - axial_displ, 1.0)

        # Fetch static invariants for the actuated lenses
        l_stat = self.lens_static_data[to_actuate]
        lr = l_stat['right']
        lu = l_stat['up']
        fwd = l_stat['forward']
        sacc_x = l_stat['sacc_x']
        sacc_y = l_stat['sacc_y']

        # Global receptor indices for all affected lenses
        global_affected_indices = (to_actuate[:, None] * R + np.arange(R)[None, :]).ravel()

        # Fetch static invariants for the affected receptors
        r_stat = self.rcpt_static_data[global_affected_indices]

        # Apply Lateral Displacement
        lat_rep = np.repeat(lateral_disp, R)
        sx_rep = np.repeat(sacc_x, R)
        sy_rep = np.repeat(sacc_y, R)

        new_x = r_stat['rot_offset'][:, 0] + lat_rep * sx_rep
        new_y = r_stat['rot_offset'][:, 1] + lat_rep * sy_rep

        # to 3D space
        lr_rep = np.repeat(lr, R, axis=0)
        lu_rep = np.repeat(lu, R, axis=0)
        fwd_rep = np.repeat(fwd, R, axis=0)
        deff_rep = np.repeat(d_eff, R)

        tip_world = (new_x[:, None] * lr_rep +
                     new_y[:, None] * lu_rep -
                     deff_rep[:, None] * fwd_rep)

        new_dirs = -tip_world
        norms = np.linalg.norm(new_dirs, axis=-1, keepdims=True)
        np.divide(new_dirs, norms, out=new_dirs, where=norms != 0)

        self.rcpt_dynamic_data['direction'][global_affected_indices] = new_dirs

        # Update acceptance angles
        wavelength_um = self._wavelength_nm * 1e-3
        diffraction_sq = (wavelength_um / self._kernel.lens_diameter_um) ** 2
        geom_rest_sq = (self._kernel.diameters_um / d_rest) ** 2
        acc_rest = np.sqrt(diffraction_sq + geom_rest_sq)

        geom_new_sq = (self._kernel.diameters_um[None, :] / d_eff[:, None]) ** 2
        acc_new = np.sqrt(diffraction_sq + geom_new_sq)

        scale = (acc_new / acc_rest).ravel()

        self.rcpt_dynamic_data['acc_axes'][global_affected_indices] = r_stat['rest_acc'] * scale[:, None]

        self.dirty_mask[global_affected_indices] = True

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
        return self.rcpt_static_data['cartridge_src'].reshape(self.lens_count, self.receptor_count) // self.receptor_count



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