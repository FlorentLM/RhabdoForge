import warnings
from pathlib import Path
from typing import Optional, Union, Tuple, Sequence
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import KDTree

from insectvision.engine.world_utils import WORLD_UP, WORLD_RIGHT
from insectvision.utils.math import normalise_vectors, tangent_frames, fibonacci_sphere, icosphere

from .datatypes import RECEPTOR_DTYPE, LENS_DTYPE
from .kernel import RhabdomereKernel
from .proxies import Eye, Ommatidium, Cartridge, _ReceptorProxyMixin


# Build helpers

def _get_lens_geometry(
        directions: Optional[ArrayLike],
        positions: Optional[ArrayLike],
        ommatidia_count: Optional[int],
        eye_radius: float,
        icosphere_method: bool
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Generates or normalises the base 3D lens positions and directions.
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


def _get_receptors_geometry(
        lens_dirs: np.ndarray,
        lens_pos: np.ndarray,
        local_right: np.ndarray,
        local_up: np.ndarray,
        kernel: RhabdomereKernel,
        bundle_orientation: np.ndarray,
        chirality: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates 3D world directions and positions for every rhabdomere behind the lenses.
    """

    N = len(lens_dirs)
    R = kernel.count

    cos_chi = np.cos(bundle_orientation)[:, np.newaxis]
    sin_chi = np.sin(bundle_orientation)[:, np.newaxis]

    dx = kernel.offsets_um[:, 0]
    dy = kernel.offsets_um[:, 1]

    # Apply chirality (mirrors the kernel horizontally before rotation)
    dx_chiral = dx[np.newaxis, :] * chirality[:, np.newaxis]

    # Rotate kernel offsets by chi (per lens)
    rot_dx = cos_chi * dx_chiral - sin_chi * dy[np.newaxis, :]
    rot_dy = sin_chi * dx_chiral + cos_chi * dy[np.newaxis, :]

    # Local tip vectors in lens frame: (N, R, 3) as [right, up, fwd]
    local_tip = np.stack([
        rot_dx,
        rot_dy,
        np.full((N, R), -kernel.nodal_distance_um, dtype=np.float32)
    ], axis=-1)

    world_tip = (
            local_tip[..., 0:1] * local_right[:, np.newaxis, :] +
            local_tip[..., 1:2] * local_up[:, np.newaxis, :] +
            local_tip[..., 2:3] * lens_dirs[:, np.newaxis, :]
    ).reshape(N * R, 3)

    # Receptor direction: from tip through lens centre = -world_tip
    rec_dirs = -world_tip
    rec_dirs = normalise_vectors(rec_dirs)

    # Receptor position: lens + tip offset
    rec_pos = np.repeat(lens_pos, R, axis=0) + world_tip

    return rec_dirs, rec_pos


def _get_acceptance_angles(
        N: int,
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

    # If physical parameters are missing (R=1 simplified) use eye_parameter (p * IOA)
    if eye_parameter is not None or kernel.nodal_distance_um < 1e-6:
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
    diffraction = wavelength_um / kernel.lens_diameter_um           # λ/D (scalar)
    geometric = kernel.diameters_um / kernel.nodal_distance_um      # d/f (R,)
    full_acceptance = np.sqrt(diffraction ** 2 + geometric ** 2)    # (R,)

    acc_1d = np.tile(full_acceptance, N)

    return np.column_stack([acc_1d, acc_1d])


def _get_lattice_properties(
        directions: np.ndarray,
        positions: np.ndarray,
        k: int = 8,
        neighbour_dist_factor: float = 1.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate local lattice properties by separating structural topology (positions)
    from optical axes (directions).
    """

    N = len(directions)
    if N <= k:
        z = np.zeros(N, dtype=np.float32)
        return z, z, z, np.zeros(N, dtype=np.uint32)

    # Topology find structural neighbours using physical 3D positions
    pos_kdtree = KDTree(positions)
    pos_distances, indices = pos_kdtree.query(positions, k=k + 1)

    nb_indices = indices[:, 1:]
    nb_pos_distances = pos_distances[:, 1:]

    if nb_indices.size == 0:
        z = np.zeros(N, dtype=np.float32)
        return z, z, z, np.zeros(N, dtype=np.uint32)

    # Filter immediate physical neighbours
    dist_to_closest = nb_pos_distances[:, 0]
    is_immediate = nb_pos_distances <= dist_to_closest[:, np.newaxis] * neighbour_dist_factor
    nb_counts = np.sum(is_immediate, axis=1)

    # Optics: Calculate optical IOA using optical directions
    dirs_main = directions[:, np.newaxis, :]
    dirs_nb = directions[nb_indices]

    dots = np.sum(dirs_main * dirs_nb, axis=2)
    angular_sep = np.arccos(np.clip(dots, -1.0, 1.0))

    # Layout: Tangent frames orthogonal to the visual axis
    local_x, local_y = tangent_frames(
        directions,  # optical axis is the normal for the layout plane
        world_up=WORLD_UP,
        world_right=WORLD_RIGHT
    )

    # Project physical neighbour vectors on the tangent plane to find lattice tilt
    pos_nb = positions[nb_indices]
    delta_pos = pos_nb - positions[:, np.newaxis, :]

    proj_x = np.sum(delta_pos * local_x[:, np.newaxis, :], axis=2)
    proj_y = np.sum(delta_pos * local_y[:, np.newaxis, :], axis=2)

    tilts = np.zeros(N, dtype=np.float32)
    ioa_major = np.zeros(N, dtype=np.float32)
    ioa_minor = np.zeros(N, dtype=np.float32)
    psi6_magnitudes = np.zeros(N, dtype=np.float32)

    for i in range(N):
        mask = is_immediate[i]
        x, y = proj_x[i, mask], proj_y[i, mask]
        sep = angular_sep[i, mask]

        if len(x) < 2:
            avg = np.mean(sep) if np.any(mask) else 0.0
            ioa_major[i], ioa_minor[i], tilts[i] = avg, avg, 0.0
            continue

        # Hexatic phase based on physical layout
        angles = np.arctan2(y, x)
        phasors = np.exp(1j * 6 * angles)
        z_avg = np.mean(phasors)
        psi6_magnitudes[i] = np.abs(z_avg)
        tilts[i] = np.angle(z_avg) / 6.0

        # IOA = mean of the 2 smallest and 2 largest visual separations
        sep_sorted = np.sort(sep)
        ioa_minor[i] = np.mean(sep_sorted[:2])
        ioa_major[i] = np.mean(sep_sorted[-2:]) if len(sep_sorted) >= 2 else ioa_minor[i]

    mean_q = np.mean(psi6_magnitudes)
    print(f"Lattice hexatic quality (Ψ6): {mean_q:.3f}"
          f" ({'Excellent' if mean_q > 0.8 else 'Irregular' if mean_q > 0.5 else 'Poor'})")

    return ioa_minor, ioa_major, tilts, nb_counts.astype(np.uint32)


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

    return (
            (eid & 0x07) |
            ((rtypes & 0x0F) << 3) |
            ((nb_rep & 0x0F) << 7) |
            ((lindex & 0xFFFF) << 11) |
            ((is_mirrored & 0x01) << 27)
    )


class ReceptorArray(_ReceptorProxyMixin):
    """
    Flat (GPU-friendly) structured array of receptors for the renderer.
    Every element is one rhabdomere. The GPU traces rays for `len(data)` receptors.
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
                 icosphere_method: bool = True):
        """
        Construct a full receptor array.

        Each of the *N* lenses contains *R* receptors whose world-space directions are
        determined by the kernel offsets and rotated by the per-lens bundle orientation (chi).

        Acceptance angles are computed from the full optical model (Snyder 1979):

            Δρ = sqrt( (λ/D)² + (d_rhab/f)² )

        where λ = wavelength_nm, D = kernel.lens_diameter_um, d_rhab = kernel.diameters_um, f = kernel.nodal_distance_um

        This can be overridden with `eye_parameter`: p = delta_rho / delta_phi

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
            acceptance_angles_rad:
            eye_radius:
            icosphere_method:
        """
        # TODO: Add missing args descriptions

        self._kernel = kernel if kernel is not None else RhabdomereKernel()

        lens_dirs, lens_positions, N = _get_lens_geometry(
            directions, positions, ommatidia_count, eye_radius, icosphere_method)
        R = self._kernel.count
        M = N * R

        self.lens_count = N
        self.receptor_count = R
        self._wavelength_nm = wavelength_nm

        if eye_ids is None:
            e_ids = np.zeros(N, dtype=np.uint32)
        else:
            e_ids = self._prepare_param(eye_ids, "eye_ids", N).astype(np.uint32)

        is_pre_expanded = N > 1 and np.allclose(lens_positions[0], lens_positions[1], atol=1e-7)

        # Lens-level lattice props

        if interommatidial_angles_rad is not None:
            ioa_arr = self._prepare_param(interommatidial_angles_rad, "interommatidial_angles", N, allow_2d=True)
            ioa_minor = ioa_arr[:, 0] if ioa_arr.ndim == 2 else ioa_arr
            ioa_major = ioa_arr[:, 1] if ioa_arr.ndim == 2 else ioa_arr
            lattice_tilts = np.zeros(N, dtype=np.float32)
            nb_counts = np.zeros(N, dtype=np.uint32)

        elif not is_pre_expanded:
            ioa_minor, ioa_major, lattice_tilts, nb_counts = _get_lattice_properties(lens_dirs, lens_positions)

        else:
            ioa_minor = ioa_major = lattice_tilts = np.zeros(N, dtype=np.float32)
            nb_counts = np.zeros(N, dtype=np.uint32)

        # Parameters derived from lattice as fallback

        if chirality is None:
            # default to equator flip
            chiral_arr = np.where(lens_positions[:, 1] >= 0, 1.0, -1.0).astype(np.float32)
        else:
            chiral_arr = self._prepare_param(chirality, "chirality", N)

        if bundle_orientation is None:
            # default to align the rhabdomere bundle with the lattice grain
            chi = lattice_tilts.copy()
        else:
            chi = self._prepare_param(bundle_orientation, "bundle_orientation", N)

        # Tangent frames
        local_right, local_up = tangent_frames(lens_dirs)

        # Build geometry
        rec_dirs, rec_pos = _get_receptors_geometry(
            lens_dirs, lens_positions, local_right, local_up,
            self._kernel, chi, chiral_arr
        )

        acc_axes = _get_acceptance_angles(
            N, self._kernel, ioa_minor, ioa_major,
            wavelength_nm, eye_parameter, acceptance_angles_rad
        )

        # Fill data structure
        self.receptor_data = np.zeros(M, dtype=RECEPTOR_DTYPE)
        self.receptor_data['position'] = rec_pos
        self.receptor_data['direction'] = rec_dirs
        self.receptor_data['acc_axes'] = acc_axes
        self.receptor_data['acc_tilt'] = np.repeat(chi, R)
        self.receptor_data['sensitivity'] = self._kernel.sensitivity
        self.receptor_data['tau'] = self._kernel.tau_s
        self.receptor_data['metadata'] = _pack_metadata(N, R, e_ids, receptor_types, nb_counts, chiral_arr)

        # Fill lens data (for visualisation mostly)
        self.lens_data = np.zeros(N, dtype=LENS_DTYPE)
        self.lens_data['ioa_axes'][:, 0] = ioa_minor
        self.lens_data['ioa_axes'][:, 1] = ioa_major
        self.lens_data['tilt'] = lattice_tilts

        self._bundle_orientation = chi
        self._lens_directions = lens_dirs
        self._lens_positions = lens_positions
        self._actuation_state = np.zeros(N, dtype=np.float32)
        self._ioa_minor_rad = ioa_minor
        self._ioa_major_rad = ioa_major
        self._lattice_tilts = lattice_tilts
        self._local_right = local_right
        self._local_up = local_up

        with np.errstate(divide='ignore', invalid='ignore'):
            self.eye_parameter_minor = self.receptor_data['acc_axes'][:, 0] / np.repeat(ioa_minor, R)
            self.eye_parameter_major = self.receptor_data['acc_axes'][:, 1] / np.repeat(ioa_major, R)

        np.nan_to_num(self.eye_parameter_minor, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(self.eye_parameter_major, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        self.dirty_mask = np.zeros(M, dtype=bool)
        self._stale_receptor_spatial = False
        self._stale_lens_spatial = False
        self._kdtree_directions = KDTree(rec_dirs) if R == 1 else None  # eager for small models, lazy for full
        self._kdtree_positions = KDTree(rec_pos) if R == 1 else None
        self._eye_cache = {}
        self._cartridge_map = None

    @classmethod
    def from_file(cls, file_path: Union[str, Path], **kwargs):
        """
        Load from .npz archive.
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
        return cls(**args)

    # Overrides and internal helpers

    def __len__(self):
        return len(self.receptor_data)

    def __repr__(self):
        return f"<ReceptorArray(lenses={self.lens_count}, R={self.receptor_count}, total={len(self.receptor_data)})>"

    @property
    def _receptor_proxy(self):
        from .proxies import Receptor
        return Receptor(self.receptor_data, slice(None), self)

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
            self._eye_cache[eye_id] = Eye(self, eye_id)
        return self._eye_cache[eye_id]

    @property
    def eyes(self) -> list:
        """
        List of Eye views for all eye_ids present.
        """
        unique = np.unique(self.receptor_data['metadata'][::self.receptor_count] & 0x07)
        return [self.eye(int(eid)) for eid in unique]

    @property
    def eye_ids(self) -> np.ndarray:
        return np.unique(self.receptor_data['metadata'][::self.receptor_count] & 0x07)

    def ommatidium(self, lens_index: int) -> Ommatidium:
        """
        Global lens index -> Ommatidium group view.
        """
        return Ommatidium(self, lens_index)

    def cartridge(self, lens_index: int) -> Cartridge:
        """
        Global lens index -> Cartridge (neural superposition unit).
        """
        return Cartridge(self, lens_index)

    # Properties

    @property
    def max_gap(self) -> float:
        """
        Largest angular gap between any receptor and its nearest neighbour.
        """
        if len(self.receptor_data) <= 1:
            return 0.0

        kd = self._ensure_global_kdtree_directions()
        d, _ = kd.query(self.receptor_data['direction'], k=2)
        return float(np.arccos(np.clip(1.0 - (np.max(d[:, 1]) ** 2) / 2.0, -1, 1)))

    @property
    def kernel(self) -> Optional[RhabdomereKernel]:
        return self._kernel

    @property
    def bundle_orientation(self) -> np.ndarray:
        """
        Bundle orientation (chi), per lens.
        """
        return self._bundle_orientation

    @property
    def chirality(self) -> np.ndarray:
        """
        Bundle chirality (+1 normal, -1 mirrored), per lens.
        """
        is_mirrored = (self.receptor_data['metadata'][::self.receptor_count] >> 27) & 0x01
        return np.where(is_mirrored, -1, 1)

    @property
    def is_full_model(self) -> bool:
        """
        True if built with a multi-receptor kernel (R > 1).
        """
        return self.receptor_count > 1

    @property
    def interommatidial_angles_rad(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._ioa_minor_rad, self._ioa_major_rad

    # Neural superposition wiring

    def build_cartridge_map(self) -> np.ndarray:
        """
        For each lens, compute which neighbour's receptor points at this lens's optical axis.
        Returns (N, R_periph) array of global lens indices, where R_periph is defined by the kernel.
        """
        # TODO: This needs to be reworked

        peripheral = self._kernel.peripheral_indices
        if self.receptor_count < 2 or len(peripheral) == 0:
            warnings.warn("Cartridge map requires a full model with peripheral receptors.")
            return np.zeros((self.lens_count, 0), dtype=np.intp)

        N = self.lens_count
        R = self.receptor_count
        lens_dirs = self._lens_directions

        cartridge = np.zeros((N, len(peripheral)), dtype=np.intp)

        for col, receptor_idx in enumerate(peripheral):
            # Directions of this receptor type across all lenses
            type_dirs = self.receptor_data['direction'][receptor_idx::R]
            tree = KDTree(type_dirs)
            _, indices = tree.query(lens_dirs)
            cartridge[:, col] = indices

        self._cartridge_map = cartridge
        return cartridge

    @property
    def cartridge_global_indices(self) -> np.ndarray:
        # TODO: rename this
        """
        Returns (N, R_periph) array of global receptor indices (for neural superposition).
        """
        if self._cartridge_map is None:
            self.build_cartridge_map()

        R = self.receptor_count
        peripheral = np.asarray(self._kernel.peripheral_indices)

        return self._cartridge_map * R + peripheral

    # Actuation

    def actuate(self, displacement_um: Union[float, ArrayLike],
                axial_um: Union[float, ArrayLike] = 0.0,
                lens_mask: Optional[ArrayLike] = None):
        """
        Displace rhabdomeres via microsaccades.

        Models the two components of rhabdomere actuation observed in vivo:
        (Juusola et al. 2017, 10.7554/eLife.26117; Kemppainen et al. 2022, 10.1073/pnas.2109717119)

        * Lateral: rhabdomere tips move sideways in the focal plane along
            the actuation axis (chi + kernel.actuation_angle_deg).
            Shifts the sampling direction.
            Typical range: 0.0 to ~1.7 μm in Drosophila

        * Axial: rhabdomeres contract away from the lens.
            This narrows the acceptance angle and widens the angular subtense of lateral offsets.
            Typical range: from ~8.1° to ~4.0° in Drosophila

        The rhabdomeres are mechanically coupled, activating one receptor likely contracts and tilts its neighbours.

        Both parameters are absolute from rest: calling `actuate(0.0, 0.0)` resets to the rest configuration.

        Args:
            displacement_um: Lateral focal-plane displacement (μm).
                Scalar for uniform, or (n_mask,) per-lens.
            axial_um: Axial contraction toward lens (μm), positive.
                Scalar for uniform, or (n_mask,) per-lens.
                Default 0 (lateral only).
            lens_mask: Global lens indices to actuate. None = all.
        """

        if self.receptor_count < 2:
            raise RuntimeError("Actuation requires a full model (R > 1).")

        kernel = self._kernel
        R = self.receptor_count
        d_rest = kernel.nodal_distance_um  # nodal distance at rest

        dx = kernel.offsets_um[:, 0]
        dy = kernel.offsets_um[:, 1]

        if lens_mask is None:
            lens_mask = np.arange(self.lens_count)

        lens_mask = np.asarray(lens_mask)
        n_act = len(lens_mask)

        lat = np.broadcast_to(np.float32(displacement_um), n_act).copy()
        axi = np.broadcast_to(np.float32(axial_um), n_act).copy()
        self._actuation_state[lens_mask] = lat

        chi = self._bundle_orientation[lens_mask]
        cos_chi = np.cos(chi)
        sin_chi = np.sin(chi)

        chiral_arr = self.chirality[lens_mask]

        # Effective nodal distance after axial contraction
        d_eff = d_rest - axi
        d_eff = np.maximum(d_eff, 1.0)  # clamp to 1 μm minimum

        # Reconstruct kernel offsets at rest (applying chirality)
        dx_chiral = dx[np.newaxis, :] * chiral_arr[:, np.newaxis]
        rot_dx = cos_chi[:, np.newaxis] * dx_chiral - sin_chi[:, np.newaxis] * dy[np.newaxis, :]
        rot_dy = sin_chi[:, np.newaxis] * dx_chiral + cos_chi[:, np.newaxis] * dy[np.newaxis, :]

        # Reconstruct saccade vector (applying chirality to its X component)
        act_angle_rad = np.radians(kernel.actuation_angle_deg)
        local_act_dx = np.cos(act_angle_rad) * chiral_arr  # mirror X if chirality is -1
        local_act_dy = np.sin(act_angle_rad)

        act_dx = cos_chi * local_act_dx - sin_chi * local_act_dy
        act_dy = sin_chi * local_act_dx + cos_chi * local_act_dy

        # Add lateral displacement along the world actuation vector
        rot_dx += lat[:, np.newaxis] * act_dx[:, np.newaxis]
        rot_dy += lat[:, np.newaxis] * act_dy[:, np.newaxis]

        if self._local_right is None:
            self._local_right, self._local_up = tangent_frames(self._lens_directions)

        lr = self._local_right[lens_mask]
        lu = self._local_up[lens_mask]
        fwd = self._lens_directions[lens_mask]

        # tip vectors (using per-lens effective nodal distance)
        local_tip = np.stack([
            rot_dx,
            rot_dy,
            np.broadcast_to(-d_eff[:, np.newaxis], (n_act, R)).copy()
        ], axis=-1)

        world_tip = (
                local_tip[..., 0:1] * lr[:, np.newaxis, :] +
                local_tip[..., 1:2] * lu[:, np.newaxis, :] +
                local_tip[..., 2:3] * fwd[:, np.newaxis, :]
        )  # (n_act, R, 3)

        new_dirs = -world_tip
        norms = np.linalg.norm(new_dirs, axis=-1, keepdims=True)
        np.divide(new_dirs, norms, out=new_dirs, where=norms != 0)

        new_positions = self._lens_positions[lens_mask, np.newaxis, :] + world_tip

        # build global receptor indices for all affected lenses
        receptor_indices = (
                lens_mask[:, np.newaxis] * R + np.arange(R)[np.newaxis, :]
        ).ravel()  # (n_act * R,)

        self.receptor_data['direction'][receptor_indices] = new_dirs.reshape(-1, 3)
        self.receptor_data['position'][receptor_indices] = new_positions.reshape(-1, 3)

        # Also change acceptance angles for any with axial displacement
        has_axial = np.any(axi != 0.0)

        if has_axial:
            wavelength_um = self._wavelength_nm * 1e-3
            diffraction = wavelength_um / kernel.lens_diameter_um
            geometric = kernel.diameters_um[np.newaxis, :] / d_eff[:, np.newaxis]
            new_acc = np.sqrt(diffraction ** 2 + geometric ** 2)  # (n_act, R)

            self.receptor_data['acc_axes'][receptor_indices, 0] = new_acc.ravel()
            self.receptor_data['acc_axes'][receptor_indices, 1] = new_acc.ravel()

        self.dirty_mask[receptor_indices] = True

        self._stale_receptor_spatial = True

    # Spatial structures: 2 levels tracked independently
    # TODO: This can be simplified. Lens level will likely never change now.
    #
    # Lens-level (_stale_lens_spatial):
    #   - Eye KDTrees (built from _lens_directions / _lens_positions)
    #   - Eye neighbour graphs
    #   - Only invalidated by scale(), translate(), or direct lens mutation
    #   - Not invalidated by actuate() (lens axes are immutable after construction)
    #
    # Receptor-level (_stale_receptor_spatial):
    #   - Global receptor KDTrees (built lazily from data['direction'] / data['position'])
    #   - Invalidated by actuate(), Receptor property setters, scale(), translate()
    #   - Built lazily

    def _resolve_lens_spatial(self):
        """Clear lens-stale flag and invalidate Eye caches."""

        if self._stale_lens_spatial:
            self._stale_lens_spatial = False

            for ev in self._eye_cache.values():
                ev._invalidate()

    def _ensure_global_kdtree_directions(self):
        """Lazy build of receptor-level direction KDTree."""

        if self._stale_receptor_spatial or self._kdtree_directions is None:
            self._kdtree_directions = KDTree(self.receptor_data['direction'])
            self._stale_receptor_spatial = False  # directions rebuilt

        return self._kdtree_directions

    def _ensure_global_kdtree_positions(self):
        """Lazy build of receptor-level position KDTree."""

        if self._stale_receptor_spatial or self._kdtree_positions is None:
            self._kdtree_positions = KDTree(self.receptor_data['position'])

        return self._kdtree_positions

    @property
    def kdtree_directions(self):
        """Global (receptor-level) direction KDTree (lazy)."""
        return self._ensure_global_kdtree_directions()

    @property
    def kdtree_positions(self):
        """Global (receptor-level) position KDTree (lazy)."""
        return self._ensure_global_kdtree_positions()

    # Global spatial queries (receptor-level, return global data indices)
    # TODO: These are duplicated, could be taken out as pure functions
    #
    #   These search over all receptors (N*R elements).
    #   In the full model (R>1), Eye-level queries (which operate on lens optical axes) should be preferred.

    def query_directions(self, directions: ArrayLike, k: int = 1) -> np.ndarray:
        """
        Find receptors with optical axis best aligned with some directions. Global indices.
        """

        if k < 1:
            raise ValueError("k must be >= 1")
        kd = self._ensure_global_kdtree_directions()
        q = np.atleast_2d(np.asarray(directions, dtype=np.float32))
        norms = np.linalg.norm(q, axis=-1, keepdims=True)
        np.divide(q, norms, out=q, where=norms != 0)
        is_single = np.asarray(directions).ndim == 1
        _, idx = kd.query(q, k=k)
        if is_single and k == 1:
            return idx.item()
        return idx.squeeze()

    def query_position(self, positions: ArrayLike, k: int = 1) -> np.ndarray:
        """
        Find receptors closest to some positions (on the eye surface). Global indices.
        """

        if k < 1:
            raise ValueError("k must be >= 1")

        kd = self._ensure_global_kdtree_positions()
        q = np.atleast_2d(np.asarray(positions, dtype=np.float32))
        is_single = np.asarray(positions).ndim == 1
        _, idx = kd.query(q, k=k)

        if is_single and k == 1:
            return idx.item()

        return idx.squeeze()

    def query_lookat(self, targets: ArrayLike, k: int = 1) -> np.ndarray:
        """
        Find receptors looking at some target points (world-space). Global indices.
        """

        if k < 1:
            raise ValueError("k must be >= 1")

        q = np.atleast_2d(np.asarray(targets, dtype=np.float32))
        is_single = np.asarray(targets).ndim == 1

        desired = q[:, np.newaxis, :] - self.receptor_data['position'][np.newaxis, :, :]

        norms = np.linalg.norm(desired, axis=-1, keepdims=True)
        np.divide(desired, norms, out=desired, where=norms != 0)
        dots = np.einsum('jk,ijk->ij', self.receptor_data['direction'], desired)

        part = np.argpartition(dots, -k, axis=1)[:, -k:]
        top = np.take_along_axis(dots, part, axis=1)

        order = np.argsort(top, axis=1)[:, ::-1]
        best = np.take_along_axis(part, order, axis=1)

        if is_single and k == 1:
            return best.item()

        return best.squeeze()

    def query_cone(self, center_direction: ArrayLike, angle: float, degrees: bool = True) -> np.ndarray:
        """
        Find all receptors within angle of a centre direction. Global indices.
        """

        kd = self._ensure_global_kdtree_directions()
        c = np.asarray(center_direction, dtype=np.float32)
        c /= np.linalg.norm(c)
        a = np.deg2rad(angle) if degrees else angle

        return kd.query_ball_point(c, r=2.0 * np.sin(a / 2.0))

    def query_ball(self, center_position: ArrayLike, radius: float) -> np.ndarray:
        """
        Find all receptors within radius of a centre position. Global indices.
        """

        kd = self._ensure_global_kdtree_positions()
        c = np.asarray(center_position, dtype=np.float32)
        return kd.query_ball_point(c, r=radius)

    # Whole-array transforms (initial unit scaling, agent setup, etc)

    def scale(self, factor: float):
        """
        Scale all receptor positions by a factor.
        """

        self.receptor_data['position'] *= factor
        self._lens_positions *= factor

        # both levels stale: lens positions changed, receptor positions changed
        self._stale_receptor_spatial = True
        self._stale_lens_spatial = True
        self._kdtree_directions = None
        self._kdtree_positions = None
        self._resolve_lens_spatial()

        return self

    def translate(self, vector: ArrayLike):
        """
        Translate all receptor positions by a vector.
        """
        v = np.asarray(vector, dtype=np.float32)

        self.receptor_data['position'] += v

        self._lens_positions += v
        self._stale_receptor_spatial = True
        self._stale_lens_spatial = True
        self._kdtree_directions = None
        self._kdtree_positions = None
        self._resolve_lens_spatial()

        return self


## Examples:
#
# # Minimal, uniform sphere model
#
# # Generates ~2000 ommatidia on a sphere
# # Defaults to R=1 (point kernel), p=1.0 (acceptance = IOA)
# eye_array = ReceptorArray(ommatidia_count=2000)
#
# # or with a custom eye parameter or radius
# eye_array = ReceptorArray(
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
# eye_array = ReceptorArray(
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
# eye_array = ReceptorArray(
#     directions=dirs,
#     positions=pos,
#     kernel=DROSOPHILA_KERNEL,       # R=7 defined kernel
#     bundle_orientation=chi_array,   # (N,) array of bundle rotations
#     chirality=chirality_array,      # (N,) array of +1 or -1 for the equator flip
#     eye_ids=eye_id_array            # (N,) mapping left/right eyes
# )
