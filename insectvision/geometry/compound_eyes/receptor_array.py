import warnings
from pathlib import Path
from typing import Optional, Union, Tuple, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import KDTree

from insectvision.geometry.compound_eyes.datatypes import GPU_RECEPTOR_DTYPE, _CLEAR_NEIGHBOURS
from insectvision.geometry.compound_eyes.kernel import RhabdomereKernel
from insectvision.geometry.compound_eyes.proxies import Eye, Ommatidium, Cartridge

from insectvision.geometry.geom_utils import estimate_lod, subdivide_icosahedron, fibonacci_sphere
from insectvision.geometry.compound_eyes.eye_utils import compute_lattice_properties, tangent_frames


class ReceptorArray:
    """
    Flat (GPU-friendly) structured array of receptors for the renderer.
    Every element is one rhabdomere. The GPU traces rays for `len(data)` receptors.

    Construction modes:
    # TODO: This will probably change

    *Full model (R receptors per lens):
        array = ReceptorArray.from_build(directions=dirs,
                                        origins=origins,
                                        kernel=DROSOPHILA_KERNEL,
                                        bundle_orientation=chi, ...)

    *Simplified (1 receptor per lens, more or less just R7/8):
        array = ReceptorArray(directions=dirs, origins=origins, ...)
    """

    @classmethod
    def from_build(cls,
                   directions: ArrayLike,
                   origins: ArrayLike,
                   kernel: RhabdomereKernel,
                   bundle_orientation: ArrayLike,
                   eye_ids: Optional[ArrayLike] = None,
                   eye_parameter: Optional[Union[float, Tuple]] = None,
                   interommatidial_angles_rad: Optional[ArrayLike] = None,
                   sensitivities: Optional[Union[ArrayLike, float]] = None,
                   wavelength_nm: float = 500.0,
                   ) -> 'ReceptorArray':
        """
        Construct a full receptor array from lens-level geometry and a rhabdomere kernel.

        Each of the *N* lenses contains *R* receptors whose world-space directions are
        determined by the kernel offsets and rotated by the per-lens bundle orientation (chi).

        Acceptance angles are computed from the full optical model (Snyder 1979):

            Δρ = sqrt( (λ/D)² + (d_rhab/f)² )

        where λ = wavelength_nm, D = kernel.lens_diameter_um, d_rhab = kernel.diameters_um, f = kernel.nodal_distance_um

        This can be overridden with `eye_parameter`: p = delta_rho / delta_phi

        Args:
            directions: (N, 3) lens optical axes
            origins: (N, 3) lens positions in head space
            kernel: Species-level rhabdomere geometry
            bundle_orientation: (N,) chi per lens (radians in tangent plane)
            eye_ids: (N,) integer eye id per lens, 0-7
            eye_parameter: Optional p = delta_rho / delta_phi override
                Bypasses the optical formula and computes acceptance as p * IOA (as in the simplified path)
            interommatidial_angles_rad: (N,) or (N,2) if known, otherwise estimated
            sensitivities: scalar or (N,) per lens (tiled to receptors) # TODO: maybe this should be a receptor-level prop?
            wavelength_nm: light wavelength for diffraction term (default 500)
        """

        dirs = np.asarray(directions, dtype=np.float32)
        origs = np.asarray(origins, dtype=np.float32)
        chi = np.asarray(bundle_orientation, dtype=np.float32)
        N = len(dirs)
        R = kernel.count
        M = N * R
        d = kernel.nodal_distance_um  # lens to rhabdomere tips (at rest) = lever arm

        # Normalise lens directions
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        lens_dirs = dirs / norms
        lens_origins = origs

        # Lens-level lattice properties
        if interommatidial_angles_rad is not None:
            ioa_arr = np.asarray(interommatidial_angles_rad, dtype=np.float32)
            if ioa_arr.ndim == 1:
                ioa_minor = ioa_arr
                ioa_major = ioa_arr
            else:
                ioa_minor = np.minimum(ioa_arr[:, 0], ioa_arr[:, 1])
                ioa_major = np.maximum(ioa_arr[:, 0], ioa_arr[:, 1])
            lattice_tilts = np.zeros(N, dtype=np.float32)
            nb_counts = np.zeros(N, dtype=np.uint32)
        else:
            ioa_minor, ioa_major, lattice_tilts, nb_counts = \
                compute_lattice_properties(lens_dirs, lens_origins)

        # Tangent frames
        local_right, local_up = tangent_frames(lens_dirs)

        # Rotate kernel offsets by chi (that is per lens)
        cos_chi = np.cos(chi)[:, np.newaxis]  # (N, 1)
        sin_chi = np.sin(chi)[:, np.newaxis]

        dx = kernel.offsets_um[:, 0]  # (R,)
        dy = kernel.offsets_um[:, 1]

        rot_dx = cos_chi * dx[np.newaxis, :] - sin_chi * dy[np.newaxis, :]  # (N, R)
        rot_dy = sin_chi * dx[np.newaxis, :] + cos_chi * dy[np.newaxis, :]

        # Local tip vectors in lens frame: (N, R, 3) as [right, up, fwd]
        local_tip = np.stack([rot_dx, rot_dy,
                              np.full((N, R), -d, dtype=np.float32)], axis=-1)

        world_tip = (
            local_tip[..., 0:1] * local_right[:, np.newaxis, :] +
            local_tip[..., 1:2] * local_up[:, np.newaxis, :] +
            local_tip[..., 2:3] * lens_dirs[:, np.newaxis, :]
        ).reshape(M, 3)

        # Receptor direction: from tip through lens centre = -world_tip
        rec_dirs = -world_tip
        rec_dirs /= np.linalg.norm(rec_dirs, axis=1, keepdims=True)

        # Receptor position: lens + tip offset
        rec_positions = np.repeat(lens_origins, R, axis=0) + world_tip

        data = np.zeros(M, dtype=GPU_RECEPTOR_DTYPE)
        data['origin'][:, :3] = rec_positions
        data['origin'][:, 3] = 1.0
        data['direction'][:, :3] = rec_dirs
        data['direction'][:, 3] = 0.0

        wavelength_um = wavelength_nm * 1e-3
        diffraction = wavelength_um / kernel.lens_diameter_um     # λ/D (scalar)
        geometric = kernel.diameters_um / d                       # d_rhab/f (R,)
        full_acceptance = np.sqrt(diffraction**2 + geometric**2)  # (R,)

        if eye_parameter is not None:
            p_min, p_maj = (eye_parameter, eye_parameter) if isinstance(eye_parameter, (float, int, np.number)) else eye_parameter
            data['acceptance_angles'][:, 0] = np.repeat(p_min * ioa_minor, R)
            data['acceptance_angles'][:, 1] = np.repeat(p_maj * ioa_major, R)

        else:
            data['acceptance_angles'][:, 0] = np.tile(full_acceptance, N)
            data['acceptance_angles'][:, 1] = np.tile(full_acceptance, N)

        data['interommatidial_angles'][:, 0] = np.repeat(ioa_minor, R)
        data['interommatidial_angles'][:, 1] = np.repeat(ioa_major, R)
        data['tilt'] = np.repeat(lattice_tilts, R)

        sens = 1.0 if sensitivities is None else sensitivities
        data['sensitivity'] = np.repeat(
            np.broadcast_to(np.float32(sens), N), R
        )

        # Packed metadata
        if eye_ids is not None:
            eid = np.repeat(np.asarray(eye_ids, dtype=np.uint32), R)
        else:
            eid = np.zeros(M, dtype=np.uint32)

        rtypes = np.tile(np.arange(R, dtype=np.uint32), N)
        lindex = np.repeat(np.arange(N, dtype=np.uint32), R)

        data['packed_data'] = (
            (eid & 0x07) |
            ((rtypes & 0x0F) << 3) |
            ((np.repeat(nb_counts, R) & 0x0F) << 7) |
            ((lindex & 0xFFFF) << 11)
        )

        # assemble object
        obj = object.__new__(cls)
        obj.data = data
        obj.lens_count = N
        obj.receptor_count = R

        obj._kernel = kernel
        obj._bundle_orientation = chi.copy()
        obj._lens_directions = lens_dirs.copy()
        obj._lens_positions = lens_origins.copy()
        obj._actuation_state = np.zeros(N, dtype=np.float32)
        obj._wavelength_nm = wavelength_nm

        obj._ioa_minor_rad = ioa_minor
        obj._ioa_major_rad = ioa_major
        obj._lattice_tilts = lattice_tilts

        obj._local_right = local_right
        obj._local_up = local_up

        with np.errstate(divide='ignore', invalid='ignore'):
            obj.eye_parameter_minor = data['acceptance_angles'][:, 0] / np.repeat(ioa_minor, R)
            obj.eye_parameter_major = data['acceptance_angles'][:, 1] / np.repeat(ioa_major, R)

        np.nan_to_num(obj.eye_parameter_minor, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(obj.eye_parameter_major, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        obj.dirty_mask = np.zeros(M, dtype=bool)
        obj._stale_receptor_spatial = False
        obj._stale_lens_spatial = False
        obj._kdtree_directions = None   # lazy
        obj._kdtree_positions = None    # also lazy
        obj._eye_cache = {}
        obj._cartridge_map = None

        return obj

    # Single-receptor constructor

    def __init__(self,
                 directions: Optional[ArrayLike] = None,
                 origins: Optional[ArrayLike] = None,
                 num_ommatidia: Optional[int] = None,
                 acceptance_angles_rad: Optional[Union[ArrayLike, Tuple, float]] = None,
                 interommatidial_angles_rad: Optional[Union[ArrayLike, Tuple, float]] = None,
                 sensitivities: Optional[Union[ArrayLike, float]] = None,
                 receptor_types: Optional[Union[ArrayLike, int]] = None,
                 eye_id: Optional[Union[int, ArrayLike]] = None,
                 eye_parameter: Optional[Union[float, Tuple]] = None,
                 lens_diameter_nm: Optional[Union[float, Tuple]] = None,
                 rhabdom_diameter_nm: Optional[Union[float, Tuple]] = None,
                 focal_length_nm: Optional[Union[float, Tuple]] = None,
                 wavelength_nm: float = 500,
                 eye_radius: float = 0.01,
                 force_isotropic: bool = False,
                 icosphere_method: bool = True,
                 ):
        """
        Simplified construction (single receptor per lens).
        """

        if directions is None and num_ommatidia is None:
            raise ValueError("Requires either 'directions' or 'num_ommatidia'.")

        if directions is not None:
            directions = np.asarray(directions, dtype=np.float32)
            N = len(directions)
        else:
            if icosphere_method:
                lod = estimate_lod(num_ommatidia)
                directions = subdivide_icosahedron(lod)
                N = len(directions)
                if abs(num_ommatidia - N) > 1:
                    print(f"Note: {N} ommatidia for subdivision level {lod}.")
            else:
                directions = fibonacci_sphere(num_ommatidia)
                N = len(directions)

        self.lens_count = N
        self.receptor_count = 1
        self.data = np.zeros(N, dtype=GPU_RECEPTOR_DTYPE)

        self._kernel = None
        self._cartridge_map = None
        self._wavelength_nm = wavelength_nm

        # Directions
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        self.data['direction'][:, :3] = directions / norms
        self.data['direction'][:, 3] = 0.0

        # Origins
        if origins is not None:
            origins_arr = np.asarray(origins, dtype=np.float32)

            if origins_arr.ndim == 1 and origins_arr.shape[0] == 3:
                self.data['origin'][:, :3] = origins_arr
            elif origins_arr.shape == (N, 3):
                self.data['origin'][:, :3] = origins_arr
            else:
                raise ValueError(f"Invalid 'origins' shape {origins_arr.shape}. Expected ({N}, 3) or (3,).")

        elif eye_radius > 0:
            self.data['origin'][:, :3] = self.data['direction'][:, :3] * eye_radius

        self.data['origin'][:, 3] = 1.0

        self.data['sensitivity'] = np.asarray(
            sensitivities if sensitivities is not None else 1.0, dtype=np.float32)

        # Packed metadata
        id_arr = np.zeros(N, dtype=np.uint32)
        if eye_id is not None:
            prepared = self._prepare_param(eye_id, "eye_id")
            if np.any(prepared > 7) or np.any(prepared < 0):
                raise ValueError("eye_id must be in [0, 7].")
            id_arr = prepared.astype(np.uint32)

        types_arr = np.zeros(N, dtype=np.uint32)
        if receptor_types is not None:
            prepared = self._prepare_param(receptor_types, "receptor_types")
            types_arr = np.clip(prepared, 0, 15).astype(np.uint32)

        lens_idx_arr = np.arange(N, dtype=np.uint32)

        self.data['packed_data'] = (
            (id_arr & 0x07) |
            ((types_arr & 0x0F) << 3) |
            ((lens_idx_arr & 0xFFFF) << 11)
        )

        self.dirty_mask = np.zeros(N, dtype=bool)
        self._stale_receptor_spatial = False
        self._stale_lens_spatial = False

        # Can eagerly build since receptor=lens, it's cheap
        self._kdtree_directions = KDTree(self.data['direction'][:, :3])
        self._kdtree_positions = KDTree(self.data['origin'][:, :3])
        self._eye_cache = {}

        # Lattice properties
        is_pre_expanded = False
        if N > 1:
            if np.allclose(self.data['origin'][0], self.data['origin'][1], atol=1e-7):
                is_pre_expanded = True

        if interommatidial_angles_rad is not None:
            angles_arr = np.asarray(interommatidial_angles_rad, dtype=np.float32)

            if angles_arr.shape == (N,):
                ioa_min = angles_arr
                ioa_maj = angles_arr
            else:
                broad = np.broadcast_to(angles_arr, (N, 2))
                ioa_min = np.minimum(broad[:, 0], broad[:, 1])
                ioa_maj = np.maximum(broad[:, 0], broad[:, 1])

            self.data['interommatidial_angles'][:, 0] = ioa_min
            self.data['interommatidial_angles'][:, 1] = ioa_maj

            self._ioa_minor_rad = ioa_min
            self._ioa_major_rad = ioa_maj
            self._lattice_tilts = np.zeros(N, dtype=np.float32)

        elif not is_pre_expanded:
            ioa_min, ioa_maj, tilts, counts = compute_lattice_properties(
                self.data['direction'][:, :3],
                self.data['origin'][:, :3]
            )

            self._ioa_minor_rad = ioa_min
            self._ioa_major_rad = ioa_maj
            self._lattice_tilts = tilts

            self.data['interommatidial_angles'][:, 0] = ioa_min
            self.data['interommatidial_angles'][:, 1] = ioa_maj
            self.data['tilt'] = tilts

            cleared = self.data['packed_data'] & _CLEAR_NEIGHBOURS
            self.data['packed_data'] = cleared | ((counts & 0x0F) << 7)
        else:
            self._ioa_minor_rad = self.data['interommatidial_angles'][:, 0]
            self._ioa_major_rad = self.data['interommatidial_angles'][:, 1]
            self._lattice_tilts = self.data['tilt'].copy()

        # Bundle orientation: in single receptor mode this defaults to lattice tilt
        # TODO: This is probably not the best approximation

        self._bundle_orientation = self._lattice_tilts.copy()
        self._lens_directions = self.data['direction'][:, :3].copy()
        self._lens_positions = self.data['origin'][:, :3].copy()
        self._actuation_state = np.zeros(N, dtype=np.float32)
        self._local_right = None
        self._local_up = None

        # Acceptance angles
        if acceptance_angles_rad is not None:
            estimated_angles = acceptance_angles_rad

        elif all(p is not None for p in [lens_diameter_nm, rhabdom_diameter_nm, focal_length_nm]):

            D_min, D_maj = self._unpack(lens_diameter_nm, "lens_diameter")
            d_min, d_maj = self._unpack(rhabdom_diameter_nm, "rhabdom_diameter")
            f_min, f_maj = self._unpack(focal_length_nm, "focal_length")

            acc_min = np.sqrt((wavelength_nm / D_min) ** 2 + (d_min / f_min) ** 2)
            acc_maj = np.sqrt((wavelength_nm / D_maj) ** 2 + (d_maj / f_maj) ** 2)
            estimated_angles = np.vstack([acc_min, acc_maj]).T

        else:
            p = eye_parameter if eye_parameter is not None else 1.0
            p_min, p_maj = (p, p) if isinstance(p, (int, float)) else p

            estimated_angles = np.vstack([
                p_min * self._ioa_minor_rad,
                p_maj * self._ioa_major_rad
            ]).T

        if force_isotropic:
            mean_a = np.mean(np.atleast_2d(estimated_angles), axis=1)
            estimated_angles = np.vstack([mean_a, mean_a]).T

        angles_arr = np.asarray(estimated_angles, dtype=np.float32)

        if angles_arr.shape == (N,):
            self.data['acceptance_angles'] = angles_arr[:, np.newaxis]
        else:
            self.data['acceptance_angles'] = angles_arr

        with np.errstate(divide='ignore', invalid='ignore'):
            self.eye_parameter_minor = self.data['acceptance_angles'][:, 0] / self._ioa_minor_rad
            self.eye_parameter_major = self.data['acceptance_angles'][:, 1] / self._ioa_major_rad

        np.nan_to_num(self.eye_parameter_minor, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(self.eye_parameter_major, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    @classmethod
    def from_file(cls, file_path: Union[str, Path], **kwargs):
        """
        Load (a R=1 simple) model from .npz archive.
        """
        # TODO: This should also load a full model

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Cannot find: {path}")

        data = np.load(path)
        if 'directions' not in data:
            raise ValueError(f"'{path}' missing required 'directions' array.")

        args = {
            'directions': data['directions'],
            'origins': data.get('origins'),
            'acceptance_angles_rad': data.get('acceptance_angles_rad'),
            'interommatidial_angles_rad': data.get('interommatidial_angles_rad'),
            'sensitivities': data.get('sensitivities'),
            'receptor_types': data.get('receptor_types'),
            'eye_id': data.get('eye_id'),
        }
        args.update(kwargs)
        return cls(**args)

    # Overrides and internal helpers

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"<ReceptorArray(lenses={self.lens_count}, R={self.receptor_count}, total={len(self.data)})>"

    def _prepare_param(self, param, name="param"):
        arr = np.asarray(param, dtype=np.float32)
        if arr.ndim == 0:
            return np.full(self.lens_count, arr.item())
        if arr.ndim == 1 and len(arr) == self.lens_count:
            return arr
        raise ValueError(
            f"'{name}' shape invalid. Need scalar or length-{self.lens_count}.")

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
        """List of Eye views for all eye_ids present."""
        unique = np.unique(self.data['packed_data'][::self.receptor_count] & 0x07)
        return [self.eye(int(eid)) for eid in unique]

    @property
    def eye_ids(self) -> np.ndarray:
        return np.unique(self.data['packed_data'][::self.receptor_count] & 0x07)

    def ommatidium(self, lens_index: int) -> Ommatidium:
        """Global lens index -> Ommatidium group view."""
        return Ommatidium(self, lens_index)

    def cartridge(self, lens_index: int) -> Cartridge:
        """Global lens index -> Cartridge (neural superposition unit)."""
        return Cartridge(self, lens_index)

    @property
    def kernel(self) -> Optional[RhabdomereKernel]:
        return self._kernel

    @property
    def bundle_orientation(self) -> np.ndarray:
        """Bundle orientation (chi), per lens."""
        return self._bundle_orientation

    @property
    def is_full_model(self) -> bool:
        """True if built with a rhabdomere kernel (R > 1)."""
        return self._kernel is not None

    @property
    def interommatidial_angles_rad(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._ioa_minor_rad, self._ioa_major_rad

    # Neural superposition wiring

    def build_cartridge_map(self) -> np.ndarray:
        """
        For each lens, compute which neighbour's receptor points at this lens's optical axis.
        Returns (N, R_outer) array of global lens indices, where R_outer = min(receptor_count, 6) (R1-R6 only)
        """

        if self.receptor_count < 2:
            warnings.warn("Cartridge map requires a full model (R > 1).")
            return np.zeros((self.lens_count, 0), dtype=np.intp)

        N = self.lens_count
        R = self.receptor_count
        R_outer = min(R, 6)
        lens_dirs = self._lens_directions

        cartridge = np.zeros((N, R_outer), dtype=np.intp)

        for k in range(R_outer):
            # Directions of receptor type k across all lenses
            type_k_dirs = self.data['direction'][k::R, :3]
            tree = KDTree(type_k_dirs)
            _, indices = tree.query(lens_dirs)
            cartridge[:, k] = indices

        self._cartridge_map = cartridge
        return cartridge

    @property
    def cartridge_global_indices(self) -> np.ndarray:
        # TODO: rename this
        """
        Returns (N, R_outer) array of global receptor indices (for neural superposition).
        """
        if self._cartridge_map is None:
            self.build_cartridge_map()

        R = self.receptor_count
        R_outer = self._cartridge_map.shape[1]

        type_offsets = np.arange(R_outer)

        return self._cartridge_map * R + type_offsets

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

        if self._kernel is None:
            raise RuntimeError("Actuation requires a full model (use from_build).")

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
        cos_chi = np.cos(chi)[:, np.newaxis]
        sin_chi = np.sin(chi)[:, np.newaxis]

        # Actuation direction: chi + kernel intrinsic angle (main_axis + saccade offset)
        act_angle = chi + np.radians(kernel.actuation_angle_deg)
        cos_act = np.cos(act_angle)[:, np.newaxis]
        sin_act = np.sin(act_angle)[:, np.newaxis]

        # Effective nodal distance after axial contraction
        d_eff = d_rest - axi  # (n_act,)
        d_eff = np.maximum(d_eff, 1.0)  # clamp to 1 μm minimum

        # Rotate kernel offsets by chi then add lateral displacement along actuation axis
        rot_dx = cos_chi * dx[np.newaxis, :] - sin_chi * dy[np.newaxis, :]
        rot_dy = sin_chi * dx[np.newaxis, :] + cos_chi * dy[np.newaxis, :]
        rot_dx += lat[:, np.newaxis] * cos_act
        rot_dy += lat[:, np.newaxis] * sin_act

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
        new_dirs /= norms

        new_origins = self._lens_positions[lens_mask, np.newaxis, :] + world_tip

        # build global receptor indices for all affected lenses
        receptor_indices = (
            lens_mask[:, np.newaxis] * R + np.arange(R)[np.newaxis, :]
        ).ravel()  # (n_act * R,)

        self.data['direction'][receptor_indices, :3] = new_dirs.reshape(-1, 3)
        self.data['direction'][receptor_indices, 3] = 0.0
        self.data['origin'][receptor_indices, :3] = new_origins.reshape(-1, 3)
        self.data['origin'][receptor_indices, 3] = 1.0

        # Also change acceptance angles for any with axial displacement
        has_axial = np.any(axi != 0.0)

        if has_axial:
            wavelength_um = self._wavelength_nm * 1e-3
            diffraction = wavelength_um / kernel.lens_diameter_um
            geometric = kernel.diameters_um[np.newaxis, :] / d_eff[:, np.newaxis]
            new_acc = np.sqrt(diffraction ** 2 + geometric ** 2)  # (n_act, R)

            self.data['acceptance_angles'][receptor_indices, 0] = new_acc.ravel()
            self.data['acceptance_angles'][receptor_indices, 1] = new_acc.ravel()

        self.dirty_mask[receptor_indices] = True

        self._stale_receptor_spatial = True

    # Spatial structures: 2 levels tracked independently
    #
    # Lens-level (_stale_lens_spatial):
    #   - Eye KDTrees (built from _lens_directions / _lens_origins)
    #   - Eye neighbour graphs
    #   - Only invalidated by scale(), translate(), or direct lens mutation
    #   - Not invalidated by actuate() (lens axes are immutable after construction)
    #
    # Receptor-level (_stale_receptor_spatial):
    #   - Global receptor KDTrees (built lazily from data['direction'] / data['origin'])
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
            self._kdtree_directions = KDTree(self.data['direction'][:, :3])
            self._stale_receptor_spatial = False  # directions rebuilt

        return self._kdtree_directions

    def _ensure_global_kdtree_positions(self):
        """Lazy build of receptor-level position KDTree."""

        if self._stale_receptor_spatial or self._kdtree_positions is None:
            self._kdtree_positions = KDTree(self.data['origin'][:, :3])

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

        desired = q[:, np.newaxis, :] - self.data['origin'][:, :3][np.newaxis, :, :]

        norms = np.linalg.norm(desired, axis=-1, keepdims=True)
        np.divide(desired, norms, out=desired, where=norms != 0)
        dots = np.einsum('jk,ijk->ij', self.data['direction'][:, :3], desired)

        part = np.argpartition(dots, -k, axis=1)[:, -k:]
        top = np.take_along_axis(dots, part, axis=1)

        order = np.argsort(top, axis=1)[:, ::-1]
        best = np.take_along_axis(part, order, axis=1)

        if is_single and k == 1:
            return best.item()

        return best.squeeze()

    def query_cone(self, center_direction: ArrayLike, angle: float, degrees: bool = True) -> np.ndarray:
        """
        Find all receptors within angle of a center direction. Global indices.
        """

        kd = self._ensure_global_kdtree_directions()
        c = np.asarray(center_direction, dtype=np.float32)
        c /= np.linalg.norm(c)
        a = np.deg2rad(angle) if degrees else angle

        return kd.query_ball_point(c, r=2.0 * np.sin(a / 2.0))

    def query_ball(self, center_position: ArrayLike, radius: float) -> np.ndarray:
        """
        Find all receptors within radius of a center position. Global indices.
        """

        kd = self._ensure_global_kdtree_positions()
        c = np.asarray(center_position, dtype=np.float32)
        return kd.query_ball_point(c, r=radius)

    def max_gap(self) -> float:
        """
        Largest angular gap between any receptor and its nearest neighbour.
        """

        if len(self.data) <= 1:
            return 0.0

        kd = self._ensure_global_kdtree_directions()
        d, _ = kd.query(self.data['direction'][:, :3], k=2)
        return float(np.arccos(np.clip(1.0 - (np.max(d[:, 1]) ** 2) / 2.0, -1, 1)))

    # Whole-array transforms (initial unit scaling, agent setup, etc)

    def scale(self, factor: float):
        """
        Scale all receptor origins by a factor.
        """

        self.data['origin'][:, :3] *= factor
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
        Translate all receptor origins by a vector.
        """
        v = np.asarray(vector, dtype=np.float32)

        self.data['origin'][:, :3] += v

        self._lens_positions += v
        self._stale_receptor_spatial = True
        self._stale_lens_spatial = True
        self._kdtree_directions = None
        self._kdtree_positions = None
        self._resolve_lens_spatial()

        return self
