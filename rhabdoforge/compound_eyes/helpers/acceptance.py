"""
Acceptance models: pluggable providers of the per-rhabdomere rest Δρ field.

An 'AcceptanceModel' is any callable that, given per-lens optics and per-rhabdomere anatomy,
returns the (N, R, 2) rest half-widths (minor, major) in radians.

    - SnyderAcceptance: consumes focal length + aperture (physical Δρ)
    - SamplingAcceptance: consumes the lattice IOA and a dimensionless ratio (eye parameter p):  Δρ = p · Δφ
    - ExplicitAcceptance: consumes a supplied array directly
    - MatchedAcceptance: derives p from the optics, then samples with it
"""
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable
import numpy as np

from rhabdoforge.compound_eyes.helpers.waveguide import LP_modes
from rhabdoforge.utils import broadcast_to_shape


# References:
#   [1] Seitz 1968: "Der Strahlengang im Appositionsauge von Calliphora erythrocephala (Meig.)", 10.1007/BF00339350
#   [2] Stavenga 1974: "Refractive index of fly rhabdomeres", 10.1007/BF00694471
#   [3] Beersma et al. 1982: "Refractive index of the fly rhabdomere", 10.1364/JOSA.72.000583
#   [4] Stavenga 2003a: "Angular and spectral sensitivity of fly photoreceptors. I. Integrated facet lens and rhabdomere optics", 10.1007/s00359-002-0370-2
#   [5] Stavenga 2003b: "Angular and spectral sensitivity of fly photoreceptors. II. Dependence on facet lens F-number and rhabdomere type in Drosophila", 10.1007/s00359-003-0390-6
#   [6] Kemppainen et al. 2022: "Binocular mirror–symmetric microsaccadic sampling enables Drosophila hyperacute 3D vision", 10.1073/pnas.2109717119

# TODO: Add Snyder to refs here

# Refractive indices of the rhabdomere interior and its surrounding medium:
#   Blowfly/housefly measurements (surround: [ref 1]; interior: [ref 2-3]),
#   considered in [ref 4] as roughly constant across flies.
#   Also used for Drosophila in [ref 5, 6].
#
# !! Not verified for non-flies !!
#
# Also: A small change here can change the mode count by a whole number...
N_SURROUND_FLY = 1.340
N_RHABDOMERE_FLY = 1.363




# Inputs an acceptance model may read

@dataclass(frozen=True)
class LensOptics:
    """
    Container for per-lens quantities available to an acceptance model.
    All shape (N,).
    """
    focal_um: np.ndarray          # focal length (scaled to the lens aperture) (μm)
    aperture_um: np.ndarray       # lens diameter (μm)
    ioa_minor: np.ndarray         # interommatidial angle, minor axis (rad)
    ioa_major: np.ndarray         # interommatidial angle, major axis (rad)

    @property
    def nb_lenses(self) -> int:
        return int(self.focal_um.shape[0])


@dataclass(frozen=True, eq=False)  # eq=False because the fields are ndarrays
class RhabdomereOptics:
    """
    Container for per-rhabdomere anatomy available to an acceptance model.
    All shape (R,).

    The refractive indices default to blowfly values and are only consumed by waveguide
    models (they set the numerical aperture, hence the V-number and mode count).
    """
    diameter_um: np.ndarray     # waveguide diameter (μm)
    wavelength_um: np.ndarray   # peak wavelength (μm)
    n_rhabdomere: Optional[np.ndarray] = None   # rhabdomere interior refractive index
    n_surround: Optional[np.ndarray] = None     # surrounding medium refractive index
    defocus_um: float = 0.0                     # rhabdomere tip's (signed) distance from the image focal plane

    def __post_init__(self):
        for name, default in (('n_rhabdomere', N_RHABDOMERE_FLY), ('n_surround', N_SURROUND_FLY)):
            value = getattr(self, name)
            value = default if value is None else value
            object.__setattr__(self, name, np.broadcast_to(
                np.asarray(value, dtype=np.float32), self.diameter_um.shape
            ))

    @property
    def nb_rhabdomeres(self) -> int:
        return int(self.diameter_um.shape[0])

    @property
    def numerical_aperture(self) -> np.ndarray:
        """Numerical Aperture (NA), the resolving power of the lens."""
        return np.sqrt(np.maximum(self.n_rhabdomere ** 2 - self.n_surround ** 2, 0.0))

    @property
    def v_number(self) -> np.ndarray:
        """
        Waveguide parameter V = (pi * D / lambda) * NA.
        (sets how many LP modes propagate)
        """
        lam = np.clip(self.wavelength_um, 1e-6, None)
        return (np.pi * self.diameter_um / lam) * self.numerical_aperture

    @property
    def nb_modes(self) -> np.ndarray:
        """Number of Linearly Polarised (LP) modes at the (rest) V-number."""
        return np.searchsorted(LP_modes.sorted_cutoffs, self.v_number, side='right').astype(np.int32)


# interface

@runtime_checkable
class AcceptanceModel(Protocol):
    """
    A provider of per-rhabdomere rest acceptance half-widths.

    Returns an (N, R, 2) float32 array of half-widths (minor, major) in radians,
    where N == lens.nb_lenses and R == rcpt.nb_rhabdomeres.
    """

    def __call__(self, lens_optics: LensOptics, rhab_optics: RhabdomereOptics) -> np.ndarray:
        ...


# Concrete models

@dataclass(frozen=True)
class SnyderAcceptance:
    """
    Δρ from diffraction + geometric blur (Snyder optics).

        rho_geom = arctan(d_rhab / f)
        rho_diff = lambda / D
        rho      = sqrt(rho_geom^2 + rho_diff^2)

    Evaluated at the resting focal length, 'rho' is the dark-resting optical Δρ
    (the widest optical state).
    Microsaccades clip it toward rho * bundle.clip_ratio at full drive.

    Isotropic: minor == major, per-axis RF anisotropy comes from the lattice.

    Consumes: focal_um, aperture_um, rhab_diameter_um, wavelength_um
    """

    def __call__(self, lens_optics: LensOptics, rhab_optics: RhabdomereOptics) -> np.ndarray:
        f = np.clip(lens_optics.focal_um, 1e-6, None)[:, None]
        D = np.clip(lens_optics.aperture_um, 1e-6, None)[:, None]
        d = rhab_optics.diameter_um[None, :]
        lam = rhab_optics.wavelength_um[None, :]

        rho_geom = np.arctan(d / f)
        rho_diff = lam / D
        rho = np.hypot(rho_geom, rho_diff).astype(np.float32)
        return np.repeat(rho[..., None], 2, axis=-1)


@dataclass(frozen=True)
class SamplingAcceptance:
    """
    Anatomical convention Δρ = p · Δφ against the local sampling lattice.

    ratio (eye parameter p = Δρ/Δφ) is the single source for the anatomical ratio.

    Lattice anisotropy (ioa_minor vs. ioa_major) supplies the per-axis RF anisotropy.
    The per-rhabdomere diameter scales it so smaller rhabdomeres get proportionally smaller RFs.

    Consumes ioa_minor, ioa_major, rhab_diameter_um, and its own 'ratio'.
    Independent of focal length, so it works with or without optics.
    """
    ratio: float = 1.0

    def __call__(self, lens_optics: LensOptics, rhab_optics: RhabdomereOptics) -> np.ndarray:
        d = rhab_optics.diameter_um.astype(np.float32)

        max_d = float(d.max()) if d.size else 1.0
        rel_d = (d / max_d) if max_d > 0 else np.ones_like(d)

        a_min = self.ratio * lens_optics.ioa_minor[:, None] * rel_d[None, :]
        a_maj = self.ratio * lens_optics.ioa_major[:, None] * rel_d[None, :]
        return np.stack([a_min, a_maj], axis=-1).astype(np.float32)


@dataclass(frozen=True, eq=False)  # eq=False because the field is a ndarray and array equality is not scalar
class ExplicitAcceptance:
    """
    Use a supplied array of half-widths verbatim.
    Accepts (N,), (R,), (N, R), or (N, R, 2), broadcast to (N, R, 2).
    """
    values_rad: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, 'values_rad',
                           np.asarray(self.values_rad, dtype=np.float32))

    def __call__(self, lens_optics: LensOptics, rhab_optics: RhabdomereOptics) -> np.ndarray:
        n, r = lens_optics.nb_lenses, rhab_optics.nb_rhabdomeres
        if n == r:
            print(f'Number of rhabdomeres and ommatidia are the same (is this a debug eye?). Broadcast of acceptance array might not work as expected.')

        return broadcast_to_shape(
            values=self.values_rad,
            shape=(n, r, 2),
            accepted=[((r,), (1,)), ((n,), (0,)), ((n, r), (0, 1)), ((n, r, 2), (0, 1, 2))],
            name='acceptance',
            dtype=np.float32
        )


@dataclass(frozen=True)
class MatchedAcceptance:
    """
    Derive p from the optics, then apply it sampling-style.

    Computes the Snyder Δρ, reads out a single 'p = Δρ/Δφ' from the median lens,
    then returns Δρ = p · Δφ everywhere. This is an example of *deriving* one
    parameter from others.

    Consumes everything SnyderAcceptance does, + the lattice IOA
    """

    def __call__(self, lens_optics: LensOptics, rhab_optics: RhabdomereOptics) -> np.ndarray:
        snyder = SnyderAcceptance()(lens_optics, rhab_optics)

        with np.errstate(divide='ignore', invalid='ignore'):
            inv_ioa = 1.0 / np.where(lens_optics.ioa_minor > 1e-9, lens_optics.ioa_minor, np.nan)
            p_per_lens = snyder[..., 0].mean(axis=1) * inv_ioa

        p = float(np.nanmedian(p_per_lens))
        if not np.isfinite(p):
            p = 1.0

        return SamplingAcceptance(ratio=p)(lens_optics, rhab_optics)