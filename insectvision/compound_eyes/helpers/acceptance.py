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
from typing import Protocol,runtime_checkable
import numpy as np

from insectvision.utils.shared import broadcast_to_shape


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


@dataclass(frozen=True)
class RhabdomereOptics:
    """
    Container for per-rhabdomere anatomy available to an acceptance model.
    All shape (R,).
    """
    diameter_um: np.ndarray     # waveguide diameter (μm)
    wavelength_um: np.ndarray   # peak wavelength (μm)

    @property
    def nb_rhabdomeres(self) -> int:
        return int(self.diameter_um.shape[0])


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
    Microsaccades narrow it toward rho * bundle.narrowing_ratio at full
    drive.

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
    Accepts (R,), (N, R), or (N, R, 2), broadcast to (N, R, 2).
    """
    values_rad: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, 'values_rad',
                           np.asarray(self.values_rad, dtype=np.float32))

    def __call__(self, lens_optics: LensOptics, rhab_optics: RhabdomereOptics) -> np.ndarray:
        n, r = lens_optics.nb_lenses, rhab_optics.nb_rhabdomeres
        return broadcast_to_shape(
            values=self.values_rad,
            shape=(n, r, 2),
            accepted=[((r,), (1,)), ((n, r), (0, 1)), ((n, r, 2), (0, 1, 2))],
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