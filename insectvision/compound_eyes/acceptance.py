"""
Acceptance models: pluggable providers of the per-receptor rest Δρ field.

An 'AcceptanceModel' is any callable that, given per-lens optics and
per-receptor anatomy, returns the (N, R, 2) rest half-widths (minor, major) in radians.

    - SnyderAcceptance   consumes focal length + aperture (physical Δρ)
    - SamplingAcceptance consumes the lattice IOA and a dimensionless ratio (eye parameter p)
                         Δρ = p · Δφ
    - ExplicitAcceptance consumes a supplied array directly
    - MatchedAcceptance  derives p from the optics, then samples with it
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike


# Inputs an acceptance model may read

@dataclass(frozen=True)
class LensOptics:
    """Per-lens quantities available to an acceptance model. All shape (N,)."""
    focal_um: np.ndarray          # per-lens focal length (already facet-scaled)
    lens_diameter_um: np.ndarray  # per-lens aperture
    ioa_minor: np.ndarray         # interommatidial angle, minor axis (rad)
    ioa_major: np.ndarray         # interommatidial angle, major axis (rad)

    @property
    def n_lenses(self) -> int:
        return int(self.focal_um.shape[0])


@dataclass(frozen=True)
class ReceptorOptics:
    """Per-receptor anatomy available to an acceptance model. All shape (R,)."""
    rhab_diameter_um: np.ndarray  # waveguide diameter
    wavelength_um: np.ndarray     # peak wavelength (micrometres)

    @property
    def n_receptors(self) -> int:
        return int(self.rhab_diameter_um.shape[0])


# interface

@runtime_checkable
class AcceptanceModel(Protocol):
    """A provider of per-receptor rest acceptance half-widths.

    Returns an (N, R, 2) float32 array of half-widths (minor, major) in radians,
    where N == lens.n_lenses and R == rcpt.n_receptors.
    """

    def __call__(self, lens: LensOptics, rcpt: ReceptorOptics) -> np.ndarray:
        ...


# Concrete models

@dataclass(frozen=True)
class SnyderAcceptance:
    """Δρ from diffraction + geometric blur (Snyder optics).

        rho_geom = arctan(d_rhab / f)
        rho_diff = lambda / D
        rho      = sqrt(rho_geom^2 + rho_diff^2)

    Evaluated at the resting focal length, 'rho' is the dark-resting optical Δρ
    (the widest optical state).
    Microsaccades narrow it toward rho * bundle.narrowing_ratio at full
    drive.

    Isotropic: minor == major; per-axis RF anisotropy comes from the lattice.

    Consumes: focal_um, lens_diameter_um, rhab_diameter_um, wavelength_um
    """

    def __call__(self, lens: LensOptics, rcpt: ReceptorOptics) -> np.ndarray:
        f = np.clip(lens.focal_um, 1e-6, None)[:, None]
        D = np.clip(lens.lens_diameter_um, 1e-6, None)[:, None]
        d = rcpt.rhab_diameter_um[None, :]
        lam = rcpt.wavelength_um[None, :]

        rho_geom = np.arctan(d / f)
        rho_diff = lam / D
        rho = np.sqrt(rho_geom ** 2 + rho_diff ** 2).astype(np.float32)
        return np.repeat(rho[..., None], 2, axis=-1)


@dataclass(frozen=True)
class SamplingAcceptance:
    """Anatomical convention Δρ = p · Δφ against the local sampling lattice.

    ratio (eye parameter p = Δρ/Δφ) is the single source for the anatomical ratio.

    Lattice anisotropy (ioa_minor vs. ioa_major) supplies the per-axis RF anisotropy.
    The per-receptor rhabdomere diameter scales it so smaller rhabdomeres get proportionally smaller RFs.

    Consumes ioa_minor, ioa_major, rhab_diameter_um, and its own 'ratio'.
    Independent of focal length, so it works with or without optics.
    """
    ratio: float = 1.5

    def __call__(self, lens: LensOptics, rcpt: ReceptorOptics) -> np.ndarray:
        d = rcpt.rhab_diameter_um.astype(np.float32)
        max_d = float(d.max()) if d.size else 1.0
        rel_d = (d / max_d) if max_d > 0 else np.ones_like(d)

        amin = self.ratio * lens.ioa_minor[:, None] * rel_d[None, :]
        amaj = self.ratio * lens.ioa_major[:, None] * rel_d[None, :]
        return np.stack([amin, amaj], axis=-1).astype(np.float32)


@dataclass(frozen=True, eq=False)    # eq=False because the field is an ndarray and array equality is not scalar
class ExplicitAcceptance:
    """Use a supplied array of half-widths verbatim.
    Accepts (R,), (N, R), or (N, R, 2), broadcast to (N, R, 2).
    """
    values_rad: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "values_rad",
                           np.asarray(self.values_rad, dtype=np.float32))

    def __call__(self, lens: LensOptics, rcpt: ReceptorOptics) -> np.ndarray:
        return broadcast_to_nr2(self.values_rad, lens.n_lenses, rcpt.n_receptors)


@dataclass(frozen=True)
class MatchedAcceptance:
    """Derive p from the optics, then apply it sampling-style.

    Computes the Snyder Δρ, reads out a single ``p = Δρ/Δφ`` from the median lens,
    then returns Δρ = p · Δφ everywhere. This is an example of *deriving* one
    parameter from others.

    Consumes everything SnyderAcceptance does, + the lattice IOA
    """

    def __call__(self, lens: LensOptics, rcpt: ReceptorOptics) -> np.ndarray:
        snyder = SnyderAcceptance()(lens, rcpt)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_ioa = 1.0 / np.where(lens.ioa_minor > 1e-9, lens.ioa_minor, np.nan)
            p_per_lens = snyder[..., 0].mean(axis=1) * inv_ioa
        p = float(np.nanmedian(p_per_lens))
        if not np.isfinite(p):
            p = 1.5
        return SamplingAcceptance(ratio=p)(lens, rcpt)


# Helpers

# TODO: Make this a more general broadcast_to(value, n, name) helper and use it elsewere
def broadcast_to_nr2(values: ArrayLike, n_lenses: int, n_receptors: int) -> np.ndarray:
    """Broadcast (R,), (N, R), or (N, R, 2) to a contiguous (N, R, 2) float32 array."""
    arr = np.asarray(values, dtype=np.float32)
    N, R = n_lenses, n_receptors

    if arr.shape == (R,):
        out = np.broadcast_to(arr[None, :, None], (N, R, 2))
    elif arr.shape == (N, R):
        out = np.broadcast_to(arr[..., None], (N, R, 2))
    elif arr.shape == (N, R, 2):
        out = arr
    else:
        raise ValueError(
            f"acceptance values must be shape ({R},), ({N}, {R}), or ({N}, {R}, 2); "
            f"got {arr.shape}"
        )
    return np.ascontiguousarray(out, dtype=np.float32)


def report_acceptance(
    acc: ArrayLike,
    lens: LensOptics,
    *,
    peripheral_indices: Optional[Sequence[int]] = None,
    narrowing_ratio: float = 1.0,
    reference_ratio: Optional[float] = None,
) -> str:
    """Human-readable diagnostic for any acceptance result. Model-agnostic.

    Reports the resting Δρ (minor axis), its narrowed/moved value, and the
    *measured* sampling ratio p = Δρ/Δφ for the peripheral and full receptor
    pools. p here is an output, not an input -- compare it to whatever reference
    you pass.

    Args:
        - acc: (N, R, 2) half-widths in radians (the model's output).
        - lens: the LensOptics used to produce ``acc`` (for ioa_minor).
        - peripheral_indices: receptor indices for the peripheral pool, if any.
        - narrowing_ratio: full-drive narrowing factor, for the moved Δρ line.
        - reference_ratio: an anatomical p to print alongside, if desired.
    """
    acc = np.asarray(acc, dtype=np.float32)
    rho = acc[..., 0]                              # minor half-width, (N, R)
    ioa_minor = np.asarray(lens.ioa_minor, dtype=np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_ioa = 1.0 / np.where(ioa_minor > 1e-9, ioa_minor, np.nan)
        p_all = rho.mean(axis=1) * inv_ioa
        if peripheral_indices is not None and len(peripheral_indices):
            p_periph = rho[:, list(peripheral_indices)].mean(axis=1) * inv_ioa
        else:
            p_periph = p_all

    def _med(a: np.ndarray) -> float:
        a = np.asarray(a)
        return float(np.nanmedian(a)) if np.isfinite(a).any() else float("nan")

    rho_deg = np.degrees(rho)
    lines = [
        "Rest acceptance (minor axis):",
        f"    Δρ rest : median {np.nanmedian(rho_deg):.2f} deg "
        f"(spread {np.nanmin(rho_deg):.2f}-{np.nanmax(rho_deg):.2f} deg) "
        f"-> moved x{narrowing_ratio:.2f}: median "
        f"{np.nanmedian(rho_deg) * narrowing_ratio:.2f} deg",
        f"    measured p = Δρ/Δφ : peripheral median {_med(p_periph):.2f}, "
        f"all-cell median {_med(p_all):.2f}",
    ]
    if reference_ratio is not None:
        lines.append(f"    (reference anatomical p ~ {reference_ratio:.2f})")
    return "\n".join(lines)

