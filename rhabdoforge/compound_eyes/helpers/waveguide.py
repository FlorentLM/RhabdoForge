"""
Waveguide optics of the facet lens.

Equation numbers refer to:

Stavenga, "Angular and spectral sensitivity of fly photoreceptors. I. Integrated facet lens and rhabdomere optics" (2003), 10.1007/s00359-002-0370-2
Stavenga, "Angular and spectral sensitivity of fly photoreceptors. III. Dependence on the pupil mechanism in the blowfly Calliphora" (2004), 10.1007/s00359-003-0477-0
"""
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Tuple, List, Iterator, Optional
import numpy as np
from scipy.optimize import brentq
from scipy.special import jv, kv, jn_zeros

if TYPE_CHECKING:
    from rhabdoforge.compound_eyes.helpers.acceptance import LensOptics, RhabdomereOptics



# Quadrature resolution for the excitation integral and the angular sweep
QUAD_NODES = 256
SWEEP_NODES = 96
SWEEP_MAX_D = 6.0      # Angular sweep limit (in units of d/b)
CUTOFF_EPS = 1e-4      # V-number margin below which a mode is treated as unbound

# Nodes per axis for the (F-number, focal length) interpolation in WaveguideAcceptance
INTERP_NODES = 6
GRID_RTOL = 1e-6        # For float32 noise

# Precomputed Gaussian quadrature nodes and weights on [-1, 1]
_GL_X, _GL_WT = np.polynomial.legendre.leggauss(QUAD_NODES)



@dataclass(frozen=True)
class LPMode:
    p: int      # Stavenga's absolute mode index p (sorted by cutoff)
    l: int
    m: int
    cutoff: float
    asymptote: float


class LPModes:
    """
    Grid and catalog of LP modes.

    Column m=0 is filled with NaN so array indices match mode numbers
    """

    def __init__(self, l_max: int = 10, m_max: int = 6):
        self.l_max = l_max
        self.m_max = m_max

        self.cutoffs = np.full((l_max + 1, m_max + 1), np.nan)
        self.asymptotes = np.full((l_max + 1, m_max + 1), np.nan)
        self.p_numbers = np.full((l_max + 1, m_max + 1), -1, dtype=int)

        self._compute_brackets()
        self._assign_p_numbers()

        valid_cutoffs = self.cutoffs[~np.isnan(self.cutoffs)]
        self.sorted_cutoffs = np.sort(valid_cutoffs)

    def _compute_brackets(self) -> None:
        """
        Zeros of Bessel J_n(x).
        """
        for l in range(self.l_max + 1):
            self.asymptotes[l, 1:] = jn_zeros(l, self.m_max)

            if l == 0:
                self.cutoffs[0, 1] = 0.0
                if self.m_max > 1:
                    self.cutoffs[0, 2:] = jn_zeros(1, self.m_max - 1)
            else:
                self.cutoffs[l, 1:] = jn_zeros(l - 1, self.m_max)

    def _assign_p_numbers(self) -> None:
        """
        Stavenga's absolute mode index p (sorted by cutoff).
        """
        modes = [
            (self.cutoffs[l, m], l, m)
            for l in range(self.l_max + 1) for m in range(1, self.m_max + 1)
        ]
        modes.sort(key=lambda x: x[0])

        for p_idx, (_, l, m) in enumerate(modes, start=1):
            self.p_numbers[l, m] = p_idx

    def bracket(self, l: int, m: int) -> Tuple[float, float]:
        """
        Returns (cutoff, asymptote) for LP(l, m).
        """
        return float(self.cutoffs[l, m]), float(self.asymptotes[l, m])

    def __getitem__(self, key: Tuple[int, int]) -> LPMode:

        l, m = key
        if not (0 <= l <= self.l_max and 1 <= m <= self.m_max):
            raise IndexError(f'Mode LP({l},{m}) out of bounds.')

        return LPMode(
            p=int(self.p_numbers[l, m]),
            l=l,
            m=m,
            cutoff=float(self.cutoffs[l, m]),
            asymptote=float(self.asymptotes[l, m]),
        )

    def up_to(self, cutoff_max: float) -> List[LPMode]:
        """
        Return every LP mode with cutoff <= 'cutoff_max'
        (sorted by cutoff)
        """
        mask = (~np.isnan(self.cutoffs)) & (self.cutoffs <= cutoff_max)
        return [self[int(l), int(m)] for l, m in zip(*np.where(mask))]

    def __iter__(self) -> Iterator[LPMode]:
        return iter(self.up_to(np.nanmax(self.cutoffs)))


LP_modes = LPModes(l_max=10, m_max=6)


# Solver

def solve_uw(v_number: float, l: int, m: int) -> Tuple[float, float]:
    """
    Roots U, W of the LP characteristic equation (Eq. 17), for mode LP(l,m) at a given V.
    """

    cutoff, hi = LP_modes.bracket(l, m)

    if v_number - cutoff < CUTOFF_EPS:
        # excited power -> 0 at cut-off (Eq. 34's W/V factor), so skipping
        raise ValueError(f'LP({l},{m}) is at or below cut-off at V={v_number:.6f}')

    lo = max(cutoff, 1e-9)
    hi = min(hi, v_number) - 1e-9
    if hi <= lo:
        raise ValueError(f'LP({l},{m}) is not bound at V={v_number:.6f}')

    def residual(u: float) -> float:
        w = math.sqrt(max(v_number ** 2 - u ** 2, 1e-18))
        return u * jv(l + 1, u) / jv(l, u) - w * kv(l + 1, w) / kv(l, w)

    if np.sign(residual(lo)) == np.sign(residual(hi)):
        raise ValueError(f'LP({l},{m}) does not bracket a root at V={v_number:.6f}')

    u = float(brentq(residual, lo, hi, xtol=1e-12, rtol=1e-14))
    return u, math.sqrt(max(v_number ** 2 - u ** 2, 0.0))


# Helpers


def mode_profile(r: np.ndarray, u: float, w: float, l: int) -> np.ndarray:
    """
    Radial field M_p(R) of a mode, R normalised to the rhabdomere radius (Eqs. 15a, 15b).
    """
    r = np.asarray(r, dtype=float)

    # Continuous at R = 1 (core/boundary-wave crossover)
    return np.where(r <= 1.0, jv(l, u * r), jv(l, u) / kv(l, w) * kv(l, w * np.maximum(r, 1e-12)))


def mode_profile_2d(x: np.ndarray, y: np.ndarray, u: float, w: float, l: int, orientation: str = 'cos') -> np.ndarray:
    """
    2D field M_p(R) * angular(l*theta) of a mode, on a Cartesian grid (x, y in rhabdomere radii),
    normalised to unit total power (Eqs. 14, 15a, 15b).
    """

    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    angular = np.cos(l * theta) if orientation == 'cos' else np.sin(l * theta)
    field = mode_profile(r, u, w, l) * angular

    dx = float(x[0, 1] - x[0, 0]) if x.ndim == 2 else float(x[1] - x[0])
    dy = float(y[1, 0] - y[0, 0]) if y.ndim == 2 else float(y[1] - y[0])

    # l > 0 modes are degenerate in cos and sin, so we sum them in power (not amplitude)
    power = np.sum(field ** 2) * dx * dy
    if power <= 0.0:
        raise ValueError(f'LP({l}) {orientation} field has no power on this grid')

    return field / np.sqrt(power)


def power_in_boundary(u: float, w: float, v_number: float, l: int) -> float:
    """
    Fraction of a mode's power travelling inside the rhabdomere boundary (Eq. 25).
    """
    denom = jv(l - 1, u) * jv(l + 1, u)  # negative for bound LP modes
    return float((w ** 2 / v_number ** 2) * (1.0 - jv(l, u) ** 2 / denom))


def outside_fraction(u: float, w: float, v_number: float, l: int, s_over_b: float) -> float:
    """
    Fraction of a mode's power beyond radius S = s/b, S >= 1 (Stavenga 2004/III Eq. A15).
    """

    def power_beyond(s: float) -> float:
        kl, klm1, klp1 = kv(l, w * s), kv(l - 1, w * s), kv(l + 1, w * s)
        return (jv(l, u) / kv(l, w)) ** 2 * (s ** 2 / 2.0) * (klm1 * klp1 - kl ** 2)

    # At S = 1, fraction is:  1 - power_in_boundary  (by construction)
    outside_at_1 = 1.0 - power_in_boundary(u, w, v_number, l)

    return float(outside_at_1 * power_beyond(s_over_b) / power_beyond(1.0))


def g_factor(x: np.ndarray, u: float, w: float, v_number: float, l: int) -> np.ndarray:
    """
    Radial coupling factor G(X) of the excitation integral (Eqs. 35a, 35b).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        num = x * jv(l, u) * jv(l + 1, x) - u * jv(l, x) * jv(l + 1, u)
        g = v_number ** 2 * num / ((x ** 2 - u ** 2) * (x ** 2 + w ** 2))

    g_at_u = 0.5 * (jv(l, u) ** 2 - jv(l - 1, u) * jv(l + 1, u))  # removable singularity at X=U
    return np.where(np.abs(x - u) < 1e-7, g_at_u, g)


def pupil_transmittance(u: float, w: float, v_number: float, l: int, b_um: float,
                        h_um: float, m_s: float = 50.0, a_s: float = 1.0) -> float:
    """
    Transmittance of a mode through the pupil mechanism (Stavenga 2004/III Eqs. A15, A16):

        T_p = exp(-m_s * a_s(lambda) * outside_fraction(s = b + h))

    h = inf: fully dark-adapted (pigment at infinity) -> T=1
    h = 0  : fully light-adapted (pigment right at the boundary)

    a_s    : pigment's absorption spectrum (Fig. 7) # TODO: digitise a_s instead of defaulting to 1.0?
    """
    return float(np.exp(-m_s * a_s * outside_fraction(u, w, v_number, l, 1.0 + h_um / b_um)))



@dataclass(frozen=True)
class RhabdomereModes:
    """
    Bound LP modes of one rhabdomere, their corresponding angular sensitivity.
    """

    v_number: float
    mode_p: np.ndarray          # Mode numbers (1-based indexing)
    u: np.ndarray
    w: np.ndarray
    eta: np.ndarray             # Fraction of the power inside the boundary (per mode)
    t_p: np.ndarray             # Pupil transmittance (per mode). 1.0 when dark-adapted
    d_half: float               # Angular half-width (dimensionless D=d/b units)
    d_sweep: np.ndarray         # Offsets the sensitivity was sampled at (dimensionless D=d/b units)
    sensitivity: np.ndarray     # Summed angular sensitivity (normalised to the peak)
    peak_power: float           # Peak pre-normalisation (for comparing across pupil states)

    @property
    def nb_modes(self) -> int:
        return int(self.mode_p.size)


def solve_modes(
        v_number: float,
        f_number: float,
        diameter_um: float,
        wavelength_um: float,
        h_um: float = np.inf,
        defocus_um: float = 0.0,
        focal_um: Optional[float] = None,
        n_image: float = 1.34,
    ) -> RhabdomereModes:
    """
    Enumerate the bound modes at 'v_number' and the angular sensitivity of their sum,
    against D = d/b (the beam's offset at the rhabdomere entrance, in rhabdomere radii)

    Note: 'defocus_um' is the tip's signed distance from the image focal plane (negative = lens
    side), and needs 'focal_um' (the object focal length).

    At 0.0, the tip is in the focal plane, Eq. 34's phase term vanishes, and the result depends on
    the lens only through its F-number.
    """

    bound = LP_modes.up_to(v_number - CUTOFF_EPS)  # all modes above cutoff
    if not bound:
        raise ValueError(f'no bound modes at V={v_number:.4f}')

    if defocus_um != 0.0 and focal_um is None:
        raise ValueError('focal_um is required when defocus_um is non-zero')

    b = 0.5 * diameter_um

    # delta is the low-Fresnel-number correction to K' (Eqs. 7e, 34)
    delta = 1.0 if defocus_um == 0.0 else 1.0 + defocus_um / (n_image * focal_um)

    # X_max = K'*rho_0 = pi*b*D_l/(lambda*f*delta) = pi*b/(lambda*F*delta) (Eqs. 4, 32)
    x_max = np.pi * b / (wavelength_um * f_number * delta)

    # Shift precomputed nodes to [0, x_max]
    x = 0.5 * x_max * (_GL_X + 1.0)
    wt = 0.5 * x_max * _GL_WT

    # Eq. 34's exp(i*Z*X^2 / 2K'), with Z = z/b and K' = 2*pi*n'*b / (lambda*delta)
    if defocus_um == 0.0:
        phase = 1.0
    else:
        phase = np.exp(1j * defocus_um * wavelength_um * delta * x ** 2
                       / (4.0 * np.pi * n_image * b ** 2))

    d_sweep = np.linspace(0.0, SWEEP_MAX_D, SWEEP_NODES)

    u_all, w_all, eta_all, t_all, p_all = [], [], [], [], []
    sensitivity = np.zeros_like(d_sweep)

    for mode in bound:
        l, m = mode.l, mode.m

        try:
            u, w = solve_uw(v_number, l, m)
        except ValueError:
            continue

        eta = power_in_boundary(u, w, v_number, l)
        t_p = 1.0 if not np.isfinite(h_um) else pupil_transmittance(u, w, v_number, l, b, h_um)

        g = g_factor(x, u, w, v_number, l)

        # Excitation coefficient (Eq. 34), tip in the focal plane (Z=0, van Hateren 1984)
        c_p = 2.0 if l == 0 else 1.0
        norm = (w / v_number) * math.sqrt(2.0 / (c_p * abs(jv(l - 1, u) * jv(l + 1, u))))

        # Weight each mode:
        #     P_eff = P_exc * T_p(h) * eta_p (Stavenga 2004/III Eq. A19)
        #   -> higher-order modes drop out first
        integral = (jv(l, np.outer(d_sweep, x)) * (g * x * wt * phase)).sum(axis=1)  # (sweep, quad) -> (sweep,)
        sensitivity += t_p * eta * norm ** 2 * np.abs(integral) ** 2

        u_all.append(u), w_all.append(w), eta_all.append(eta), t_all.append(t_p), p_all.append(mode.p)

    if not p_all:
        raise ValueError(f'no bound modes converged at V={v_number:.4f}')

    peak = float(sensitivity.max())
    if peak <= 0.0:
        raise ValueError(f'pupil absorbs all power at V={v_number:.4f}, h={h_um}')

    return RhabdomereModes(
        v_number=v_number,
        mode_p=np.asarray(p_all, dtype=np.int32),
        u=np.asarray(u_all),
        w=np.asarray(w_all),
        eta=np.asarray(eta_all),
        t_p=np.asarray(t_all),
        d_half=half_width(d_sweep, sensitivity),
        d_sweep=d_sweep,
        sensitivity=sensitivity / peak,
        peak_power=peak,
    )


def half_width(d_sweep: np.ndarray, sensitivity: np.ndarray) -> float:
    """
    FWHM of the angular sensitivity measured from its peak (not the on-axis value
    because a multi-mode profile can be double-peaked off-axis)
    """

    peak = sensitivity.max()
    if peak <= 0.0:
        raise ValueError('angular sensitivity has no power')

    normalised = sensitivity / peak
    below = np.flatnonzero(normalised < 0.5)
    if below.size == 0:
        raise ValueError('sensitivity does not fall to half max within the sweep range')

    i = int(below[0])
    y0, y1 = normalised[i - 1], normalised[i]
    t = (y0 - 0.5) / max(y0 - y1, 1e-12)
    return float(d_sweep[i - 1] + t * (d_sweep[i] - d_sweep[i - 1]))


# TODO: Move these three in utils probably

def to_angle(d_half, diameter_um, focal_um):
    """
    Convert a half-width in D = d/b units to an acceptance angle (rad, FWHM): d = f*tan(theta)
    at the rhabdomere entrance (Eq. 31, tip in the focal plane).
    """
    return 2.0 * np.arctan(d_half * 0.5 * diameter_um / np.asarray(focal_um))


def _interp_grid(values: np.ndarray, nodes: int = INTERP_NODES) -> np.ndarray:
    """
    Interpolation nodes over 'values'.
    An axis whose values are all equal collapses to a single node,
    so a quantity that doesn't vary across the eye costs one solve.
    """
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo <= GRID_RTOL * max(abs(lo), abs(hi), 1.0):
        return np.array([0.5 * (lo + hi)])
    return np.linspace(lo, hi, nodes)


def _bilinear(x_grid: np.ndarray, y_grid: np.ndarray, table: np.ndarray,
              x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Interpolate 'table', shape (len(x_grid), len(y_grid)), at the points (x, y).
    Either grid may be a single node. Queries outside the grid are clamped to the edge.
    """
    if y_grid.size == 1:
        rows = np.repeat(table[:, :1], x.size, axis=1)
    else:
        rows = np.stack([np.interp(y, y_grid, table[i]) for i in range(x_grid.size)])

    if x_grid.size == 1:
        return rows[0]

    i = np.clip(np.searchsorted(x_grid, x) - 1, 0, x_grid.size - 2)
    t = np.clip((x - x_grid[i]) / (x_grid[i + 1] - x_grid[i]), 0.0, 1.0)
    cols = np.arange(x.size)

    return rows[i, cols] * (1.0 - t) + rows[i + 1, cols] * t


@lru_cache(maxsize=8192)
def _solve_scalar(attr: str, v_number: float, f_number: float, diameter_um: float,
                  wavelength_um: float, h_um: float, defocus_um: float, focal_um: float) -> float:
    """
    One scalar attribute of a mode solve (baking passes share the rest state so it's lru-cached).
    """
    modes = solve_modes(v_number, f_number, diameter_um, wavelength_um,
                        h_um=h_um, defocus_um=defocus_um, focal_um=focal_um)
    return float(getattr(modes, attr))


def solve_per_lens(attr: str, f_number: np.ndarray, focal_um: np.ndarray, *,
                   v_number: float, diameter_um: float, wavelength_um: float,
                   h_um: float = np.inf, defocus_um: float = 0.0) -> np.ndarray:
    """
    Per-lens value of a scalar 'RhabdomereModes' attribute (for one rhabdomere type).

    Solving per lens is way too slow, so solve on a grid spanning the (F-number, focal length)
    values the eye has and interpolate.
    """
    fn_grid = _interp_grid(f_number)
    f_grid = _interp_grid(focal_um)

    table = np.array([
        [_solve_scalar(attr, v_number, float(fn), diameter_um, wavelength_um,
                       h_um, defocus_um, float(fl))
         for fl in f_grid]
        for fn in fn_grid
    ])

    return _bilinear(fn_grid, f_grid, table, f_number, focal_um)



##


@dataclass(frozen=True)
class WaveguideAcceptance:
    """
    Δρ from the [facet lens - rhabdomere waveguide] overlap integral (Stavenga 2003a).

    Sums the excited power (inside the rhabdomere boundary) of all bound LP modes,
    and takes the half-width of the resulting angular sensitivity.

    Note: this is isotropic (minor == major), per-axis RF anisotropy comes from the lattice    # TODO: Decide if it's worth keeping
    """

    def __call__(self, lens_optics: 'LensOptics', rhab_optics: 'RhabdomereOptics') -> np.ndarray:

        f = np.clip(np.asarray(lens_optics.focal_um, dtype=np.float64), 1e-6, None)
        aperture = np.clip(np.asarray(lens_optics.aperture_um, dtype=np.float64), 1e-6, None)
        f_number = f / aperture

        d_rhab = np.asarray(rhab_optics.diameter_um, dtype=np.float64)
        v_number = np.asarray(rhab_optics.v_number, dtype=np.float64)
        wavelengths = np.asarray(rhab_optics.wavelength_um, dtype=np.float64)

        defocus = float(rhab_optics.defocus_um)

        rho = np.empty((f.size, d_rhab.size), dtype=np.float32)

        for r in range(d_rhab.size):
            d_val = float(d_rhab[r])

            d_half = solve_per_lens('d_half', f_number, f,
                                    v_number=float(v_number[r]),
                                    diameter_um=d_val,
                                    wavelength_um=float(wavelengths[r]),
                                    defocus_um=defocus)

            rho[:, r] = to_angle(d_half, d_val, f)

        return np.repeat(rho[..., None], 2, axis=-1)