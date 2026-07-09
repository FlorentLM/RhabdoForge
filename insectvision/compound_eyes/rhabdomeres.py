import warnings
from typing import TYPE_CHECKING, Optional, Tuple, Union
import numpy as np
from numpy.typing import ArrayLike
from insectvision.utils import broadcast_1d

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


RHAB_COLOURS = [
    '#ffad13',  # R1
    '#FF1C25',  # R2
    '#880015',  # R3
    '#8000ff',  # R4
    '#008000',  # R5
    '#0000ff',  # R6
    '#aaa712',  # R7/8
]


class RhabdomereBundle:
    """
    Model of the rhabdomere bundle (inside a single ommatidium).

    Args:
        - name: str, Identifier (e.g. species name).
        - offsets_um: (R, 2) array_like, Rhabdomere offsets in the focal plane (μm),
            relative to the optical axis. R is the bundle size.
        - diameters_um: float or (R,) array_like, Waveguide diameters (μm). Scalar broadcasts.
        - focal_um: float (optional), Nodal->tip distance at the rhabdomere's dark-resting
            position (μm). Lever arm converting focal-plane offsets into angular shifts, and
            the distance at which the Snyder acceptance angle is evaluated. Required when R > 1.
        - sensitivity: scalar, (3,), (R,), or (R, 3) array_like, Spectral multipliers (UV, Green, Blue).
            UV (slot 0) is rendered in the red sub-pixel of the output.
                scalar: applied uniformly to (R, 3)
                (3,)  : per-channel, broadcast to all receptors
                (R,)  : per-receptor grayscale, tiled across channels
                (R, 3): explicit per-receptor, per-channel values
        - wavelengths_nm: float or (R,) array_like, Per-receptor peak wavelengths (nm). Scalar broadcasts.
        - tau_membrane: float, Membrane RC integration time (s).
        - tau_rise, tau_relax: float, Mechanical rise / relaxation time constants of the microsaccade (s).
        - tau_fast, tau_adapt: float, Fast and slow adaptation EMA times (s).
        - ampl_lat_um, ampl_ax_um: float, Max lateral / axial tip displacement at full microsaccade drive (μm).
        - extra_narrowing_ratio: float, Extra, non-optical RF narrowing at full saccade (1.0 = pure optics).
        - center_index: int, Index of the central rhabdomere (e.g. R7/8 in Drosophila is index 6).
        - main_axis_indices: (int, int), optional, Indices of the two rhabdomeres defining the
            bundle's main structural axis. If None, the main axis is taken as the bundle's longest axis.
            For a collapsed or symmetrical bundle this is degenerate, and an arbitrary axis (0 rad) is assigned.
        - alignment_offset: float or (int, int), Optional. Target angle (deg, mod 180) between the main axis and the
            external alignment reference. If (int, int), rhabdomere indices: then the offset is derived from th
            axis defined by these two rhabdomeres indices.
    """

    def __init__(self,
                 name: str = 'Simple',
                 offsets_um: ArrayLike = ((0.0, 0.0),),
                 diameters_um: Union[float, ArrayLike] = 2.0,
                 focal_um: Optional[float] = 20.0,
                 sensitivity: Union[float, ArrayLike] = 1.0,
                 wavelengths_nm: Union[float, ArrayLike] = 540.0,
                 fused_rhabdoms: bool = False,
                 tau_membrane: float = 0.0,
                 tau_rise: float = 0.015,
                 tau_relax: float = 0.060,
                 tau_fast: float = 0.005,
                 tau_adapt: float = 0.050,
                 ampl_lat_um: float = 2.0,
                 ampl_ax_um: float = 2.0,
                 extra_narrowing_ratio: float = 1.0,
                 center_index: int = 0,
                 main_axis_indices: Optional[Tuple[int, int]] = None,
                 alignment_offset: float = 0.0,
                 saccade_offset: Union[float, Tuple[int, int], None] = None,
                 ):

        self.name = str(name)
        self.focal_um = float(focal_um) if focal_um is not None else None
        self.fused_rhabdoms = bool(fused_rhabdoms)

        self.tau_membrane = float(tau_membrane)
        self.tau_rise = float(tau_rise)
        self.tau_relax = float(tau_relax)
        self.tau_fast = float(tau_fast)
        self.tau_adapt = float(tau_adapt)
        self.ampl_lat_um = float(ampl_lat_um)
        self.ampl_ax_um = float(ampl_ax_um)

        self._extra_narrowing_ratio = float(min(max(0.0, extra_narrowing_ratio), 1.0))

        self.offsets_um = np.atleast_2d(np.asarray(offsets_um, dtype=np.float32)).reshape(-1, 2)
        R = self.offsets_um.shape[0]

        self.center_index = int(center_index)

        # Alignment: main axis is a line (mod 180) so the offset is |.| mod 180
        self.alignment_offset_deg = abs(float(alignment_offset)) % 180.0

        self.diameters_um = broadcast_1d(diameters_um, R, 'diameters_um')
        self.wavelengths_nm = broadcast_1d(wavelengths_nm, R, 'wavelengths_nm')
        self.sensitivity = self._resolve_sensitivity(sensitivity, R)

        # Validation
        if not (0 <= self.center_index < R):
            raise ValueError(f'center_index={self.center_index} out of range for R={R}')

        if main_axis_indices is None:
            self.main_axis_indices = None
        else:
            i1, i2 = int(main_axis_indices[0]), int(main_axis_indices[1])
            if not (0 <= i1 < R and 0 <= i2 < R):
                raise ValueError(f"main_axis_indices=({i1},{i2}) out of range for R={R}")
            if R > 1 and i1 == i2:
                warnings.warn(
                    f"RhabdomereBundle '{self.name}': main_axis_indices=({i1},{i2}) are identical, "
                    "falling back to an arbitrary main axis.",
                    stacklevel=2,
                )
            self.main_axis_indices = (i1, i2)

        # Resolve main axis angle (offsets are fixed at construction)
        self._main_axis_rad = self._resolve_main_axis_rad()

        # Saccade axis: an offset (deg, mod 180) from the main axis
        self.saccade_offset_deg = self._resolve_saccade_offset_deg(saccade_offset)

        if R > 1 and self.focal_um is None:
            warnings.warn(
                f"RhabdomereBundle '{self.name}': R={R} but focal_um is None. "
                "Receptor directions cannot be computed without it.",
                stacklevel=2,
            )

    # Construction helpers

    @staticmethod
    def _resolve_sensitivity(sensitivity: Union[float, ArrayLike], R: int) -> np.ndarray:

        spec = np.atleast_1d(np.asarray(sensitivity, dtype=np.float32))

        if spec.ndim == 1:
            if spec.size == 1:
                return np.full((R, 3), spec.item(), dtype=np.float32)
            if spec.size == 3:
                return np.tile(spec, (R, 1)).astype(np.float32)
            if spec.size == R:
                return np.column_stack([spec, spec, spec]).astype(np.float32)
            raise ValueError(f'1D sensitivity must be size 1, 3, or R={R}, got {spec.size}.')

        if spec.ndim == 2:
            if spec.shape == (R, 3):
                return spec.astype(np.float32).copy()
            raise ValueError(f'2D sensitivity must be shape ({R}, 3), got {spec.shape}.')

        raise ValueError('Sensitivity must be a scalar, 1D, or 2D array.')

    def _resolve_main_axis_rad(self) -> float:
        """
        Derive the main-axis angle (rad, canonicalised to [0, pi)) from the offsets.

        - If main_axis_indices is given: the line through those two rhabdomeres.
        - Otherwise: the line from the optical axis (origin) to the furthest rhabdomere.
        - Degenerate (collapsed / coincident points): arbitrary axis, 0.0 rad.
        """
        off = self.offsets_um

        if self.main_axis_indices is not None:
            i1, i2 = self.main_axis_indices
            delta = off[i2] - off[i1]
        else:
            r2 = np.einsum('ij,ij->i', off, off)
            delta = off[int(np.argmax(r2))]

        if float(delta @ delta) < 1e-12:
            return 0.0  # collapsed / symmetrical -> arbitrary

        return float(np.arctan2(delta[1], delta[0]) % np.pi)

    def _resolve_saccade_offset_deg(self, spec: Union[float, Tuple[int, int], None]) -> float:
        """
        Resolve the saccade axis offset from the main axis (deg, in [0, 180)).

        - float: an explicit magnitude, stored as ``abs(spec) % 180``.
        - (int, int): the two rhabdomeres defining the saccade axis, its angle is
            derived from the offsets and reduced modulo the main axis.
        - None: the bundle's minor principal axis (smallest spatial spread, i.e.
            least steric hindrance), reduced modulo the main axis.
        """
        R = self.offsets_um.shape[0]

        # Index pair -> derive from structure

        if spec is not None and hasattr(spec, '__len__') and not isinstance(spec, str) and len(spec) == 2:
            j1, j2 = int(spec[0]), int(spec[1])
            if not (0 <= j1 < R and 0 <= j2 < R):
                raise ValueError(f"saccade_offset indices=({j1},{j2}) out of range for R={R}")
            delta = self.offsets_um[j2] - self.offsets_um[j1]
            if float(delta @ delta) < 1e-12:
                return 0.0
            axis_rad = np.arctan2(delta[1], delta[0])
            return float(np.degrees((axis_rad - self._main_axis_rad) % np.pi))

        # Explicit magnitude
        if spec is not None:
            return abs(float(spec)) % 180.0

        # Default: minor principal axis of the offset cloud
        pts = self.offsets_um - self.offsets_um.mean(axis=0)
        if float(np.max(np.einsum('ij,ij->i', pts, pts))) < 1e-12:
            return 0.0  # collapsed -> arbitrary

        cov = pts.T @ pts
        _, evecs = np.linalg.eigh(cov)   # ascending eigenvalues
        minor = evecs[:, 0]              # smallest spread

        axis_rad = np.arctan2(minor[1], minor[0])

        return float(np.degrees((axis_rad - self._main_axis_rad) % np.pi))

    # Size / lookup / repr

    def __repr__(self) -> str:
        return (
            f"RhabdomereBundle(name={self.name!r}, R={self.count}, center={self.center_index}, "
            f"main axis={self.main_axis_deg:.1f}°, "
            f"alignment offset={self.alignment_offset_deg:.1f}°, "
            f"saccade offset={self.saccade_offset_deg:.1f}°)"
        )

    def __len__(self) -> int:
        return self.offsets_um.shape[0]

    @property
    def count(self) -> int:
        """Number of rhabdomeres in this bundle."""
        return len(self)

    @property
    def center(self) -> np.ndarray:
        """Central rhabdomere position (x, y) in the focal plane (μm)."""
        return self.offsets_um[self.center_index]

    @property
    def indices(self) -> np.ndarray:
        """Indices of all rhabdomeres."""
        return np.arange(self.count, dtype=np.intp)

    @property
    def peripheral_indices(self) -> np.ndarray:
        """Indices of peripheral rhabdomeres (all except the central one)."""
        return np.array([i for i in self.indices if i != self.center_index])

    # Axes / alignment

    @property
    def main_axis_rad(self) -> float:
        """Main structural axis angle in the focal plane (rad, in [0, pi), it is a line)."""
        return self._main_axis_rad

    @property
    def main_axis_deg(self) -> float:
        return float(np.rad2deg(self._main_axis_rad))

    @property
    def alignment_offset_rad(self) -> float:
        """Target main-axis offset from the external reference (rad, in [0, pi))."""
        return float(np.deg2rad(self.alignment_offset_deg))

    @property
    def saccade_offset_rad(self) -> float:
        """Saccade axis offset from the main axis (rad, in [0, pi))."""
        return float(np.deg2rad(self.saccade_offset_deg))

    @property
    def saccade_axis_rad(self) -> float:
        """Saccade axis angle in the anatomical focal plane (rad, in [0, pi), it is a line)."""
        return float((self._main_axis_rad + self.saccade_offset_rad) % np.pi)

    @property
    def saccade_axis_deg(self) -> float:
        return float(np.rad2deg(self.saccade_axis_rad))

    # Phenomenological extra narrowing

    @property
    def extra_narrowing_ratio(self) -> float:
        return self._extra_narrowing_ratio

    @extra_narrowing_ratio.setter
    def extra_narrowing_ratio(self, val: float):
        self._extra_narrowing_ratio = float(min(max(0.0, val), 1.0))

    # Spectral accessors

    @property
    def sensitivity_uv(self) -> np.ndarray:
        return self.sensitivity[:, 0]

    @property
    def sensitivity_g(self) -> np.ndarray:
        return self.sensitivity[:, 1]

    @property
    def sensitivity_b(self) -> np.ndarray:
        return self.sensitivity[:, 2]

    # Geometry transforms

    def rotated_offsets(
            self,
            chi: ArrayLike,
            chirality: ArrayLike = 1.0,
            scale: Optional[Union[float, ArrayLike]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Orient the bundle so its main axis points at chi in each lens's
        (right, up) tangent frame, optionally mirrored by chirality.

        Args:
            - chi: (N,) array_like, Per-lens main-axis bearing (rad).
            - chirality: (N,) array_like or float, Per-lens handedness (+1 or -1). Defaults to +1 (no flip).
            - scale: (N,) array, float, or None. If provided, scales the spread of
                the receptors about the bundle's center_index.

        Returns:
            rot_dx, rot_dy: (N, R) arrays, Per-(lens, receptor) focal-plane offsets.
        """
        off = np.asarray(self.offsets_um)

        # Optional scaling about the centre
        if scale is not None:
            s = np.asarray(scale)
            if s.ndim > 0:
                s = s.reshape(-1, 1, 1)
            c = off[self.center_index]
            off = c + s * (off - c)

        dx = off[..., 0]
        dy = off[..., 1]

        # To canonical frame: rotate by -main_axis_rad so the main axis lies along +x
        m = self.main_axis_rad
        cos_m, sin_m = np.cos(m), np.sin(m)
        u = cos_m * dx + sin_m * dy
        v = -sin_m * dx + cos_m * dy

        # Chirality: reflect across the main axis (v -> -v) where chirality == -1
        chir = np.asarray(chirality, dtype=np.float32).reshape(-1, 1)
        v = chir * v

        # Rotate the (possibly mirrored) canonical bundle to chi
        chi = np.asarray(chi, dtype=np.float32).reshape(-1)
        cos_c = np.cos(chi)[:, np.newaxis]
        sin_c = np.sin(chi)[:, np.newaxis]

        rot_dx = cos_c * u - sin_c * v
        rot_dy = sin_c * u + cos_c * v

        return rot_dx, rot_dy

    # Visualisation

    def plot(self,
            spectral: bool = False,
            raw_orientation: bool = True,
            chi: float = 0.0,
            chirality: float = 1.0,
            degrees: bool = False,
        ) -> Tuple['Figure', 'Axes']:
        """
        Quick visual of the bundle and its main / saccade / alignment axes.

        Args:
            spectral: colour receptors by spectral sensitivity instead of index.
            raw_orientation: if True, shows offsets as supplied (anatomical frame).
                If False, shows the oriented view via `rotated_offsets` (the bundle
                taken to canonical form then placed at `chi` / `chirality`) inside a
                mod-180 orientation dial with chi = 0 at the top.
            chi: main-axis bearing (rad) to place the bundle at. Only when raw_orientation=False.
            chirality: +1 or -1 handedness. Only when raw_orientation=False.
        """
        import matplotlib.pyplot as plt

        if degrees:
            chi = np.deg2rad(chi)

        if raw_orientation:
            offsets = np.asarray(self.offsets_um, dtype=float)
            main_angle = self.main_axis_rad
            saccade_angle = self.saccade_axis_rad
            align_angle = (self.main_axis_rad - self.alignment_offset_rad) % np.pi
            disp_rot = 0.0
        else:
            rot_dx, rot_dy = self.rotated_offsets(
                np.full(1, chi, dtype=np.float32),
                chirality=np.full(1, chirality, dtype=np.float32),
            )
            offsets = np.stack([rot_dx[0], rot_dy[0]], axis=-1).astype(float)
            main_angle = float(chi)
            # Chirality reflects saccade / alignment axes across the main axis
            saccade_angle = float(chi) + float(chirality) * self.saccade_offset_rad
            align_angle = float(chi) - float(chirality) * self.alignment_offset_rad
            # Display rotation: put chi = 0 at the top (pure +90°, so chirality is preserved)
            disp_rot = np.pi / 2.0

        # Rotate the receptor cloud into display coordinates
        cr, sr = np.cos(disp_rot), np.sin(disp_rot)
        offsets_disp = offsets @ np.array([[cr, sr], [-sr, cr]])

        fig, ax = plt.subplots(figsize=(6, 6))
        max_sens = float(np.max(self.sensitivity))
        norm = max_sens if max_sens > 0 else 1.0

        radii = np.linalg.norm(offsets, axis=1) if self.count else np.zeros(1)
        max_r = max(float(radii.max()), float(self.diameters_um.max()) * 0.5)
        if max_r < 1e-6:
            max_r = 1.0
        has_axes = self.count > 1 and float(radii.max()) > 1e-6
        dial_r = max_r * 1.3
        extent = dial_r * (1.28 if not raw_orientation else 1.05)

        # Orientation dial: mod-180 clock face, 0 at top (aligned mode only)
        if not raw_orientation:
            ax.add_patch(plt.Circle((0, 0), dial_r, fill=False, ec='0.6', lw=1.0, zorder=0))
            for s_deg in range(0, 360, 15):
                s = np.radians(s_deg)
                dvec = np.array([np.cos(s), np.sin(s)])
                major = (s_deg % 30 == 0)
                t0 = dial_r * (0.93 if major else 0.965)
                ax.plot([t0 * dvec[0], dial_r * dvec[0]], [t0 * dvec[1], dial_r * dvec[1]],
                        color='0.55', lw=1.2 if major else 0.7, zorder=0)
                if major:
                    # Dial value: 0 at top (screen 90°), increasing CCW with chi, mod 180
                    val = int(round((s_deg - 90) % 180))
                    ax.text(*(dial_r * 1.1 * dvec), f'{val}', color='0.4', fontsize=6.5,
                            ha='center', va='center', zorder=0)

        # Receptors
        for r in range(self.count):
            pos = offsets_disp[r]
            d = self.diameters_um[r]

            if spectral:
                uv, g, b = self.sensitivity[r] / norm
                color = [float(uv), float(g), float(min(1.0, b + uv))]
            else:
                color = plt.cm.viridis(r / max(1, self.count - 1))

            ax.add_patch(plt.Circle(pos, d / 2, alpha=0.5, color=color, label=f"R{r + 1}"))
            edge = 'white' if spectral else color
            ax.scatter(*pos, color=color, edgecolor=edge, s=30, linewidth=0.5, zorder=3)

        # Axes: main (black), saccade (green), alignment (blue) + small coloured labels
        if has_axes:
            for angle, name, col, ls in (
                (main_angle,    'main',  'black', '--'),
                (saccade_angle, 'sacc',  'green', ':'),
                (align_angle,   'align', 'blue',  ':'),
            ):
                a = angle + disp_rot
                dx, dy = np.cos(a), np.sin(a)
                ax.plot([-extent * dx, extent * dx], [-extent * dy, extent * dy],
                        color=col, linestyle=ls, alpha=0.85, lw=1.4, zorder=2)
                val = np.deg2rad(angle) % 180.0
                ax.text(dial_r * 0.82 * dx, dial_r * 0.82 * dy, f'{name} {val:.0f}°',
                        color=col, fontsize=7, ha='center', va='center', zorder=4,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=col, lw=0.6, alpha=0.85))

        ax.set_aspect('equal')
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)

        mode = 'as supplied' if raw_orientation else 'aligned'
        title = f'Bundle: {self.name} (R={self.count}, {mode}'
        if not raw_orientation:
            title += f', $\\chi$={np.deg2rad(chi):.0f}°, chir={int(chirality):+d}'
        title += ')'
        ax.set_title(title)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7, title='Receptors')

        if spectral:
            ax.set_facecolor('#222222')
            fig.patch.set_facecolor('#222222')
            ax.grid(True, alpha=0.2, color='white')
            ax.title.set_color('white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
        else:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax


## =====================================================================================================================
#
# Built-in bundle: Drosophila melanogaster
# (values inspired from Kemppainen et al., 2022)

def drosophila_bundle(name: str = 'Drosophila') -> RhabdomereBundle:
    """Reference Drosophila melanogaster bundle."""

    # R1-R6 panchromatic, R7/8 strong UV bias
    sensitivity = np.ones((7, 3), dtype=np.float32)
    sensitivity[6] = [1.0, 0.2, 0.6]

    return RhabdomereBundle(
        name=name,
        offsets_um=[
            [-1.6881,  1.0273],   # R1
            [-1.8046, -0.9934],   # R2
            [-1.7111, -2.9717],   # R3
            [-0.0025, -1.9261],   # R4
            [ 1.6690, -0.9493],   # R5
            [ 1.6567,  0.9762],   # R6
            [ 0.0045, -0.0113],   # R7/8 (central)
        ],
        diameters_um=[1.8, 1.6, 1.6, 1.6, 1.6, 1.8, 1.0],  # Kemppainen 2022: R1/R6 1.8, R2-R5 1.6, R7/8 1.0
        sensitivity=sensitivity,
        focal_um=21.36,     # lens nodal point -> rhabdomere tip distance
        tau_membrane=0.005,
        tau_rise=0.015,
        tau_relax=0.060,
        tau_fast=0.005,
        tau_adapt=0.100,
        ampl_lat_um=2.0,    # upper-range microsaccade, ~6° RF shift
        ampl_ax_um=2.0,     # axial move 17->19 μm (Kemppainen 2022, Table S6)
        center_index=6,
        main_axis_indices=(2, 5),   # R3-R6 line
        alignment_offset=81.0,      # main axis sits ~81° (mod 180) off the flow reference
        saccade_offset=28.6,        # ~R1-R2-R3 line, offset from the main axis
    )


def honeybee_bundle(name: str = 'Honeybee') -> RhabdomereBundle:
    """
    Reference Apis mellifera (honeybee) worker bundle.
    Fused rhabdoms: all 9 photoreceptors contribute to a single central waveguide.
    """

    # TODO: double check the values in this bundle

    # 9 photoreceptors (R1-R9)
    # Most common ommatidium type (Type I):
    # - 4 Green cells (R1, R4, R5, R8)
    # - 2 UV cells (R2, R3)
    # - 2 Blue cells (R6, R7)
    # - 1 Basal cell (R9) - typically UV or Green

    # Sensitivities in [UV, Green, Blue] channels
    sens = np.zeros((9, 3), dtype=np.float32)
    sens[[1, 2]] = [1.0, 0.0, 0.0]              # UV
    sens[[5, 6]] = [0.0, 0.0, 1.0]              # Blue
    sens[[0, 3, 4, 7, 8]] = [0.0, 1.0, 0.0]     # Green

    wavelengths = np.zeros(9, dtype=np.float32)
    wavelengths[[1, 2]] = 340.0
    wavelengths[[5, 6]] = 436.0
    wavelengths[[0, 3, 4, 7, 8]] = 540.0

    # Fused rhabdoms: all receptors share the optical center
    offsets = np.zeros((9, 2), dtype=np.float32)

    return RhabdomereBundle(
        name=name,
        fused_rhabdoms=True,
        offsets_um=offsets,
        diameters_um=2.2,   # fused rhabdom diameter
        focal_um=55.0,      # apposition eye focal length
        sensitivity=sens,
        wavelengths_nm=wavelengths,
        ampl_lat_um=0.0,
        ampl_ax_um=0.0,
        center_index=8,             # R9 is the basal cell
        main_axis_indices=None,     # collapsed bundle -> arbitrary axis
        alignment_offset=0.0,
    )


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    b = drosophila_bundle()

    print(b)
    print(f"  Main axis:         {b.main_axis_deg:.1f}°")
    print(f"  Alignment offset:  {b.alignment_offset_deg:.1f}°")
    print(f"  Saccade offset:    {b.saccade_offset_deg:.1f}° (axis at {b.saccade_axis_deg:.1f}°)")
    print(f"  UV sensitivity per cell: {b.sensitivity_uv}")

    b.plot(raw_orientation=False, chi=0.0, chirality=1.0)

    plt.show()