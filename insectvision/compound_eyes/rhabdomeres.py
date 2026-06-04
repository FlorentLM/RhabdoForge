import warnings
from typing import Optional, Tuple, Union
import numpy as np
from numpy.typing import ArrayLike


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
        - name: str, Identifier (e.g. species name)
        - offsets_um: (R, 2) array_like, Rhabdomere offsets in the focal plane (μm). R is the bundle size.
        - diameters_um: float or (R,) array_like, Waveguide diameters (μm). Scalar broadcasts to all receptors.
        - focal_um: float (optional), Effective nodal->tip distance at the rhabdomere's dark-resting position (μm).
            This is the lever arm converting lateral focal-plane offsets into angular shifts,
            and the distance at which the optical (Snyder) acceptance angle is evaluated.
            Required when R > 1.
        - sensitivity: scalar, (3,), (R,), or (R, 3) array_like, Spectral multipliers in the order (UV, Green, Blue).
            UV (slot 0) is rendered in the red sub-pixel of the output.
            Shapes:
                scalar: applied uniformly to (R, 3)
                (3,)  : per-channel, broadcast to all receptors
                (R,)  : per-receptor grayscale, tiled across channels
                (R, 3): explicit per-receptor, per-channel values
        - wavelengths_nm: float or (R,) array_like, Per-receptor peak wavelengths (nm).
            Used by the Snyder diffraction term. Scalar broadcasts.
        - tau_membrane: float, Membrane RC integration time (s).
            The receptor's output is an EMA with this time constant. ~0.012 s in Drosophila.
        - tau_rise, tau_relax: float, Mechanical rise / relaxation time constants of the microsaccade (s).
        - tau_fast, tau_adapt: float, Fast (~PIP2 hydrolysis, ~5 ms) and slow (~Ca²+, ~50 ms) adaptation EMA times (s).
        - ampl_lat_um, ampl_ax_um: float, Max lateral / axial rhabdomere tip displacement at full microsaccade drive (μm).
        - extra_narrowing_ratio: float, Extra, non-optical RF narrowing at full saccade (i.e. ~ voltage-level hyperacuity).
            1.0 = pure optics
        - center_index: int, Index of the central rhabdomere (e.g. R7/8 in Drosophila is index 6).
        - main_axis_indices: (int, int), Indices of the two rhabdomeres defining the bundle's main structural axis.
            R3, R6 in Drosophila, indices (2, 5).
        - flow_axis_deg: float, Optic flow alignment axis, in degrees offset from the main axis.
            In Drosophila this is ~-81° (more or less the R2-R5 line).
        - saccade_offset_deg: float, Microsaccade actuation axis, in degrees offset from the main axis.
            In Drosophila this is ~28.6° (more or less the R1-R2-R3 line).
    """

    def __init__(self,
                 name: str = 'Simple',
                 offsets_um: ArrayLike = ((0.0, 0.0),),
                 diameters_um: Union[float, ArrayLike] = 2.0,
                 focal_um: Optional[float] = 20.0,
                 sensitivity: Union[float, ArrayLike] = 1.0,
                 wavelengths_nm: Union[float, ArrayLike] = 540.0,
                 tau_membrane: float = 0.005,
                 tau_rise: float = 0.015,
                 tau_relax: float = 0.060,
                 tau_fast: float = 0.005,
                 tau_adapt: float = 0.050,
                 ampl_lat_um: float = 2.0,
                 ampl_ax_um: float = 2.0,
                 extra_narrowing_ratio: float = 1.0,
                 center_index: int = 0,
                 main_axis_indices: Tuple[int, int] = (0, 0),
                 flow_axis_deg: float = -81.0,
                 saccade_offset_deg: float = 0.0,
                 ):

        self.name = str(name)
        self.focal_um = float(focal_um)

        self.tau_membrane = float(tau_membrane)
        self.tau_rise = float(tau_rise)
        self.tau_relax = float(tau_relax)
        self.tau_fast = float(tau_fast)
        self.tau_adapt = float(tau_adapt)
        self.ampl_lat_um = float(ampl_lat_um)
        self.ampl_ax_um = float(ampl_ax_um)

        self._extra_narrowing_ratio = float(min(max(0.0, extra_narrowing_ratio), 1.0))

        self.center_index = int(center_index)
        self.main_axis_indices = (int(main_axis_indices[0]), int(main_axis_indices[1]))
        self.flow_axis_deg = float(flow_axis_deg)
        self.saccade_offset_deg = float(saccade_offset_deg)

        self.offsets_um = np.atleast_2d(np.asarray(offsets_um, dtype=np.float32)).reshape(-1, 2)
        R = self.offsets_um.shape[0]

        self.diameters_um = self._broadcast_to_R(diameters_um, R, 'diameters_um')
        self.wavelengths_nm = self._broadcast_to_R(wavelengths_nm, R, 'wavelengths_nm')
        self.sensitivity = self._resolve_sensitivity(sensitivity, R)

        # Validation
        if not (0 <= self.center_index < R):
            raise ValueError(f"center_index={self.center_index} out of range for R={R}")

        i1, i2 = self.main_axis_indices
        if not (0 <= i1 < R and 0 <= i2 < R):
            raise ValueError(f"main_axis_indices=({i1},{i2}) out of range for R={R}")

        if R > 1 and i1 == i2:
            warnings.warn(
                f"RhabdomereBundle '{self.name}': R={R} but main_axis_indices=({i1},{i2}) "
                "are identical; bundle orientation pipeline cannot derive an alignment axis.",
                stacklevel=2,
            )

        if R > 1 and self.focal_um is None:
            warnings.warn(
                f"RhabdomereBundle '{self.name}': R={R} but focal_um is None. "
                "Receptor directions cannot be computed without it.",
                stacklevel=2,
            )

    # Construction helpers

    @staticmethod
    def _broadcast_to_R(value: Union[float, ArrayLike], R: int, name: str) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(value, dtype=np.float32))
        if arr.size == 1:
            return np.full(R, arr.item(), dtype=np.float32)
        if arr.size == R:
            return arr.flatten().astype(np.float32)
        raise ValueError(f"{name} size ({arr.size}) must be 1 or match receptor count R={R}")

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
            raise ValueError(f"1D sensitivity must be size 1, 3, or R={R}; got {spec.size}.")

        if spec.ndim == 2:
            if spec.shape == (R, 3):
                return spec.astype(np.float32).copy()
            raise ValueError(f"2-D sensitivity must be shape ({R}, 3); got {spec.shape}.")
        raise ValueError("sensitivity must be a scalar, 1-D, or 2-D array.")

    # Size / lookup / repr

    def __repr__(self) -> str:
        return (
            f"RhabdomereBundle(name={self.name!r}, R={self.count}, "
            f"center={self.center_index}, "
            f"main_axis={self.main_axis_indices}, flow_axis_deg={self.flow_axis_deg:g})"
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

    # Phenomenological extra narrowing

    @property
    def extra_narrowing_ratio(self) -> float:
        return self._extra_narrowing_ratio

    @extra_narrowing_ratio.setter
    def extra_narrowing_ratio(self, val: float):
        self._extra_narrowing_ratio = float(min(max(0.0, val), 1.0))

    # Axes

    @property
    def main_axis_rad(self) -> float:
        """Main structural axis angle in the focal plane (rad)."""
        i1, i2 = self.main_axis_indices
        if i1 == i2:
            return 0.0
        delta = self.offsets_um[i2] - self.offsets_um[i1]
        return float(np.arctan2(delta[1], delta[0]))

    @property
    def main_axis_deg(self) -> float:
        return float(np.degrees(self.main_axis_rad))

    @property
    def flow_axis_rad(self) -> float:
        """Optic flow alignment axis angle in the focal plane (rad)."""
        return self.main_axis_rad + float(np.radians(self.flow_axis_deg))

    @property
    def saccade_axis_rad(self) -> float:
        """Microsaccade actuation axis angle in the focal plane (rad)."""
        return self.main_axis_rad + float(np.radians(self.saccade_offset_deg))

    @property
    def saccade_axis_deg(self) -> float:
        return self.main_axis_deg + self.saccade_offset_deg

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

    def rotated_offsets(self, chi: np.ndarray, chirality: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-lens rotation and chirality flip of focal-plane offsets.

        Args:
            - chi: (N,) array, Bundle yaw in each lens tangent frame (rad).
            - chirality: (N,) array, Per-lens chirality (+1 or -1).
                The chirality flip mirrors the bundle across the x-axis *before* rotating by chi.

        Returns:
            rot_dx, rot_dy: (N, R) arrays, Per-(lens, receptor) focal-plane offsets.
        """
        cos_chi = np.cos(chi)[:, None]
        sin_chi = np.sin(chi)[:, None]
        dx_chiral = self.offsets_um[:, 0][None, :] * chirality[:, None]
        dy = self.offsets_um[:, 1]
        rot_dx = cos_chi * dx_chiral - sin_chi * dy[None, :]
        rot_dy = sin_chi * dx_chiral + cos_chi * dy[None, :]
        return rot_dx, rot_dy

    # Visualisation

    def plot(self, spectral: bool = False):
        """Quick visual of the bundle."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        max_sens = float(np.max(self.sensitivity))
        norm = max_sens if max_sens > 0 else 1.0

        for r in range(self.count):
            pos = self.offsets_um[r]
            d = self.diameters_um[r]

            if spectral:
                uv, g, b = self.sensitivity[r] / norm
                # UV -> red, with hint of UV into blue so UV shows as purple
                color = [float(uv), float(g), float(min(1.0, b + uv))]
            else:
                color = plt.cm.viridis(r / max(1, self.count - 1))

            ax.add_patch(plt.Circle(pos, d / 2, alpha=0.5, color=color, label=f"R{r + 1}"))
            edge = 'white' if spectral else color
            ax.scatter(*pos, color=color, edgecolor=edge, s=30, linewidth=0.5, zorder=3)

        if self.count > 1 or self.saccade_offset_deg != 0:
            angle = np.radians(self.saccade_axis_deg)
            dx, dy = np.cos(angle), np.sin(angle)
            extent = float(np.max(self.diameters_um)) * 1.5
            p0 = self.center - extent * np.array([dx, dy])
            p1 = self.center + extent * np.array([dx, dy])
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                    color='black', linestyle='--', alpha=0.7, label='Microsaccade axis')

        ax.set_aspect('equal')
        ax.set_title(f'Bundle: {self.name} (R={self.count})')
        ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1))

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
        tau_membrane = 0.005,
        tau_rise=0.015,
        tau_relax=0.060,
        tau_fast=0.005,
        tau_adapt=0.100,
        ampl_lat_um=2.0,    # upper-range microsaccade, ~6° RF shift
        ampl_ax_um=2.0,     # axial move 17->19 μm (Kemppainen 2022, Table S6)
        center_index=6,
        main_axis_indices=(2, 5),    # R3-R6
        flow_axis_deg=-81.0,
        saccade_offset_deg=28.6,
    )


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    b = drosophila_bundle()

    print(b)
    print(f"  Main axis: {b.main_axis_deg:.1f}°")
    print(f"  Flow axis: {np.degrees(b.flow_axis_rad):.1f}°")
    print(f"  Saccade axis: {b.saccade_axis_deg:.1f}°")
    print(f"  UV sensitivity per cell: {b.sensitivity_uv}")
    b.plot(spectral=True)

    plt.show()
