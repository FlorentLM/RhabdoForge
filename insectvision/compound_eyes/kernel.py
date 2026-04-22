import warnings
from typing import Tuple, Optional, Union
import numpy as np
from numpy.typing import ArrayLike


class RhabdomereKernel:
    """
    Model of the rhabdomere bundle inside a single ommatidium.

    The offsets describe each rhabdomere's position behind the lens in the focal plane (micrometres).

    - nodal_distance_um: physical distance from the lens inner surface (nodal point) to the rhabdomere tips (at rest).
        This is the lever arm that converts a lateral displacement (μm) into an angular shift.
        For Drosophila this is ~20-21 μm (Kemppainen et al. 2022; Stavenga 2003).

    - diameters_um: Rhabdomere waveguide diameters. Can be a scalar (all identical) or an array of length R.

    - sensitivity: Multipliers for [UV, Green, Blue] channels. Can be:
        - A scalar (e.g. 1.0 -> applied to all channels, all receptors)
        - A list of 3 floats (e.g. [1.0, 0.5, 0.0] -> applied to all receptors)
        - An (R, 3) array mapping exact spectral sensitivities per receptor

    - main_axis_indices: the main structural axis of the bundle (longest axis, e.g. R3-R6 in Drosophila).

    - flow_axis_deg: angular offset (from the main axis) of the optic flow alignment vector.
        In Drosophila, alignment is about -81° off of the main axis (roughly aligned with R2-R5 line).

    - saccade_offset_deg: angular offset (from the main axis) of the microsaccade actuation direction.
        In Drosophila, microsaccade axis is about 28.6° off of the main axis (roughly aligned with R1-R2-R3).

    - tau_s: Temporal integration in seconds. The receptor's output is an exponential moving average.
        Realistic values (e.g., 0.012 sec for Drosophila) absorb ray noise and create realistic motion blur.
    """

    def __init__(
            self,
            name: str = 'Simple',
            offsets_um: ArrayLike = np.array([[0.0, 0.0]]),
            nodal_distance_um: Optional[float] = 21.0,
            diameters_um: Union[float, np.floating, ArrayLike] = 2.0,
            lens_diameter_um: float = 16.0,
            tau_s: float = 0.0,
            sensitivity: Union[float, np.floating, ArrayLike] = 1.0,
            center_index: int = 0,
            peripheral_indices: Optional[ArrayLike] = None,
            main_axis_indices: ArrayLike = np.array([0, 0]),
            flow_axis_deg: float = -81.0,
            saccade_offset_deg: float = 0.0
    ):

        self.name = name

        self.nodal_distance_um = nodal_distance_um
        self.lens_diameter_um = lens_diameter_um
        self.tau_s = tau_s

        self.center_index = center_index
        self.main_axis_indices = main_axis_indices
        self.flow_axis_deg = flow_axis_deg
        self.saccade_offset_deg = saccade_offset_deg

        self.offsets_um = np.atleast_2d(np.asarray(offsets_um, dtype=np.float32)).reshape(-1, 2)

        R = self.offsets_um.shape[0]

        diams = np.atleast_1d(np.asarray(diameters_um, dtype=np.float32))
        if diams.size == 1:
            self.diameters_um = np.full(R, diams.item(), dtype=np.float32)
        elif diams.size == R:
            self.diameters_um = diams.flatten()
        else:
            raise ValueError(f"diameters_um size ({diams.size}) must be 1 or match receptor count R={R}")

        self.peripheral_indices = np.asarray(peripheral_indices) if peripheral_indices is not None else np.arange(R)

        spec = np.atleast_1d(np.asarray(sensitivity, dtype=np.float32))

        if spec.ndim == 1:
            if spec.size == 1:
                # scalar
                self.sensitivity = np.full((R, 3), spec.item(), dtype=np.float32)
            elif spec.size == 3:
                # global RGB: e.g., [1.0, 0.5, 0.0] -> applied to all R receptors
                self.sensitivity = np.tile(spec, (R, 1))
            elif spec.size == R:
                # grayscale per receptor -> duplicate across RGB for each receptor
                self.sensitivity = np.column_stack([spec, spec, spec])
            else:
                raise ValueError(f"sensitivity 1D array must be size 1, 3, or R={R}. Got {spec.size}.")
        elif spec.ndim == 2:
            if spec.shape == (R, 3):
                self.sensitivity = spec.copy()
            else:
                raise ValueError(f"sensitivity 2D array must be shape ({R}, 3). Got {spec.shape}.")
        else:
            raise ValueError("sensitivity must be a scalar, 1D, or 2D array.")

        # Validation
        i1, i2 = self.main_axis_indices
        if R > 1 and i1 == i2:
            warnings.warn(
                f"RhabdomereKernel '{self.name}' has {R} receptors but "
                f"main_axis_indices=({i1}, {i2}) are identical. Tissue alignment calculations may fail.",
                stacklevel=2
            )

    def __len__(self) -> int:
        return self.offsets_um.shape[0]

    @property
    def count(self) -> int:
        """Number of rhabdomeres in this kernel."""
        return len(self)

    @property
    def center(self) -> np.ndarray:
        """Central rhabdomere position (x, y) in the focal plane."""
        return self.offsets_um[self.center_index]

    @property
    def main_axis_rad(self) -> float:
        """Angle of the main structural axis in radians (e.g. R3-R6)."""
        i1, i2 = self.main_axis_indices
        if i1 == i2 or i2 >= self.count:
            return 0.0
        delta = self.offsets_um[i2] - self.offsets_um[i1]
        return float(np.arctan2(delta[1], delta[0]))

    @property
    def main_axis_deg(self) -> float:
        """Angle of the main structural axis in degrees (e.g. R3-R6)."""
        return float(np.degrees(self.main_axis_rad))

    @property
    def flow_axis_rad(self) -> float:
        """Optic flow alignment axis angle in radians."""
        return self.main_axis_rad + float(np.radians(self.flow_axis_deg))

    @property
    def saccade_axis_rad(self) -> float:
        """Microsaccade actuation axis angle in radians."""
        return self.main_axis_rad + float(np.radians(self.saccade_offset_deg))

    @property
    def saccade_axis_deg(self) -> float:
        """Microsaccade actuation axis angle in degrees."""
        return self.main_axis_deg + self.saccade_offset_deg

    def rotated_offsets(
            self,
            chi: np.ndarray,
            chirality: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns rotated and mirrored offsets by per-lens bundle orientation, for this kernel."""

        cos_chi = np.cos(chi)[:, None]
        sin_chi = np.sin(chi)[:, None]

        dx_chiral = self.offsets_um[:, 0][None, :] * chirality[:, None]
        dy = self.offsets_um[:, 1]

        rot_dx = cos_chi * dx_chiral - sin_chi * dy[None, :]
        rot_dy = sin_chi * dx_chiral + cos_chi * dy[None, :]
        return rot_dx, rot_dy

    def plot(self, spectral=False):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        max_sens = np.max(self.sensitivity)
        norm_factor = max_sens if max_sens > 0 else 1.0

        for r in range(self.count):
            pos = self.offsets_um[r]
            d = self.diameters_um[r]

            if spectral:
                # Spectral mapping: [UV, Green, Blue]
                uv_red, green, blue = self.sensitivity[r] / norm_factor

                # Make UV appear purple
                disp_r = uv_red
                disp_g = green
                disp_b = min(1.0, blue + uv_red)

                color = [disp_r, disp_g, disp_b]
            else:
                color = plt.cm.viridis(r / max(1, self.count - 1))

            circle = plt.Circle(pos, d / 2, alpha=0.5, color=color, label=f"R{r + 1}")
            ax.add_patch(circle)

            edge_color = 'white' if spectral else color
            ax.scatter(*pos, color=color, edgecolor=edge_color, s=30, linewidth=0.5, zorder=3)

        # Actuation vector
        if self.count > 1 or self.saccade_offset_deg != 0:
            angle_rad = np.radians(self.saccade_axis_deg)
            dx, dy = np.cos(angle_rad), np.sin(angle_rad)
            extent = np.max(self.diameters_um) * 1.5
            p0 = self.center - extent * np.array([dx, dy])
            p1 = self.center + extent * np.array([dx, dy])
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color='black', linestyle='--', alpha=0.7, label="Microsaccade axis")

        ax.set_aspect('equal')
        ax.set_title(f'Kernel: {self.name} (R={self.count})')
        ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1))

        if spectral:
            ax.set_facecolor('#222222')
            fig.patch.set_facecolor('#222222')
            plt.grid(True, alpha=0.2, color='white')
            ax.title.set_color('white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
        else:
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


##

if __name__ == '__main__':
    # Example: Drosophila data from Kemppainen et al. 2022

    # Illustrative spectral profile where R1-6 are broad/panchromatic
    # and the central R7/8 has a strong UV/Blue bias
    spec_profile = np.ones((7, 3), dtype=np.float32)
    spec_profile[6] = [1.0, 0.2, 0.6]  # Strong UV, some Green, more Blue

    droso = RhabdomereKernel(
        name="Drosophila",
        offsets_um=[
            [-1.6881, 1.0273],      # R1
            [-1.8046, -0.9934],     # R2
            [-1.7111, -2.9717],     # R3
            [-0.0025, -1.9261],     # R4
            [1.6690, -0.9493],      # R5
            [1.6567, 0.9762],       # R6
            [0.0045, -0.0113]       # R7/8 (central)
        ],
        diameters_um=[1.8627, 1.8627, 1.8627, 1.8627, 1.8627, 1.8627, 1.5743],
        sensitivity=spec_profile,
        nodal_distance_um=21.0,
        center_index=6,             # R7/8
        main_axis_indices=(2, 5),   # R3-R6
        flow_axis_deg=-81.0,
        saccade_offset_deg=28.6,
    )
    droso.plot(spectral=False)