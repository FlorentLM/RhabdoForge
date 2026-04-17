import warnings
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Union
import numpy as np
from numpy.typing import ArrayLike


@dataclass
class RhabdomereKernel:
    """
    Model of thr rhabdomere bundle inside a single ommatidium.

    The offsets describe each rhabdomere's position behind the lens in the focal plane (micrometres).

    - nodal_distance_um: physical distance from the lens inner surface (nodal point) to the rhabdomere tips (at rest).
        This is the lever arm that converts a lateral displacement (μm) into an angular shift.
        For Drosophila this is ~20-21 μm (Kemppainen et al. 2022, 10.1073/pnas.2109717119; Stavenga 2003, 10.1007/s00359-002-0370-2)
        https://github.com/JuusolaLab/Hyperacute_Stereopsis_paper/blob/main/CG-Compound-Eye/model_init.py#L213

    - main_axis: the main structural axis of the bundle (longest axis, R3-R6 in Drosophila).

    - optic_flow_offset_deg: angular offset (from the main axis) of the optic flow alignment vector.
        In Drosophila, alignment is about 81° off of the main axis (so roughly aligned with R2-R5 line.

    - saccade_axis_deg: angular offset (from the main axis) of the microsaccade actuation direction.
        In Drosophila, microsaccade axis is about 28.6° off of the main axis, so roughly aligned with R1-R2-R3 line.

    - tau_ms: Temporal integration in milliseconds. The receptor's output is an exponential moving average across
        this time. High values create 'motion blur'. Realistic values (0.012 sec = 12 ms for Drosophila) absorb noise
        from rays (particularly useful when used with quasi-random sampling).

    - sensitivity: Photometric sensitivity: multiplicative weight on luminance, linear, fixed per receptor
    """
    # TODO: modeling light adaptation would be nice, but it would likely be a separate pass after linear sensitivity

    name: str = 'Simple'

    # XY coords of rhabdomere tips in focal plane
    offsets_um: ArrayLike = field(
        default_factory=lambda: np.zeros((1, 2), dtype=np.float32)  # default single receptor at the centre
    )

    # Distance from lens nodal point to rhabdomere tips (lever arm for shift/optics)
    nodal_distance_um: Optional[float] = 21.0

    # Rhabdomere waveguide diameters
    diameters_um: Union[float, ArrayLike] = 2.0

    # Facet lens diameter (used for the diffraction component of acceptance angle)
    lens_diameter_um: float = 16.0

    # Temporal integration (s)
    tau_s: float = 0.0

    # Photometric sensitivity in [0.0, 1.0]
    sensitivity: float = 1.0

    # Structural props
    center_index: int = 0
    peripheral_indices: Optional[List[int]] = None
    main_axis_indices: Tuple[int, int] = (0, 0)

    # Alignment axes (relative to main_axis)
    flow_axis_deg: float = -81.0
    saccade_offset_deg: float = 0.0

    def __post_init__(self):

        offsets = np.atleast_2d(np.asarray(self.offsets_um, dtype=np.float32))
        if offsets.shape[0] == 1 and offsets.shape[1] > 2:
            offsets = offsets.reshape(-1, 2)
        self.offsets_um = offsets

        R = len(self)

        diams = np.atleast_1d(np.asarray(self.diameters_um, dtype=np.float32))
        if diams.size == 1:
            self.diameters_um = np.full(R, diams.item(), dtype=np.float32)
        elif diams.size == R:
            self.diameters_um = diams.flatten()
        else:
            raise ValueError(f"diameters_um size ({diams.size}) must be 1 or match offsets ({R})")

        if self.peripheral_indices is None:
            self.peripheral_indices = list(range(R))

        # Validate main_axis_indices for multi-receptor kernels
        i1, i2 = self.main_axis_indices
        if R > 1 and i1 == i2:
            warnings.warn(
                f"RhabdomereKernel '{self.name}' has {R} receptors but "
                f"main_axis_indices=({i1}, {i2}) are identical. Tissue alignment may be wrong.",
                stacklevel=2
            )

    def __len__(self):
        return self.offsets_um.shape[0]

    @property
    def count(self) -> int:
        return len(self)

    @property
    def center(self) -> np.ndarray:
        """Central rhabdomere position."""
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
        return self.main_axis_rad + float(np.deg2rad(self.flow_axis_deg))

    @property
    def saccade_axis_rad(self) -> float:
        """Microsaccade actuation axis angle in radians."""

        return self.main_axis_rad + float(np.deg2rad(self.saccade_offset_deg))

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

    def plot(self):

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        for r in range(self.count):
            pos = self.offsets_um[r]
            d = self.diameters_um[r]
            color = plt.cm.viridis(r / max(1, self.count - 1))

            circle = plt.Circle(pos, d / 2, alpha=0.3, color=color, label=f"R{r + 1}")
            ax.add_patch(circle)
            ax.scatter(*pos, color=color, s=20)

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
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.grid(True, alpha=0.3)
        plt.show()


##

if __name__ == '__main__':

    # Drosophila data from Kemppainen et al. 2022
    droso = RhabdomereKernel(
        name="Drosophila",
        offsets_um=np.array([
            [-1.6881,  1.0273],    # R1
            [-1.8046, -0.9934],    # R2
            [-1.7111, -2.9717],    # R3
            [-0.0025, -1.9261],    # R4
            [ 1.6690, -0.9493],    # R5
            [ 1.6567,  0.9762],    # R6
            [ 0.0045, -0.0113]     # R7/8 (central)
        ]),
        diameters_um=np.array([1.8627, 1.8627, 1.8627, 1.8627, 1.8627, 1.8627, 1.5743]),
        nodal_distance_um=21.0,
        center_index=6,            # R7/8
        main_axis_indices=(2, 5),  # R3-R6
        flow_axis_deg=-81.0,
        saccade_offset_deg=28.6,
    )
    droso.plot()