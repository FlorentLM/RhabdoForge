from dataclasses import dataclass
import numpy as np


@dataclass
class RhabdomereKernel:
    """
    Model of thr rhabdomere bundle inside a single ommatidium.

    The offsets describe each rhabdomere's position behind the lens in the focal plane (micrometres).

    - `nodal_distance_um`: physical distance from the lens inner surface (nodal point) to the rhabdomere tips (at rest).
        This is the lever arm that converts a lateral displacement (μm) into an angular shift
        For Drosophila this is ~20-21 μm (Kemppainen et al. 2022, 10.1073/pnas.2109717119; Stavenga 2003, 10.1007/s00359-002-0370-2)
        https://github.com/JuusolaLab/Hyperacute_Stereopsis_paper/blob/main/CG-Compound-Eye/model_init.py#L213

    - `main_axis` is the main axis of the bundle, defined as the R3-R6 axis

    - `saccade_axis_deg`: angular offset (from the main axis) of the microsaccade actuation direction
        In world space the full actuation angle is `chi + main_axis + saccade_axis_deg`
    """
    name: str
    offsets_um: np.ndarray          # (R, 2) XY in focal plane
    nodal_distance_um: float        # lens to rhabdomere tip distance (at rest)
    diameters_um: np.ndarray        # (R,) waveguide diameter
    lens_diameter_um: float         # facet lens diameter (for diffraction term)
    saccade_axis_deg: float = 0.0   # microsaccade axis delta (relative to main_axis)

    @property
    def count(self) -> int:
        return len(self.offsets_um)

    @property
    def center(self) -> np.ndarray:
        """
        Central rhabdomere position (R7/R8).
        """
        return self.offsets_um[6]

    @property
    def main_axis_deg(self) -> float:
        """
        Bundle axis angle (R3->R6 line) in degrees.
        """
        delta = self.offsets_um[5] - self.offsets_um[2]
        return float(np.degrees(np.arctan2(delta[1], delta[0])))

    @property
    def actuation_angle_deg(self) -> float:
        """
        Full actuation angle in the kernel's local frame (degrees).
        """
        return self.main_axis_deg + self.saccade_axis_deg

    def plot(self):

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        for r, pos in enumerate(self.offsets_um):
            col = plt.cm.tab10(r / 8)
            txt = "R7/R8" if r == 6 else f"R{r + 1}"
            circle = plt.Circle(pos, self.diameters_um[r] / 2,
                                alpha=0.3, color=col, label=txt)
            ax.add_patch(circle)
            ax.scatter(*pos, color=col, s=10)

        angle_rad = np.radians(self.actuation_angle_deg)
        dx, dy = np.cos(angle_rad), np.sin(angle_rad)
        ax.axline(self.center, slope=dy / dx, color='black', linestyle='--', alpha=0.7, label="Microsaccade axis")

        ax.set_aspect('equal')
        ax.set_title(f'Rhabdomere layout ({self.name})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.show()
