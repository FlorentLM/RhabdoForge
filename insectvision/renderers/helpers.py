from typing import Tuple
import numpy as np

from insectvision.compound_eyes import Model
from insectvision.geometry.circular import wrap_angle


class VisualOutput:
    """
    Per-receptor output array with various biological pathway mappings conveniences.

    The renderers return a float array where the last axis is (R/UV, G, B, radiance).
    This class supports both single snapshots (shape: N, 4) and timeseries (shape: T, N, 4).

    Layouts / pathways:
        .per_ommatidium    -> (..., N, R, 4) Physical ommatidia grouping.
        .per_cartridge     -> (..., N, R, 4) Neural superposition grouping.
        .per_receptor(i)   -> (..., N, 4)    Specific receptor type across all cartridges.
        .peripheral_signal -> (..., N, 4)    Pooled R1-R6 (LMC pathway for motion).
        .central_signal    -> (..., N, 4)    Central R7/R8 (Medulla colour pathway).
        .lmc_input         -> (..., N, 4)    Alias for peripheral_signal.
        .pale_input        -> (..., N, 4)    Alias for central_signal.

    Signal analysis:
        .colours      -> (..., 3) The adapted spectral response.
        .adaptation   -> (..., )  The gain factor (state of the biological system).
        .radiance     -> (..., )  Adapted intensity (mean of adapted RGB).
        .raw_radiance -> (..., )  Physical light intensity recovered by 'un-baking'
                                  the adaptation factor.
    """
    __slots__ = ('_data', '_model', '_shape', '_is_time_series')

    def __init__(self, data: np.ndarray, model: 'Model'):

        if data.shape[-2] % model.shape[1] != 0:
            raise ValueError(f'data length {data.shape[-2]} not divisible by R={model.shape[1]}')

        self._data = data
        self._model = model

        if data.ndim == 3:
            self._is_time_series = True
            self._shape = int(data.shape[0]), int(data.shape[-2] // model.shape[1]), int(model.shape[1])
        else:
            self._is_time_series = False
            self._shape = int(data.shape[-2] // model.shape[1]), int(model.shape[1])

    # TODO: __bool__ overload

    @property
    def shape(self) -> Tuple[int, int] | Tuple[int, int, int]:
        return self._shape

    @property
    def size(self) -> int:
        return int(np.prod(self._shape))

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @classmethod
    def from_history(cls, history: list['VisualOutput']) -> 'VisualOutput':
        """Stacks a list of single-frame VisualOutputs into one timeseries VisualOutput."""
        if not history:
            raise ValueError('History list is empty')
        return cls(np.stack([vo.data for vo in history if vo], axis=0), history[0]._model)
    # TODO: support stacking list of already-timeseries VO

    @property
    def data(self) -> np.ndarray:
        """The raw per-receptor array."""
        return self._data

    # Channel helpers

    @property
    def colours(self) -> np.ndarray:
        """The adapted spectral response (Photoreceptor output)."""
        return self._data[..., :3]

    @property
    def adaptation(self) -> np.ndarray:
        """
        The adaptation state (gain factor) of the receptors.
        This is the value calculated by the Naka-Rushton equations (0.0 to 1.0+).
        """
        return self._data[..., 3]

    @property
    def gain(self) -> np.ndarray:
        """Alias to adaptation"""
        return self.adaptation

    @property
    def raw_radiance(self) -> np.ndarray:
        """
        The physical light intensity hitting the eye before adaptation.
        Recovered by 'un-baking' the adaptation factor.
        """
        return np.mean(self.colours, axis=-1) / (self.adaptation + 1e-6)

    @property
    def radiance(self) -> np.ndarray:
        """
        The mean intensity of the adapted signal.
        Calculated as the mean of the RGB channels.
        """
        return np.mean(self.colours, axis=-1)

    # Level 1: raw grids
    @property
    def per_ommatidium(self) -> np.ndarray:
        """Returns (..., N, R, 4) array of all receptor outputs, per ommatidium."""
        return self._data.reshape(*self.shape, 4)

    @property
    def per_cartridge(self) -> np.ndarray:
        """Returns (..., N, R, 4) array of all receptor outputs, per cartridge."""
        if not self._model.neural_superposition:
            return self.per_ommatidium   # fallback for R=1 models

        if self._data.ndim == 2:
            return self._data[self._model.cartridge_indices]
        return self._data[:, self._model.cartridge_indices, :]

    # Level 2: type-based access
    def per_receptor(self, index: int) -> np.ndarray:
        """Returns (..., N, 4) array for a specific receptor index (e.g. 0 for R1)."""
        return self.per_cartridge[..., index, :]

    # Level 3: biological pathways
    @property
    def peripheral_signal(self) -> np.ndarray:
        """The pooled response of all peripheral rhabdomeres (LMC-pathway)."""
        if self.shape[-1] == 1:
            return self.per_ommatidium[..., 0, :]

        periph_indices = self._model.bundle.peripheral_indices
        if len(periph_indices) == 0:
            return self.per_cartridge[..., 0, :]

        return np.mean(self.per_cartridge[..., periph_indices, :], axis=-2)

    @property
    def central_signal(self) -> np.ndarray:
        """The response of the central rhabdomere (Medulla color pathway)."""
        if self.shape[-1] == 1:
            return self.per_ommatidium[..., 0, :]
        return self.per_cartridge[..., getattr(self._model.bundle, 'center_index', 0), :]

    @property
    def lmc_input(self):
        """Alias to peripheral_signal"""
        return self.peripheral_signal

    @property
    def pale_input(self):
        """Alias to central_signal"""
        return self.central_signal

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        t_str = f', T={self.shape[0]}' if self.ndim == 3 else ''
        return f"VisualOutput(N={self.shape[-2]}, R={self.shape[-1]}{t_str}, shape={self.shape})"

    # Plotting methods

    def plot(self, pathway: str = 'all', projection: str = 'equirectangular',
             ax=None, false_colors: bool = False, uv_encoding: bool = False,
             draw_edges: bool = False, dark_mode: bool = False):
        """
        Displays the visual output as a gapless Voronoi tessellation (like in the first person shader).
        If this VisualOutput contains multiple timesteps, this plots the last one.
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from scipy.spatial import Voronoi

        if ax is None:
            if projection == 'equirectangular':
                fig, ax = plt.subplots(figsize=(10, 5))
            else:
                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(111, projection=projection)

        is_ts = self._data.ndim == 3

        # Extract spatial data and colour
        if pathway == 'all':
            rgb = self.colours[-1] if is_ts else self.colours
            az = self._model.receptors.azimuth
            el = self._model.receptors.elevation

        elif pathway in ('peripheral', 'central', 'ommatidium', 'cartridge'):
            match pathway:
                case 'peripheral':
                    sig = self.peripheral_signal
                case 'central':
                    sig = self.central_signal
                case 'ommatidium':
                    sig = np.mean(self.per_ommatidium, axis=-2)
                case 'cartridge':
                    sig = np.mean(self.per_cartridge, axis=-2)

            rgb = sig[-1, :, :3] if is_ts else sig[:, :3]
            az = self._model.ommatidia.azimuth
            el = self._model.ommatidia.elevation

            # Peripheral is greyscale
            if pathway == 'peripheral':
                rgb = np.repeat(np.mean(rgb, axis=-1, keepdims=True), 3, axis=-1)   # TODO: use weights?
        else:
            raise ValueError("Invalid pathway, must be must be 'all', 'peripheral', 'central', 'ommatidium', or 'cartridge'")

        # False colour modes
        rgb = np.clip(rgb, 0.0, 1.0).copy()
        if uv_encoding:
            rgb[:, 2] = np.clip(rgb[:, 2] + rgb[:, 0], 0.0, 1.0)
        elif false_colors:
            rgb[:, 0] = 0.0

        # TODO: use polygons.py module
        # Voronoi cells on the unwrapped cylinder
        pts = np.column_stack((az, el))
        pts_left = np.column_stack((az - 2 * np.pi, el))
        pts_right = np.column_stack((az + 2 * np.pi + 1e-6, el))
        cap_x = np.linspace(-2 * np.pi, 2 * np.pi, max(100, len(az) // 10))
        pts_top = np.column_stack((cap_x, np.full_like(cap_x, np.pi / 2 + 0.5)))
        pts_bot = np.column_stack((cap_x, np.full_like(cap_x, -np.pi / 2 - 0.5)))

        all_pts = np.vstack((pts, pts_left, pts_right, pts_top, pts_bot))
        vor = Voronoi(all_pts)

        polygons = []
        for i in range(len(pts)):
            polygons.append(vor.vertices[vor.regions[vor.point_region[i]]])

        if projection == 'equirectangular':
            polygons = [np.degrees(p) for p in polygons]
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_xlabel('Azimuth (deg)')
            ax.set_ylabel('Elevation (deg)')
            ax.set_aspect('equal', adjustable='box')
        else:
            for p in polygons:
                p[:, 0] = wrap_angle(p[:, 0])
            ax.grid(True, alpha=0.3)

        # Anti-aliasing workaround: 'face' draws a mini border of the polygon's
        # own color over the pixel gaps left by matplotlib
        edge_c = ('white' if dark_mode else 'black') if draw_edges else 'face'
        ax.add_collection(PolyCollection(polygons, facecolors=rgb, edgecolors=edge_c, linewidths=0.5, antialiaseds=True))

        if dark_mode:
            ax.set_facecolor('black')
            if ax.figure is not None:
                ax.figure.patch.set_facecolor('black')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('white')

        return ax

    def plot_time_series(self, pathway: str = 'peripheral',
                         max_items: int = 100, sort_by: str = 'azimuth',
                         false_colors: bool = False, uv_encoding: bool = False,
                         dark_mode: bool = False, ax=None):
        """
        Plots a spatio-temporal heatmap of the visual output over time.
        """
        import matplotlib.pyplot as plt

        if self._data.ndim != 3:
            raise ValueError("This visualisation requires a VisualOutput containing multiple timesteps.")
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, max(4, min(10, max_items * 0.1))))

        if pathway == 'all':
            data = self.colours
            az = self._model.receptors.azimuth

        elif pathway in ('peripheral', 'central', 'ommatidium', 'cartridge'):

            match pathway:
                case 'peripheral':
                    data = np.repeat(np.mean(self.peripheral_signal[..., :3], axis=-1, keepdims=True), 3, axis=-1)
                case 'central':
                    data = self.central_signal[..., :3]
                case 'ommatidium':
                    data = np.mean(self.per_ommatidium, axis=-2)[..., :3]
                case 'cartridge':
                    data = np.mean(self.per_cartridge, axis=-2)[..., :3]

            az = self._model.ommatidia.azimuth

        else:
            raise ValueError("Invalid pathway, must be 'all', 'peripheral', 'central', 'ommatidium', or 'cartridge'")

        data = np.clip(data, 0.0, 1.0)
        if uv_encoding:
            data[..., 2] = np.clip(data[..., 2] + data[..., 0], 0.0, 1.0)
        elif false_colors:
            data[..., 0] = 0.0

        # Transpose to (Space, Time, RGB) for imshow
        data = np.transpose(data, (1, 0, 2))
        if sort_by == 'azimuth':
            data = data[np.argsort(az)]

        # downsample (spatially) if needed
        S, T, C = data.shape
        if S > max_items:
            block_size = S // max_items
            data = data[:block_size * max_items].reshape(max_items, block_size, T, C).mean(axis=1)

        ax.imshow(data, aspect='auto', origin='lower', interpolation='none')
        ax.set_xlabel("Time step")

        ylabel = 'Items'
        if pathway == 'all':
            ylabel = 'Receptors'
        elif pathway in ('cartridge', 'peripheral', 'central'):
            ylabel = 'Cartridges'
        elif pathway == 'ommatidium':
            ylabel = 'Ommatidia'

        ax.set_ylabel(ylabel + (' (sorted Left to Right)' if sort_by == 'azimuth' else ''))

        if dark_mode:
            ax.set_facecolor('black')
            if ax.figure is not None:
                ax.figure.patch.set_facecolor('black')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('white')

        return ax
