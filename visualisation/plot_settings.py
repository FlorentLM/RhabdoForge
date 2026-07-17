"""
Shared figure settings.
"""
from dataclasses import dataclass
from typing import Optional, Sequence
import matplotlib.pyplot as plt


MM_PER_IN = 25.4


@dataclass
class PlotSettings:

    width_mm: float = 183.0            # Nature double column (PLOS max 190.5)
    height_mm: Optional[float] = None  # if None, derived from 'aspect'
    aspect: float = 0.95               # height / width when height_mm is None
    dpi: int = 600                          # Raster image export (PNG/TIFF)
    raster_dpi: int = 300                   # Resolution of rasterised layers inside EPS/PDF
    screen_dpi: int = 100                   # dpi for the interactive viewer (not the export dpi)

    font_family: Sequence[str] = ('Arial', 'Helvetica', 'DejaVu Sans')
    base: float = 7.0                  # body text + tick labels
    small: float = 6.5                 # secondary annotations, legends
    tiny: float = 6.0                  # Smallest labels
    title: float = 8.0                 # Panel titles
    header: float = 8.5                # column / row-group headers
    initials: float = 14               # Big letters for subfigures (A, B, C ...)

    axis_lw: float = 0.6
    curve_lw: float = 0.8
    grid_lw: float = 0.5

    bg: str = 'white'
    yellorange: str = '#FFC32F'
    green: str = '#27AE60'
    blue: str = '#2980B9'
    red: str = '#C0392B'
    dark: str = '#34495E'
    frame: str = '#7F8C8D'
    grid: str = '#EAECEE'

    formats: Sequence[str] = ('eps', 'pdf', 'png')
    rasterize: bool = True

    @property
    def width_in(self) -> float:
        return self.width_mm / MM_PER_IN

    @property
    def height_in(self) -> float:
        if self.height_mm is not None:
            return self.height_mm / MM_PER_IN
        return self.width_in * self.aspect

    @property
    def figsize(self):
        return (self.width_in, self.height_in)

    def apply(self) -> 'PlotSettings':
        import shutil
        import platform
        if 'Windows' in platform.system():
            gs_bin = 'gswin64c'
        else:
            gs_bin = 'gs'
        if shutil.which(gs_bin):
            plt.rcParams['ps.usedistiller'] = 'ghostscript'

        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': list(self.font_family),
            'font.size': self.base,
            'axes.titlesize': self.title,
            'axes.titleweight': 'normal',
            'axes.labelsize': self.base,
            'xtick.labelsize': self.base,
            'ytick.labelsize': self.base,
            'legend.fontsize': self.small,
            'axes.linewidth': self.axis_lw,
            'xtick.major.width': self.axis_lw,
            'ytick.major.width': self.axis_lw,
            'lines.linewidth': self.curve_lw,
            'mathtext.fontset': 'dejavusans',
            'axes.unicode_minus': True,
            'svg.fonttype': 'none',
            'pdf.fonttype': 42,   # embed TrueType in PDF
            'ps.fonttype': 42,    # embed TrueType in EPS/PS
            'ps.useafm': False,
            'figure.dpi': self.screen_dpi,   # only interactive window, export dpi is set in savefig()
            'savefig.dpi': self.dpi,
            'figure.facecolor': self.bg,
            'savefig.facecolor': self.bg,
        })
        return self

    def new_figure(self) -> plt.Figure:
        return plt.figure(figsize=self.figsize, dpi=self.screen_dpi, facecolor=self.bg)

    def savefig(self, fig: plt.Figure, name: str, formats: Optional[Sequence[str]] = None):
        raster_exts = {'png', 'tif', 'tiff', 'jpg', 'jpeg'}
        for ext in (formats or self.formats):
            dpi = self.dpi if ext.lower() in raster_exts else self.raster_dpi
            fig.savefig(f'{name}.{ext}', format=ext, dpi=dpi, facecolor=self.bg)

    @classmethod
    def nature_double(cls, **kw) -> 'PlotSettings':
        """Nature double column: 183 mm, 5-7 pt type."""
        return cls(width_mm=183.0, base=7.0, small=6.5, tiny=6.0,
                   title=8.0, header=8.5, dpi=600, **kw)

    @classmethod
    def nature_single(cls, **kw) -> 'PlotSettings':
        """Nature single column: 89 mm."""
        return cls(width_mm=89.0, base=6.0, small=5.5, tiny=5.0,
                   title=7.0, header=7.0, dpi=600, **kw)

    @classmethod
    def plos(cls, **kw) -> 'PlotSettings':
        """PLOS: up to 190.5 mm, 8-12 pt type."""
        return cls(width_mm=183.0, base=8.0, small=7.0, tiny=7.0,
                   title=9.0, header=9.5, dpi=600, **kw)


## Some shared drawing helpers

# zorder threshold: everything below this goes into one rasterised layer per ax
# (keeps translucent overlays out of the vector stream, so EPS/PDF stay small)
Z_RASTER = 8.0
Z_TEXT = 12.0   # text remains vector


def panel_letter(fig, s: PlotSettings, letter: str, ref, dx: float = 0.0, dy: float = 0.012):
    """
    Bold panel initial (A, B, C ...) at the top-left of 'ref'.
    ('ref' is anything with a bounding box in figure coordinates: an Axes, a SubplotSpec or a Bbox)
    """
    pos = ref.get_position() if hasattr(ref, 'get_position') else ref
    fig.text(pos.x0 + dx, pos.y1 + dy, letter, ha='right', va='bottom',
             fontsize=s.initials, fontweight='bold', color='black', zorder=Z_TEXT)


def column_header(fig, s: PlotSettings, ax, text: str, dy: float = 0.008, **kw):
    """Centred header above a column of panels."""
    p = ax.get_position()
    return fig.text((p.x0 + p.x1) / 2, p.y1 + dy, text,
                    ha='center', va='bottom', fontsize=s.header, zorder=Z_TEXT, **kw)


def row_header(fig, s: PlotSettings, ax, text: str, dx: float = 0.022, colour: str = 'black', **kw):
    """Rotated bold header to the left of a row of panels."""
    p = ax.get_position()
    return fig.text(p.x0 - dx, (p.y0 + p.y1) / 2, text, rotation=90,
                    va='center', ha='center', fontweight='bold',
                    fontsize=s.header, color=colour, zorder=Z_TEXT, **kw)


def despine(ax, keep=('left', 'bottom')):
    for name, spine in ax.spines.items():
        spine.set_visible(name in keep)


def placeholder(ax, s: PlotSettings, text: str = '(schematic)'):
    """Dashed empty box to drop an illustrator-drawn schematic into later."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(s.frame)
        spine.set_linewidth(s.axis_lw)
        spine.set_linestyle((0, (4, 3)))

    ax.text(0.5, 0.5, text, ha='center', va='center',
            fontsize=s.header, color=s.frame, transform=ax.transAxes)
    return ax