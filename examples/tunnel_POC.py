from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

from insectvision.compound_eyes import Model
from insectvision.compound_eyes.helpers.alignment import BundlesAligner
from insectvision.compound_eyes.rhabdomeres import honeybee_bundle
from insectvision.renderers import Renderer
from insectvision.engine import Context, Agent, Scene, Asset
from insectvision.types import WORLD_FORWARD, OverlayColormap, WORLD_BACKWARD
from insectvision.engine.meshes import plane_geom
from insectvision.engine.materials_utils import checkerboard_texture
from insectvision.neuromorphic.basic_models import HassensteinReichardtEMD, GradientFlowDetector

from visualisation.plot_settings import (
    PlotSettings, Z_RASTER, Z_TEXT, panel_letter, column_header, row_header, placeholder
)

# Configuration

COL_YAW = '#c2a5cf'
COL_STRAFE = '#a6dba0'

COL_LEFT = '#EA5E00'
COL_RIGHT = '#0094D6'

MODE_YAW = 'yaw'
MODE_STRAFE = 'strafe'
MODE_LABEL = {MODE_YAW: 'Non-holonomic (yaw)', MODE_STRAFE: 'Holonomic (strafe)'}
MODE_COLOUR = {MODE_YAW: COL_YAW, MODE_STRAFE: COL_STRAFE}

WALL_SYMMETRIC = 'symmetric'
WALL_ASYMMETRIC = 'asymmetric'

MODEL_HRC = 'HRC'
MODEL_GRADIENT = 'gradient'
MODEL_LABEL = {MODEL_HRC: 'Hassenstein-Reichardt', MODEL_GRADIENT: 'Gradient (ratio)'}
MODEL_CLASS = {MODEL_HRC: HassensteinReichardtEMD, MODEL_GRADIENT: GradientFlowDetector}


@dataclass
class Configuration:

    # Tunnel geometry (metres)
    width: float = 0.2
    height: float = 0.2
    length: float = 10.0

    time_step: float = 0.01       # 10 ms time step: at 500 fps this means 5 times faster than real time

    # Texture
    block_size: int = 4                # fine chequer size
    asym_factor: int = 4               # larger chequer = block_size * asym_factor
    checkerboard_ratio: float = 0.5
    texture_res: int = 1024

    # Flight / control
    flight_speed: float = 0.5          # m/s base forward speed
    control_authority: float = 1.0     # k: lateral correction at full error, as a fraction of flight_speed
    yaw_damping: float = 3.0
    strafe_damping: float = 3.0
    speed_modulation: float = 0.5       # slow forward speed when steering hard (0 = off, 1 = full)
    motor_noise_frac: float = 0.30      # motor noise sd as a fraction of full-scale correction
    motor_noise: bool = False

    # Eyes param
    tau_membrane: float = 0.012

    # EMD params
    emd_pooling: Optional[int] = None       # auto

    # Batch
    n_trials: int = 30
    time_limit_s: float = 30.0          # per-trial time limit
    nb_samples: int = 256
    seed: Optional[int] = 0

    # Plotting
    n_quivers: int = 10                 # quiver samples per trajectory


# Logging

@dataclass
class RunLog:

    time: List[float] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    z: List[float] = field(default_factory=list)
    left_flow: List[float] = field(default_factory=list)
    right_flow: List[float] = field(default_factory=list)
    yaw: List[float] = field(default_factory=list)
    error: List[float] = field(default_factory=list)

    def arr(self, name: str) -> np.ndarray:
        return np.asarray(getattr(self, name), dtype=float)

    @property
    def dist(self) -> np.ndarray:
        """Distance travelled down tunnel (0 -> length)."""
        return -self.arr('z')


Results = Dict[Tuple[str, str, str], List[RunLog]]


# Scene construction

def _add_wall(scene: Scene, name: str, corners, block_size: int, cfg: Configuration):
    v, uv, idx = plane_geom(*corners)

    tex_h, tex_w = int(cfg.texture_res / cfg.height), int(cfg.texture_res / cfg.length)

    tex = checkerboard_texture(tex_w, tex_h, block_size=block_size, ratio=cfg.checkerboard_ratio)
    scene.add_instance(Asset.from_arrays(name=name, vertices=v, faces=idx, uv_coords=uv, texture=tex))


def build_tunnel(cfg: Configuration, wall_condition: str) -> Scene:

    w, h, l = cfg.width, cfg.height, cfg.length
    right_bs = cfg.block_size * cfg.asym_factor if wall_condition == WALL_ASYMMETRIC else cfg.block_size

    scene = Scene(background_color=(0.15, 0.15, 0.3))
    scene.add_sky('assets/textures/kloppenheim_05_4k.exr')

    _add_wall(scene, 'left_wall',
              ([-w/2, 0.0, -l], [-w/2, h, -l], [-w/2, h, 0.0], [-w/2, 0.0, 0.0]),
              cfg.block_size, cfg)
    _add_wall(scene, 'right_wall',
              ([w/2, 0.0, 0.0], [w/2, h, 0.0], [w/2, h, -l], [w/2, 0.0, -l]),
              right_bs, cfg)
    _add_wall(scene, 'bottom_wall',
              ([-w/2, 0.0, 0.0], [w/2, 0.0, 0.0], [w/2, 0.0, -l], [-w/2, 0.0, -l]),
              cfg.block_size, cfg)
    _add_wall(scene, 'top_wall',
              ([-w/2, h, 0.0], [w/2, h, 0.0], [w/2, h, -l], [-w/2, h, -l]),
              cfg.block_size, cfg)
    return scene


def random_tunnel_start(width, height, margin_pct: float = 0.5, randomise_height=False,
                        rng: Optional[np.random.Generator] = None):

    rng = rng or np.random.default_rng()
    keep = 1.0 - np.clip(margin_pct, 0.0, 1.0)
    start_x = (rng.random() - 0.5) * (width * keep)
    start_y = rng.random() * (height * keep) if randomise_height else (height * keep) / 2.0

    return start_x, start_y, 0.0


def randomise_wall_textures(scene: Scene, cfg: Configuration, wall_condition: str):
    """Regenerates random checkerboard patterns for all walls and updates the GPU."""

    tex_h, tex_w = int(cfg.texture_res / cfg.height), int(cfg.texture_res / cfg.length)
    right_bs = cfg.block_size * cfg.asym_factor if wall_condition == WALL_ASYMMETRIC else cfg.block_size

    for asset in scene.assets.values():
        if not asset.name.endswith('_wall'):
            continue

        bs = right_bs if asset.name == 'right_wall' else cfg.block_size
        new_tex = checkerboard_texture(tex_w, tex_h, block_size=bs, ratio=cfg.checkerboard_ratio)

        asset.set_texture(new_tex)


# Runner functions ---------------------------------------------------------------------------

def run_trial(cfg: Configuration, context: Context, renderer: 'Renderer',
              agent: Agent, left_eye, right_eye, mode: str, model_name: str,
              rng: np.random.Generator, draw: bool = False) -> RunLog:
    """One closed-loop run from a random lateral start to the end of the tunnel."""

    # Randomise lateral start
    sx, _, sz = random_tunnel_start(cfg.width, cfg.height, randomise_height=False, rng=rng)

    agent.position = (sx, cfg.height / 2.0, sz)
    agent.yaw = agent.pitch = agent.roll = 0.0

    EMD = MODEL_CLASS[model_name]

    left_emd = EMD(eye=left_eye, direction=-WORLD_FORWARD, pooling_k=cfg.emd_pooling)
    right_emd = EMD(eye=right_eye, direction=-WORLD_FORWARD, pooling_k=cfg.emd_pooling)

    log = RunLog()
    w, l = cfg.width, cfg.length
    t0 = context.total_time

    summ_ref = 0.001

    while context.run_interactive():

        context.input()
        dt = context.dt

        output = renderer.step()

        if output is None:
            continue

        left_motion = left_emd.process(output, dt)
        right_motion = right_emd.process(output, dt)

        left_estim = left_emd.last_estimate
        right_estim = right_emd.last_estimate

        diff = right_estim - left_estim
        summ = abs(right_estim) + abs(left_estim)
        summ_ref += 0.01 * (summ - summ_ref)    # slow EMA of the bilateral signal scale
        error = diff / (summ + 0.1 * summ_ref)

        k = cfg.control_authority
        u = k * error
        # u = k * np.tanh(error * 5.0)

        if cfg.motor_noise:
            u += rng.normal(0.0, cfg.motor_noise_frac * k)

        # Slow forward speed when steering hard (more time to correct before overshoot)
        v_fwd = cfg.flight_speed * (1.0 - cfg.speed_modulation * min(abs(error), 1.0))

        if mode == MODE_YAW:
            sign = 1.0

            psi_des = np.rad2deg(np.arcsin(np.clip(u * sign, -1.0, 1.0)))  # deg
            turn_rate = psi_des - cfg.yaw_damping * agent.yaw  # deg/s

            agent.rotate(yaw=turn_rate * dt).translate(agent.forward * v_fwd * dt)

        elif mode == MODE_STRAFE:
            sign = -1.0

            strafe_rate = (agent.right * u * sign * cfg.flight_speed) / max(1.0, cfg.strafe_damping)

            agent.translate((agent.forward * v_fwd + strafe_rate) * dt)

        else:
            raise ValueError(f'Unknown mode {mode!r}')

        # Stay inside the tunnel
        pos = agent.position
        pos.x = np.clip(pos.x, -(w * 0.48), (w * 0.48))
        pos.y = np.clip(pos.y, 0.1, cfg.height - 0.1)
        agent.position = pos

        if draw:
            renderer.set_overlay(
                {left_eye: left_motion, right_eye: right_motion},
                colormap=OverlayColormap.Diverging, compression=0.8)

            context.display()

        log.time.append(context.total_time)
        log.x.append(float(agent.position.x))
        log.y.append(float(agent.position.y))
        log.z.append(float(agent.position.z))
        log.left_flow.append(left_estim)
        log.right_flow.append(right_estim)
        log.yaw.append(float(agent.yaw))
        log.error.append(error)

        if agent.position.z < -l or (context.total_time - t0) >= cfg.time_limit_s:
            break

    return log


def run_all_trials(cfg: Configuration) -> Results:
    """Run n_trials per (wall condition x motion model x control mode)."""

    context = Context(window_size=(1280, 720))
    context.mouse_captured = False

    context.time_step = cfg.time_step

    aligner = BundlesAligner(
        equatorial_discontinuity=False,  # no equatorial discontinuity in the bee
        ref_direction=WORLD_BACKWARD,   # optic flow in flight
        combing_strength=1.0,
        combing_angle_deg=45.0,
        alignment_smoothing_iter=4,
        saccade_smoothing_iter=5,
    )

    model = Model.from_file(
        'assets/honeybee_scaffold_s10.npz',
        bundle=honeybee_bundle(),
        orientation=aligner,
        neural_superposition=False,     # Apposition eyes
    )

    model.scale(1e-6)

    model.tau_membrane = cfg.tau_membrane

    left_eye, right_eye = model.eyes

    agent = Agent()

    results: Results = {}

    for wall_condition in (WALL_SYMMETRIC, WALL_ASYMMETRIC):

        scene = build_tunnel(cfg, wall_condition)
        scene.sun.elevation, scene.sun.azimuth, scene.sun.color = 0.0, 0.0, (1.0, 1.0, 1.0)

        renderer = Renderer(
            model=model, scene=scene, agent=agent,
            nb_samples=cfg.nb_samples, time_dithering=True,
            randomness_mode='Halton', enable_shadows=False
        )

        for model_name in (MODEL_HRC, MODEL_GRADIENT):

            rng = np.random.default_rng(cfg.seed)

            for mode in (MODE_YAW, MODE_STRAFE):
                runs: List[RunLog] = []

                randomise_wall_textures(scene, cfg, wall_condition)

                for t in range(cfg.n_trials):
                    print(f'  {wall_condition:11s} / {MODEL_LABEL[model_name]:22s} / '
                          f'{MODE_LABEL[mode]:24s}  trial {t+1}/{cfg.n_trials}')

                    runs.append(
                        run_trial(cfg, context, renderer, agent,
                                  left_eye, right_eye, mode, model_name, rng, draw=True)
                    )

                results[(wall_condition, mode, model_name)] = runs

    context.free()

    return results


# Figures ---------------------------------------------------------------------------

def _representative(runs: List[RunLog]) -> RunLog:
    """Pick the trial whose steady-state is the median (least cherry-picked)."""

    if not runs:
        return RunLog()

    key = []
    for r in runs:
        if r.dist.size == 0:
            continue

        xs = r.arr('x')[-1:]
        mean_xs = np.mean(xs)
        key.append(np.abs(mean_xs))

    return runs[int(np.argsort(key)[len(key) // 2])]


NEEDLE = Path([(0.0, 0.0), (1.0, 0.0)])

def _steady_offset(run: RunLog, cfg: Configuration, last_frac: float = 0.25) -> float:
    """Signed mean lateral position over the final `last_frac` of the tunnel."""

    x = run.arr('x')
    d = run.dist
    if x.size == 0:
        return np.nan

    return float(np.mean(x[d >= d.max() - last_frac * cfg.length]))


def _trajectory_panel(
        ax,
        results: Results,
        wall_condition: str,
        model_name: str,
        cfg: Configuration,
        s: PlotSettings,
        show_xlabel: bool = True,
        show_ylabel: bool = True,
        show_legend: bool = False
    ):

    # Everything below Z_RASTER is flattened to pixels on PDF/EPS export
    ax.set_rasterization_zorder(Z_RASTER)

    for mode in (MODE_YAW, MODE_STRAFE):
        runs = results.get((wall_condition, mode, model_name), [])
        colour = MODE_COLOUR[mode]

        for r in runs:
            ax.plot(r.dist, r.arr('x'), color=colour, lw=0.5, alpha=0.22, zorder=2)

        rep = _representative(runs)

        # Quivers for representative trial
        d, x = rep.dist, rep.arr('x')
        if d.size < 2:
            continue

        offset = 0 if mode == MODE_YAW else 3
        samples = np.linspace(d.min(), d.max(), cfg.n_quivers + offset)
        idx = np.clip(np.searchsorted(d, samples), 0, d.size - 1)

        yaws = rep.arr('yaw')

        for i in idx:
            ax.plot(d[i], x[i],
                    marker='o',
                    markersize=3.2,
                    markerfacecolor=colour,
                    markeredgewidth=0.5,
                    markeredgecolor='white',
                    zorder=Z_RASTER + 2)

            ax.plot(d[i], x[i],
                    marker=MarkerStyle(NEEDLE, transform=Affine2D().scale(0.5).rotate_deg(yaws[i])),
                    markersize=5.0,
                    color=colour,
                    markeredgewidth=0.8,
                    zorder=Z_RASTER + 3)

    half_width = cfg.width / 2
    lim = half_width * 1.25

    ax.axhline(0.0, color=s.frame, ls='--', lw=0.5, zorder=1)
    ax.axhline(-half_width, color=s.dark, lw=1.0, zorder=1)
    ax.axhline(half_width, color=s.dark, lw=1.0, zorder=1)

    ax.set_xlim(0, cfg.length)
    ax.set_ylim(-lim, lim)
    ax.set_yticks([-half_width, 0.0, half_width])
    ax.tick_params(labelsize=s.base)

    if show_xlabel:
        ax.set_xlabel('Distance down tunnel (m)', fontsize=s.small)
    else:
        ax.tick_params(labelbottom=False)

    if show_ylabel:
        ax.set_ylabel('R  $\\leftarrow$  Position x (m)  $\\rightarrow$  L', fontsize=s.small)
    else:
        ax.tick_params(labelleft=False)

    ax.invert_yaxis()   # Left on top

    if show_legend:
        handles = [Line2D([0], [0], color=MODE_COLOUR[m], lw=1.4, label=MODE_LABEL[m])
                   for m in (MODE_YAW, MODE_STRAFE)]
        leg = ax.legend(handles=handles, fontsize=s.tiny, loc='lower right',
                        ncol=2, handlelength=1.1, handletextpad=0.4,
                        columnspacing=0.9, borderpad=0.25)
        leg.get_frame().set_edgecolor(s.frame)
        leg.get_frame().set_linewidth(0.3)
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_alpha(0.92)
        leg.set_zorder(Z_TEXT)


def _flow_panel(ax, results: Results, cfg: Configuration, s: PlotSettings):
    """Bilateral flow + balance error for one representative run."""

    ax.set_rasterization_zorder(Z_RASTER)

    rep = _representative(results.get((WALL_SYMMETRIC, MODE_YAW, MODEL_HRC), []))
    dist = rep.dist

    l_raw = rep.arr('left_flow')
    r_raw = rep.arr('right_flow')

    # min/max across both eyes for this trial
    g_min = min(l_raw.min(), r_raw.min())
    g_max = max(l_raw.max(), r_raw.max())

    left_flow_norm = (l_raw - g_min) / (g_max - g_min + 1e-6)
    right_flow_norm = (r_raw - g_min) / (g_max - g_min + 1e-6)

    ax.plot(dist, left_flow_norm, color=COL_LEFT, lw=s.curve_lw, alpha=0.85,
            label='left eye', zorder=4)
    ax.plot(dist, right_flow_norm, color=COL_RIGHT, lw=s.curve_lw, alpha=0.85,
            label='right eye', zorder=4)

    ax.axhline(0.0, color=s.frame, ls='--', lw=0.5, zorder=1)
    ax.set_xlim(0, cfg.length)
    ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel('Distance down tunnel (m)', fontsize=s.small)
    ax.set_ylabel('Mean EMD response', fontsize=s.small)
    ax.tick_params(labelsize=s.base)

    ax.set_title('Bilateral flow & balance error', fontsize=s.title,
                 loc='left', pad=11)

    # -> which of the eight cells this representative run came from
    ax.annotate(f'{MODEL_LABEL[MODEL_HRC]} / {MODE_LABEL[MODE_YAW].lower()} / symmetric',
                xy=(0.0, 1.012), xycoords='axes fraction', ha='left', va='bottom',
                fontsize=s.tiny, color=s.frame)

    axErr = ax.twinx()
    axErr.plot(dist, rep.arr('error'), color=s.frame, lw=0.7, alpha=0.9,
               zorder=3, label='balance error')
    axErr.set_ylabel('Balance error', color=s.frame, fontsize=s.small)
    axErr.set_ylim(-1.05, 1.05)
    axErr.tick_params(axis='y', colors=s.frame, labelsize=s.base)
    axErr.spines['right'].set_color(s.frame)

    handles = [Line2D([0], [0], color=COL_LEFT, lw=1.2, label='Left eye'),
               Line2D([0], [0], color=COL_RIGHT, lw=1.2, label='Right eye'),
               Line2D([0], [0], color=s.frame, lw=1.0, label='Balance error')]
    leg = ax.legend(handles=handles, fontsize=s.tiny, loc='upper left',
                    handlelength=1.1, handletextpad=0.4, borderpad=0.25,
                    labelspacing=0.2)

    leg.get_frame().set_edgecolor(s.frame)
    leg.get_frame().set_linewidth(0.3)

    leg.set_zorder(Z_TEXT)


def _offset_panel(ax, results: Results, cfg: Configuration, s: PlotSettings):
    """
    Steady-state lateral offset per cell: does the agent hold the midline?
    """

    cells = [(WALL_SYMMETRIC, MODEL_HRC), (WALL_ASYMMETRIC, MODEL_HRC),
             (WALL_SYMMETRIC, MODEL_GRADIENT), (WALL_ASYMMETRIC, MODEL_GRADIENT)]
    half_width = cfg.width / 2

    ax.axhspan(-half_width, half_width, color=s.grid, zorder=0)
    ax.axhline(0.0, color=s.frame, ls='--', lw=0.5, zorder=1)

    for j, cell in enumerate(cells):
        for k, mode in enumerate((MODE_YAW, MODE_STRAFE)):

            offs = np.array(
                [_steady_offset(r, cfg) for r in results.get((cell[0], mode, cell[1]), [])]
            )

            offs = offs[np.isfinite(offs)]
            if offs.size == 0:
                continue

            xpos = j + (k - 0.5) * 0.30
            jitter = np.zeros_like(offs) if offs.size == 1 else np.linspace(-0.055, 0.055, offs.size)

            ax.plot(xpos + jitter, offs, ls='none', marker='o', ms=2.4,
                    mfc=MODE_COLOUR[mode], mec='white', mew=0.3, alpha=0.9, zorder=4)
            ax.plot([xpos - 0.10, xpos + 0.10], [np.median(offs)] * 2,
                    color=s.dark, lw=1.1, solid_capstyle='butt', zorder=5)

    ax.set_xticks(range(len(cells)))

    ax.set_xticklabels(['Sym', 'Asym', 'Sym', 'Asym'], fontsize=s.small)

    ax.set_xlim(-0.5, len(cells) - 0.5)
    ax.set_ylim(-half_width * 1.25, half_width * 1.25)

    ax.set_yticks([-half_width, 0.0, half_width])

    ax.invert_yaxis()

    ax.tick_params(labelsize=s.base)
    ax.set_ylabel('R  $\\leftarrow$  x (m)  $\\rightarrow$  L', fontsize=s.small)

    ax.set_title('Steady-state offset (final 25% of tunnel)',
                 fontsize=s.title, loc='left', pad=3)

    # Group the four cells into their two models along the bottom.
    for xc, label in ((0.5, MODEL_LABEL[MODEL_HRC]), (2.5, MODEL_LABEL[MODEL_GRADIENT])):
        ax.annotate(label, xy=(xc, -0.135), xycoords=('data', 'axes fraction'),
                    ha='center', va='top', fontsize=s.tiny, color=s.dark,
                    annotation_clip=False)

    ax.annotate('median', xy=(0.99, 0.03), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=s.tiny, color=s.dark)


def make_figure(results: Results, cfg: Configuration, s: Optional[PlotSettings] = None) -> plt.Figure:

    s = (s or PlotSettings.nature_double(height_mm=170.0)).apply()

    fig = s.new_figure()

    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.00, 2.00, 0.90], hspace=0.40)
    gsB = gs[1].subgridspec(2, 2, hspace=0.14, wspace=0.10)
    gsCD = gs[2].subgridspec(1, 2, wspace=0.46)

    # A: task-summary placeholder
    axA = fig.add_subplot(gs[0])
    placeholder(axA, s,'(placeholder: tunnel schematic, insect view)')

    # B: 2 models x 2 wall conditions
    cells = {}
    for i, model_name in enumerate((MODEL_HRC, MODEL_GRADIENT)):
        for j, wall in enumerate((WALL_SYMMETRIC, WALL_ASYMMETRIC)):
            ax = fig.add_subplot(gsB[i, j])
            _trajectory_panel(ax, results, wall, model_name, cfg, s,
                              show_xlabel=(i == 1),
                              show_ylabel=(j == 0),
                              show_legend=(i == 0 and j == 0))
            cells[(i, j)] = ax

    # C / D: mechanism and outcome
    axC = fig.add_subplot(gsCD[0])
    _flow_panel(axC, results, cfg, s)

    axD = fig.add_subplot(gsCD[1])
    _offset_panel(axD, results, cfg, s)

    fig.subplots_adjust(left=0.135, right=0.955, top=0.945, bottom=0.075)

    column_header(fig, s, cells[(0, 0)], 'Symmetric walls')
    column_header(fig, s, cells[(0, 1)],
                  f'Asymmetric walls (right {cfg.asym_factor}$\\times$ larger)')
    row_header(fig, s, cells[(0, 0)], MODEL_LABEL[MODEL_HRC], dx=0.098, colour=s.dark)
    row_header(fig, s, cells[(1, 0)], MODEL_LABEL[MODEL_GRADIENT], dx=0.098, colour=s.dark)

    panel_letter(fig, s, 'A', axA, dx=-0.076, dy=0.0)
    panel_letter(fig, s, 'B', cells[(0, 0)], dx=-0.076, dy=0.030)
    panel_letter(fig, s, 'C', axC, dx=-0.014)
    panel_letter(fig, s, 'D', axD, dx=-0.014)

    return fig


## --------------------------------------------------------------------------

if __name__ == '__main__':

    cfg = Configuration()

    results = run_all_trials(cfg)

    settings = PlotSettings.nature_double(height_mm=170.0)
    fig = make_figure(results, cfg, settings)
    settings.savefig(fig, 'centering_figure', formats=['png', 'pdf', 'svg'])