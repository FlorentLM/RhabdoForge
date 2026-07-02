from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from insectvision.compound_eyes import Model
from insectvision.utils import WORLD_FORWARD
from insectvision.renderers import Renderer
from insectvision.engine import Context, Agent, Scene, Asset
from insectvision.utils import Colormap, norm_minmax
from insectvision.engine.meshes import plane_geom
from insectvision.engine.materials_utils import checkerboard_texture
from insectvision.neuromorphic.basic_models import HassensteinReichardtEMD, GradientFlowDetector

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
    eye_model_path: str = 'assets/drosophila_scaffold.npz'
    # eye_model_path: str = 'assets/honeybee_scaffold_s10.npz'

    # EMD params
    emd_pooling: int = 1

    # Batch
    n_trials: int = 1
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

    # Tuning the EMD to the expected forward velocity
    target_v = cfg.flight_speed / (cfg.width / 2.0)  # expected angular velocity at tunnel centre

    left_emd = EMD(eye=left_eye, direction=-WORLD_FORWARD, target_velocity=target_v, pooling_k=cfg.emd_pooling)
    right_emd = EMD(eye=right_eye, direction=-WORLD_FORWARD, target_velocity=target_v, pooling_k=cfg.emd_pooling)

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
                colormap=Colormap.Diverging, compression=0.8)

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

    model = Model.from_file(cfg.eye_model_path)
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


def _trajectory_panel(
        ax,
        results: Results,
        wall_condition: str,
        model_name: str,
        cfg: Configuration,
        title: str
    ):

    for mode in (MODE_YAW, MODE_STRAFE):
        runs = results.get((wall_condition, mode, model_name), [])
        colour = MODE_COLOUR[mode]

        for r in runs:
            ax.plot(r.dist, r.arr('x'), color=colour, lw=0.8, alpha=0.22, zorder=2)

        rep = _representative(runs)

        # Quivers for representative trial
        d, x = rep.dist, rep.arr('x')
        if d.size < 2:
            continue

        offset = 0 if mode == MODE_YAW else 3
        samples = np.linspace(d.min(), d.max(), cfg.n_quivers + offset)
        idx = np.clip(np.searchsorted(d, samples), 0, d.size - 1)

        for i in idx:
            ax.plot(d[i], x[i],
                    marker='o',
                    markersize=8,
                    markerfacecolor=colour,
                    markeredgewidth=1.5,
                    markeredgecolor='white',
                    zorder=6)

            yw = rep.arr('yaw')[i]

            t = matplotlib.markers.MarkerStyle(marker='_')
            t._transform = t.get_transform().translate(0.5, 0.0).rotate_deg(yw)

            ax.plot(d[i], x[i],
                    marker=t,
                    markersize=10,
                    color=colour,
                    zorder=7)

        ax.plot([], [], color=colour, label=mode, lw=1.1, zorder=0)

    half_width = cfg.width / 2
    lim = half_width * 1.25

    ax.axhline(0.0, color='k', ls='--', lw=0.6)
    ax.axhline(-half_width, color='grey', lw=1.0)
    ax.axhline(half_width, color='grey', lw=1.0)

    ax.set_xlim(0, cfg.length)
    ax.set_ylim(-lim, lim)

    ax.set_xlabel('Distance down tunnel (m)')
    ax.set_ylabel('R ←  Position x (m)  → L')
    ax.invert_yaxis()   # Left on top

    ax.set_title(title)

    ax.legend(fontsize=7, loc='upper right')


def make_figure(
        results: Results,
        cfg: Configuration
) -> plt.Figure:

    fig = plt.figure(figsize=(11, 17), constrained_layout=True)
    fig.suptitle('Optic-flow centering', fontsize=14, fontweight='bold')

    # Row 0: summary, Rows 1-2: HRC / gradient trajectories, Row 3: flow + error
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.0, 0.9, 0.9, 0.9])

    # A: task-summary placeholder
    axA = fig.add_subplot(gs[0, :])
    axA.text(0.5, 0.5, '(graphical summary)',
             ha='center', va='center', fontsize=12, color='grey',
             transform=axA.transAxes)

    axA.set_xticks([])
    axA.set_yticks([])

    asym_title = f'Asymmetric walls (right {cfg.asym_factor}x larger)'

    # Row 1: Hassenstein-Reichardt correlator
    _trajectory_panel(fig.add_subplot(gs[1, 0]), results, WALL_SYMMETRIC, MODEL_HRC, cfg,
                      f'{MODEL_LABEL[MODEL_HRC]} — Symmetric walls')
    _trajectory_panel(fig.add_subplot(gs[1, 1]), results, WALL_ASYMMETRIC, MODEL_HRC, cfg,
                      f'{MODEL_LABEL[MODEL_HRC]} — {asym_title}')

    # Row 2: Gradient (ratio) detector
    _trajectory_panel(fig.add_subplot(gs[2, 0]), results, WALL_SYMMETRIC, MODEL_GRADIENT, cfg,
                      f'{MODEL_LABEL[MODEL_GRADIENT]} — Symmetric walls')
    _trajectory_panel(fig.add_subplot(gs[2, 1]), results, WALL_ASYMMETRIC, MODEL_GRADIENT, cfg,
                      f'{MODEL_LABEL[MODEL_GRADIENT]} — {asym_title}')

    # D: bilateral flow + balance error (representative HRC symmetric-yaw)
    axD = fig.add_subplot(gs[3, :])
    rep = _representative(results.get((WALL_SYMMETRIC, MODE_YAW, MODEL_HRC), []))

    left_flow = rep.arr('left_flow')
    right_flow = rep.arr('right_flow')

    left_flow_norm = norm_minmax(left_flow)
    right_flow_norm = norm_minmax(right_flow)

    err = rep.arr('error')
    dist = rep.dist

    axD.plot(dist, left_flow_norm, color=COL_LEFT, lw=0.9, alpha=0.8, label='left eye')
    axD.plot(dist, right_flow_norm, color=COL_RIGHT, lw=0.9, alpha=0.8, label='right eye')

    axD.axhline(0.0, color='k', ls='--', lw=0.5)
    axD.set_xlabel('Distance down tunnel (m)')
    axD.set_ylabel('Mean EMD response')
    axD.set_ylim(-0.05, 1.05)

    axD.set_title('Bilateral flow & balance error')
    axD.legend(fontsize=7, loc='upper left')

    axErr = axD.twinx()

    axErr.plot(dist, err, color='grey', lw=0.8, alpha=0.7)
    axErr.axhline(0.0, color='grey', ls='--', lw=0.5)
    axErr.set_ylabel('balance error', color='grey')
    axErr.set_ylim(-1.05, 1.05)
    axErr.tick_params(axis='y', colors='grey')

    return fig

# --------------------------------------------------------------------------

if __name__ == '__main__':

    cfg = Configuration()

    results = run_all_trials(cfg)

    fig = make_figure(results, cfg)
    fig.savefig('centering_figure.png', dpi=200)
    # fig.savefig('centering_figure.svg', format='svg')

    plt.show()