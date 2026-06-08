from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from insectvision.engine import Context, Agent, Scene, Asset
from insectvision.compound_eyes import CompoundEyeModel
from insectvision.renderers import Raytracer
from insectvision.utils import Colormap
from insectvision.geometry import plane_geom
from insectvision.geometry.materials_utils import checkerboard_texture
from insectvision.neuromorphic.basic_models import HassensteinReichardtEMD


# Configuration

COL_YAW = '#E3BE6B'
COL_STRAFE = '#3DB1A6'

COL_LEFT = '#DB4C77'
COL_RIGHT = '#10559a'

MODE_YAW = 'yaw'
MODE_STRAFE = 'strafe'
MODE_LABEL = {MODE_YAW: 'Non-holonomic (yaw)', MODE_STRAFE: 'Holonomic (strafe)'}
MODE_COLOUR = {MODE_YAW: COL_YAW, MODE_STRAFE: COL_STRAFE}

WALL_SYMMETRIC = 'symmetric'
WALL_ASYMMETRIC = 'asymmetric'


@dataclass
class Configuration:
    # Tunnel geometry (metres)
    width: float = 0.2
    height: float = 0.2
    length: float = 6.0

    time_step: float = 0.01            # 10 ms time step: at 500 fps this means 5 times faster than real time

    # Texture
    block_size: int = 4                # fine chequer size
    asym_factor: int = 4               # larger chequer = block_size * asym_factor
    checkerboard_ratio: float = 0.5
    texture_res: int = 1024

    # Flight / control
    flight_speed: float = 0.5          # m/s
    yaw_gain: float = 30.0
    damping_gain: float = 5.0
    strafe_gain: float = 0.05
    motor_noise: bool = True

    eye_model_path: str = 'species_models/drosophila_custom.npz'

    # Batch
    n_trials: int = 25
    time_limit_s: float = 30.0          # per-trial time limit
    nb_samples: int = 256
    seed: Optional[int] = 0

    # Plotting
    heading_quiver_gain: float = 1.0    # visual exaggeration of yaw quivers
    n_quivers: int = 20                 # quiver samples per trajectory


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


Results = Dict[Tuple[str, str], List[RunLog]]


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
    scene.add_skybox('assets/textures/bright_day_nosun')

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


def randomise_wall_textures(scene: Scene, cfg: Configuration, wall_condition: str, renderer: Raytracer):
    """Regenerates random checkerboard patterns for all walls and updates the GPU."""

    tex_h, tex_w = int(cfg.texture_res / cfg.height), int(cfg.texture_res / cfg.length)
    right_bs = cfg.block_size * cfg.asym_factor if wall_condition == WALL_ASYMMETRIC else cfg.block_size

    for asset in scene.assets.values():
        if not asset.name.endswith('_wall'):
            continue

        bs = right_bs if asset.name == 'right_wall' else cfg.block_size
        new_tex = checkerboard_texture(tex_w, tex_h, block_size=bs, ratio=cfg.checkerboard_ratio)

        asset.set_texture(new_tex)
        renderer.update_texture(asset)


# Runner functions ---------------------------------------------------------------------------

def run_trial(cfg: Configuration, context: Context, scene: Scene, renderer: Raytracer,
              agent: Agent, left_eye, right_eye, mode: str,
              rng: np.random.Generator, draw: bool = False) -> RunLog:
    """One closed-loop run from a random lateral start to the end of the tunnel."""

    # Randomise lateral start
    sx, _, sz = random_tunnel_start(cfg.width, cfg.height, randomise_height=False, rng=rng)

    agent.position = (sx, cfg.height / 2.0, sz)
    agent.yaw = agent.pitch = agent.roll = 0.0

    left_emd = HassensteinReichardtEMD(eye=left_eye, direction=(-1.0, 0.0), coordinate='spherical')
    right_emd = HassensteinReichardtEMD(eye=right_eye, direction=(1.0, 0.0), coordinate='spherical')

    log = RunLog()
    w, l = cfg.width, cfg.length
    t0 = context.total_time

    while context.run_interactive(agent=agent, scene=scene, renderer=renderer):

        context.input()

        visual_output = renderer.step()

        left_motion = left_emd.process(visual_output,)
        right_motion = right_emd.process(visual_output)
        mean_left = float(np.mean(left_motion))
        mean_right = float(np.mean(right_motion))

        diff = mean_right - mean_left
        summ = abs(mean_right) + abs(mean_left) + 1e-6
        error = diff / summ

        dt = context.dt

        if mode == MODE_YAW:
            turn_rate = error * cfg.yaw_gain - agent.yaw * cfg.damping_gain

            if cfg.motor_noise:
                turn_rate += rng.normal(loc=0.0, scale=cfg.yaw_gain)

            agent.rotate(yaw=turn_rate * dt)
            agent.translate(agent.forward * cfg.flight_speed * dt)

        elif mode == MODE_STRAFE:
            strafe_speed = -1.0 * error * cfg.strafe_gain

            if cfg.motor_noise:
                strafe_speed += rng.normal(loc=0.0, scale=cfg.strafe_gain)

            agent.translate((agent.forward * cfg.flight_speed + agent.right * strafe_speed) * dt)
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

            context.draw()

        log.time.append(context.total_time)
        log.x.append(float(agent.position.x))
        log.y.append(float(agent.position.y))
        log.z.append(float(agent.position.z))
        log.left_flow.append(mean_left)
        log.right_flow.append(mean_right)
        log.yaw.append(float(agent.yaw))
        log.error.append(error)

        if agent.position.z < -l or (context.total_time - t0) >= cfg.time_limit_s:
            break

    return log


def run_all_trials(cfg: Configuration) -> Results:
    """Run n_trials per (wall condition x control mode)."""

    rng = np.random.default_rng(cfg.seed)

    context = Context(window_size=(1280, 720))
    context.mouse_captured = False

    context.time_step = cfg.time_step

    model = CompoundEyeModel.from_file(cfg.eye_model_path)
    model.scale(1e-6)

    with model.unlock(receptors=True):
        model.receptors.tau_membrane = 0.012

    left_eye, right_eye = model.eyes
    agent = Agent()

    results: Results = {}

    for wall_condition in (WALL_SYMMETRIC, WALL_ASYMMETRIC):

        scene = build_tunnel(cfg, wall_condition)

        renderer = Raytracer(
            model=model, scene=scene, agent=agent, context=context,
            nb_samples=cfg.nb_samples, time_dithering=True,
            randomness_mode='Halton', enable_shadows=False
        )

        for mode in (MODE_YAW, MODE_STRAFE):
            runs: List[RunLog] = []

            randomise_wall_textures(scene, cfg, wall_condition, renderer)

            for t in range(cfg.n_trials):
                print(f'  {wall_condition:11s} / {MODE_LABEL[mode]:24s}  trial {t+1}/{cfg.n_trials}')

                runs.append(
                    run_trial(cfg, context, scene, renderer, agent,
                                      left_eye, right_eye, mode, rng, draw=True)
                )

            results[(wall_condition, mode)] = runs

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
        cfg: Configuration,
        title: str
    ):

    for mode in (MODE_YAW, MODE_STRAFE):
        runs = results.get((wall_condition, mode), [])
        colour = MODE_COLOUR[mode]

        for r in runs:
            ax.plot(r.dist, r.arr('x'), color=colour, lw=0.8, alpha=0.22, zorder=2)

        rep = _representative(runs)

        # Quivers for representative trial
        d, x = rep.dist, rep.arr('x')
        if d.size < 2:
            continue

        samples = np.linspace(d.min(), d.max(), cfg.n_quivers)
        idx = np.clip(np.searchsorted(d, samples), 0, d.size - 1)

        dx = ax.get_xlim()
        span = abs(dx[1] - dx[0])
        ticklength = 0.02 * span

        for i in idx:
            ax.plot(d[i], x[i], 'o', color=colour, ms=2.6, zorder=6)

            yw = rep.arr('yaw')[i]
            yw = np.deg2rad(yw)
            yw *= cfg.heading_quiver_gain

            ax.plot([d[i], d[i] + np.cos(yw) * ticklength],
                    [x[i], x[i] - np.sin(yw) * ticklength],
                    color=colour, lw=1.1, zorder=6)

        ax.plot([], [], color=colour, label=mode, lw=1.1, zorder=6)

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

    fig = plt.figure(figsize=(11, 15), constrained_layout=True)
    fig.suptitle('Optic-flow centering', fontsize=14, fontweight='bold')

    # A spans the top, D spans the bottom
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.10, 1.0, 1.0])

    # A: task-summary placeholder
    axA = fig.add_subplot(gs[0, :])
    axA.text(0.5, 0.5, '(graphical summary)',
             ha='center', va='center', fontsize=12, color='grey',
             transform=axA.transAxes)

    axA.set_xticks([])
    axA.set_yticks([])

    # for s in axA.spines.values():
    #     s.set_linestyle((0, (4, 4))); s.set_color('grey')

    # B / C: trajectory + heading quiver
    _trajectory_panel(fig.add_subplot(gs[1, 0]), results, WALL_SYMMETRIC, cfg,
                      'Symmetric walls')
    axC = fig.add_subplot(gs[1, 1])
    _trajectory_panel(axC, results, WALL_ASYMMETRIC, cfg, f'Asymmetric walls (right {cfg.asym_factor}x larger)')

    # D: bilateral flow + balance error (representative asym-yaw)
    axD = fig.add_subplot(gs[2, :])
    rep = _representative(results.get((WALL_SYMMETRIC, MODE_YAW), []))

    left_flow = rep.arr('left_flow')
    right_flow = rep.arr('right_flow')
    err = rep.arr('error')
    dist = rep.dist

    axD.plot(dist, left_flow, color=COL_LEFT, lw=0.9, alpha=0.8, label='left eye')
    axD.plot(dist, right_flow, color=COL_RIGHT, lw=0.9, alpha=0.8, label='right eye')

    axD.axhline(0.0, color='k', ls='--', lw=0.5)
    axD.set_xlabel('Distance down tunnel (m)')
    axD.set_ylabel('Mean EMD response')
    axD.set_ylim(-0.005, 0.025)   # TODO: have this automatic

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
    # fig.savefig('centering_figure.pdf')

    plt.show()