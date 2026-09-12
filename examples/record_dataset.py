import json
from pathlib import Path
from datetime import datetime
import numpy as np
import polars as pl

from rhabdoforge.engine import get_context, Agent, Scene, Asset
from rhabdoforge.compound_eyes import Model
from rhabdoforge.compound_eyes.rhabdomeres import drosophila_bundle
from rhabdoforge.compound_eyes.helpers.waveguide import WaveguideAcceptance
from rhabdoforge.renderers import Renderer
from rhabdoforge.types import RandomnessMode, SamplingMode


# Config
# See record_dataset.md

ENVIRONMENT = 'seville'        # 'seville' or 'canberra'
ENABLE_DYNAMICS = False        # microsaccades + pupil adaptation (pupil forced halfway when on)
SAMPLES_PER_RHABDOMERE = 64
OUTPUT_DIR = Path('datasets')
RECORD_KEY = 'f9'

PUPIL_DRIVE = 0.5

ENV_ASSETS = {
    'seville': 'assets/seville_filtered.ply',
    'canberra': 'assets/canberra_filtered.ply',
}

# -----------------------------------------------


def save_take(rows: list, layout: dict, meta: dict) -> None:

    if not rows:
        print('[record] nothing to save (empty take)')
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = f"{meta['environment']}_{datetime.now():%Y%m%d_%H%M%S}"

    df = pl.DataFrame(rows).with_columns(pl.col('visual_output').cast(pl.List(pl.Float32)))
    df.write_parquet(OUTPUT_DIR / f'{stem}.parquet')

    np.savez(OUTPUT_DIR / f'{stem}_layout.npz', **layout)

    with open(OUTPUT_DIR / f'{stem}_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'[record] saved {len(rows)} frames -> {stem}.parquet')


if __name__ == '__main__':

    context = get_context()

    scene = Scene(background_color=(0.15, 0.15, 0.3))
    scene.add_sky('assets/textures/kloppenheim_05_4k.exr')

    env_asset = Asset.from_file(name=ENVIRONMENT, file_path=ENV_ASSETS[ENVIRONMENT])
    scene.add_instance(env_asset)

    model = Model.from_file(
        'assets/drosophila_scaffold.npz',
        bundle=drosophila_bundle(),
        acceptance=WaveguideAcceptance(),
        neural_superposition=True,
    )
    model.scale(1e-6)

    agent = Agent(position=(0.0, 0.0, 4.0))

    renderer = Renderer(
        model=model, scene=scene, agent=agent,
        nb_samples=SAMPLES_PER_RHABDOMERE,
        time_dithering=True,
        randomness_mode=RandomnessMode.Halton,
        sampling_mode=SamplingMode.Waveguide,
        enable_microsaccades=ENABLE_DYNAMICS,
        enable_direct=True, enable_shadows=True, enable_ambient=True,
    )
    renderer.hybrid_sampling = True
    renderer.pupil_drive = PUPIL_DRIVE if ENABLE_DYNAMICS else None

    # Cartridge (post neural-superposition) layout: saved with every take so a
    # frame's flat visual_output array can be placed back in visual space without needing rhabdoforge
    cart = model.ommatidia.cartridges
    N, R = model.shape
    layout = {
        'N': N,
        'R': R,
        'azimuth': np.asarray(cart.rhabdomere_azimuth, dtype=np.float32),      # (N, R) rad
        'elevation': np.asarray(cart.rhabdomere_elevation, dtype=np.float32),  # (N, R) rad
    }

    meta = {
        'environment': ENVIRONMENT,
        'dynamics_enabled': ENABLE_DYNAMICS,
        'pupil_drive': renderer.pupil_drive,
        'nb_samples_per_rhabdomere': SAMPLES_PER_RHABDOMERE,
        'sampling_mode': 'Waveguide',
        'model_scale': 1e-6,
        'visual_output_channels': ['R', 'G', 'B', 'gain'],
        'visual_output_shape': [N, R, 4],
    }

    recording = False
    rows = []

    def toggle_recording():
        global recording
        recording = not recording
        if recording:
            rows.clear()
            print('[record] started')
        else:
            print(f'[record] stopped ({len(rows)} frames)')
            save_take(rows, layout, meta)

    context.bind_key(RECORD_KEY, toggle_recording, description='Toggle recording')

    print(f"Environment: {ENVIRONMENT} | dynamics: {'on' if ENABLE_DYNAMICS else 'off'}")
    print(f'Fly around with WASD + mouse. Press {RECORD_KEY.upper()} to start/stop recording.')

    while context.run_interactive(use_dashboard=True):

        context.input()

        output = renderer.step()

        if recording and output is not None:
            pos = agent.position
            gaze = agent.forward
            rows.append({
                'frame': context.frame_count,
                't': context.total_time,
                'pos_x': float(pos.x), 'pos_y': float(pos.y), 'pos_z': float(pos.z),
                'yaw': float(agent.yaw), 'pitch': float(agent.pitch), 'roll': float(agent.roll),
                'gaze_x': float(gaze.x), 'gaze_y': float(gaze.y), 'gaze_z': float(gaze.z),
                'visual_output': output.per_cartridge.data.reshape(-1).tolist(),
            })

        context.display()

    if recording:
        print(f'[record] window closed mid-take, saving ({len(rows)} frames)')
        save_take(rows, layout, meta)

    context.free()
