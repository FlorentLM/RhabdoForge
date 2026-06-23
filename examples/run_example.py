import numpy as np

from insectvision.engine import Context, Agent, Scene, Asset
from insectvision.engine.meshes import CUBE_VERTICES, CUBE_INDICES
from insectvision.compound_eyes import Model
from insectvision.compound_eyes.rhabdomeres import drosophila_bundle
from insectvision.renderers import Renderer
from insectvision.interactive.debug import DebugBox, AxesGizmo
from insectvision.renderers.helpers import VisualOutput
from insectvision.utils.shared import RandomnessMode


def main():

    USE_POINT_CLOUD = True
    SAMPLES_PER_RHABDOMERE = 64
    HEADLESS = False
    BATCH_SIZE = 1000
    SHOW_DEBUG_OBJECTS = False
    USE_NEURAL_SUPERPOSITION = True

    # -----------------------------------------------

    # This always needs to be the first thing called
    context = Context()

    # # Example: use a gamepad
    # from insectvision.interactive import Gamepad
    # context.controls = Gamepad()

    scene = Scene(background_color=(0.15, 0.15, 0.3))

    if USE_POINT_CLOUD:
        # Create a point cloud asset from a file
        point_cloud_asset = Asset.from_file(name='seville', file_path='assets/seville_filtered.ply', radii=0.01)
        scene.add_instance(point_cloud_asset)

    cube_positions, cube_uvs = np.split(CUBE_VERTICES.reshape(-1, 5), [3], axis=1)
    cube_faces = CUBE_INDICES.reshape(-1, 3)

    # Create a mesh asset from raw vertex and index data
    crate_asset = Asset.from_arrays(name='crate', vertices=cube_positions, faces=cube_faces, uv_coords=cube_uvs, texture='assets/textures/wood.jpg', sRGB=False)

    # Add multiple instances of the same asset
    static_crate_1 = scene.add_instance(asset=crate_asset, transform=(-3.0, 0.0, 0.0))
    static_crate_2 = scene.add_instance(asset=crate_asset, transform=(3.0, 0.0, 0.0))

    # A crate that will move
    dynamic_crate = scene.add_instance(asset=crate_asset, dynamic=True, transform=(0.0, 0.0, 2.0))

    # Add a skybox
    scene.add_skybox('assets/textures/kloppenheim_05_4k.exr')   # from https://polyhaven.com/a/kloppenheim_05

    # Example debug objects (wireframes, grid etc)
    if SHOW_DEBUG_OBJECTS:
        context.debug.add(AxesGizmo(size=0.4))
        context.debug.add(DebugBox(static_crate_1))
        context.debug.add(DebugBox(static_crate_2))
        context.debug.add(DebugBox(dynamic_crate))

    # Setup eye model
    eye_file_path = 'species_models/drosophila_custom.npz'
    # eye_file_path = 'species_models/bee_Sturzl.npz'
    # eye_file_path = 'species_models/drosophila_Kemppainen.npz'

    model = Model.from_file(
        eye_file_path,
        # bundle=drosophila_bundle() if USE_NEURAL_SUPERPOSITION else None,
        # neural_superposition=USE_NEURAL_SUPERPOSITION
    )
    model.scale(1e-6)

    # Example setting time adaptation
    model.tau_membrane = 0.012   # 12 ms is good for Drosophila

    # It is also possible to unlock CPU writes persistently
    # model.allow_receptor_writes = True

    # Setup Agent
    agent = Agent(position=(0.0, 0.0, 4.0))

    print('before renderer')
    # Setup renderer
    renderer = Renderer(model=model, scene=scene, agent=agent, context=context,
                         nb_samples=SAMPLES_PER_RHABDOMERE,
                         time_dithering=True,
                         randomness_mode=RandomnessMode.Halton,
                         enable_microsaccades=True,
                         enable_direct=True, enable_shadows=True, enable_ambient=True)

    print('after renderer')
    renderer.photon_concentration = 0.5

    # The BVH can also be displayed in debug
    if SHOW_DEBUG_OBJECTS:
        for blas in renderer.blases:
            context.debug.add(DebugBox(blas, color=(1.0, 1.0, 0.0)))


    # Example custom key binding:
    def cycle_randomness():
        current = renderer.randomness_mode
        modes = list(RandomnessMode)
        next_mode = modes[(current.value + 1) % len(modes)]
        renderer.randomness_mode = next_mode

    context.bind_key('m', cycle_randomness)



    # Example fixed timing: simulation steps by exactly 10 ms regardless of render speed
    # context.time_step = 1/100.0
    # Note: with 1/100.0, nb_samples needs to be high enough (32 or 64 or more),
    # or tau_fast should be at least 2x the dt (set tau_fast to 0.02s)
    # Biological 5 ms responses are difficult to simulate cleanly with Monte Carlo noise at 100 Hz

    context.time_step = 1 / 100.0


    # Tune / disable luminance boost on RF narrowing
    renderer.photon_concentration = 0.0
    # renderer.photon_concentration = 0.2


    # Run
    all_timesteps = []

    if not HEADLESS:

        while context.run_interactive(renderer=renderer, use_dashboard=True):

            context.input()  # Processes mouse and keyboard, optional

            # Rotate dynamic test crate at 45 deg/s (framerate-independent)
            dynamic_crate.rotate_axis(45 * context.dt, 'up')

            # Render one biological step
            visual_output = renderer.step()

            context.draw(visual_output)  # draws to the viewport, also optional

    else:
        # Headless and batched mode

        print(f"Running headless simulation for {BATCH_SIZE} steps...")

        for i in range(BATCH_SIZE):

            # Important: clock must be advanced manually in headless mode
            context.tick()

            # Move the agent at 0.5 m/s and yaw at 25 deg/s (framerate-independent)
            agent.translate(agent.forward * 0.5 * context.dt).rotate(yaw=25.0 * context.dt, degrees=True)

            # Render one biological step
            visual_output = renderer.step()

            # If the return value is not None, it's a valid chunk of data (either a single frame or a full batch)
            if visual_output is not None:
                all_timesteps.append(visual_output)

        # After the loop, flush() gets the last partial batch from async mode
        # (this is harmless in sync mode, it will just return None)
        final_chunk = renderer.flush()

        if final_chunk is not None:
            all_timesteps.append(final_chunk)

    print(f'Ran for {context.frame_count} frames in {context.wall_time:.2f}s (avg. {context.frame_count / context.wall_time:.2f} fps).')

    if all_timesteps:
        full_dataset = VisualOutput.from_history(all_timesteps)
        print(f"Final concatenated dataset shape: {full_dataset.shape}")

    # Cleanup
    context.free()


if __name__ == "__main__":
    main()