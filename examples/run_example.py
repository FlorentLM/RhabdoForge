import numpy as np
from insectvision.engine import Context, Agent, Scene, Asset
from insectvision.engine.meshes import CUBE_VERTICES, CUBE_INDICES
from insectvision.compound_eyes import Model
from insectvision.compound_eyes.rhabdomeres import drosophila_bundle
from insectvision.renderers import Renderer
from insectvision.interactive.debug import DebugBox, AxesGizmo
from insectvision.renderers.helpers import VisualOutput
from insectvision.utils import RandomnessMode


if __name__ == "__main__":

    SAMPLES_PER_RHABDOMERE = 64
    HEADLESS = False
    BATCH_SIZE = 1000
    SHOW_DEBUG_OBJECTS = False
    USE_NEURAL_SUPERPOSITION = True

    # -----------------------------------------------

    # This always needs to be the first thing called
    context = Context()

    # # If you want to use a gamepad
    # from insectvision.interactive import Gamepad
    # context.controls = Gamepad()

    scene = Scene(background_color=(0.15, 0.15, 0.3))

    # Add a sky
    scene.add_sky('assets/textures/kloppenheim_05_4k.exr')  # from https://polyhaven.com/a/kloppenheim_05

    # Create a point cloud asset from a file
    seville = Asset.from_file(name='seville', file_path='assets/seville_filtered.ply', radii=0.01)
    scene.add_instance(seville)


    # Create a mesh asset from raw vertex and index data
    cube_positions, cube_uvs = np.split(CUBE_VERTICES.reshape(-1, 5), [3], axis=1)
    cube_faces = CUBE_INDICES.reshape(-1, 3)
    crate = Asset.from_arrays(name='crate', vertices=cube_positions, faces=cube_faces, uv_coords=cube_uvs, texture='assets/textures/wood.jpg', sRGB=False)


    # Add multiple instances of the same asset
    crate_instance_1 = scene.add_instance(asset=crate, transform=(-3.0, 0.0, 0.0))
    crate_instance_2 = scene.add_instance(asset=crate, transform=(3.0, 0.0, 0.0))
    crate_instance_3 = scene.add_instance(asset=crate, transform=(0.0, 0.0, 2.0), dynamic=True)  # this one can move


    # Setup compound eyes model
    scaffold_file = 'assets/drosophila_scaffold.npz'
    # scaffold_file = 'assets/honeybee_scaffold_s10.npz'
    # scaffold_file = 'assets/drosophila_scaffold_k22.npz'

    model = Model.from_file(
        scaffold_file,
        # bundle=drosophila_bundle() if USE_NEURAL_SUPERPOSITION else None,
        # neural_superposition=USE_NEURAL_SUPERPOSITION
    )
    model.scale(1e-6)

    # Example: setting time adaptation (generates motion blur)
    model.tau_membrane = 0.012   # 12 ms is good for Drosophila

    # Setup the agent
    agent = Agent(position=(0.0, 0.0, 4.0))

    # Setup the renderer
    renderer = Renderer(
        model=model, scene=scene, agent=agent,
        nb_samples=SAMPLES_PER_RHABDOMERE,
        time_dithering=True,
        randomness_mode=RandomnessMode.Halton,
        enable_microsaccades=True,
        enable_direct=True, enable_shadows=True, enable_ambient=True
    )


    # Example: add debug objects (wireframes, grid etc)
    if SHOW_DEBUG_OBJECTS:
        context.debug.add(AxesGizmo(size=0.4))
        context.debug.add(DebugBox(crate_instance_1))
        context.debug.add(DebugBox(crate_instance_2))
        context.debug.add(DebugBox(crate_instance_3))

        # The BVH can also be displayed in debug
        for blas in renderer.blases:
            context.debug.add(DebugBox(blas, color=(1.0, 1.0, 0.0)))



    # Example: define your own custom key binding
    def cycle_randomness():
        current = renderer.randomness_mode
        modes = list(RandomnessMode)
        next_mode = modes[(current.value + 1) % len(modes)]
        renderer.randomness_mode = next_mode

    context.bind_key('m', cycle_randomness)



    # # Example: Set a fixed simulation step
    context.time_step = 1/100.0       # each simulated step will correspond to exactly 10 ms regardless of framerate

    # Note: with 1/100.0, nb_samples needs to be high enough (64 or more), or tau_fast should be at least 2x the dt
    # (set tau_fast to 0.02s)


    # Tune / disable luminance boost on RF narrowing
    renderer.photon_concentration = 0.0
    # renderer.photon_concentration = 0.2


    # Run
    if not HEADLESS:

        while context.run_interactive(use_dashboard=True):

            context.input()  # processes inputs from keyboard / gamepad etc (optional)

            # Rotate dynamic test crate at 45 deg/s
            crate_instance_3.rotate_axis(45 * context.dt, 'up')

            # Render one biological step
            output = renderer.step()

            context.display()  # displays to the viewport (also optional)
    
    else:
        # Headless and batched mode

        print(f"Running headless simulation for {BATCH_SIZE} steps...")

        all_data = []

        for dt in context.run_headless(BATCH_SIZE):

            # Move the agent forward at 0.5 m/s and yaw at 25 deg/s
            agent.translate(agent.forward * 0.5 * dt).rotate(yaw=25.0 * dt, degrees=True)

            # Render one biological step
            output = renderer.step()

            # If the return value is not None, it's a valid chunk of data (either a single frame or a full batch)
            if output is not None:
                all_data.append(output)

        # Grab the final partial batch (harmless in sync mode, it will just return None)
        final_chunk = renderer.flush()
        if final_chunk is not None:
            all_data.append(final_chunk)

        full_dataset = VisualOutput.from_history(all_data)
        print(f"Final concatenated dataset shape: {full_dataset.shape}")

    print(f'Ran for {context.frame_count} frames in {context.wall_time:.2f}s (avg. {context.frame_count / context.wall_time:.2f} fps).')

    # Cleanup
    context.free()
