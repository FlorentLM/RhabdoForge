import numpy as np

from insectvision.engine import Context, Agent, Scene, Asset
from insectvision.geometry.meshes import CUBE_VERTICES, CUBE_INDICES
from insectvision.compound_eyes import CompoundEyeModel
from insectvision.renderers import Rasterizer, Raytracer
from insectvision.interactive.debug import DebugBox, AxesGizmo


def main():

    USE_RAYTRACER = True
    USE_POINT_CLOUD = True

    SAMPLES_PER_RECEPTOR = 64
    HEADLESS = False

    USE_ASYNC_BATCHING = True
    BATCH_SIZE = 1000

    SHOW_DEBUG_OBJECTS = False

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
    crate_asset = Asset.from_arrays(name='crate', vertices=cube_positions, faces=cube_faces, uv_coords=cube_uvs, texture='assets/textures/wood.jpg')

    # Add multiple instances of the same asset
    static_crate_1 = scene.add_instance(asset=crate_asset, transform=(-3.0, 0.0, 0.0))
    static_crate_2 = scene.add_instance(asset=crate_asset, transform=(3.0, 0.0, 0.0))

    # A crate that will move
    dynamic_crate = scene.add_instance(asset=crate_asset, dynamic=True, transform=(0.0, 0.0, 2.0))

    # Add a skybox
    scene.add_skybox('assets/textures/bright_day_nosun')

    # Example debug objects (wireframes, grid etc)
    if SHOW_DEBUG_OBJECTS:
        context.debug.add(AxesGizmo(size=0.4))
        context.debug.add(DebugBox(static_crate_1))
        context.debug.add(DebugBox(static_crate_2))
        context.debug.add(DebugBox(dynamic_crate))

    # Setup eye model

    # model = CompoundEyeModel.from_sphere(n=1962, force_isotropic=True)
    # model = CompoundEyeModel.from_file('species_models/drosophila_custom.npz', eye_parameter=1.5)
    # model = CompoundEyeModel.from_file('species_models/drosophila_Kemppainen.npz', eye_parameter=1.5)
    model = CompoundEyeModel.from_file('species_models/bee_Sturzl.npz', eye_parameter=1.1)
    model.scale(1e-6)

    # Example setting time adaptation
    with model.unlock(receptors=True):      # temporarily allow writing from the CPU
        model.receptors.tau_membrane = 0.012   # 12 ms is good for Drosophila

    # It is also possible to unlock CPU writes persistently
    # model.allow_receptor_writes = True

    # Setup Agent
    agent = Agent(position=(0.0, 0.0, 4.0))

    # Setup Renderers
    batch_size = BATCH_SIZE if (HEADLESS and USE_ASYNC_BATCHING) else 1

    if USE_RAYTRACER:
        renderer = Raytracer(model=model, scene=scene, agent=agent, context=context,
                                 nb_samples=SAMPLES_PER_RECEPTOR,
                                 time_dithering=True,
                                 quasi_random=True,
                                 enable_actuation=True,
                                 enable_direct=True,
                                 enable_shadows=True,
                                 enable_ambient=True)

        # The BVH can also be displayed in debug
        if SHOW_DEBUG_OBJECTS:
            for blas in renderer.BLASes:
                context.debug.add(DebugBox(blas, color=(1.0, 1.0, 0.0)))

    else:
        renderer = Rasterizer(model=model, scene=scene, agent=agent, context=context,
                                  nb_samples=SAMPLES_PER_RECEPTOR,
                                  time_dithering=True,
                                  batch_size=batch_size,
                                  enable_actuation=False,
                                  enable_direct=True,
                                  enable_shadows=True,
                                  enable_ambient=True)

    # Example custom key binding:
    def toggle_halton():
        renderer.quasi_random = not renderer.quasi_random

    context.bind_key('m', toggle_halton)



    # Example fixed timing: simulation steps by exactly 10 ms regardless of render speed
    # context.time_step = 1/100.0
    # Note: with 1/100.0, nb_samples needs to be high enough (32 or 64 or more),
    # or tau_fast should be at least 2x the dt (set tau_fast to 0.02s)
    # Biological 5 ms responses are difficult to simulate cleanly with Monte Carlo noise at 100 Hz

    context.time_step = 1 / 1000.0


    # Tune / disable luminance boost on RF narrowing
    renderer.photon_concentration = 0.0
    # renderer.photon_concentration = 0.2


    # Run

    start_time = context.current_wall_time
    nb_frames = 0
    all_ommatidia_data = []

    if not HEADLESS:

        while context.run_interactive(agent=agent, scene=scene, renderer=renderer, use_dashboard=False):

            context.input()  # Processes mouse and keyboard, optional

            # Rotate dynamic test crate at 45 deg/s (framerate-independent)
            dynamic_crate.rotate_axis(45 * context.dt, 'up')

            # Render one biological step
            visual_output = renderer.step()

            context.draw(visual_output)  # draws to the viewport, also optional

            nb_frames += 1

    else:
        # Headless and batched mode

        max_steps = 10000
        all_ommatidia_data = []

        print(f"Running headless simulation for {max_steps} steps...")

        for i in range(max_steps):

            # Important: clock must be advanced manually in headless mode
            context.tick()

            # Move the agent at 0.5 m/s and yaw at 25 deg/s (framerate-independent)
            agent.translate(agent.forward * 0.5 * context.dt).rotate(yaw=25.0 * context.dt, degrees=True)

            # Render one biological step
            visual_output = renderer.step()

            # If the return value is not None, it's a valid chunk of data (either a single frame or a full batch)
            if visual_output is not None:
                all_ommatidia_data.append(visual_output.per_cartridge)

            nb_frames += 1

        # After the loop, flush() gets the last partial batch from async mode
        # (this is harmless in sync mode, it will just return an empty array)
        final_chunk = renderer.flush()  # TODO: This might actually be done automatically

        if final_chunk.size > 0:
            all_ommatidia_data.append(final_chunk)

    total_time = context.current_wall_time - start_time

    print(f"Ran for {nb_frames} frames in {total_time:.2f}s (avg. {nb_frames / total_time:.2f} fps).")

    if all_ommatidia_data:
        # In sync mode, this combines 10,000 arrays of shape (19362, 4)
        # In async mode, this might combine 10 arrays of shape (1000, 19362, 4)

        full_dataset = np.concatenate(all_ommatidia_data, axis=0)
        print(f"Final concatenated dataset shape: {full_dataset.shape}")

    # Cleanup
    renderer.free()
    scene.free()
    context.free()


if __name__ == "__main__":
    main()