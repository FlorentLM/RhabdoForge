import numpy as np

from insectvision.engine import Context, Agent, Scene, Asset
from insectvision.geometry import CUBE_VERTICES, CUBE_INDICES
from insectvision.geometry.compound_eyes import ReceptorArray
from insectvision.renderers import Rasterizer, Raytracer
from insectvision.debug import DebugBox, AxesGizmo


def main():

    USE_RAYTRACER = True
    USE_POINT_CLOUD = True

    SAMPLES_PER_RECEPTOR = 16
    HEADLESS = False

    USE_ASYNC_BATCHING = True
    BATCH_SIZE = 1000

    SHOW_DEBUG_OBJECTS = False

    # -----------------------------------------------

    # This always needs to be the first thing called
    context = Context()

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

    # eye_model = ReceptorArray(num_ommatidia=1962, force_isotropic=True)
    eye_model = ReceptorArray.from_file('species_models/drosophila_custom.npz', eye_parameter=1.5)
    # eye_model = ReceptorArray.from_file('species_models/drosophila_Kemppainen.npz', eye_parameter=1.5)
    # eye_model = ReceptorArray.from_file('species_models/bee_Sturzl.npz', eye_parameter=1.1)

    eye_model.scale(0.01)

    # Setup Agent
    agent = Agent(position=(0.0, 0.0, 4.0))

    # Setup Renderers
    batch_size = BATCH_SIZE if (HEADLESS and USE_ASYNC_BATCHING) else 1

    if USE_RAYTRACER:
        eye_renderer = Raytracer(receptor_array=eye_model, scene=scene,
                                 nb_samples=SAMPLES_PER_RECEPTOR,
                                 time_dithering=False,
                                 # time_accumulation=0.012,  # 12 ms
                                 quasi_random=False,
                                 enable_shadows=False)

        # The BVH can also be displayed in debug
        if SHOW_DEBUG_OBJECTS:
            for blas in eye_renderer._scene_baked.BLASes:
                context.debug.add(DebugBox(blas, color=(1.0, 1.0, 0.0)))

    else:
        eye_renderer = Rasterizer(receptor_array=eye_model, scene=scene,
                                  nb_samples=SAMPLES_PER_RECEPTOR,
                                  time_dithering=False,
                                  batch_size=batch_size,
                                  enable_direct=True,
                                  enable_shadows=True,
                                  enable_ambient=True)

    # Example custom key binding:
    def toggle_halton():
        eye_renderer.quasi_random = not eye_renderer.quasi_random

    context.bind_key('m', toggle_halton)

    # Run

    start_time = context.current_time
    nb_frames = 0
    all_ommatidia_data = []

    if not HEADLESS:

        while context.run_interactive(agent=agent, scene=scene, renderer=eye_renderer):

            context.input()  # this processes mouse and keyboard, it can be omitted to run headless

            # Rotate dynamic test crate
            dynamic_crate.dt(context.delta_time).rotate_axis(45, 'up')

            # Get sensory data from the compound eye renderer
            ommatidia_data = eye_renderer.get_visual_output(agent)

            context.draw()  # this draws to the viewport, it can be omitted to run headless

            nb_frames += 1

    else:
        # Headless and batched mode

        max_steps = 10000
        all_ommatidia_data = []

        print(f"Running headless simulation for {max_steps} steps...")

        for i in range(max_steps):

            # Move the agent or whatever
            agent.translate(agent.forward * 0.05).rotate(yaw_delta=-0.5, pitch_delta=0, roll_delta=0, degrees=False)

            # Get sensory data from the compound eye renderer
            view = eye_renderer.get_visual_output(agent)

            # If the return value is not None, it's a valid chunk of data (either a single frame or a full batch)
            if view is not None:
                all_ommatidia_data.append(view.per_ommatidium)

            nb_frames += 1

        # After the loop, flush() gets the last partial batch from async mode
        # (this is harmless in sync mode, it will just return an empty array)
        final_chunk = eye_renderer.flush()  # TODO: This might actually be done automatically

        if final_chunk.size > 0:
            all_ommatidia_data.append(final_chunk)

    total_time = context.current_time - start_time

    print(f"Ran for {nb_frames} frames in {total_time:.2f}s (avg. {nb_frames / total_time:.2f} fps).")

    if all_ommatidia_data:
        # In sync mode, this combines 10,000 arrays of shape (19362, 4)
        # In async mode, this might combine 10 arrays of shape (1000, 19362, 4)

        full_dataset = np.concatenate(all_ommatidia_data, axis=0)
        print(f"Final concatenated dataset shape: {full_dataset.shape}")

    # Cleanup
    eye_renderer.free()
    scene.free()
    context.free()


if __name__ == "__main__":
    main()