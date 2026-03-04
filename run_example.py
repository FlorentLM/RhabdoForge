import numpy as np
from graphics.debug import DebugBox, AxesGizmo
from graphics.scene import Scene, Asset
from graphics.agent import Agent
from geometry.compound_eyes import CompoundEye
from geometry.primitives import CUBE_VERTICES, CUBE_INDICES
from graphics.renderers.rasterizer import Rasterizer
from graphics.renderers.raytracer import Raytracer
from graphics.context import Context


def main():
    # Configuration
    USE_RAYTRACER = True
    USE_POINT_CLOUD = True
    NB_OMMATIDIA = 1962
    NB_SAMPLES = 16
    HEADLESS = False
    ENABLE_SHADOWS = True   # basic shadows (only for ray tracing for now)

    USE_ASYNC_BATCHING = True
    BATCH_SIZE = 1000

    # -----------------------------------------------

    # This needs to be the first thing called
    context = Context()

    scene = Scene(background_color=(0.15, 0.15, 0.3))

    if USE_POINT_CLOUD:
        # Create a point cloud asset from a file
        point_cloud_asset = Asset.from_file(name='seville', file_path='assets/seville_filtered.ply', radii=0.01)
        scene.add_instance(point_cloud_asset)

    cube_positions, cube_uvs = np.split(CUBE_VERTICES.reshape(-1, 5), [3], axis=1)
    cube_faces = CUBE_INDICES.reshape(-1, 3)

    # Create a mesh asset from raw vertex and index data
    crate_asset = Asset.from_arrays(name='crate', vertices=cube_positions, faces=cube_faces, uv_coords=cube_uvs, texture='textures/wood.jpg')

    # Add multiple instances of the same asset
    static_crate_1 = scene.add_instance(asset=crate_asset, transform=(-3.0, 0.0, 0.0))
    static_crate_2 = scene.add_instance(asset=crate_asset, transform=(3.0, 0.0, 0.0))

    # A crate that will move
    dynamic_crate = scene.add_instance(asset=crate_asset, dynamic=True, transform=(0.0, 0.0, 2.0))

    # Add a skybox
    scene.add_skybox('textures/bright_day_nosun')

    context.debug.add(AxesGizmo(size=0.4))
    context.debug.add(DebugBox(static_crate_1))
    context.debug.add(DebugBox(static_crate_2))
    context.debug.add(DebugBox(dynamic_crate))

    # Setup eye model
    # eye_model = CompoundEye(num_ommatidia=1962, force_isotropic=True)
    eye_model = CompoundEye.from_file('species_models/drosophila_custom.npz', eye_parameter=1.5)
    # eye_model = CompoundEye.from_file('species_models/drosophila_Kemppainen.npz', eye_parameter=1.5)
    # eye_model = CompoundEye.from_file('species_models/bee_Sturzl.npz', eye_parameter=1.1)


    # Setup Agent
    agent = Agent(position=(0.0, 0.0, 4.0))

    # Setup Renderers
    batch_size = BATCH_SIZE if (HEADLESS and USE_ASYNC_BATCHING) else 1

    if USE_RAYTRACER:
        eye_renderer = Raytracer(eye_model=eye_model, scene=scene,
                                 nb_samples=NB_SAMPLES,
                                 time_dithering=False,
                                 batch_size=batch_size,
                                 enable_shadows=ENABLE_SHADOWS)

        for blas in eye_renderer._scene_baked.BLASes:
            context.debug.add(DebugBox(blas, color=(1.0, 1.0, 0.0)))

    else:
        eye_renderer = Rasterizer(eye_model=eye_model, scene=scene,
                                  nb_samples=NB_SAMPLES,
                                  time_dithering=False,
                                  batch_size=batch_size,
                                  enable_shadows=ENABLE_SHADOWS)

    # ================== Example moving ommatidia ==================

    # Let's pick the one most aligned with the agent's forward direction
    # o = eye_model.query_directions(agent.forward, k=1)

    # or a bunch of them near this direction
    foveal_indices = eye_model.query_directions_angle(agent.forward, angle=5.0, degrees=True)

    # ================== End example moving ommatidia ==================

    # Run
    start_time = context.current_time
    nb_frames = 0
    all_ommatidia_data = []

    if not HEADLESS:

        while context.run_interactive(agent=agent, scene=scene, renderer=eye_renderer):

            context.input()  # this processes mouse and keyboard, it can be omitted to run headless

            # Rotate dynamic test crate
            dynamic_crate.dt(context.delta_time).rotate_axis(45, 'up')

            # ================== Example moving ommatidia ==================

            # Animate the foveal patch scanning horizontally
            # eye_model.ommatidia[foveal_indices].dt(context.delta_time).rotate(yaw_delta=5.0)

            # Send the updates to the GPU
            eye_renderer.update()           # TODO: This could be called unconditionally since it is a no-op when nothing changed

            # ================== End example moving ommatidia ==================

            # Get sensory data from the compound eye renderer
            ommatidia_data = eye_renderer.get_ommatidia_data(agent)

            # ommatidia_data is the array that you'd feed to your neuromorphic model

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
            ommatidia_data = eye_renderer.get_ommatidia_data(agent)

            # If the return value is not None, it's a valid chunk of data (either a single frame or a full batch)
            if ommatidia_data is not None:
                all_ommatidia_data.append(ommatidia_data)

            nb_frames += 1

        # After the loop, flush() gets the last partial batch from async mode
        # (this is harmless in sync mode, it will just return an empty array)
        final_chunk = eye_renderer.flush()
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