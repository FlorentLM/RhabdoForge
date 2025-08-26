import time

from graphics.scene import Scene, PointsAsset, MeshAsset
from graphics.agent import Agent
from geometry.compound_eyes import CompoundEye
from geometry.primitives import CUBE_VERTICES, CUBE_INDICES

from graphics.renderers.rasterizer import EyeRendererRaster
from graphics.renderers.raytracer import EyeRendererRay
from graphics.context import Context


def main():

    # Configuration
    USE_RAYTRACER = True
    USE_POINT_CLOUD = True
    NB_OMMATIDIA = 19362
    NB_SAMPLES = 16
    HEADLESS = False

    context = Context()

    scene = Scene()

    if USE_POINT_CLOUD:
        point_cloud_asset = PointsAsset('canberra', file_path='assets/canberra_filtered.ply')
        scene.add_instance(point_cloud_asset, point_radius=5)

    # Load the mesh asset data
    crate_asset = MeshAsset('crate', vertices=CUBE_VERTICES, indices=CUBE_INDICES, texture_path='textures/wood.jpg')

    # Add multiple instances of the same asset
    scene.add_instance(asset=crate_asset, transform=(-3.0, 0.0, 0.0))
    scene.add_instance(asset=crate_asset, transform=(3.0, 0.0, 0.0))

    # A crate that will move
    dynamic_crate = scene.add_instance(asset=crate_asset, dynamic=True, transform=(0.0, 0.0, 2.0))

    # Add a skybox
    scene.add_skybox('textures/bright_day')

    # Setup eye model
    eye_model = CompoundEye(num_ommatidia=NB_OMMATIDIA, force_isotropic=True)

    # Setup Agent
    agent = Agent(position=(0.0, 0.0, 4.0))

    # Setup Renderers
    if USE_RAYTRACER:
        renderer = EyeRendererRay(eye_model=eye_model, scene=scene, nb_samples=NB_SAMPLES, time_dithering=False)

    else:
        renderer = EyeRendererRaster(eye_model=eye_model, scene=scene, nb_samples=NB_SAMPLES, time_dithering=False)

    # Run
    if not HEADLESS:

        while context.run_interactive(agent=agent, scene=scene, renderer=renderer):

            context.input()

            # Rotate dynamic test crate
            dynamic_crate.dt(context.elapsed_time).rotate_axis(0.5, 'roll').rotate_axis(0.5, 'up')

            # Get sensory data from the renderer
            ommatidia_values = renderer.get_ommatidia_data(agent, to_cpu=True)

            context.draw()

    else:
        # Run headless experiment loop

        max_steps = 10000

        print(f"Running headless simulation for {max_steps} steps...")
        start_time = time.time()

        for i in range(max_steps):

            # Programmatically control the agent
            agent.translate(agent.forward * 0.05)
            agent.rotate(yaw_delta=-0.5, pitch_delta=0)

            # Get sensory data from the renderer
            ommatidia_values = renderer.get_ommatidia_data(agent, to_cpu=True)

        total_time = time.time() - start_time
        print(f"Finished. {max_steps} frames in {total_time:.2f}s ({max_steps / total_time:.2f} FPS).")

    # Cleanup
    renderer.free()
    scene.free()
    context.free()

if __name__ == "__main__":
    main()