import time

from pyglm import glm

from graphics.scene import Scene, PointsAsset, MeshAsset
from graphics.agent import Agent
from geometry.compound_eyes import CompoundEye
from geometry.primitives import CUBE_VERTICES

from graphics.renderers.rasterizer import EyeRendererRaster
from graphics.renderers.raytracer import EyeRendererRay
from graphics.interactive.context import Context


def main():

    # Configuration
    USE_RAYTRACER = False
    USE_POINT_CLOUD = True
    NB_OMMATIDIA = 119362
    NB_SAMPLES = 16
    HEADLESS = False

    context = Context()

    scene = Scene()

    if USE_POINT_CLOUD:
        point_cloud_asset = PointsAsset('canberra', file_path='assets/canberra_filtered.ply')
        scene.add_instance(point_cloud_asset, point_radius=0.15)

    # Load the mesh asset data
    crate_asset = MeshAsset('crate', vertex_data=CUBE_VERTICES, texture_path='textures/wood.jpg')
    # rock_asset = MeshAsset('rock', vertex_data=CUBE_VERTICES, texture_path='textures/rock.jpg')

    # Add multiple instances of the same asset
    scene.add_instance(asset=crate_asset, transform=(-3.0, 0.0, 0.0))
    scene.add_instance(asset=crate_asset, transform=(3.0, 0.0, 0.0))
    scene.add_instance(asset=crate_asset, transform=(0.0, 2.0, 6.0))
    scene.add_instance(asset=crate_asset, transform=(0.0, -2.0, 6.0))

    # A crate that will move
    dynamic_crate = scene.add_instance(asset=crate_asset, dynamic=True, transform=(0.0, 0.0, 2.0))

    # Add a skybox
    scene.add_skybox('textures/bright_day')
    # scene.add_skybox('textures/black')

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
            spin_speed = 0.1
            angle = context.elapsed_time * spin_speed

            dynamic_crate.rotate(angle, (0.0, 1.0, 0.0))

            # Get sensory data from the renderer
            ommatidia_values = renderer.get_ommatidia_data(agent, to_cpu=True)

            context.draw()

    else:
        # Run headless experiment loop

        max_steps = 1000

        print(f"Running headless simulation for {max_steps} steps...")
        start_time = time.time()

        for i in range(max_steps):

            # Programmatically control the agent
            agent.move(agent.forward * 0.05)
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