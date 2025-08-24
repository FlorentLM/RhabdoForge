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
    WINDOW_SIZE = (1280, 720)
    USE_RAYTRACER = True
    USE_POINT_CLOUD = True
    NB_OMMATIDIA = 119362
    HEADLESS = False

    context = Context()

    # Setup Scene
    scene = Scene()

    if USE_POINT_CLOUD:
        # Load the asset data once
        point_cloud_asset = PointsAsset('canberra', file_path='assets/canberra_filtered.ply')

        # Add an instance of it with specific properties
        scene.add_instance(point_cloud_asset, point_radius=0.15)

    # Load the mesh asset data
    crate_asset = MeshAsset('crate', vertex_data=CUBE_VERTICES, texture_path='textures/wood.jpg')
    # rock_asset = MeshAsset('rock', vertex_data=CUBE_VERTICES, texture_path='textures/rock.jpg')

    # Add multiple instances of the same asset
    scene.add_instance(asset=crate_asset, transform=glm.translate(glm.vec3(-3.0, 0.0, 0.0)))
    scene.add_instance(asset=crate_asset, transform=glm.translate(glm.vec3(3.0, 0.0, 0.0)))
    scene.add_instance(asset=crate_asset, transform=glm.translate(glm.vec3(0.0, 2.0, 6.0)))
    scene.add_instance(asset=crate_asset, transform=glm.translate(glm.vec3(0.0, -2.0, 6.0)))

    # A crate that will move
    initial_transform = glm.translate(glm.vec3(0.0, 0.0, 2.0))

    dynamic_crate = scene.add_instance(asset=crate_asset, dynamic=True, transform=initial_transform)

    # Add a skybox
    scene.add_skybox('textures/bright_day')
    # scene.add_skybox('textures/black')

    # Setup eye model
    eye_model = CompoundEye(num_ommatidia=NB_OMMATIDIA, force_isotropic=True)

    # Setup Agent
    agent = Agent(position=(0.0, 0.0, 4.0))

    # Setup Renderers
    if USE_RAYTRACER:
        renderer = EyeRendererRay(eye_model=eye_model, scene=scene, window_size=WINDOW_SIZE, nb_samples=2, time_dithering=False)
        # The debug renderer allows to see the scene geometry without raytracing
        debug_renderer = EyeRendererRaster(eye_model=eye_model, scene=scene, window_size=WINDOW_SIZE)

    else:
        renderer = EyeRendererRaster(eye_model=eye_model, scene=scene, window_size=WINDOW_SIZE)
        debug_renderer = None

    # Run

    if not HEADLESS:

        while context.run_interactive(agent=agent, scene=scene, renderer=renderer, debug_renderer=debug_renderer):

            context.input()

            # Rotate dynamic test crate
            spin_speed = 1.5
            angle = context.elapsed_time * spin_speed

            rotation = glm.rotate(glm.mat4(1.0), angle, glm.vec3(0.0, 1.0, 0.0))

            dynamic_crate.transform = initial_transform * rotation

            # Get sensory data from the renderer via the context
            ommatidia_values = context.active_renderer.get_ommatidia_data(agent, to_cpu=True)

            context.draw()

    else:
        # Run headless experiment loop

        max_steps = 1000
        results = []

        print(f"Running headless simulation for {max_steps} steps...")
        start_time = time.time()

        for i in range(max_steps):

            # Programmatically control the agent
            agent.move(agent.forward * 0.05)
            agent.rotate(yaw_delta=-0.5, pitch_delta=0)

            # Get sensory data from the renderer directly
            ommatidia_values = renderer.get_ommatidia_data(agent, to_cpu=True)

            results.append(ommatidia_values)

        total_time = time.time() - start_time
        print(f"Finished. {max_steps} frames in {total_time:.2f}s ({max_steps / total_time:.2f} FPS).")

    # Cleanup
    print("Cleaning up resources...")

    renderer.free()
    if debug_renderer: debug_renderer.free()
    scene.free()
    context.free()

if __name__ == "__main__":
    main()