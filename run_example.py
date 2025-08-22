import time
from pyglm import glm

from graphics.scene import Scene
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
    NB_OMMATIDIA = 19362
    NB_SAMPLES = 16
    TIME_DITHERING = False
    HEADLESS = False

    context = Context(window_size=WINDOW_SIZE, headless=HEADLESS)

    # Setup Scene
    scene = Scene()

    if USE_POINT_CLOUD:
        scene.add_point_cloud('canberra', 'assets/canberra_filtered.ply')

    crate_asset = scene.load_mesh("crate", CUBE_VERTICES, 'textures/wood.jpg')
    scene.add_instance(asset=crate_asset)
    scene.add_instance(asset=crate_asset, transform=glm.translate(glm.vec3(-3.0, 0.0, 0.0)))
    scene.add_instance(asset=crate_asset, transform=glm.translate(glm.vec3(3.0, 0.0, 0.0)))

    scene.add_skybox('textures/bright_day')

    # Setup eye model
    eye_model = CompoundEye(num_ommatidia=NB_OMMATIDIA, force_isotropic=True)

    # Setup Agent
    agent = Agent(position=(0.0, 0.0, 4.0))

    # Setup Renderers

    renderer = None
    debug_renderer = None

    if USE_RAYTRACER:
        renderer = EyeRendererRay(eye_model=eye_model, scene=scene,
                                  window_size=WINDOW_SIZE,
                                  nb_samples=NB_SAMPLES, time_dithering=TIME_DITHERING)

        # The debug renderer allows us to see the scene geometry without raytracing
        debug_renderer = EyeRendererRaster(eye_model=eye_model, scene=scene,
                                           window_size=WINDOW_SIZE,
                                           nb_samples=NB_SAMPLES, time_dithering=TIME_DITHERING)
    else:
        renderer = EyeRendererRaster(eye_model=eye_model, scene=scene,
                                     window_size=WINDOW_SIZE,
                                     nb_samples=NB_SAMPLES, time_dithering=TIME_DITHERING)

    if not HEADLESS:
        # Setup and run interactive viewer

        while context.interactive(agent=agent, scene=scene, renderer=renderer, debug_renderer=debug_renderer):

            context.handle_input()

            # Get sensory data from the renderer via the context
            ommatidia_values = context.active_renderer.get_ommatidia_data(agent.camera, to_cpu=True)

            context.draw()

    else:
        # Run headless experiment loop

        max_steps = 1000
        results = []

        print(f"Running headless simulation for {max_steps} steps...")
        start_time = time.time()

        for i in range(max_steps):

            # Programmatically control the agent
            agent.move(agent.camera.forward * 0.05)
            agent.rotate(yaw_delta=-0.5, pitch_delta=0)

            # Get sensory data from the renderer directly
            ommatidia_values = renderer.get_ommatidia_data(agent.camera, to_cpu=True)

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