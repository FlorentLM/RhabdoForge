from pathlib import Path
import numpy as np

from graphics.scene import Scene, Asset
from graphics.agent import Agent
from geometry.compound_eyes import CompoundEye
from graphics.renderers.raytracer import EyeRendererRay
from graphics.context import Context


def main():

    # Configuration
    NB_OMMATIDIA = 1962
    NB_SAMPLES = 16
    ENABLE_SHADOWS = True   # basic shadows (only for ray tracing for now)

    # -----------------------------------------------

    # This needs to be the first thing called
    context = Context()

    scene = Scene(background_color=[0.45, 0.6, 0.7])

    example_obj = Path().home() / 'Downloads/gapArray.obj'
    scene.load(example_obj)

    # Add a skybox
    # scene.add_skybox('textures/bright_day')

    # Setup eye model
    # eye_model = CompoundEye(num_ommatidia=NB_OMMATIDIA, force_isotropic=True)   # Uniform spherical eye
    eye_model = CompoundEye.from_file('drosophila_eye.npz', eye_parameter=1.5)  # Manually mapped drosophila eye
    # eye_model = CompoundEye.from_file('bee_eye.npz', eye_parameter=1.1)  # Procedurally-generated bee eye

    # Setup Agent
    agent = Agent(position=(0.0, 0.0, 0.0))


    eye_renderer = EyeRendererRay(eye_model=eye_model, scene=scene,
                                  nb_samples=NB_SAMPLES,
                                  time_dithering=False,
                                  enable_shadows=ENABLE_SHADOWS)

    while context.run_interactive(agent=agent, scene=scene, renderer=eye_renderer):

        context.input()  # this processes mouse and keyboard, it can be omitted to run headless

        # Get sensory data from the compound eye renderer
        ommatidia_data = eye_renderer.get_ommatidia_data(agent)
        # ommatidia_data is the array that you'd feed to your neuromorphic model

        context.draw()  # this draws to the viewport, it can be omitted to run headless

    # Cleanup
    eye_renderer.free()
    scene.free()
    context.free()


if __name__ == "__main__":
    main()