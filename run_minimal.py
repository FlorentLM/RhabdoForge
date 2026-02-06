from pathlib import Path
from graphics.scene import Scene
from graphics.agent import Agent
from geometry.compound_eyes import CompoundEye
from graphics.renderers.raytracer import Raytracer
from graphics.context import Context


def main():
    context = Context()

    scene = Scene(background_color=[0.45, 0.6, 0.7])

    example_obj = Path().home() / 'Downloads/gapArray.obj'
    scene.load(example_obj)

    scene.add_skybox('textures/bright_day_nosun')

    # eye_model = CompoundEye(num_ommatidia=1962, force_isotropic=True)
    eye_model = CompoundEye.from_file('species_models/drosophila_custom.npz', eye_parameter=1.5)
    # eye_model = CompoundEye.from_file('species_models/drosophila_Kemppainen.npz', eye_parameter=1.5)
    # eye_model = CompoundEye.from_file('species_models/bee_Sturzl.npz', eye_parameter=1.1)

    agent = Agent(position=(0.0, 0.0, 4.0))

    eye_renderer = Raytracer(
        eye_model=eye_model, scene=scene,
        nb_samples=16,
        time_dithering=False,
        enable_shadows=True
    )

    while context.run_interactive(agent=agent, scene=scene, renderer=eye_renderer):
        context.input()
        ommatidia_data = eye_renderer.get_ommatidia_data(agent)
        context.draw()

    eye_renderer.free()
    scene.free()
    context.free()


if __name__ == "__main__":
    main()