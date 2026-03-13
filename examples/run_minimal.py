from pathlib import Path

from insectvision.engine.scene import Scene
from insectvision.engine.agent import Agent
from insectvision.engine.context import Context
from insectvision.engine.movement import Curve, Trajectory
from insectvision.engine.utils import extract_obj_curves

from insectvision.geometry.compound_eyes.receptor_array import ReceptorArray
from insectvision.renderers.raytracer import Raytracer


def main():
    context = Context()

    scene = Scene(background_color=[0.45, 0.6, 0.7])

    example_obj = Path().home() / 'Desktop/gapArray.obj'
    scene.load(example_obj)

    scene.add_skybox('assets/textures/bright_day_nosun')

    # eye_model = ReceptorArray(num_ommatidia=1962, force_isotropic=True)
    # eye_model = ReceptorArray.from_file('species_models/drosophila_custom.npz', eye_parameter=1.5)
    # eye_model = ReceptorArray.from_file('species_models/drosophila_Kemppainen.npz', eye_parameter=1.5)
    eye_model = ReceptorArray.from_file('species_models/bee_Sturzl.npz', eye_parameter=1.1)

    eye_model.scale(0.01)

    agent = Agent(position=(0.0, 0.0, 4.0))

    eye_renderer = Raytracer(
        receptor_array=eye_model, scene=scene,
        nb_samples=16,
        time_dithering=False,
        enable_shadows=True
    )

    # Extract curve coords from the .obj file
    nurbs = extract_obj_curves(example_obj)
    curve_coords = nurbs['NurbsPath']

    # Instantiate a Curve and a Trajectory
    curve = Curve(curve_coords)
    agent_path = Trajectory(curve, speed=2.5, loop=True)

    while context.run_interactive(agent=agent, scene=scene, renderer=eye_renderer):
        context.input()

        # Make the agent follow the trajectory
        agent.dt(context.delta_time).follow(agent_path, align_orientation=True)

        view = eye_renderer.get_visual_output(agent)

        context.draw()

    eye_renderer.free()
    scene.free()
    context.free()


if __name__ == "__main__":
    main()