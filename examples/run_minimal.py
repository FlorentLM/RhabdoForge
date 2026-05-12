from pathlib import Path

from insectvision.engine import Context, Agent, Scene
from insectvision.engine.movement import Curve, Trajectory, extract_obj_curves
from insectvision.compound_eyes import ReceptorArray
from insectvision.renderers import Raytracer


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
        receptor_array=eye_model, scene=scene, agent=agent,
        context=context,
        nb_samples=16,
        time_dithering=False,
        enable_shadows=True
    )

    # Extract curve coords from the .obj file # TODO: This could be done internally by the scene's loader
    nurbs = extract_obj_curves(example_obj)
    curve_coords = nurbs['NurbsPath']

    # Instantiate a Curve and a Trajectory
    curve = Curve(curve_coords)
    agent_path = Trajectory(curve, speed=2.5, loop=True)

    while context.run_interactive(agent=agent, scene=scene, renderer=eye_renderer):
        context.input()

        # Advance clocks
        dt = context.tick()

        # Make the agent follow the trajectory
        agent.dt(dt).follow(agent_path, align_orientation=True)

        output = eye_renderer.step()

        context.draw()
        # context.draw(output)  # eye output must be passed when using the dashboard so it can plot


    eye_renderer.free()
    scene.free()
    context.free()


if __name__ == "__main__":
    main()