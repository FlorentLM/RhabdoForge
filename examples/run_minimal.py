from pathlib import Path

from insectvision.engine import Context, Agent, Scene
from insectvision.engine.movement import Curve, Trajectory, extract_obj_curves
from insectvision.compound_eyes import Model
from insectvision.renderers import Renderer


def main():
    context = Context()

    scene = Scene(background_color=[0.45, 0.6, 0.7])

    example_obj = Path().home() / 'Desktop/gapArray.obj'
    scene.load(example_obj)

    scene.add_skybox('assets/textures/bright_day_nosun')

    # eye_model = CompoundEyeModel(n=1962, force_isotropic=True)
    # eye_model = CompoundEyeModel.from_file('species_models/drosophila_custom.npz')
    # eye_model = CompoundEyeModel.from_file('species_models/drosophila_Kemppainen.npz')
    eye_model = Model.from_file('species_models/bee_Sturzl.npz')

    eye_model.scale(1e-6)

    agent = Agent(position=(0.0, 0.0, 4.0))

    renderer = Renderer(
        model=eye_model, scene=scene, agent=agent,
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

    while context.run_interactive(agent=agent, scene=scene, renderer=renderer):
        context.input()

        # Make the agent follow the trajectory
        agent.follow(agent_path, dt=context.dt, align=True)

        visual_output = renderer.step()

        context.draw()
        # context.draw(output)  # eye output must be passed when using the dashboard so it can plot


    renderer.free()
    scene.free()
    context.free()


if __name__ == "__main__":
    main()