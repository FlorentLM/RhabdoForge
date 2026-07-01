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

    eye_file_path = 'assets/drosophila_scaffold.npz'
    # eye_file_path = 'assets/honeybee_scaffold_s10.npz'
    # eye_file_path = 'assets/drosophila_scaffold_k22.npz'

    eye_model = Model.from_file(eye_file_path)
    eye_model.scale(1e-6)

    eye_model = Model.from_sphere(n=1962, force_isotropic=True)

    agent = Agent(position=(0.0, 0.0, 4.0))

    renderer = Renderer(
        model=eye_model,
        scene=scene,
        agent=agent,
        context=context,
        nb_samples=16,
        time_dithering=False,
        enable_shadows=True
    )

    # Extract curve coords from the .obj file
    # TODO: This should be done internally by the scene's loader
    nurbs = extract_obj_curves(example_obj)
    curve_coords = nurbs['NurbsPath']

    # Instantiate a Curve and a Trajectory
    curve = Curve(curve_coords)
    agent_path = Trajectory(curve, speed=2.5, loop=True)

    while context.run_interactive(renderer=renderer):
        context.input()

        # Make the agent follow the trajectory
        agent.follow(agent_path, dt=context.dt, align=True)

        output = renderer.step()

        context.draw()

    renderer.free()
    scene.free()
    context.free()


if __name__ == "__main__":
    main()