from pathlib import Path
from rhabdoforge.engine import Context, Agent, Scene
from rhabdoforge.engine.movement import Curve, Trajectory, extract_obj_curves
from rhabdoforge.compound_eyes import Model
from rhabdoforge.renderers import Renderer


if __name__ == "__main__":

    context = Context()

    scene = Scene(background_color=[0.45, 0.6, 0.7])

    example_obj = Path().home() / 'Desktop/gapArray.obj'
    scene.load(example_obj)

    scene.add_sky('assets/textures/bright_day_nosun')

    eye_model = Model.from_sphere(n=2000, force_isotropic=True)

    agent = Agent(position=(0.0, 0.0, 4.0))

    renderer = Renderer(
        model=eye_model,
        scene=scene,
        agent=agent,
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

    while context.run_interactive():

        context.input()

        # Make the agent follow the trajectory
        agent.follow(agent_path, dt=context.dt, align=True)

        output = renderer.step()

        context.display()

    context.free()
