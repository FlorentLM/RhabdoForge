import numpy as np

from insectvision.engine import Context, Agent, Scene, Asset
from insectvision.compound_eyes import ReceptorArray, RhabdomereKernel
from insectvision.renderers import Raytracer
from insectvision.geometry import plane_geom

# Bar config
BAR_WIDTH = 0.1
BAR_SEPARATION = 0.4  # center-to-center distance between the bars
BAR_HEIGHT = 10.0
DISTANCE = 2.0  # Distance in front of agent (-Z axis)

EYE_MODEL_PATH = 'species_models/drosophila_custom.npz'


def create_vertical_bar(name, x_pos, width, height, distance, texture):
    """Procedurally generates a vertical plane Asset."""

    v_0 = [x_pos - width / 2, -height / 2, -distance]
    v_1 = [x_pos - width / 2, height / 2, -distance]
    v_2 = [x_pos + width / 2, height / 2, -distance]
    v_3 = [x_pos + width / 2, -height / 2, -distance]

    vertices, uv_coords, faces = plane_geom(v_0, v_1, v_2, v_3)

    return Asset.from_arrays(
        name=name,
        vertices=vertices,
        faces=faces,
        uv_coords=uv_coords,
        texture=texture
    )


def main():

    context = Context()
    context.mouse_captured = False

    context.fixed_sim_dt = 1/200.0

    # empty pure black world
    scene = Scene(background_color=(0.0, 0.0, 0.0))
    # scene = Scene(background_color=(0.15, 0.15, 0.3))

    # put the sun right behind the agent to fully illuminate bars
    scene.sun.elevation = 1.0
    scene.sun.azimuth = 0.0
    scene.sun.color = (1.0, 1.0, 1.0)


    # pure white texture array
    white_tex = np.ones((32, 32), dtype=np.uint8) * 255


    # Calculate positions and add the two bars to the scene
    left_x = -BAR_SEPARATION / 2.0
    right_x = BAR_SEPARATION / 2.0

    bar_left = create_vertical_bar('bar_left', left_x, BAR_WIDTH, BAR_HEIGHT, DISTANCE, white_tex)
    bar_right = create_vertical_bar('bar_right', right_x, BAR_WIDTH, BAR_HEIGHT, DISTANCE, white_tex)

    scene.add_instance(bar_left)
    scene.add_instance(bar_right)

    droso = RhabdomereKernel(
        name='Drosophila',
        offsets_um=np.array([
            [-1.6881, 1.0273],
            [-1.8046, -0.9934],
            [-1.7111, -2.9717],
            [-0.0025, -1.9261],
            [1.6690, -0.9493],
            [1.6567, 0.9762],
            [0.0045, -0.0113],
        ]),
        diameters_um=np.array([1.8627, 1.8627, 1.8627, 1.8627, 1.8627, 1.8627, 1.5743]),
        nodal_distance_um=21.0,
        center_index=6,
        main_axis_indices=(2, 5),
        saccade_offset_deg=28.6,
    )

    eye_model = ReceptorArray.from_file(EYE_MODEL_PATH, kernel=droso)
    eye_model.scale(1e-6)
    eye_model.receptors.tau_membrane = 0.012

    agent = Agent(position=(0.0, 0.0, 0.0))

    renderer = Raytracer(
        receptor_array=eye_model,
        scene=scene,
        agent=agent,
        context=context,
        nb_samples=16,
        time_dithering=True,
        quasi_random=True,
        enable_actuation=True,  # starts with True, toggled manually from the dashboard
        enable_ambient=True,
        enable_direct=True,
        enable_shadows=False
    )

    # boost ambient intensity so the white bars are very bright
    renderer.ambient_intensity = 1.5

    renderer.selected_lenses = [648, 2]

    while context.run_interactive(agent=agent, scene=scene, renderer=renderer, use_dashboard=True):

        context.input()

        dt = context.tick()

        visual_output = renderer.step()

        context.draw(visual_output)

    renderer.free()
    scene.free()
    context.free()


if __name__ == '__main__':
    main()