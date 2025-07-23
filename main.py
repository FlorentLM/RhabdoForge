from graphics.glm import translation_mat
from graphics.engine import Engine
from graphics.scene import Instance
from geometry.primitives import CUBE_VERTICES
from graphics.skybox import Skybox
from graphics.utils import load_cubemap


def main():

    eng = Engine(width=1280, height=720, headless=False)
    # eng = Engine(width=1024, height=1024, headless=True)

    # --- Load Assets and Create Scene ---

    crate_mesh = eng.load_mesh(
        name="crate",
        vertex_data=CUBE_VERTICES,
        vert_shader_path='shaders/base.vert',
        frag_shader_path='shaders/base.frag',
        texture_path='textures/wood.jpg'
    )

    eng.skybox = Skybox()
    eng.skybox_texture_id = load_cubemap('textures/bright_day')

    # Create instances of the loaded mesh
    eng.add_instance(Instance(asset=crate_mesh, transform=translation_mat([ 0.0, 0.0, 0.0])))
    eng.add_instance(Instance(asset=crate_mesh, transform=translation_mat([-3.0, 0.0, 0.0])))
    eng.add_instance(Instance(asset=crate_mesh, transform=translation_mat([ 3.0, 0.0, 0.0])))

    # --- Run the Simulation/Visualization ---

    # Option A: Run the interactive preview mode

    print("Starting interactive mode. Press ESC to quit.")
    eng.run_interactive()

    # Option B: Run a scripted simulation

    # print("Running scripted simulation for 100 frames.")
    # for i in range(100):
    #     # Update object transforms
    #     rotation = glm.rotation_mat(np.deg2rad(i * 2), engine_utils.WORLD_UP)
    #     eng.scene.instances[0].transform = rotation
    #
    #     # Render one frame (to the hidden buffer if headless)
    #     eng.render_frame()


if __name__ == "__main__":
    main()