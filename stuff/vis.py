import pyvista as pv
import numpy as np


def plot_eye_3d(eye_model, glyph_length=0.02, show_origins=False):

    # Extract origins and directions from the eye model
    origins = np.copy(eye_model.ommatidia[:].origin)
    directions = np.copy(eye_model.ommatidia[:].direction)

    if np.allclose(origins, 0):
        print("Warning: Eye model has zero radius. Glyphs will all start from the center.")
        print("Creating a temporary surface for plotting.")
        radius = np.max(np.linalg.norm(directions, axis=1))
        origins = directions * radius  # place origins on the surface

    points_polydata = pv.PolyData(origins)
    points_polydata['directions'] = -directions
    points_polydata['scale'] = np.full(eye_model.ommatidia_count, glyph_length)

    # "glyph" filter to place a cone at each point
    geom = pv.Cone(radius=0.3, resolution=12)  # radius is a ratio of the length

    cones = points_polydata.glyph(
        orient='directions',
        scale='scale',
        geom=geom
    )

    plotter = pv.Plotter(window_size=[800, 800])

    if show_origins:
        plotter.add_mesh(
            points_polydata,
            color='blue',
            render_points_as_spheres=True,
            point_size=10
        )

    plotter.add_mesh(cones, color='red')
    plotter.add_axes()
    plotter.show_grid()

    # Set the camera to a nice isometric view
    plotter.camera_position = 'xy'
    plotter.camera.elevation = 30
    plotter.camera.azimuth = 30
    plotter.camera.zoom(1.5)

    plotter.show()


if __name__ == "__main__":
    from geometry.compound_eyes import CompoundEye

    # Create an eye with a non-zero radius so the origins are on the surface
    NB_OMMATIDIA_PLOT = 1962
    EYE_RADIUS_PLOT = 1.0
    eye = CompoundEye(num_ommatidia=NB_OMMATIDIA_PLOT, eye_radius=EYE_RADIUS_PLOT)
    plot_eye_3d(eye, glyph_length=EYE_RADIUS_PLOT / 5, show_origins=True)