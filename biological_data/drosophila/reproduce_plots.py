import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
matplotlib.use('QtAgg')

plot_1 = False
plot_2 = True
plot_1_all_lines = False

background_image = 'biological_data/drosophila/drosophila_Heisenberg_and_Wolf_1984.png'
overlay_image = 'biological_data/drosophila/overlay.png'

crop_pixels_top = 92
crop_pixels_bottom = 92
crop_pixels_left = 91
crop_pixels_right = 88

# -----------------------------------------------------------------------------


def spherical_to_opengl(lon_deg, lat_deg):
    """Converts spherical coordinates to 3D Cartesian (OpenGL)."""
    lon_rad, lat_rad = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    x = np.cos(lat_rad) * np.sin(lon_rad)
    y = np.sin(lat_rad)
    z = -np.cos(lat_rad) * np.cos(lon_rad)
    return np.array([x, y, z])


def opengl_to_spherical(x, y, z):
    """Converts 3D Cartesian (OpenGL) to spherical coordinates."""
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r[r == 0] = 1e-9
    lat_deg = np.rad2deg(np.arcsin(y / r))
    lon_deg = np.rad2deg(np.arctan2(x, -z))
    return lon_deg, lat_deg


def equatorial_stereographic_projection(lon_deg, lat_deg, center_lon_deg):
    lon_rad, lat_rad = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    lon0_rad, lat0_rad = np.deg2rad(center_lon_deg), np.deg2rad(0)
    cos_lat, cos_d_lon = np.cos(lat_rad), np.cos(lon_rad - lon0_rad)
    k = 2.0 / (1.0 + np.cos(lat0_rad) * cos_lat * cos_d_lon + np.sin(lat0_rad) * np.sin(lat_rad))
    if isinstance(k, np.ndarray):
        k[k < 0] = np.nan
    elif k < 0:
        k = np.nan
    x = k * cos_lat * np.sin(lon_rad - lon0_rad)
    y = k * (np.cos(lat0_rad) * np.sin(lat_rad) - np.sin(lat0_rad) * cos_lat * cos_d_lon)
    return x, y


def generate_small_circle(plane_normal, angular_offset_deg, num_points=400):
    """Generates the 3D points for a small circle on the unit sphere."""
    offset_rad = np.deg2rad(angular_offset_deg)
    d = np.sin(offset_rad)
    r = np.cos(offset_rad)
    circle_center = d * plane_normal
    z_axis = np.array([0, 0, 1])
    if np.abs(np.dot(plane_normal, z_axis)) > 0.99:
        z_axis = np.array([1, 0, 0])
    u_vec = np.cross(plane_normal, z_axis)
    u_vec /= np.linalg.norm(u_vec)
    w_vec = np.cross(plane_normal, u_vec)
    theta = np.linspace(0, 2 * np.pi, num_points)
    points = (circle_center[:, np.newaxis] +
              r * np.cos(theta) * u_vec[:, np.newaxis] +
              r * np.sin(theta) * w_vec[:, np.newaxis])
    return points

def centered_linspace(maxval, inc=1):
    x = np.arange(inc, maxval, inc)
    if x[-1] != maxval:
        x = np.r_[x, maxval]
    return np.r_[-x[::-1], 0, x]

# -----------------------------------------------------------------------------
# ORIGIN AND AXIS DIRECTIONS
# -----------------------------------------------------------------------------

# These values come from the 1984 graph
grid_origin_lon = -68.144360022
grid_origin_lat = -1.1835184140

nb_x_lines = 33
nb_y_lines = 32
nb_v_lines = 29

angle_x = np.radians(-23)
angle_y = np.radians(31.6)
angle_v = np.radians(-87.2)

dir_x = np.array([np.cos(angle_x), np.sin(angle_x)])
dir_y = np.array([np.cos(angle_y), np.sin(angle_y)])
dir_v = np.array([np.cos(angle_v), np.sin(angle_v)])

# 3D geometry
p_origin_3d = spherical_to_opengl(grid_origin_lon, grid_origin_lat)

lon_rad, lat_rad = np.deg2rad(grid_origin_lon), np.deg2rad(grid_origin_lat)
h_vec_3d = np.array([np.cos(lat_rad) * np.cos(lon_rad), 0, np.cos(lat_rad) * np.sin(lon_rad)])
v_vec_3d = np.array([-np.sin(lat_rad) * np.sin(lon_rad), np.cos(lat_rad), np.sin(lat_rad) * np.cos(lon_rad)])

v_axis_3d = (dir_v[0] * h_vec_3d + dir_v[1] * v_vec_3d)
x_axis_3d = (dir_x[0] * h_vec_3d + dir_x[1] * v_vec_3d)
y_axis_3d = (dir_y[0] * h_vec_3d + dir_y[1] * v_vec_3d)

N_v = np.cross(p_origin_3d, v_axis_3d)
N_v /= np.linalg.norm(N_v)
N_x = np.cross(p_origin_3d, x_axis_3d)
N_x /= np.linalg.norm(N_x)
N_y = np.cross(p_origin_3d, y_axis_3d)
N_y /= np.linalg.norm(N_y)

# Generate grid
all_grid_curves_stereo = []
x_step = 5
y_step = 5
v_step = 6

grid_x_offsets = centered_linspace(90, x_step)
grid_y_offsets = centered_linspace(90, y_step)
grid_v_offsets = centered_linspace(90, v_step)

# Determine the offsets for the stereographic plot based on the user's choice
if not plot_1_all_lines:
    grid_x_offsets_plot_1 = grid_y_offsets_plot_1 = grid_v_offsets_plot_1 = [0.0]
else:
    grid_x_offsets_plot_1 = grid_x_offsets
    grid_y_offsets_plot_1 = grid_y_offsets
    grid_v_offsets_plot_1 = grid_v_offsets

# Generate curves for the first plot
for normal, color, offsets in zip([N_x, N_y, N_v], ['red', 'green', 'blue'], [grid_x_offsets_plot_1, grid_y_offsets_plot_1, grid_v_offsets_plot_1]):
    for offset in offsets:
        curve_3d = generate_small_circle(normal, offset)
        lon, lat = opengl_to_spherical(curve_3d[0], curve_3d[1], curve_3d[2])
        all_grid_curves_stereo.append({'lon': lon, 'lat': lat, 'color': color})

# Generate the full set of curves for the second plot
all_grid_curves_cartesian = []
for normal, color, offsets in zip([N_x, N_y, N_v], ['red', 'green', 'blue'], [grid_x_offsets, grid_y_offsets, grid_v_offsets]):
    for offset in offsets:
        curve_3d = generate_small_circle(normal, offset)
        lon, lat = opengl_to_spherical(curve_3d[0], curve_3d[1], curve_3d[2])
        all_grid_curves_cartesian.append({'lon': lon, 'lat': lat, 'color': color})

# Plotting
if plot_1:
    stereo_center_lon = -90.0

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Subplot 1: Equatorial Stereographic Projection
    ax.set_title("Stereographic Projection", fontsize=16)

    for lat in np.arange(-80, 81, 10):
        lons = np.linspace(-180, 0, 200)
        x, y = equatorial_stereographic_projection(lons, lat, stereo_center_lon)
        ax.plot(x, y, color='black', linewidth=1)

    for lon in np.arange(-180, 1, 10):
        lats = np.linspace(-90, 90, 200)
        x, y = equatorial_stereographic_projection(lon, lats, stereo_center_lon)
        ax.plot(x, y, color='black', linewidth=1)

    boundary_lons = np.concatenate([np.full(100, 0), np.full(100, -180)])
    boundary_lats = np.concatenate([np.linspace(-90, 90, 100), np.linspace(90, -90, 100)])
    bx, by = equatorial_stereographic_projection(boundary_lons, boundary_lats, stereo_center_lon)
    ax.plot(bx, by, color='black', linewidth=1.5)

    # Text labels
    for lat in np.arange(-80, 81, 20):
        lx, ly = equatorial_stereographic_projection(5, lat, stereo_center_lon)
        ax.text(lx, ly, f'{lat}', rotation=lat, ha='left', va='center', fontweight='bold', fontsize=12)

    for lon in [180, 170]:
        lx, ly = equatorial_stereographic_projection(lon, 180, stereo_center_lon)
        ax.text(lx - 0.02, ly -0.05, f'{lon}', ha='right', va='center', fontweight='bold', fontsize=10)

    # Plot colored grid lines (either full or single) with clipping
    for curve in all_grid_curves_stereo:
        lon_clipped, lat_clipped = curve['lon'].copy(), curve['lat'].copy()
        lon_clipped[(lon_clipped < -185) | (lon_clipped > 5)] = np.nan
        lat_clipped[(lat_clipped < -95) | (lat_clipped > 95)] = np.nan
        proj_x, proj_y = equatorial_stereographic_projection(lon_clipped, lat_clipped, stereo_center_lon)
        ax.plot(proj_x, proj_y, color=curve['color'], lw=1.5, alpha=0.9)

    ax.set_aspect('equal')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.axis('off')

    plt.show()

if plot_2:
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    # Subplot 2: Cartesian projection
    ax.set_title("Cartesian (Equirectangular) Projection", fontsize=16)

    # Display the background image
    try:
        img = mpimg.imread(background_image)
        height, width, _ = img.shape
        img_cropped = img[crop_pixels_top:height - crop_pixels_bottom, crop_pixels_left:width - crop_pixels_right, :]
        ax.imshow(img_cropped, extent=[-180, 180, -90, 90], aspect='auto', zorder=0)
    except FileNotFoundError:
        print(f"Warning: Background image not found at '{background_image}'. Plotting without it.")

    # Plot the colored grid lines
    for curve in all_grid_curves_cartesian:
        lon_clipped, lat_clipped = curve['lon'].copy(), curve['lat'].copy()
        lon_clipped[(lon_clipped < -180) | (lon_clipped > 0)] = np.nan
        lat_clipped[np.isnan(lon_clipped)] = np.nan
        ax.plot(lon_clipped, lat_clipped, color=curve['color'], lw=1.5, alpha=0.9, zorder=2)

    # Display the overlay image
    if overlay_image:
        try:
            overlay_img = mpimg.imread(overlay_image)
            height, width, _ = overlay_img.shape
            overlay_img_cropped = overlay_img[crop_pixels_top:height - crop_pixels_bottom, crop_pixels_left:width - crop_pixels_right, :]

            ax.imshow(overlay_img, extent=[-180, 180, -90, 90], aspect='auto',
                       alpha=1.0, zorder=3)
        except FileNotFoundError:
            print(f"Warning: Overlay image not found at '{overlay_image}'. Skipping overlay.")

    # Plot markers on top of everything
    ax.scatter(grid_origin_lon, grid_origin_lat, color='k', zorder=5)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    ax.set_xlabel("Longitude (ψ) [degrees]", fontsize=12)
    ax.set_ylabel("Latitude (ϑ) [degrees]", fontsize=12)

    ax.set_xticks(np.arange(-180, 181, 20))
    ax.set_yticks(np.arange(-90, 91, 10))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')

    plt.show()