import numpy as np
from svg.path import parse_path
import xml.etree.ElementTree as ET
import re


def parse_drosophila_svg(svg_file: str) -> dict:

    tree = ET.parse(svg_file)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    data = {'ommatidia': [], 'stars': [], 'axes': {}, 'boundary': {}}

    for circle in root.findall('svg:circle', ns):
        if circle.get('id') != 'boundary':
            data['ommatidia'].append((float(circle.get('cx')), float(circle.get('cy'))))

    boundary_circle = root.find(".//*[@id='boundary']", ns)
    if boundary_circle is not None:
        data['boundary'] = {
            'cx': float(boundary_circle.get('cx')),
            'cy': float(boundary_circle.get('cy')),
            'r': float(boundary_circle.get('r'))
        }
    else:
        raise ValueError("Boundary circle with id='boundary' not found in SVG.")

    for i in range(1, 6):
        star_path_element = root.find(f".//*[@id='star-{i}']", ns)
        if star_path_element is not None:
            path_d = star_path_element.get('d')
            cleaned_path = re.sub(r"[A-Za-z,]", " ", path_d)
            coords = [float(c) for c in cleaned_path.split()]
            points = np.array(coords).reshape(-1, 2)
            data['stars'].append(tuple(np.mean(points, axis=0)))

    for axis_id in ['axis-x', 'axis-y', 'axis-v']:
        axis_path_element = root.find(f".//*[@id='{axis_id}']", ns)
        if axis_path_element is not None:
            # Store the raw path object for later sampling
            data['axes'][axis_id] = parse_path(axis_path_element.get('d'))

    print(f"Parsed SVG: Found {len(data['ommatidia'])} ommatidia")
    return data


def get_rotation_matrix(angle_rad, axis):
    """
    Calculates a rotation matrix for a given angle and axis
    """
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    one_minus_cos_a = 1 - cos_a

    return np.array([
        [cos_a + one_minus_cos_a * x ** 2, one_minus_cos_a * x * y - sin_a * z, one_minus_cos_a * x * z + sin_a * y],
        [one_minus_cos_a * y * x + sin_a * z, cos_a + one_minus_cos_a * y ** 2, one_minus_cos_a * y * z - sin_a * x],
        [one_minus_cos_a * z * x - sin_a * y, one_minus_cos_a * z * y + sin_a * x, cos_a + one_minus_cos_a * z ** 2]
    ])


def stereographic_to_spherical(x, y):
    """
    Converts 2D stereographic coordinates to 3D spherical coordinates (long, lat)
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    colatitude = 2 * np.arctan(rho / 2.0)
    latitude = np.pi / 2.0 - colatitude
    longitude = theta
    return longitude, latitude


def spherical_to_cartesian(longitude, latitude):
    """
    Converts spherical coordinates (long, lat) to 3D cartesian vectors
    """
    colatitude = np.pi / 2.0 - latitude
    x = np.sin(colatitude) * np.cos(longitude)
    y = np.sin(colatitude) * np.sin(longitude)
    z = np.cos(colatitude)
    return np.vstack([x, y, z]).T


##

PLOT = True

print("Generating Drosophila Eye Model (from Buchner 1971 data)")

svg_data = parse_drosophila_svg('stuff/biological_data/drosophila/drosophila_buchner1971_redigitized.svg')

bc = svg_data['boundary']

def to_stereo(coords_px):
    coords_px = np.atleast_2d(coords_px)
    x_stereo = -(coords_px[:, 0] - bc['cx']) / bc['r']
    y_stereo = -(coords_px[:, 1] - bc['cy']) / bc['r']
    return np.stack([x_stereo, y_stereo], axis=-1)

ommatidia_positions = to_stereo(svg_data['ommatidia'])

h_long, h_lat = stereographic_to_spherical(ommatidia_positions[:, 0], ommatidia_positions[:, 1])

# TODO: Use data from svg to orient the eyes correctly
transform_matrix = np.array([
    [ 0, 0, -1],
    [ 0, 1,  0],
    [ 1, 0,  0]
])

heisenberg_coords = spherical_to_cartesian(h_long, h_lat)
left_eye_dirs = heisenberg_coords @ transform_matrix.T

# Create the right eye by mirroring on X
right_eye_dirs = left_eye_dirs.copy()
right_eye_dirs[:, 0] *= -1

num_ommatidia = len(left_eye_dirs)
print(f"Generated {left_eye_dirs} unique ommatidia.")

# Generate eye_id for each eye (left = 0, right = 1)
left_eye_ids = np.zeros(num_ommatidia, dtype=int)
right_eye_ids = np.ones(num_ommatidia, dtype=int)

# Define plausible origins for the eyes
eye_radius = 0.00035        # 0.35 mm radius
eye_separation = 0.0003     # 0.3 mm separation

# Separate the eyes along the X-axis
left_eye_origins = left_eye_dirs * eye_radius - np.array([eye_separation / 2, 0, 0])
right_eye_origins = right_eye_dirs * eye_radius + np.array([eye_separation / 2, 0, 0])

both_eyes_origs = np.concatenate((right_eye_origins, left_eye_origins))
both_eyes_dirs = np.concatenate((right_eye_dirs, left_eye_dirs))
both_eyes_ids = np.concatenate((right_eye_ids, left_eye_ids))

# Save as npz
np.savez_compressed(
    "drosophila_eye.npz",
    directions=both_eyes_dirs,
    origins=both_eyes_origs,
    eye_id=both_eyes_ids
)

if PLOT:
    import matplotlib.pyplot as plt
    from PIL import Image

    image = Image.open('stuff/biological_data/drosophila/drosophila_buchner1971.png')
    image_array = np.array(image)
    aspect_ratio = image.size[0] / image.size[1]
    center = (0.6975, 0.5)
    im_scale = 2.275

    im_extent = (im_scale * (aspect_ratio - center[0]),
                 im_scale * (0 - center[0]),
                 im_scale * (0 - center[1]),
                 im_scale * (1 - center[1]))

    # Plotting
    fig, ax = plt.subplots(figsize=(18, 15))

    ax.imshow(image, origin='upper', extent=im_extent, aspect='equal', cmap='gray')

    ax.scatter(ommatidia_positions[:, 0], ommatidia_positions[:, 1], s=50, alpha=0.8, facecolors='none', edgecolors='red',
               label='Digitized')

    ax.set_title("Verification of Digitized Drosophila Eye Data", fontsize=16)
    ax.set_xlabel("Stereographic X Coordinate", fontsize=12)
    ax.set_ylabel("Stereographic Y Coordinate", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.set_xlim(im_extent[0], im_extent[1])
    ax.set_ylim(im_extent[2], im_extent[3])

    plt.show()

print("\nDrosophila eye models generated successfully!")
