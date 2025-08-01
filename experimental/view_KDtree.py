import pyvista as pv
import numpy as np
from pathlib import Path

OUTPUT_PREFIX = 'assets/scene_pointcloud'
MAX_VISUALIZATION_DEPTH = 10

# GLSL Struct constants
# (32-bit words)
KD_NODE_STRIDE_FLOATS = 8
POINT_STRIDE_FLOATS = 12


def generate_tree_boxes(kdtree_buffer, node_idx, bounds, depth, max_depth):
    """
    Recursively traverses the KD-tree and generates pyvista.Box objects for each node's bounding box
    """
    if depth > max_depth or node_idx >= len(kdtree_buffer):
        return []

    # Create a box for the current node
    current_box = pv.Box(bounds)
    all_boxes = [current_box]

    node = kdtree_buffer[node_idx]
    is_leaf = node.view(np.uint32)[0]

    # If it's an internal node, recurse to its children
    if not is_leaf:
        split_axis = node.view(np.uint32)[1]
        split_pos = node[4]
        right_child_offset = node.view(np.uint32)[5]

        # Calculate bounds for the left child
        left_bounds = bounds.copy()
        left_bounds[split_axis * 2 + 1] = split_pos  # Modify max_coord
        all_boxes.extend(generate_tree_boxes(
            kdtree_buffer, node_idx + 1, left_bounds, depth + 1, max_depth
        ))

        # Calculate bounds for the right child
        right_bounds = bounds.copy()
        right_bounds[split_axis * 2] = split_pos  # Modify min_coord
        all_boxes.extend(generate_tree_boxes(
            kdtree_buffer, node_idx + right_child_offset, right_bounds, depth + 1, max_depth
        ))

    return all_boxes

def main():
    kdtree_path = Path(f"{OUTPUT_PREFIX}.kdtree.npy")
    points_path = Path(f"{OUTPUT_PREFIX}.points.npy")

    if not kdtree_path.exists() or not points_path.exists():
        print(f"Error: Could not find pre-processed data at '{OUTPUT_PREFIX}.*'")
        print("Please run DEBUG_pointcloud_BVH.py first.")
        return

    print("Loading pre-processed data...")
    kdtree_nodes = np.load(kdtree_path)
    point_attributes = np.load(points_path)

    # Extract positions and colors for plotting
    positions = point_attributes[:, 0:3]
    colors = point_attributes[:, 8:11]

    # Create a PyVista object for the point cloud
    cloud = pv.PolyData(positions)
    cloud['colors'] = colors * 255  # PyVista expects RGB in 0-255 range

    print(f"Visualizing {len(positions)} points and KD-tree up to depth {MAX_VISUALIZATION_DEPTH}.")

    # --- Generate KD-Tree Boxes ---
    # Calculate the initial bounding box of the whole scene
    min_bounds = np.min(positions, axis=0)
    max_bounds = np.max(positions, axis=0)
    initial_bounds = [
        min_bounds[0], max_bounds[0],
        min_bounds[1], max_bounds[1],
        min_bounds[2], max_bounds[2],
    ]

    # Recursively generate all the box meshes
    box_meshes = generate_tree_boxes(kdtree_nodes, 0, initial_bounds, 0, MAX_VISUALIZATION_DEPTH)

    # Plotting
    plotter = pv.Plotter(window_size=[1200, 900])
    plotter.add_mesh(
        cloud,
        scalars='colors',
        rgb=True,
        style='points',
        point_size=2,
        render_points_as_spheres=True
    )

    # Add all generated bounding boxes for KD-tree nodes
    for i, box in enumerate(box_meshes):
        plotter.add_mesh(box, style='wireframe', color='cyan', opacity=0.5)

    plotter.add_axes()
    plotter.show_grid()
    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.2)

    print("Showing 3D point cloud and KD-tree plot. Close the window to continue.")
    plotter.show()


if __name__ == '__main__':
    main()