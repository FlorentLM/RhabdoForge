import numpy as np
import open3d as o3d
from pathlib import Path

POINT_CLOUD_FILE = 'C:/Users/flolm/Downloads/Seville_high_resolution/ground_inner_high_res.ply'
OUTPUT_PREFIX = 'assets/scene_pointcloud'
DOWNSAMPLE_SIZE = 0.1
# Maximum number of points in a leaf node of the KD tree
MAX_LEAF_SIZE = 128     # smaller = deeper tree, larger = shallower tree
# Maximum recursion depth to prevent infinite loops on weird data
MAX_TREE_DEPTH = 30

# GPU Layout constants (in terms of 32-bit words/floats)

# GLSL struct KdNode:
# uint is_leaf;        // word 0
# uint split_axis;     // word 1
# uint point_count;    // word 2
# uint start_index;    // word 3
# float split_pos;     // word 4
# uint right_child;    // word 5
# uint pad0, pad1;     // word 6, 7
# TOTAL: 8 words = 32 bytes
KD_NODE_STRIDE_FLOATS = 8

# GLSL struct Point:
# vec4 pos;            // words 0-3
# vec4 normal;         // words 4-7
# vec4 color;          // words 8-11
# TOTAL: 12 words = 48 bytes
POINT_STRIDE_FLOATS = 12

def build_recursive(points, indices, depth):
    """ Recursively builds a KD tree and returns nodes and sorted point indices """

    if len(indices) <= MAX_LEAF_SIZE or depth > MAX_TREE_DEPTH:
        # Leaf node
        leaf_node_data = np.zeros(KD_NODE_STRIDE_FLOATS, dtype=np.float32)
        leaf_node_uint_view = leaf_node_data.view(np.uint32)

        leaf_node_uint_view[0] = 1  # is_leaf = true
        leaf_node_uint_view[2] = len(indices)  # point_count
        # 'start_index' (at word 3) will be set in a final pass

        return leaf_node_data.flatten(), indices

    # Internal node
    # Choose axis to split on
    split_axis = depth % 3

    # Sort points along the chosen axis and find the median
    current_points = points[indices]
    sorted_local_indices = np.argsort(current_points[:, split_axis])
    median_idx = len(indices) // 2

    # Partition indices into left and right children
    left_indices = indices[sorted_local_indices[:median_idx]]
    right_indices = indices[sorted_local_indices[median_idx:]]
    split_pos = points[indices[sorted_local_indices[median_idx]]][split_axis]

    # Recurse on children
    left_nodes_data, left_indices_sorted = build_recursive(points, left_indices, depth + 1)
    right_nodes_data, right_indices_sorted = build_recursive(points, right_indices, depth + 1)

    # Create the root node for this subtree
    root_node_data = np.zeros(KD_NODE_STRIDE_FLOATS, dtype=np.float32)
    root_node_uint_view = root_node_data.view(np.uint32)

    root_node_uint_view[0] = 0  # is_leaf = false
    root_node_uint_view[1] = split_axis
    root_node_data[4] = split_pos  # This is a float so write to float array directly

    # The right child starts after the current node (1) and all nodes in the left subtree
    num_left_nodes = len(left_nodes_data) // KD_NODE_STRIDE_FLOATS
    root_node_uint_view[5] = 1 + num_left_nodes  # right_child index

    # Combine nodes and indices from children
    all_nodes = np.concatenate(([root_node_data], left_nodes_data.reshape(-1, KD_NODE_STRIDE_FLOATS),
                                right_nodes_data.reshape(-1, KD_NODE_STRIDE_FLOATS)))
    all_indices = np.concatenate((left_indices_sorted, right_indices_sorted))

    return all_nodes.flatten(), all_indices

def main():

    input_file = Path(POINT_CLOUD_FILE)

    if not input_file.exists():
        print(f"Error: Point cloud file not found at {input_file}")
        return

    print(f"Loading point cloud from {input_file}...")
    pcd = o3d.io.read_point_cloud(input_file)

    if not pcd.has_points():
        print("Error: Point cloud is empty.")
        return

    # Downsample (optional)
    if DOWNSAMPLE_SIZE:
        pcd = pcd.voxel_down_sample(voxel_size=0.1)


    # R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    # pcd = pcd.rotate(R, center=(0, 0, 0))

    # Ensure normals exist
    if not pcd.has_normals():
        print("Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    points_np = np.asarray(pcd.points, dtype=np.float32)
    normals_np = np.asarray(pcd.normals, dtype=np.float32)
    colors_np = np.asarray(pcd.colors, dtype=np.float32)

    print(f"Building KD tree from {len(points_np)} points...")
    flat_tree_buffer, sorted_indices = build_recursive(points_np, np.arange(len(points_np)), 0)

    # Reshape for easier indexing
    flat_tree_buffer = flat_tree_buffer.reshape(-1, KD_NODE_STRIDE_FLOATS)
    tree_uint_view = flat_tree_buffer.view(np.uint32)

    print("Reordering point attribute buffers...")
    sorted_points = points_np[sorted_indices]
    sorted_normals = normals_np[sorted_indices]
    sorted_colors = colors_np[sorted_indices]

    print("Finalizing leaf node start indices...")
    current_pos = 0
    for i in range(len(flat_tree_buffer)):
        node_is_leaf = tree_uint_view[i, 0]
        if node_is_leaf == 1:
            # set start_index (word 3) for this leaf node
            tree_uint_view[i, 3] = current_pos
            point_count = tree_uint_view[i, 2]
            current_pos += point_count

    print("Creating interleaved point attribute buffer...")
    # Create the flat buffer for point data
    point_attributes = np.zeros((len(sorted_points), POINT_STRIDE_FLOATS), dtype=np.float32)

    # Place data into the flat buffer
    point_attributes[:, 0:3] = sorted_points    # pos.xyz
    point_attributes[:, 4:7] = sorted_normals   # normal.xyz
    point_attributes[:, 8:11] = sorted_colors   # color.rgb

    # Save the final GPU-ready buffers to disk
    output_dir = Path(OUTPUT_PREFIX).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    kdtree_path = f"{OUTPUT_PREFIX}.kdtree.npy"
    points_path = f"{OUTPUT_PREFIX}.points.npy"

    print(f"Saving pre-processed binary assets...")
    np.save(kdtree_path, flat_tree_buffer)
    np.save(points_path, point_attributes)

    print(f"Pre-processing complete.")
    print(f"  - Saved k-d tree with {len(flat_tree_buffer)} nodes to {kdtree_path}")
    print(f"  - Saved point data for {len(point_attributes)} points to {points_path}")

if __name__ == '__main__':
    main()