import numpy as np
import open3d as o3d
from pathlib import Path
import time
import pytinybvh


# TODO: Move this debug script to the Scene class for proper implementation


# --- Configuration ---
# Set one of these to True
PROCESS_POINT_CLOUD = True
PROCESS_MESHES = False

# --- Point Cloud Settings ---
POINT_CLOUD_FILE = 'C:/Users/flolm/Downloads/Seville_high_resolution/ground_inner_high_res.ply'
DOWNSAMPLE_SIZE = 0.1
OUTPUT_PREFIX = 'assets/scene_pointcloud_bvh'

# --- Mesh Settings ---
# specify mesh files and their transforms here
# MESH_FILES = [...]
# OUTPUT_PREFIX_MESH = 'assets/scene_mesh_bvh'


def main():
    input_file = Path(POINT_CLOUD_FILE)
    if not input_file.exists():
        print(f"Error: Input file not found at {input_file}")
        return

    primitives_np = None
    build_func = None
    prim_type = ""

    # Load geometry into a primitive numpy array
    if PROCESS_POINT_CLOUD:
        print(f"Loading and processing point cloud from {input_file}...")
        prim_type = "points"
        pcd = o3d.io.read_point_cloud(input_file)

        aabb = pcd.get_axis_aligned_bounding_box()
        print("--- Point Cloud Stats ---")
        print(f"Center: {aabb.get_center()}")
        print(f"Min bound: {aabb.get_min_bound()}")
        print(f"Max bound: {aabb.get_max_bound()}")
        print("-------------------------")

        pcd = pcd.translate(-aabb.get_center(), relative=True)

        if DOWNSAMPLE_SIZE:
            pcd = pcd.voxel_down_sample(voxel_size=DOWNSAMPLE_SIZE)

        if not pcd.has_normals(): pcd.estimate_normals()

        points = np.asarray(pcd.points, dtype=np.float32)
        normals = np.asarray(pcd.normals, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32)

        # primitives_np is the interleaved buffer that will be reordered
        # GLSL struct Point: vec4 pos, vec4 normal, vec4 color (48 bytes)
        primitives_np = np.zeros((len(points), 12), dtype=np.float32)
        primitives_np[:, 0:3] = points
        primitives_np[:, 4:7] = normals
        primitives_np[:, 8:11] = colors

        # The build function needs the point positions (N, 3)
        build_input = points
        build_func = pytinybvh.from_points

    elif PROCESS_MESHES:
        print("Processing meshes...")
        prim_type = "triangles"

        # This part is pretty much what RaytracingScene._pack_static_geometry() and update() do
        # (loading all the scene meshes, applying their instance transforms, and collating all world-space triangles)

        print("Mesh processing not fully implemented in this debug script.")
        return

    if primitives_np is None or build_func is None:
        print("No primitives were loaded. Exiting.")
        return

    # Build BVH using pytinybvh
    print(f"\nBuilding BVH from {len(build_input)} {prim_type}...")

    start_time = time.time_ns()
    bvh_nodes, prim_indices = build_func(build_input)
    gen_time_ms = (time.time_ns() - start_time) / 1e6
    print(f"BVH build complete! Time: {gen_time_ms:.4f} ms")
    print(f"Generated {len(bvh_nodes)} BVH nodes.")

    # Reorder the full primitive data buffer
    print("Reordering primitive data for GPU...")
    start_time = time.time_ns()
    reordered_primitives = primitives_np[prim_indices]
    reorder_time_ms = (time.time_ns() - start_time) / 1e6
    print(f"Reordering took: {reorder_time_ms:.4f} ms")

    # Save GPU-ready buffers to disk
    output_dir = Path(OUTPUT_PREFIX).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    bvh_path = f"{OUTPUT_PREFIX}.bvh.npy"
    primitives_path = f"{OUTPUT_PREFIX}.primitives.npy"

    print("\nSaving pre-processed binary assets...")
    np.save(bvh_path, bvh_nodes)
    np.save(primitives_path, reordered_primitives)

    print("Pre-processing complete.")
    print(f"  - Saved BVH nodes to: {bvh_path}")
    print(f"  - Saved reordered primitive data to: {primitives_path}")


if __name__ == '__main__':
    main()