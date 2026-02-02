#ifndef RAYTRACING_COMMON_GLSL
#define RAYTRACING_COMMON_GLSL

#include "commons.glsl"
#include "pytinybvh_preamble.glsl"

// --------------------------------------------------- Structs ---------------------------------------------------------

struct Ray {
    vec3 origin;
    vec3 inv_direction;
    float t;            // max travel distance
};

struct HitInfo {
    bool found;
    float t;            // distance along ray
    vec3 barycentric_coords; // for triangle hits
    uint primitive_idx;
    uint instance_id;        // ID of the instance hit in the TLAS
    bool is_point_hit;
};

// A struct to hold all information about an instance
struct InstanceInfo {
    mat4 transform;
    mat4 inverse_transform;
    uint blas_node_offset;
    uint vertex_or_point_offset; // primitive_offset
    uint index_offset;
    uint material_id;
    uint is_points;    // 0 = no (i.e. mesh), 1 = points
    uint prim_index_offset;
    float radius_factor;
    uint pad0;
};  // Total size 160 bytes

// -------------------------------------------- Uniforms and textures --------------------------------------------------

layout(binding = 0) uniform samplerCube skybox;
layout(binding = 1) uniform sampler2DArray scene_textures;

uniform uint nb_tlas_nodes;
uniform vec3 background_color;
uniform vec3 sun_direction;
uniform float shadow_intensity;
uniform bool use_skybox;

// ----------------------------------------------- SSBO Bindings -------------------------------------------------------
// TODO: Add compile-time ifs around indexed geometry if point cloud only (and vice versa)??


// Node access (Standard layout)

#if TBVH_LAYOUT_STANDARD || TBVH_LAYOUT_BVH_GPU // BVH_GPU uses the same StdNode layout on the GPU

// Read nodes as raw 32-bit words to be layout-agnostic at the API level
layout(std430, binding = 6) readonly buffer AllBlasNodesBuffer { uint blas_nodes32[]; };
layout(std430, binding = 7) readonly buffer TlasNodesBuffer    { uint tlas_nodes32[]; };

const uint TBVH_WORDS_PER_NODE = TBVH_NODE_STRIDE_FLOATS; // 1 float == 1 uint (32-bit)

struct StdNode {
    vec4 data1; // .xyz = AABB Min, .w = Left/First index (bitcast uint)
    vec4 data2; // .xyz = AABB Max, .w = Primitive count (0 for internal)
};

// Node index is always relative to the base, which is passed in

// Load a Standard-layout node from the BLAS pool
StdNode load_blas_node(uint base_node_offset, uint node_index) {
    uint w = (base_node_offset + node_index) * TBVH_WORDS_PER_NODE;
    StdNode n;
    n.data1 = vec4(
        uintBitsToFloat(blas_nodes32[w+0]), uintBitsToFloat(blas_nodes32[w+1]),
        uintBitsToFloat(blas_nodes32[w+2]), uintBitsToFloat(blas_nodes32[w+3]));
    n.data2 = vec4(
        uintBitsToFloat(blas_nodes32[w+4]), uintBitsToFloat(blas_nodes32[w+5]),
        uintBitsToFloat(blas_nodes32[w+6]), uintBitsToFloat(blas_nodes32[w+7]));
    return n;
}

// Load a Standard-layout node from the TLAS pool (base = 0)
StdNode load_tlas_node(uint node_index) {
    uint w = node_index * TBVH_WORDS_PER_NODE;
    StdNode n;
    n.data1 = vec4(
        uintBitsToFloat(tlas_nodes32[w+0]), uintBitsToFloat(tlas_nodes32[w+1]),
        uintBitsToFloat(tlas_nodes32[w+2]), uintBitsToFloat(tlas_nodes32[w+3]));
    n.data2 = vec4(
        uintBitsToFloat(tlas_nodes32[w+4]), uintBitsToFloat(tlas_nodes32[w+5]),
        uintBitsToFloat(tlas_nodes32[w+6]), uintBitsToFloat(tlas_nodes32[w+7]));
    return n;
}

#else

// Fallback for non-Standard layouts
struct BvhNode {
    vec4 data1; // .xyz = AABB Min, .w = Left child/First primitive index
    vec4 data2; // .xyz = AABB Max, .w = Primitive count (0 for internal nodes)
}; // total size = 32 bytes (2x vec4)

layout(std430, binding = 6) readonly buffer AllBlasNodesBuffer { BvhNode blas_nodes[]; };
layout(std430, binding = 7) readonly buffer TlasNodesBuffer    { BvhNode tlas_nodes[]; };

#endif // TBVH_LAYOUT_STANDARD || TBVH_LAYOUT_BVH_GPU

// Bindings

layout(std430, binding = 2) readonly buffer VertexBuffer { float v[]; };
layout(std430, binding = 3) readonly buffer IndexBuffer { uint indices[]; };
layout(std430, binding = 4) readonly buffer MaterialBuffer { Material materials[]; };
layout(std430, binding = 5) readonly buffer PointsBuffer { float points_data[]; };

// BVH bindings start at 6
layout(row_major, std430, binding = 8) readonly buffer InstancesBuffer { InstanceInfo instances[]; };   // row-major!
layout(std430, binding = 9) readonly buffer TlasPrimIndexBuffer { uint tlas_prim_indices[]; };
layout(std430, binding = 10) readonly buffer BlasPrimIndexBuffer { uint blas_prim_indices[]; };

// ------------------------------------- Forward declarations and helpers ----------------------------------------------

float intersect_aabb(in Ray r, vec3 aabb_min, vec3 aabb_max);
HitInfo intersect_triangle(inout Ray r, vec3 direction, vec3 v0, vec3 v1, vec3 v2);
HitInfo intersect_sphere(inout Ray r, vec3 direction, vec3 center, float radius);
void traverse_tlas(inout Ray r_world, vec3 dir_world, out HitInfo closest_hit);
void traverse_blas(inout Ray r_obj, vec3 dir_obj, out HitInfo blas_hit, InstanceInfo inst);

// Mini helpers to get vertex data from the buffer
vec3 getPos(uint i){ uint b = i*5u; return vec3(v[b], v[b+1], v[b+2]); }
vec2 getUV (uint i){ uint b = i*5u; return vec2(v[b+3], v[b+4]); }

// Mini helper to get all point data
Point getPoint(uint point_idx) {
    uint base_offset = point_idx * 12u; // 12 floats per point
    Point p;
    p.position = vec3(points_data[base_offset + 0], points_data[base_offset + 1], points_data[base_offset + 2]);
    p.radius = points_data[base_offset + 3];
    p.normal = vec3(points_data[base_offset + 4], points_data[base_offset + 5], points_data[base_offset + 6]);
    p.color = vec3(points_data[base_offset + 7], points_data[base_offset + 8], points_data[base_offset + 9]);
    return p;
}

// Mini helper to get only position + radius (avoids fetching normals/colors in the hot loop)
void fast_getPoint(uint point_idx, out vec3 pos, out float radius) {
    uint base_offset = point_idx * 12u; // 12 floats per point
    pos.x = points_data[base_offset + 0u];
    pos.y = points_data[base_offset + 1u];
    pos.z = points_data[base_offset + 2u];
    radius = points_data[base_offset + 3u];
}

// ------------------------------------------ Traversal implementation -------------------------------------------------

void traverse_tlas(inout Ray r_world, vec3 dir_world, out HitInfo closest_hit) {
    closest_hit.found = false;
    if (nb_tlas_nodes == 0u) return;

    uint stack[64];
    uint stack_ptr = 0;
    stack[stack_ptr++] = 0u;    // TLAS traversal always starts at node 0

    while (stack_ptr > 0u) {
        uint node_idx = stack[--stack_ptr];
        #if TBVH_LAYOUT_STANDARD
            StdNode node = load_tlas_node(node_idx);
        #else
            BvhNode node = tlas_nodes[node_idx];
        #endif

        if (intersect_aabb(r_world, node.data1.xyz, node.data2.xyz) >= r_world.t) continue;

        uint prim_count = floatBitsToUint(node.data2.w);
        uint first_idx = floatBitsToUint(node.data1.w);

        if (prim_count > 0u) { // TLAS leaf node
            for (uint j = 0u; j < prim_count; ++j) {
                uint instance_id = tlas_prim_indices[first_idx + j];
                InstanceInfo inst = instances[instance_id];

                // Transform ray to object space
                Ray r_obj;
                r_obj.origin = (inst.inverse_transform * vec4(r_world.origin, 1.0)).xyz;
                vec3 dir_obj = (inst.inverse_transform * vec4(dir_world, 0.0)).xyz;
                r_obj.inv_direction = 1.0 / dir_obj;
                r_obj.t = 1.0/0.0;

                HitInfo blas_hit;
                traverse_blas(r_obj, dir_obj, blas_hit, inst);

                if (blas_hit.found) {
                    // Transform hit point back to world space
                    vec3 hit_point_obj = r_obj.origin + dir_obj * blas_hit.t;
                    vec3 hit_point_world = (inst.transform * vec4(hit_point_obj, 1.0)).xyz;
                    float new_world_t = distance(r_world.origin, hit_point_world);

                    if (new_world_t < r_world.t) {
                        r_world.t = new_world_t;
                        closest_hit = blas_hit;
                        closest_hit.found = true;
                        closest_hit.t = new_world_t;
                        closest_hit.instance_id = instance_id;
                    }
                }
            }
        } else { // TLAS internal node
            uint left_idx = first_idx;
            uint right_idx = first_idx + 1;
            #if TBVH_LAYOUT_STANDARD
                StdNode leftNode = load_tlas_node(left_idx);
                StdNode rightNode = load_tlas_node(right_idx);
                float d1 = intersect_aabb(r_world, leftNode.data1.xyz, leftNode.data2.xyz);
                float d2 = intersect_aabb(r_world, rightNode.data1.xyz, rightNode.data2.xyz);
            #else
                float d1 = intersect_aabb(r_world, tlas_nodes[left_idx].data1.xyz,  tlas_nodes[left_idx].data2.xyz);
                float d2 = intersect_aabb(r_world, tlas_nodes[right_idx].data1.xyz, tlas_nodes[right_idx].data2.xyz);
            #endif

            if (d1 > d2) { float temp_d = d1; d1 = d2; d2 = temp_d; uint temp_i = left_idx; left_idx = right_idx; right_idx = temp_i; }
            if (d2 < r_world.t && stack_ptr < 64) stack[stack_ptr++] = right_idx;
            if (d1 < r_world.t && stack_ptr < 64) stack[stack_ptr++] = left_idx;
        }
    }
}

void traverse_blas(inout Ray r_obj, vec3 dir_obj, out HitInfo blas_hit, InstanceInfo inst) {
    blas_hit.found = false;

    uint stack[64];
    uint stack_ptr = 0;
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0u) {
        uint node_idx = stack[--stack_ptr];
         #if TBVH_LAYOUT_STANDARD
            StdNode node = load_blas_node(inst.blas_node_offset, node_idx);
        #else
            BvhNode node = blas_nodes[inst.blas_node_offset + node_idx];
        #endif

        if (intersect_aabb(r_obj, node.data1.xyz, node.data2.xyz) >= r_obj.t) continue;

        uint prim_count = floatBitsToUint(node.data2.w);
        uint first_idx = floatBitsToUint(node.data1.w);

        if (prim_count > 0u) { // BLAS leaf node
            uint prim_base = inst.prim_index_offset + first_idx;

            for (uint i = 0; i < prim_count; ++i) {
                // Get the primitive index within this BLAS
                // 'first_idx' for a leaf is the start of prims in the BLAS's prim_indices list
                uint blas_prim_id = blas_prim_indices[prim_base + i];

                if (inst.is_points == 1u) {
                     uint point_id = inst.vertex_or_point_offset + blas_prim_id;
                     vec3 c; float rad;
                     fast_getPoint(point_id, c, rad);
                     rad *= inst.radius_factor;
                     HitInfo p_hit = intersect_sphere(r_obj, dir_obj, c, rad);

                     if (p_hit.found) {
                         blas_hit.found = true;
                         blas_hit.is_point_hit = true;
                         blas_hit.primitive_idx = blas_prim_id; // This is the local index
                         blas_hit.t = p_hit.t;
                         r_obj.t = p_hit.t;
                     }

                } else {
                    // For triangles, blas_prim_id is the triangle index within the asset
                    // Each triangle uses 3 indices, so we need to multiply by 3
                    uint base_idx = inst.index_offset + blas_prim_id * 3;
                    uint i0 = indices[base_idx + 0];
                    uint i1 = indices[base_idx + 1];
                    uint i2 = indices[base_idx + 2];

                    uint base_vtx = inst.vertex_or_point_offset;
                    vec3 v0 = getPos(base_vtx + i0);
                    vec3 v1 = getPos(base_vtx + i1);
                    vec3 v2 = getPos(base_vtx + i2);

                    HitInfo tri_hit = intersect_triangle(r_obj, dir_obj, v0, v1, v2);
                    if (tri_hit.found) {
                        blas_hit.found = true;
                        blas_hit.is_point_hit = false;
                        blas_hit.primitive_idx = blas_prim_id;  // Store the BLAS-local triangle ID
                        blas_hit.barycentric_coords = tri_hit.barycentric_coords;
                        blas_hit.t = tri_hit.t;
                        r_obj.t = tri_hit.t;
                    }
                }
            }
        } else { // BLAS internal node
            // 'first_idx' for an internal node is the BLAS-local index of the left child
            uint left_idx = first_idx;
            uint right_idx = left_idx + 1;

            #if TBVH_LAYOUT_STANDARD

                StdNode leftNode  = load_blas_node(inst.blas_node_offset, left_idx);
                StdNode rightNode = load_blas_node(inst.blas_node_offset, right_idx);
                float d1 = intersect_aabb(r_obj, leftNode.data1.xyz,  leftNode.data2.xyz);
                float d2 = intersect_aabb(r_obj, rightNode.data1.xyz, rightNode.data2.xyz);

            #else

                float d1 = intersect_aabb(r_obj, blas_nodes[inst.blas_node_offset + left_idx].data1.xyz,  blas_nodes[inst.blas_node_offset + left_idx].data2.xyz);
                float d2 = intersect_aabb(r_obj, blas_nodes[inst.blas_node_offset + right_idx].data1.xyz, blas_nodes[inst.blas_node_offset + right_idx].data2.xyz);

            #endif // TBVH_LAYOUT_STANDARD

            // Push the farther node first, so the closer one is processed next
            if (d1 > d2) {
                float temp_d = d1; d1 = d2; d2 = temp_d;
                uint temp_i = left_idx; left_idx = right_idx; right_idx = temp_i;
            }
            if (d2 < r_obj.t && stack_ptr < 64) stack[stack_ptr++] = right_idx;
            if (d1 < r_obj.t && stack_ptr < 64) stack[stack_ptr++] = left_idx;
        }
    }
}

// -------------------------------------- Intersection implementation --------------------------------------------------

HitInfo intersect_triangle(inout Ray r, vec3 direction, vec3 v0, vec3 v1, vec3 v2) {
    HitInfo hit;
    hit.found = false;

    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;

    vec3 h = cross(direction, edge2);
    float a = dot(edge1, h);

    if (a > -1e-6 && a < 1e-6) {
        return hit;
    }

    float f = 1.0 / a;
    vec3 s = r.origin - v0;
    float u = f * dot(s, h);

    if (u < 0.0 || u > 1.0) {
        return hit;
    }

    vec3 q = cross(s, edge1);
    float v = f * dot(direction, q);

    if (v < 0.0 || u + v > 1.0) {
        return hit;
    }

    float t = f * dot(edge2, q);
    if (t > 1e-6 && t < r.t) {
        hit.found = true;
        hit.t = t;
        hit.barycentric_coords = vec3(1.0 - u - v, u, v);
    }
    return hit;
}

// Minimal sphere hit (any/closest)
bool hit_sphere(vec3 O, vec3 D, vec3 C, float r2, float tMax, out float tHit) {
    vec3  oc = O - C;
    float b  = dot(oc, D);
    float c  = dot(oc, oc) - r2;
    float disc = b*b - c;
    if (disc <= 0.0) return false;
    float t = -b - sqrt(disc);
    if (t <= 0.0 || t >= tMax) return false;
    tHit = t; return true;
}

HitInfo intersect_sphere(inout Ray r, vec3 direction, vec3 center, float radius) {
    HitInfo hit;
    hit.found = false;

    vec3 oc = r.origin - center;
    float a = dot(direction, direction);
    float b = 2.0 * dot(oc, direction);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant < 0) {
        return hit;
    }

    float sqrt_d = sqrt(discriminant);
    float t1 = (-b - sqrt_d) / (2.0 * a);
    float t2 = (-b + sqrt_d) / (2.0 * a);
    float t = -1.0;

    if (t1 > 1e-6 && t1 < r.t) {
        t = t1;
    }
    if (t2 > 1e-6 && t2 < r.t && (t < 0.0 || t2 < t)) {
        t = t2;
    }
    if (t > 0.0) {
        hit.found = true;
        hit.t = t;
    }
    return hit;
}

float intersect_aabb(in Ray r, vec3 aabb_min, vec3 aabb_max) {
    vec3 t1 = (aabb_min - r.origin) * r.inv_direction;
    vec3 t2 = (aabb_max - r.origin) * r.inv_direction;

    vec3 tmin_v = min(t1, t2);
    vec3 tmax_v = max(t1, t2);

    float tmin = max(tmin_v.x, max(tmin_v.y, tmin_v.z));
    float tmax = min(tmax_v.x, min(tmax_v.y, tmax_v.z));

    if (tmax < 0.0 || tmin > tmax) {
        return 1.0/0.0;
    }
    if (tmin < 0.0) {
        return 0.0;
    }
    return tmin;
}

float compute_shadow(vec3 hit_pos, vec3 light_dir) {
    // Offset slightly along normal to avoid self-intersection
    vec3 shadow_origin = hit_pos + light_dir * 0.001;

    Ray shadow_ray;
    shadow_ray.origin = shadow_origin;
    shadow_ray.inv_direction = 1.0 / light_dir;
    shadow_ray.t = 1e10;  // Large distance (sun is infinitely far)

    HitInfo shadow_hit;
    traverse_tlas(shadow_ray, light_dir, shadow_hit);

    if (shadow_hit.found) {
        return shadow_intensity;  // In shadow
    }
    return 1.0;  // Fully lit
}

// General-purpose trace and shade function
vec3 trace(Ray r) {

    vec3 direction = 1.0 / r.inv_direction; // Reconstruct direction
    HitInfo closest_hit;
    traverse_tlas(r, direction, closest_hit);

    vec3 final_color;
    if (closest_hit.found) {
        InstanceInfo hit_inst = instances[closest_hit.instance_id];
        vec3 surface_color;

        if (closest_hit.is_point_hit) {
            // For point clouds, primitive_idx is the point index within the asset
            uint point_id = hit_inst.vertex_or_point_offset + closest_hit.primitive_idx;
            Point hit_point = getPoint(point_id);
            surface_color = pow(hit_point.color.rgb, vec3(2.2));

        } else {
            // For triangles primitive_idx is the triangle index within the asset
            uint blas_prim_id = closest_hit.primitive_idx;

            // Calculate the base index in the indices buffer for this triangle
            uint base_idx = hit_inst.index_offset + blas_prim_id * 3;
            uint i0 = indices[base_idx + 0];
            uint i1 = indices[base_idx + 1];
            uint i2 = indices[base_idx + 2];

            // Get the vertex data using the asset's vertex offset
            uint base_vtx = hit_inst.vertex_or_point_offset;

            // Interpolate UV coordinates using barycentric coordinates
            vec2 hit_uv = getUV(base_vtx + i0) * closest_hit.barycentric_coords.x +
                          getUV(base_vtx + i1) * closest_hit.barycentric_coords.y +
                          getUV(base_vtx + i2) * closest_hit.barycentric_coords.z;

            Material hit_mat = materials[hit_inst.material_id];

            if (hit_mat.texture_idx == 0xFFFFFFFFu) {
                // No texture: use base color
                surface_color = unpack_color(hit_mat.base_color).rgb;
            } else {
                // Texture: sample from it
                surface_color = texture(scene_textures, vec3(hit_uv, hit_mat.texture_idx)).rgb;
            }
        }

        // TESTING CRAPPY SHADOWS
        vec3 hit_pos = r.origin + direction * closest_hit.t;
        float shadow = compute_shadow(hit_pos, sun_direction);
        final_color = surface_color * shadow;

    } else {
        if (use_skybox) {
            final_color = texture(skybox, direction).rgb;
        } else {
            final_color = background_color;
        }
    }
    return final_color;
}

#endif // RAYTRACING_COMMON_GLSL