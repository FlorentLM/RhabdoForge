#ifndef RAYTRACING_COMMON_GLSL
#define RAYTRACING_COMMON_GLSL

#include "commons.glsl"
#include "pytinybvh_preamble.glsl"

// ============================================ Structs =============================================

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
    uint is_points;    // 0 = no (= mesh), 1 = points
    uint prim_index_offset;
    float radius_factor;
    uint pad0;
};  // Total size 160 bytes

// ================================== Textures (fixed bindings) =====================================

layout(binding = 0) uniform samplerCube skybox;
layout(binding = 1) uniform sampler2DArray scene_textures;

// =================================== Common uniforms (to all ray modes) ===========================

// Scene structure
uniform uint nb_tlas_nodes;

// Background / environment
uniform vec3 background_color;
uniform bool use_skybox;

// Path tracing configuration
uniform bool enable_path_tracing;
uniform int max_bounces;
uniform float sky_intensity;
uniform float sun_intensity;
uniform vec3 sun_direction;         // direction to the sun (normalised)
uniform float sun_angular_radius;   // angular size for soft shadows

// Legacy shadow mode (for simple non-path-tracing)
uniform bool enable_shadows;
uniform float shadow_intensity;

// ====================================== SSBO Bindings =============================================
// Bindings 0-1 are shader-specific (e.g. ommatidia input, ray output)
// Bindings 2-10 are scene data, bound by _bind_scene_ssbos()

// Node access (Standard layout)

#if TBVH_LAYOUT_STANDARD || TBVH_LAYOUT_BVH_GPU

layout(std430, binding = 6) readonly buffer AllBlasNodesBuffer { uint blas_nodes32[]; };
layout(std430, binding = 7) readonly buffer TlasNodesBuffer    { uint tlas_nodes32[]; };

const uint TBVH_WORDS_PER_NODE = TBVH_NODE_STRIDE_FLOATS;

struct StdNode {
    vec4 data1;
    vec4 data2;
};

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

struct BvhNode {
    vec4 data1;
    vec4 data2;
};

layout(std430, binding = 6) readonly buffer AllBlasNodesBuffer { BvhNode blas_nodes[]; };
layout(std430, binding = 7) readonly buffer TlasNodesBuffer    { BvhNode tlas_nodes[]; };

#endif

// Geometry and materials
layout(std430, binding = 2) readonly buffer VertexBuffer { float v[]; };
layout(std430, binding = 3) readonly buffer IndexBuffer { uint indices[]; };
layout(std430, binding = 4) readonly buffer MaterialBuffer { Material materials[]; };
layout(std430, binding = 5) readonly buffer PointsBuffer { float points_data[]; };

layout(row_major, std430, binding = 8) readonly buffer InstancesBuffer { InstanceInfo instances[]; };
layout(std430, binding = 9) readonly buffer TlasPrimIndexBuffer { uint tlas_prim_indices[]; };
layout(std430, binding = 10) readonly buffer BlasPrimIndexBuffer { uint blas_prim_indices[]; };

// ================================= Forward declarations ==========================================

float intersect_aabb(in Ray r, vec3 aabb_min, vec3 aabb_max);
HitInfo intersect_triangle(inout Ray r, vec3 direction, vec3 v0, vec3 v1, vec3 v2);
HitInfo intersect_sphere(inout Ray r, vec3 direction, vec3 center, float radius);
void traverse_tlas(inout Ray r_world, vec3 dir_world, out HitInfo closest_hit);
void traverse_blas(inout Ray r_obj, vec3 dir_obj, out HitInfo blas_hit, InstanceInfo inst);
bool is_occluded(Ray r_world);

// ==================================== Data accessors =============================================

vec3 getPos(uint i){ uint b = i*5u; return vec3(v[b], v[b+1], v[b+2]); }
vec2 getUV (uint i){ uint b = i*5u; return vec2(v[b+3], v[b+4]); }

Point getPoint(uint point_idx) {
    uint base_offset = point_idx * 12u;
    Point p;
    p.position = vec3(points_data[base_offset + 0], points_data[base_offset + 1], points_data[base_offset + 2]);
    p.radius = points_data[base_offset + 3];
    p.normal = vec3(points_data[base_offset + 4], points_data[base_offset + 5], points_data[base_offset + 6]);
    p.color = vec3(points_data[base_offset + 7], points_data[base_offset + 8], points_data[base_offset + 9]);
    return p;
}

void fast_getPoint(uint point_idx, out vec3 pos, out float radius) {
    uint base_offset = point_idx * 12u;
    pos.x = points_data[base_offset + 0u];
    pos.y = points_data[base_offset + 1u];
    pos.z = points_data[base_offset + 2u];
    radius = points_data[base_offset + 3u];
}

// PCG-style hash for RNG
uint pcg_hash(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float random_float(inout uint rng_state) {
    rng_state = pcg_hash(rng_state);
    return float(rng_state) / 4294967295.0;
}

// Cosine-weighted hemisphere sampling (importance sampling for diffuse)
// Returns direction in tangent space (z-up)
vec3 cosine_sample_hemisphere(float r1, float r2) {
    float phi = TWOPI * r1;
    float cos_theta = sqrt(r2);
    float sin_theta = sqrt(1.0 - r2);

    return vec3(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta
    );
}

// Orthonormal basis from normal
void build_basis(vec3 N, out vec3 T, out vec3 B) {
    vec3 up = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    T = normalize(cross(up, N));
    B = cross(N, T);
}

// Transform direction from tangent space to world space
vec3 tangent_to_world(vec3 dir, vec3 N, vec3 T, vec3 B) {
    return dir.x * T + dir.y * B + dir.z * N;
}

// Sample a direction toward an area light (sun disk)
vec3 sample_sun_direction(vec3 sun_dir, float angular_radius, float r1, float r2) {
    if (angular_radius <= 0.0) {
        return sun_dir;
    }

    vec3 T, B;
    build_basis(sun_dir, T, B);

    float phi = TWOPI * r1;
    float r = angular_radius * sqrt(r2);  // sqrt for uniform disk

    return normalize(sun_dir + T * cos(phi) * r + B * sin(phi) * r);
}

// Get sky color for a direction (environment lighting)
vec3 get_sky_color(vec3 direction) {
    if (use_skybox) {
        return texture(skybox, direction).rgb * sky_intensity;
    } else {
        return background_color * sky_intensity;
    }
}

// ================================= BVH traversal ======================================

void traverse_tlas(inout Ray r_world, vec3 dir_world, out HitInfo closest_hit) {
    closest_hit.found = false;
    if (nb_tlas_nodes == 0u) return;

    uint stack[64];
    uint stack_ptr = 0;
    stack[stack_ptr++] = 0u;

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

        if (prim_count > 0u) {
            for (uint j = 0u; j < prim_count; ++j) {
                uint instance_id = tlas_prim_indices[first_idx + j];
                InstanceInfo inst = instances[instance_id];

                Ray r_obj;
                r_obj.origin = (inst.inverse_transform * vec4(r_world.origin, 1.0)).xyz;
                vec3 dir_obj = (inst.inverse_transform * vec4(dir_world, 0.0)).xyz;
                r_obj.inv_direction = 1.0 / dir_obj;
                r_obj.t = 1.0/0.0;

                HitInfo blas_hit;
                traverse_blas(r_obj, dir_obj, blas_hit, inst);

                if (blas_hit.found) {
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
        } else {
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

        if (prim_count > 0u) {
            uint prim_base = inst.prim_index_offset + first_idx;

            for (uint i = 0; i < prim_count; ++i) {
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
                        blas_hit.primitive_idx = blas_prim_id;
                        blas_hit.t = p_hit.t;
                        r_obj.t = p_hit.t;
                    }
                } else {
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
                        blas_hit.primitive_idx = blas_prim_id;
                        blas_hit.barycentric_coords = tri_hit.barycentric_coords;
                        blas_hit.t = tri_hit.t;
                        r_obj.t = tri_hit.t;
                    }
                }
            }
        } else {
            uint left_idx = first_idx;
            uint right_idx = left_idx + 1;

            #if TBVH_LAYOUT_STANDARD
                StdNode leftNode = load_blas_node(inst.blas_node_offset, left_idx);
                StdNode rightNode = load_blas_node(inst.blas_node_offset, right_idx);
                float d1 = intersect_aabb(r_obj, leftNode.data1.xyz, leftNode.data2.xyz);
                float d2 = intersect_aabb(r_obj, rightNode.data1.xyz, rightNode.data2.xyz);
            #else
                float d1 = intersect_aabb(r_obj, blas_nodes[inst.blas_node_offset + left_idx].data1.xyz,  blas_nodes[inst.blas_node_offset + left_idx].data2.xyz);
                float d2 = intersect_aabb(r_obj, blas_nodes[inst.blas_node_offset + right_idx].data1.xyz, blas_nodes[inst.blas_node_offset + right_idx].data2.xyz);
            #endif

            if (d1 > d2) {
                float temp_d = d1; d1 = d2; d2 = temp_d;
                uint temp_i = left_idx; left_idx = right_idx; right_idx = temp_i;
            }
            if (d2 < r_obj.t && stack_ptr < 64) stack[stack_ptr++] = right_idx;
            if (d1 < r_obj.t && stack_ptr < 64) stack[stack_ptr++] = left_idx;
        }
    }
}

// ================================ Intersections ====================================

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
    float vv = f * dot(direction, q);

    if (vv < 0.0 || u + vv > 1.0) {
        return hit;
    }

    float t = f * dot(edge2, q);

    if (t > 0.0 && t < r.t) {
        hit.found = true;
        hit.t = t;
        float w = 1.0 - u - vv;
        hit.barycentric_coords = vec3(w, u, vv);
        r.t = t;
    }

    return hit;
}

HitInfo intersect_sphere(inout Ray r, vec3 direction, vec3 center, float radius) {
    HitInfo hit;
    hit.found = false;

    vec3 oc = r.origin - center;
    float a = dot(direction, direction);
    float half_b = dot(oc, direction);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = half_b * half_b - a * c;

    if (discriminant < 0.0) {
        return hit;
    }

    float sqrt_d = sqrt(discriminant);
    float t = (-half_b - sqrt_d) / a;

    if (t < 0.0) {
        t = (-half_b + sqrt_d) / a;
    }

    if (t > 0.0 && t < r.t) {
        hit.found = true;
        hit.t = t;
        r.t = t;
    }

    return hit;
}

float intersect_aabb(in Ray r, vec3 aabb_min, vec3 aabb_max) {
    vec3 t0 = (aabb_min - r.origin) * r.inv_direction;
    vec3 t1 = (aabb_max - r.origin) * r.inv_direction;
    vec3 tmin_v = min(t0, t1);
    vec3 tmax_v = max(t0, t1);
    float tmin = max(max(tmin_v.x, tmin_v.y), tmin_v.z);
    float tmax = min(min(tmax_v.x, tmax_v.y), tmax_v.z);
    return (tmax >= max(tmin, 0.0)) ? tmin : 1e30;
}


// Occlusion test for shadow rays
bool is_occluded(Ray r_world) {
    if (nb_tlas_nodes == 0u) return false;

    vec3 dir_world = 1.0 / r_world.inv_direction;

    uint stack[64];
    uint stack_ptr = 0;
    stack[stack_ptr++] = 0u;

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

        if (prim_count > 0u) {
            for (uint j = 0u; j < prim_count; ++j) {
                uint instance_id = tlas_prim_indices[first_idx + j];
                InstanceInfo inst = instances[instance_id];

                Ray r_obj;
                r_obj.origin = (inst.inverse_transform * vec4(r_world.origin, 1.0)).xyz;
                vec3 dir_obj = (inst.inverse_transform * vec4(dir_world, 0.0)).xyz;
                r_obj.inv_direction = 1.0 / dir_obj;
                r_obj.t = 1e10;

                HitInfo blas_hit;
                traverse_blas(r_obj, dir_obj, blas_hit, inst);
                if (blas_hit.found) return true;
            }
        } else {
            uint left_idx = first_idx;
            uint right_idx = first_idx + 1;
            #if TBVH_LAYOUT_STANDARD
                StdNode leftNode = load_tlas_node(left_idx);
                StdNode rightNode = load_tlas_node(right_idx);
                float d1 = intersect_aabb(r_world, leftNode.data1.xyz, leftNode.data2.xyz);
                float d2 = intersect_aabb(r_world, rightNode.data1.xyz, rightNode.data2.xyz);
            #else
                float d1 = intersect_aabb(r_world, tlas_nodes[left_idx].data1.xyz, tlas_nodes[left_idx].data2.xyz);
                float d2 = intersect_aabb(r_world, tlas_nodes[right_idx].data1.xyz, tlas_nodes[right_idx].data2.xyz);
            #endif
            if (d1 > d2) { float td=d1; d1=d2; d2=td; uint ti=left_idx; left_idx=right_idx; right_idx=ti; }
            if (d2 < r_world.t && stack_ptr < 64) stack[stack_ptr++] = right_idx;
            if (d1 < r_world.t && stack_ptr < 64) stack[stack_ptr++] = left_idx;
        }
    }
    return false;
}

// ==================================== Surface properties =========================================

// Get surface color at hit point
vec3 get_surface_color(HitInfo hit) {
    InstanceInfo hit_inst = instances[hit.instance_id];

    if (hit.is_point_hit) {
        uint point_id = hit_inst.vertex_or_point_offset + hit.primitive_idx;
        Point hit_point = getPoint(point_id);
        return pow(hit_point.color.rgb, vec3(2.2));  // sRGB to linear
    } else {
        uint blas_prim_id = hit.primitive_idx;
        uint base_idx = hit_inst.index_offset + blas_prim_id * 3;
        uint i0 = indices[base_idx + 0];
        uint i1 = indices[base_idx + 1];
        uint i2 = indices[base_idx + 2];

        uint base_vtx = hit_inst.vertex_or_point_offset;

        vec2 hit_uv = getUV(base_vtx + i0) * hit.barycentric_coords.x +
                      getUV(base_vtx + i1) * hit.barycentric_coords.y +
                      getUV(base_vtx + i2) * hit.barycentric_coords.z;

        Material hit_mat = materials[hit_inst.material_id];

        if (hit_mat.texture_idx == 0xFFFFFFFFu) {
            return unpack_color(hit_mat.base_color).rgb;
        } else {
            return texture(scene_textures, vec3(hit_uv, hit_mat.texture_idx)).rgb;
        }
    }
}

// Get surface normal at hit point (world space)
vec3 get_surface_normal(HitInfo hit, vec3 ray_dir) {
    InstanceInfo hit_inst = instances[hit.instance_id];
    vec3 normal_obj;

    if (hit.is_point_hit) {
        uint point_id = hit_inst.vertex_or_point_offset + hit.primitive_idx;
        Point p = getPoint(point_id);
        normal_obj = p.normal;
        if (length(normal_obj) < 0.001) {
            normal_obj = vec3(0.0, 1.0, 0.0);
        }
    } else {
        uint blas_prim_id = hit.primitive_idx;
        uint base_idx = hit_inst.index_offset + blas_prim_id * 3;
        uint i0 = indices[base_idx + 0];
        uint i1 = indices[base_idx + 1];
        uint i2 = indices[base_idx + 2];

        uint base_vtx = hit_inst.vertex_or_point_offset;
        vec3 v0 = getPos(base_vtx + i0);
        vec3 v1 = getPos(base_vtx + i1);
        vec3 v2 = getPos(base_vtx + i2);

        normal_obj = normalize(cross(v1 - v0, v2 - v0));
    }

    // Transform normal to world space
    mat3 normal_matrix = mat3(hit_inst.transform);
    vec3 normal_world = normalize(normal_matrix * normal_obj);

    // Make sure normal faces ray
    if (dot(normal_world, ray_dir) > 0.0) {
        normal_world = -normal_world;
    }

    return normal_world;
}

// ===================================== Trace ===========================================


// Simple trace (direct lighting only)
float compute_shadow(vec3 hit_pos, vec3 light_dir) {
    vec3 shadow_origin = hit_pos + light_dir * 0.001;

    Ray shadow_ray;
    shadow_ray.origin = shadow_origin;
    shadow_ray.inv_direction = 1.0 / light_dir;
    shadow_ray.t = 1e10;

    if (is_occluded(shadow_ray)) {
        return shadow_intensity;
    }
    return 1.0;
}

vec3 trace_simple(Ray r) {
    vec3 direction = 1.0 / r.inv_direction;
    HitInfo closest_hit;
    traverse_tlas(r, direction, closest_hit);

    vec3 final_color;
    if (closest_hit.found) {
        vec3 surface_color = get_surface_color(closest_hit);

        float shadow = 1.0;
        if (enable_shadows) {
            vec3 hit_pos = r.origin + direction * closest_hit.t;
            shadow = compute_shadow(hit_pos, sun_direction);
        }
        final_color = surface_color * shadow;
    } else {
        final_color = get_sky_color(direction);
    }
    return final_color;
}

// Path trace
vec3 trace_path(Ray r, inout uint rng_state) {
    vec3 throughput = vec3(1.0);
    vec3 radiance = vec3(0.0);

    vec3 direction = 1.0 / r.inv_direction;

    for (int bounce = 0; bounce <= max_bounces; bounce++) {
        HitInfo hit;
        traverse_tlas(r, direction, hit);

        if (!hit.found) {
            radiance += throughput * get_sky_color(direction);
            break;
        }

        vec3 hit_pos = r.origin + direction * hit.t;
        vec3 surface_color = get_surface_color(hit);
        vec3 normal = get_surface_normal(hit, direction);

        // Direct lighting (Next Event Estimation)
        if (sun_intensity > 0.0) {
            float r1 = random_float(rng_state);
            float r2 = random_float(rng_state);
            vec3 light_dir = sample_sun_direction(sun_direction, sun_angular_radius, r1, r2);

            float NdotL = max(dot(normal, light_dir), 0.0);

            if (NdotL > 0.0) {
                Ray shadow_ray;
                shadow_ray.origin = hit_pos + normal * 0.001;
                shadow_ray.inv_direction = 1.0 / light_dir;
                shadow_ray.t = 1e10;

                if (!is_occluded(shadow_ray)) {
                    radiance += throughput * surface_color * sun_intensity * NdotL;
                }
            }
        }

        // Russian roulette
        if (bounce >= 3) {
            float survival_prob = min(max(throughput.r, max(throughput.g, throughput.b)), 0.95);
            if (random_float(rng_state) > survival_prob) {
                break;
            }
            throughput /= survival_prob;
        }

        // Sample next bounce direction (cosine-weighted for diffuse)
        float r1 = random_float(rng_state);
        float r2 = random_float(rng_state);

        vec3 T, B;
        build_basis(normal, T, B);
        vec3 local_dir = cosine_sample_hemisphere(r1, r2);
        vec3 new_direction = tangent_to_world(local_dir, normal, T, B);

        throughput *= surface_color;

        r.origin = hit_pos + normal * 0.001;
        direction = new_direction;
        r.inv_direction = 1.0 / direction;
        r.t = 1e10;
    }

    return radiance;
}

// Main entry func
vec3 trace(Ray r, inout uint rng_state) {
    if (enable_path_tracing) {
        return trace_path(r, rng_state);
    } else {
        return trace_simple(r);
    }
}

#endif // RAYTRACING_COMMON_GLSL
