#ifndef BVH_GLSL
#define BVH_GLSL

// BVH traversal and scene geometry access

#include "commons.glsl"
#include "pytinybvhPreamble.glsl"

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
};  // 160 bytes

// ================================== Textures (fixed bindings) =====================================

layout(binding = 0) uniform samplerCube skybox;
layout(binding = 1) uniform sampler2DArray scene_textures;

// ====================================== Scene uniforms ============================================

uniform uint nb_tlas_nodes;

// ====================================== SSBO Bindings =============================================
// Bindings 0-4 are shader-specific (e.g. ommatidia input, ray output)
// Bindings 5+ are scene geometry data

// Node access (Standard layout)

#if TBVH_LAYOUT_STANDARD || TBVH_LAYOUT_BVH_GPU

layout(std430, binding = 9) readonly buffer AllBlasNodesBuffer { uint blas_nodes32[]; };
layout(std430, binding = 10) readonly buffer TlasNodesBuffer    { uint tlas_nodes32[]; };

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

layout(std430, binding = 9) readonly buffer AllBlasNodesBuffer { BvhNode blas_nodes[]; };
layout(std430, binding = 10) readonly buffer TlasNodesBuffer    { BvhNode tlas_nodes[]; };

#endif

// Geometry and materials
layout(std430, binding = 5) readonly buffer VertexBuffer { float v[]; };
layout(std430, binding = 6) readonly buffer IndexBuffer { uint indices[]; };
layout(std430, binding = 7) readonly buffer MaterialBuffer { Material materials[]; };
layout(std430, binding = 8) readonly buffer PointsBuffer { float points_data[]; };

layout(row_major, std430, binding = 11) readonly buffer InstancesBuffer { InstanceInfo instances[]; };
layout(std430, binding = 12) readonly buffer TlasPrimIndexBuffer { uint tlas_prim_indices[]; };
layout(std430, binding = 13) readonly buffer BlasPrimIndexBuffer { uint blas_prim_indices[]; };

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

// ================================ Primitive intersection functions ================================

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
    return (tmax >= max(tmin, 0.0)) ? tmin : 1.0/0.0;
}

// ================================= Forward declarations ==========================================

void traverse_blas(inout Ray r_obj, vec3 dir_obj, out HitInfo blas_hit, InstanceInfo inst);

// ================================= Traversal functions ===========================================

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

// ================================= Shadow/occlusion tests =========================================

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

#endif // BVH_GLSL
