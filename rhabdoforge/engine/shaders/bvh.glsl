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
    uint is_srgb;
};  // 160 bytes

// ================================== Textures (fixed bindings) =====================================

layout(binding = 0) uniform sampler2D sky_texture;

// Material images in size buckets
layout(binding = 1) uniform sampler2DArray scene_textures_0;
layout(binding = 2) uniform sampler2DArray scene_textures_1;
layout(binding = 3) uniform sampler2DArray scene_textures_2;

vec4 sample_material(Material mat, vec2 uv) {
    uv *= mat.uv_scale;  // map mesh UV onto the texture's used sub-rect of its (square) tier canvas
    if (mat.texture_tier == 0u) return texture(scene_textures_0, vec3(uv, mat.texture_idx));
    if (mat.texture_tier == 1u) return texture(scene_textures_1, vec3(uv, mat.texture_idx));
    return texture(scene_textures_2, vec3(uv, mat.texture_idx));
}

uniform float pixel_angular_size;   // for the LOD tests

vec4 sample_material_lod(Material mat, vec2 uv, float lod) {
    uv *= mat.uv_scale;
    if (mat.texture_tier == 0u) return textureLod(scene_textures_0, vec3(uv, mat.texture_idx), lod);
    if (mat.texture_tier == 1u) return textureLod(scene_textures_1, vec3(uv, mat.texture_idx), lod);
    return textureLod(scene_textures_2, vec3(uv, mat.texture_idx), lod);
}

// Cheap LOD estimate from the hit triangle's world-vs-UV area ratio (texel density)
// TODO: This will do for now... but micromaps and bit-lookup in a SSBO would completely avoid the texture lookup
float estimate_lod(Material mat, float world_area, float uv_area, float hit_dist) {
    if (uv_area <= 0.0 || world_area <= 0.0) return 0.0;

    ivec3 dims = (mat.texture_tier == 0u) ? textureSize(scene_textures_0, 0)
               : (mat.texture_tier == 1u) ? textureSize(scene_textures_1, 0)
               : textureSize(scene_textures_2, 0);

    float texel_world_size = sqrt(world_area / uv_area) / float(dims.x);
    float footprint_world = hit_dist * pixel_angular_size;

    return max(0.0, log2(max(footprint_world / texel_world_size, 1.0)));
}

// ========================================== Consts ================================================

const uint MAX_ALPHA_DISCARDS = 32u;  // cap on transparent hits per traversal

// ====================================== Scene uniforms ============================================

uniform uint nb_tlas_nodes;

// ====================================== SSBO Bindings =============================================

// Node access (BVH standard layout)

#if TBVH_LAYOUT_STANDARD || TBVH_LAYOUT_BVH_GPU

layout(std430, binding = BINDING_BLAS_NODES) readonly buffer AllBlasNodesBuffer { uint blas_nodes32[]; };
layout(std430, binding = BINDING_TLAS_NODES) readonly buffer TlasNodesBuffer    { uint tlas_nodes32[]; };

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

layout(std430, binding = BINDING_BLAS_NODES) readonly buffer AllBlasNodesBuffer { BvhNode blas_nodes[]; };
layout(std430, binding = BINDING_TLAS_NODES) readonly buffer TlasNodesBuffer    { BvhNode tlas_nodes[]; };

#endif

// Geometry and materials
layout(std430, binding = BINDING_VERTS) readonly buffer VertexBuffer { float v[]; };
layout(std430, binding = BINDING_INDICES) readonly buffer IndexBuffer { uint indices[]; };
layout(std430, binding = BINDING_INST_VISIBLE) readonly buffer InstanceVisibleBuffer { uint inst_visible[]; };
layout(std430, binding = BINDING_MATERIALS) readonly buffer MaterialBuffer { Material materials[]; };
layout(std430, binding = BINDING_POINTS) readonly buffer PointsBuffer { float points_data[]; };

layout(row_major, std430, binding = BINDING_INST_INFO) readonly buffer InstancesBuffer { InstanceInfo instances[]; };
layout(std430, binding = BINDING_TLAS_INDICES) readonly buffer TlasPrimIndexBuffer { uint tlas_prim_indices[]; };
layout(std430, binding = BINDING_BLAS_INDICES) readonly buffer BlasPrimIndexBuffer { uint blas_prim_indices[]; };

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

// ==================================== Skybox equirect sampler =========================================

vec3 sample_env(vec3 dir) {

    const float MAX_ENV = 1.0e4;

    vec2 uv = vec2(atan(dir.x, dir.z) / TWOPI + 0.5, 0.5 - asin(clamp(dir.y, -1.0, 1.0)) / PI);
    return min(texture(sky_texture, uv).rgb, vec3(MAX_ENV));
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

// Alpha-cutout test for MASK/BLEND materials. No true blending support (yet?) so both use a hard cutoff
// true if the ray should pass through the triangle (= no hit)
bool alpha_discard(uint material_id, uint base_vtx, uint i0, uint i1, uint i2, vec3 bary) {
    Material mat = materials[material_id];
    if (mat.alpha_cutoff <= 0.0) return false;

    float a;
    if (mat.texture_idx == 0xFFFFFFFFu) {
        a = unpack_color(mat.base_color).a;
    } else {
        vec2 uv = getUV(base_vtx + i0) * bary.x + getUV(base_vtx + i1) * bary.y + getUV(base_vtx + i2) * bary.z;
        a = sample_material(mat, uv).a;
    }
    return a < mat.alpha_cutoff;
}

// ================================= Forward declarations ==========================================

void traverse_blas(inout Ray r_obj, vec3 dir_obj, out HitInfo blas_hit, InstanceInfo inst, bool any_hit);

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
                if (inst_visible[instance_id] == 0u) continue;
                InstanceInfo inst = instances[instance_id];

                Ray r_obj;
                r_obj.origin = (inst.inverse_transform * vec4(r_world.origin, 1.0)).xyz;
                vec3 dir_obj = (inst.inverse_transform * vec4(dir_world, 0.0)).xyz;
                r_obj.inv_direction = 1.0 / dir_obj;
                r_obj.t = 1.0/0.0;

                HitInfo blas_hit;
                traverse_blas(r_obj, dir_obj, blas_hit, inst, false); // Closest-hit

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

void traverse_blas(inout Ray r_obj, vec3 dir_obj, out HitInfo blas_hit, InstanceInfo inst, bool any_hit) {
    blas_hit.found = false;
    uint discards_left = MAX_ALPHA_DISCARDS;

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

                    float prev_t = r_obj.t;
                    HitInfo tri_hit = intersect_triangle(r_obj, dir_obj, v0, v1, v2);

                    if (tri_hit.found && discards_left > 0u &&
                        alpha_discard(inst.material_id, base_vtx, i0, i1, i2, tri_hit.barycentric_coords)) {
                        discards_left--;
                        r_obj.t = prev_t;  // ray keeps going past this (transparent) texel
                        continue;
                    }

                    if (tri_hit.found) {
                        blas_hit.found = true;
                        blas_hit.is_point_hit = false;
                        blas_hit.primitive_idx = blas_prim_id;
                        blas_hit.barycentric_coords = tri_hit.barycentric_coords;
                        blas_hit.t = tri_hit.t;
                        r_obj.t = tri_hit.t;

                        if (any_hit) return;
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
                if (inst_visible[instance_id] == 0u) continue;
                InstanceInfo inst = instances[instance_id];

                Ray r_obj;
                r_obj.origin = (inst.inverse_transform * vec4(r_world.origin, 1.0)).xyz;
                vec3 dir_obj = (inst.inverse_transform * vec4(dir_world, 0.0)).xyz;
                r_obj.inv_direction = 1.0 / dir_obj;
                r_obj.t = 1e10;

                HitInfo blas_hit;
                traverse_blas(r_obj, dir_obj, blas_hit, inst, true); // any hit
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
