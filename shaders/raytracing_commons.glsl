#ifndef RAYTRACING_COMMON_GLSL
#define RAYTRACING_COMMON_GLSL

#include "commons.glsl"

// A struct to hold all information about an instance
struct InstanceInfo {
    mat4 transform;
    mat4 inverse_transform;
    uint blas_node_offset;
    uint primitive_offset;
    uint material_id;
    uint is_point_cloud;
};

// Bindings (To be used by including shaders)
layout(std430, binding = 1) readonly buffer TriangleBuffer { Triangle triangles[]; };
layout(std430, binding = 2) readonly buffer MaterialBuffer { Material materials[]; };
layout(std430, binding = 3) readonly buffer PrimitiveBuffer { Point points[]; };
layout(std430, binding = 5) readonly buffer AllBlasNodesBuffer { BvhNode blas_nodes[]; };
layout(std430, binding = 6) readonly buffer TlasNodesBuffer { BvhNode tlas_nodes[]; };
layout(row_major, std430, binding = 7) readonly buffer InstancesBuffer { InstanceInfo instances[]; }; // row major!!!!
layout(std430, binding = 8) readonly buffer TlasPrimIndexBuffer { uint tlas_prim_indices[]; };

layout(binding = 0) uniform samplerCube skybox;
layout(binding = 1) uniform sampler2DArray scene_textures;
uniform uint nb_tlas_nodes;
uniform float point_radius;

// Forward declarations

float intersect_aabb(in Ray r, vec3 aabb_min, vec3 aabb_max);
HitInfo intersect_triangle(inout Ray r, vec3 direction, Triangle tri);
HitInfo intersect_sphere(inout Ray r, vec3 direction, vec3 center, float radius);
void traverse_blas(inout Ray r_obj, vec3 dir_obj, out HitInfo blas_hit, InstanceInfo inst);
void find_closest_hit(inout Ray r_world, vec3 dir_world, out HitInfo closest_hit);

// Traversal implementation

void find_closest_hit(inout Ray r_world, vec3 dir_world, out HitInfo closest_hit) {
    closest_hit.found = false;
    if (nb_tlas_nodes == 0u) return;

    uint stack[64];
    uint stack_ptr = 0;
    stack[stack_ptr++] = 0u;

    while (stack_ptr > 0u) {
        uint node_idx = stack[--stack_ptr];
        BvhNode node = tlas_nodes[node_idx];

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
                        closest_hit.found = true;
                        closest_hit.t = new_world_t;
                        closest_hit.barycentric_coords = blas_hit.barycentric_coords;
                        // blas_hit.primitive_idx is the index into the global primitive buffer (triangles or points)
                        closest_hit.primitive_idx = blas_hit.primitive_idx;
                        closest_hit.is_point_hit = blas_hit.is_point_hit;
                        closest_hit.instance_id = instance_id;
                    }
                }
            }
        } else { // TLAS internal node
            uint left_idx = first_idx;
            uint right_idx = first_idx + 1;
            float d1 = intersect_aabb(r_world, tlas_nodes[left_idx].data1.xyz, tlas_nodes[left_idx].data2.xyz);
            float d2 = intersect_aabb(r_world, tlas_nodes[right_idx].data1.xyz, tlas_nodes[right_idx].data2.xyz);

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
    stack[stack_ptr++] = inst.blas_node_offset;

    while (stack_ptr > 0u) {
        uint node_idx = stack[--stack_ptr];
        BvhNode node = blas_nodes[node_idx];

        if (intersect_aabb(r_obj, node.data1.xyz, node.data2.xyz) >= r_obj.t) continue;

        uint prim_count = floatBitsToUint(node.data2.w);
        uint first_idx = floatBitsToUint(node.data1.w);

        if (prim_count > 0u) { // BLAS leaf node
            for (uint i = 0; i < prim_count; ++i) {
                // Calculate the correct primitive index:
                // first_idx is relative to the BLAS so we need to add the primitive_offset
                uint prim_idx = inst.primitive_offset + first_idx + i;

                if (inst.is_point_cloud == 1u) {
                    HitInfo p_hit = intersect_sphere(r_obj, dir_obj, points[prim_idx].pos.xyz, point_radius);
                    if (p_hit.found) {
                        blas_hit.found = true;
                        blas_hit.is_point_hit = true;
                        blas_hit.primitive_idx = prim_idx;  // this is the global index
                        blas_hit.t = p_hit.t;
                        r_obj.t = p_hit.t;
                    }
                } else {
                    HitInfo tri_hit = intersect_triangle(r_obj, dir_obj, triangles[prim_idx]);
                    if (tri_hit.found) {
                        blas_hit.found = true;
                        blas_hit.is_point_hit = false;
                        blas_hit.primitive_idx = prim_idx;  // this is the global index
                        blas_hit.barycentric_coords = tri_hit.barycentric_coords;
                        blas_hit.t = tri_hit.t;
                        r_obj.t = tri_hit.t;
                    }
                }
            }
        } else { // BLAS internal node
            uint left_idx = inst.blas_node_offset + first_idx;
            uint right_idx = left_idx + 1;

            float d1 = intersect_aabb(r_obj, blas_nodes[left_idx].data1.xyz, blas_nodes[left_idx].data2.xyz);
            float d2 = intersect_aabb(r_obj, blas_nodes[right_idx].data1.xyz, blas_nodes[right_idx].data2.xyz);

            if (d1 > d2) { float temp_d = d1; d1 = d2; d2 = temp_d; uint temp_i = left_idx; left_idx = right_idx; right_idx = temp_i; }
            if (d2 < r_obj.t && stack_ptr < 64) stack[stack_ptr++] = right_idx;
            if (d1 < r_obj.t && stack_ptr < 64) stack[stack_ptr++] = left_idx;
        }
    }
}

// Intersection implementation

HitInfo intersect_triangle(inout Ray r, vec3 direction, Triangle tri) {
    HitInfo hit;
    hit.found = false;

    vec3 edge1 = tri.v1.xyz - tri.v0.xyz;
    vec3 edge2 = tri.v2.xyz - tri.v0.xyz;

    vec3 h = cross(direction, edge2);
    float a = dot(edge1, h);

    if (a > -1e-6 && a < 1e-6) {
        return hit;
    }

    float f = 1.0 / a;
    vec3 s = r.origin - tri.v0.xyz;
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


// General-purpose trace and shade function
vec3 trace(Ray r) {

    vec3 direction = 1.0 / r.inv_direction; // Reconstruct direction
    HitInfo closest_hit;
    find_closest_hit(r, direction, closest_hit);

    vec3 final_color;
    if (closest_hit.found) {
        InstanceInfo hit_inst = instances[closest_hit.instance_id];
        if (closest_hit.is_point_hit) {
            Point hit_point = points[closest_hit.primitive_idx];
            final_color = hit_point.color.rgb;
        } else {
            Triangle hit_tri = triangles[closest_hit.primitive_idx];
            Material hit_mat = materials[hit_tri.material_idx];
            vec2 hit_uv = hit_tri.uv0 * closest_hit.barycentric_coords.x +
                          hit_tri.uv1 * closest_hit.barycentric_coords.y +
                          hit_tri.uv2 * closest_hit.barycentric_coords.z;
            final_color = texture(scene_textures, vec3(hit_uv, hit_mat.texture_idx)).rgb;
        }
    } else {
        final_color = texture(skybox, direction).rgb;
    }
    return final_color;
}

#endif // RAYTRACING_COMMON_GLSL