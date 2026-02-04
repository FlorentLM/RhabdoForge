#ifndef LIGHTING_GLSL
#define LIGHTING_GLSL

// Lighting system with conditional multi-light support
//#define MULTI_LIGHT   // TODO: Inject this def with the python-side preprocessor

#include "bvh.glsl"

// ============================================ Light Structs =======================================

const int MAX_LIGHTS_PER_PASS = 16;

struct PointLightData {
    vec3 position;
    float radius;           // physical radius for soft shadows
    vec3 color;
    float intensity;
    float constant_atten;   // attenuation factors
    float linear_atten;
    float quadratic_atten;
    uint cast_shadows;      // 0 = no shadow rays, 1 = cast shadows
}; // 48 bytes

struct AreaLightData {
    vec3 position;
    float width;
    vec3 normal;
    float height;
    vec3 tangent;
    float intensity;
    vec3 bitangent;
    uint cast_shadows;      // 0 = no shadow rays, 1 = cast shadows
    vec3 color;
    uint two_sided;         // 0 = one-sided, 1 = two-sided
}; // 64 bytes

// =================================== Lighting uniforms ============================================

// Background / environment
uniform vec3 background_color;
uniform bool use_skybox;
uniform float sky_intensity;        // intensity for sky color and sun disk

// Global lighting controls
uniform bool enable_ambient;        // toggle ambient/fill lighting from sky
uniform bool enable_direct;         // toggle all direct lighting
uniform bool enable_shadows;        // global shadow override (false = skip all shadow rays)
uniform float ambient_intensity;    // multiplier for ambient (independent of sky_intensity)

// Primary light
uniform bool primary_light_enabled;
uniform int primary_light_type;     // 0 = directional, 1 = point, 2 = area
uniform vec3 primary_light_dir;     // direction *to* the light (for directional)
uniform vec3 primary_light_pos;     // position (for point/area)
uniform vec3 primary_light_color;
uniform float primary_light_intensity;
uniform float primary_light_radius; // angular radius (directional) or physical radius (point)
uniform bool primary_light_shadows;

// Additional light counts (only used when MULTI_LIGHT is defined)
uniform int point_lights_count;
uniform int area_lights_count;

// ====================================== Light SSBOs ===============================================

#ifdef MULTI_LIGHT
layout(std430, binding = 11) readonly buffer PointLightsBuffer { PointLightData point_lights[]; };
layout(std430, binding = 12) readonly buffer AreaLightsBuffer { AreaLightData area_lights[]; };
#endif

// ==================================== RNG ========================================================

uint pcg_hash(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float random_float(inout uint rng_state) {
    rng_state = pcg_hash(rng_state);
    return float(rng_state) / 4294967295.0;
}

// ==================================== Sampling ===================================================

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

void build_basis(vec3 N, out vec3 T, out vec3 B) {
    vec3 up = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    T = normalize(cross(up, N));
    B = cross(N, T);
}

vec3 tangent_to_world(vec3 dir, vec3 N, vec3 T, vec3 B) {
    return dir.x * T + dir.y * B + dir.z * N;
}

// Sample direction toward a disk (for soft shadows)
vec3 sample_disk_direction(vec3 center_dir, float angular_radius, float r1, float r2) {
    if (angular_radius <= 0.0) {
        return center_dir;
    }

    vec3 T, B;
    build_basis(center_dir, T, B);

    float phi = TWOPI * r1;
    float r = angular_radius * sqrt(r2);

    return normalize(center_dir + T * cos(phi) * r + B * sin(phi) * r);
}

// ======================================= Light sampling ==========================================

#ifdef MULTI_LIGHT

float point_light_attenuation(PointLightData light, float distance) {
    return 1.0 / (light.constant_atten + light.linear_atten * distance + light.quadratic_atten * distance * distance);
}

vec3 sample_point_light_direction(PointLightData light, vec3 hit_pos, float r1, float r2, out float dist) {
    vec3 to_light = light.position - hit_pos;
    dist = length(to_light);
    vec3 light_dir = to_light / dist;

    if (light.radius > 0.0) {
        vec3 T, B;
        build_basis(light_dir, T, B);

        float sin_theta_max = light.radius / dist;
        float cos_theta = 1.0 - r2 * (1.0 - sqrt(1.0 - sin_theta_max * sin_theta_max));
        float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        light_dir = normalize(
            light_dir * cos_theta +
            T * cos(r1 * TWOPI) * sin_theta +
            B * sin(r1 * TWOPI) * sin_theta
        );
    }

    return light_dir;
}

vec3 sample_area_light_point(AreaLightData light, float r1, float r2) {
    float u = (r1 - 0.5) * light.width;
    float v = (r2 - 0.5) * light.height;
    return light.position + light.tangent * u + light.bitangent * v;
}

#endif // MULTI_LIGHT

// ===================================== Sky / background ==========================================

vec3 get_sun_disk_color(vec3 direction) {
    if (!primary_light_enabled || primary_light_type != 0 || primary_light_intensity <= 0.0) {
        return vec3(0.0);
    }

    float cos_angle = dot(direction, primary_light_dir);
    float cos_radius = cos(primary_light_radius);

    if (cos_angle < cos_radius) {
        return vec3(0.0);
    }

    float r = acos(cos_angle) / primary_light_radius;
    float limb = 1.0 - 0.6 * (1.0 - sqrt(1.0 - r*r));
    float edge_softness = 1.0 - smoothstep(0.85, 1.0, r);

    return primary_light_color * limb * edge_softness;
}

vec3 get_sky_color(vec3 direction) {
    vec3 sky;

    if (use_skybox) {
        sky = texture(skybox, direction).rgb * sky_intensity;
    } else {
        sky = background_color * sky_intensity;
    }

    // Add sun disk for directional lights
    if (primary_light_enabled && primary_light_type == 0 && primary_light_intensity > 0.0) {
        sky += get_sun_disk_color(direction) * primary_light_intensity;
    }

    return sky;
}

vec3 get_ambient_light() {
    vec3 ambient;

    if (use_skybox) {
        ambient = vec3(0.0);
        ambient += texture(skybox, vec3(0.0, 1.0, 0.0)).rgb;
        ambient += texture(skybox, vec3(1.0, 0.3, 0.0)).rgb;
        ambient += texture(skybox, vec3(-1.0, 0.3, 0.0)).rgb;
        ambient += texture(skybox, vec3(0.0, 0.3, 1.0)).rgb;
        ambient += texture(skybox, vec3(0.0, 0.3, -1.0)).rgb;
        ambient /= 5.0;
    } else {
        ambient = background_color;
    }

    // Desaturate to reduce color cast
    float ambient_saturation = 0.3;
    float luma = dot(ambient, vec3(0.2126, 0.7152, 0.0722));
    ambient = mix(vec3(luma), ambient, ambient_saturation);

    return ambient * sky_intensity * ambient_intensity;
}

// ===================================== Surface data ==============================================

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

    mat3 normal_matrix = mat3(hit_inst.transform);
    vec3 normal_world = normalize(normal_matrix * normal_obj);

    if (dot(normal_world, ray_dir) > 0.0) {
        normal_world = -normal_world;
    }

    return normal_world;
}

// ========================================= Shadows ===============================================

bool test_shadow(vec3 hit_pos, vec3 normal, vec3 light_dir, float max_dist) {
    if (!enable_shadows) return false;

    vec3 shadow_origin = hit_pos + normal * 0.001;

    Ray shadow_ray;
    shadow_ray.origin = shadow_origin;
    shadow_ray.inv_direction = 1.0 / light_dir;
    shadow_ray.t = max_dist;

    return is_occluded(shadow_ray);
}

// ===================================== Direct lighting ===========================================

vec3 calculate_direct_lighting(vec3 hit_pos, vec3 normal, inout uint rng_state) {
    vec3 direct_light = vec3(0.0);

    // Primary light
    if (primary_light_enabled && primary_light_intensity > 0.0) {
        
        if (primary_light_type == 0) {
            // Directional light (sun-like)
            float r1 = random_float(rng_state);
            float r2 = random_float(rng_state);
            vec3 light_dir = sample_disk_direction(primary_light_dir, primary_light_radius, r1, r2);

            float NdotL = max(dot(normal, light_dir), 0.0);

            if (NdotL > 0.0) {
                bool shadowed = false;
                if (primary_light_shadows) {
                    shadowed = test_shadow(hit_pos, normal, light_dir, 1e10);
                }

                if (!shadowed) {
                    direct_light += primary_light_color * primary_light_intensity * NdotL;
                }
            }
        }
        else if (primary_light_type == 1) {
            // Point light as primary
            vec3 to_light = primary_light_pos - hit_pos;
            float dist = length(to_light);
            vec3 light_dir = to_light / dist;

            // Sample soft shadow direction if radius > 0
            if (primary_light_radius > 0.0) {
                float r1 = random_float(rng_state);
                float r2 = random_float(rng_state);
                
                vec3 T, B;
                build_basis(light_dir, T, B);
                float sin_theta_max = primary_light_radius / dist;
                float cos_theta = 1.0 - r2 * (1.0 - sqrt(1.0 - sin_theta_max * sin_theta_max));
                float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
                light_dir = normalize(
                    light_dir * cos_theta +
                    T * cos(r1 * TWOPI) * sin_theta +
                    B * sin(r1 * TWOPI) * sin_theta
                );
            }

            float NdotL = max(dot(normal, light_dir), 0.0);

            if (NdotL > 0.0) {
                bool shadowed = false;
                if (primary_light_shadows) {
                    shadowed = test_shadow(hit_pos, normal, light_dir, dist - 0.002);
                }

                if (!shadowed) {
                    float atten = 1.0 / (dist * dist); // Simple quadratic falloff
                    direct_light += primary_light_color * primary_light_intensity * NdotL * atten;
                }
            }
        }

        // TODO: area light as primary
    }

#ifdef MULTI_LIGHT
    // Additional point lights
    for (int i = 0; i < min(point_lights_count, MAX_LIGHTS_PER_PASS); i++) {
        float intensity = point_lights[i].intensity;
        if (intensity <= 0.0) continue;

        float r1 = random_float(rng_state);
        float r2 = random_float(rng_state);
        float dist;

        vec3 light_dir = sample_point_light_direction(point_lights[i], hit_pos, r1, r2, dist);

        float NdotL = max(dot(normal, light_dir), 0.0);

        if (NdotL > 0.0) {
            bool shadowed = false;

            if (enable_shadows && point_lights[i].cast_shadows != 0u) {
                shadowed = test_shadow(hit_pos, normal, light_dir, dist - 0.002);
            }

            if (!shadowed) {
                PointLightData light = point_lights[i];
                float atten = point_light_attenuation(light, dist);
                direct_light += light.color * intensity * NdotL * atten;
            }
        }
    }

    // Additional area lights
    for (int i = 0; i < min(area_lights_count, MAX_LIGHTS_PER_PASS); i++) {
        float intensity = area_lights[i].intensity;
        if (intensity <= 0.0) continue;

        float r1 = random_float(rng_state);
        float r2 = random_float(rng_state);

        AreaLightData light = area_lights[i];

        vec3 sample_pos = sample_area_light_point(light, r1, r2);
        vec3 to_light = sample_pos - hit_pos;
        float dist = length(to_light);
        vec3 light_dir = to_light / dist;

        float NdotL = max(dot(normal, light_dir), 0.0);
        float light_cos = -dot(light.normal, light_dir);

        if (light.two_sided != 0u) {
            light_cos = abs(light_cos);
        }

        if (NdotL > 0.0 && light_cos > 0.0) {
            bool shadowed = false;
            if (enable_shadows && light.cast_shadows != 0u) {
                shadowed = test_shadow(hit_pos, normal, light_dir, dist - 0.002);
            }

            if (!shadowed) {
                float area = light.width * light.height;
                float geometry = (NdotL * light_cos * area) / (dist * dist);
                direct_light += light.color * intensity * geometry;
            }
        }
    }
#endif // MULTI_LIGHT

    return direct_light;
}

#endif // LIGHTING_GLSL
