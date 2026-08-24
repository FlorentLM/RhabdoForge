#ifndef LIGHTING_GLSL
#define LIGHTING_GLSL

// defines injected at compile time based on light counts:
// HAS_DIRECTIONAL_LIGHT / MULTI_DIRECTIONAL
// HAS_POINT_LIGHT / MULTI_POINT
// HAS_AREA_LIGHT / MULTI_AREA

#include "commons.glsl"
#include "bvh.glsl"

const int MAX_LIGHTS_PER_PASS = 16;

// ============================================ Structs =======================================

struct DirectionalLightData {
    vec3 direction;
    float angular_radius;
    vec3 color;
    float intensity;
    uint cast_shadows;
    uint _pad0, _pad1, _pad2;
};

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

// =================================== Uniforms =========================================

uniform vec3 background_color;
uniform bool use_sky;
uniform float sky_intensity;        // intensity for sky color and sun disk
uniform bool enable_ambient;        // toggle ambient/fill lighting from sky
uniform bool enable_direct;         // toggle all direct lighting
uniform bool enable_shadows;        // global shadow override (false = skip all shadow rays)
uniform float ambient_intensity;    // multiplier for ambient (independent of sky_intensity)

uniform int directional_lights_count;
uniform int point_lights_count;
uniform int area_lights_count;

uniform vec3 sh_irradiance_coeffs[9];

// ====================================== Light SSBOs ===============================================

#ifdef HAS_DIRECTIONAL_LIGHT
layout(std430, binding = BINDING_DIR) readonly buffer DirectionalLightsBuffer {
    DirectionalLightData directional_lights[];
};
#endif

#ifdef HAS_POINT_LIGHT
layout(std430, binding = BINDING_POINT) readonly buffer PointLightsBuffer {
    PointLightData point_lights[];
};
#endif

#ifdef HAS_AREA_LIGHT
layout(std430, binding = BINDING_AREA) readonly buffer AreaLightsBuffer {
    AreaLightData area_lights[];
};
#endif

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

#ifdef HAS_POINT_LIGHT
float point_light_attenuation(PointLightData light, float distance) {
    return 1.0 / (light.constant_atten
                 + light.linear_atten * distance
                 + light.quadratic_atten * distance * distance);
}

vec3 sample_point_light_direction(PointLightData light, vec3 hit_pos,
                                  float r1, float r2, out float dist) {
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
#endif

#ifdef HAS_AREA_LIGHT
vec3 sample_area_light_point(AreaLightData light, float r1, float r2) {
    float u = (r1 - 0.5) * light.width;
    float v = (r2 - 0.5) * light.height;
    return light.position + light.tangent * u + light.bitangent * v;
}
#endif

// ===================================== Sky / background ==========================================

vec3 sun_disk_contribution(DirectionalLightData light, vec3 direction) {
    if (light.intensity <= 0.0 || light.angular_radius <= 0.0)
        return vec3(0.0);

    float cos_angle = dot(direction, light.direction);
    float cos_radius = cos(light.angular_radius);
    if (cos_angle < cos_radius)
        return vec3(0.0);

    float r = acos(cos_angle) / light.angular_radius;
    float limb = 1.0 - 0.6 * (1.0 - sqrt(1.0 - r * r));
    float edge = 1.0 - smoothstep(0.85, 1.0, r);
    return light.color * light.intensity * limb * edge;
}

vec3 get_sun_disk_color(vec3 direction) {
#ifndef HAS_DIRECTIONAL_LIGHT
    return vec3(0.0);
#elif defined(MULTI_DIRECTIONAL)
    vec3 color = vec3(0.0);
    for (int i = 0; i < directional_lights_count; i++)
        color += sun_disk_contribution(directional_lights[i], direction);
    return color;
#else
    return sun_disk_contribution(directional_lights[0], direction);
#endif
}

vec3 get_sky_color(vec3 direction, bool with_sun) {
    vec3 sky = use_sky ? sample_env(direction) * sky_intensity
                          : background_color * sky_intensity;
    if (with_sun) sky += get_sun_disk_color(direction);
    return sky;
}

vec3 sh_irradiance(vec3 n) {
    vec3 r = sh_irradiance_coeffs[0] * 0.282095
           + sh_irradiance_coeffs[1] * (0.488603 * n.y)
           + sh_irradiance_coeffs[2] * (0.488603 * n.z)
           + sh_irradiance_coeffs[3] * (0.488603 * n.x)
           + sh_irradiance_coeffs[4] * (1.092548 * n.x * n.y)
           + sh_irradiance_coeffs[5] * (1.092548 * n.y * n.z)
           + sh_irradiance_coeffs[6] * (0.315392 * (3.0 * n.z * n.z - 1.0))
           + sh_irradiance_coeffs[7] * (1.092548 * n.x * n.z)
           + sh_irradiance_coeffs[8] * (0.546274 * (n.x * n.x - n.y * n.y));
    return max(r, vec3(0.0));   // clamp SH ringing -> no negative ambient near bright suns
}

// ===================================== Surface data ==============================================

vec3 get_surface_color(HitInfo hit) {
    InstanceInfo hit_inst = instances[hit.instance_id];

    vec3 albedo;

    if (hit.is_point_hit) {
        Point p = getPoint(hit_inst.vertex_or_point_offset + hit.primitive_idx);
        albedo = p.color.rgb;

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
            albedo = unpack_color(hit_mat.base_color).rgb;
        } else {
            albedo = texture(scene_textures, vec3(hit_uv, hit_mat.texture_idx)).rgb;
        }
    }

    if (hit_inst.is_srgb != 0u) albedo = pow(albedo, vec3(2.2));
    return albedo;
}

vec3 get_surface_normal(HitInfo hit, vec3 ray_dir, vec3 hit_pos) {
    InstanceInfo hit_inst = instances[hit.instance_id];

    vec3 normal_world;

    if (hit.is_point_hit) {
        uint point_id = hit_inst.vertex_or_point_offset + hit.primitive_idx;
        Point p = getPoint(point_id);
        vec3 center_world = (hit_inst.transform * vec4(p.position, 1.0)).xyz;
        normal_world = normalize(hit_pos - center_world);   // sphere normal, world space

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

        vec3 normal_obj = normalize(cross(v1 - v0, v2 - v0));
        mat3 normal_matrix = mat3(hit_inst.transform);

        normal_world = normalize(normal_matrix * normal_obj);
    }

    if (dot(normal_world, ray_dir) > 0.0) normal_world = -normal_world; // no-op for near-side hit
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

vec3 calculate_direct_lighting(vec3 hp, vec3 N, inout uint rng) {
    int total_lights = directional_lights_count + point_lights_count + area_lights_count;

    if (total_lights == 0) return vec3(0.0);

    // Pick random light uniformly
    int pick = int(random_float(rng) * float(total_lights));
    pick = clamp(pick, 0, total_lights - 1); // safety clamp

    float pdf = 1.0 / float(total_lights);     // probability of picking this light
    vec3 result = vec3(0.0);

#ifdef HAS_DIRECTIONAL_LIGHT
    if (pick < directional_lights_count) {
        DirectionalLightData dl = directional_lights[pick];
        if (dl.intensity > 0.0) {
            vec3 ld = sample_disk_direction(dl.direction, dl.angular_radius, random_float(rng), random_float(rng));
            float NdL = max(dot(N, ld), 0.0);
            if (NdL > 0.0) {
                bool sh = dl.cast_shadows != 0u && test_shadow(hp, N, ld, 1e10);
                if (!sh) result = dl.color * dl.intensity * NdL;
            }
        }
        return result / pdf; // weight by inverse proba
    }
    pick -= directional_lights_count;
#endif

#ifdef HAS_POINT_LIGHT
    if (pick < point_lights_count) {
        PointLightData pl = point_lights[pick];
        if (pl.intensity > 0.0) {
            float dist;
            vec3 ld = sample_point_light_direction(pl, hp, random_float(rng), random_float(rng), dist);
            float NdL = max(dot(N, ld), 0.0);
            if (NdL > 0.0) {
                bool sh = pl.cast_shadows != 0u && test_shadow(hp, N, ld, dist - 0.002);
                if (!sh) result = pl.color * pl.intensity * NdL * point_light_attenuation(pl, dist);
            }
        }
        return result / pdf;
    }
    pick -= point_lights_count;
#endif

#ifdef HAS_AREA_LIGHT
    if (pick < area_lights_count) {
        AreaLightData al = area_lights[pick];
        if (al.intensity > 0.0) {
            vec3 sp = sample_area_light_point(al, random_float(rng), random_float(rng));
            vec3 tl = sp - hp;
            float dist = length(tl);
            vec3 ld = tl / dist;
            float NdL = max(dot(N, ld), 0.0);
            float lcos = al.two_sided != 0u ? abs(dot(al.normal, ld)) : max(-dot(al.normal, ld), 0.0);

            if (NdL > 0.0 && lcos > 0.0) {
                bool sh = al.cast_shadows != 0u && test_shadow(hp, N, ld, dist - 0.002);
                float surface_area = al.width * al.height;
                if (!sh) result = al.color * al.intensity * (NdL * lcos * surface_area) / (dist * dist);
            }
        }
        return result / pdf;
    }
#endif

    return vec3(0.0);
}

#endif // LIGHTING_GLSL
