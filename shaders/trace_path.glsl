#ifndef TRACE_PATH_GLSL
#define TRACE_PATH_GLSL

// Path tracing: multiple bounces with Monte Carlo integration

#include "lighting.glsl"

uniform int max_bounces;

vec3 trace_path(Ray r, inout uint rng_state) {
    vec3 throughput = vec3(1.0);
    vec3 radiance = vec3(0.0);

    vec3 direction = 1.0 / r.inv_direction;

    for (int bounce = 0; bounce <= max_bounces; bounce++) {
        HitInfo hit;
        traverse_tlas(r, direction, hit);

        if (!hit.found) {
            // Sky contribution (includes sun disk for directional lights)
            radiance += throughput * get_sky_color(direction);
            break;
        }

        vec3 hit_pos = r.origin + direction * hit.t;
        vec3 surface_color = get_surface_color(hit);
        vec3 normal = get_surface_normal(hit, direction);

        // Ambient (first bounce only)
        if (bounce == 0 && enable_ambient) {
            vec3 ambient_light = get_ambient_light();
            float hemisphere_factor = dot(normal, vec3(0.0, 1.0, 0.0)) * 0.4 + 0.6;
            radiance += throughput * surface_color * ambient_light * hemisphere_factor;
        }

        // Direct lighting (Next Event Estimation)
        if (enable_direct) {
            vec3 incoming_light = calculate_direct_lighting(hit_pos, normal, rng_state);
            radiance += throughput * surface_color * incoming_light;
        }

        // Russian roulette for path termination
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

vec3 trace(Ray r, inout uint rng_state) {
    return trace_path(r, rng_state);
}

#endif // TRACE_PATH_GLSL
