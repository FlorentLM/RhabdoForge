#ifndef COMMONS_GLSL
#define COMMONS_GLSL

const int RNG_PSEUDO     = 0;
const int RNG_HALTON     = 1;
const int RNG_STRATIFIED = 2;

const int MODE_GAUSSIAN  = 0;
const int MODE_AIRY      = 1;

const float PI = 3.14159265359;
const float HPI = PI * 0.5;
const float TWOPI = 2.0 * PI;

// Constant k = 4 * log(2) for a Gaussian with FWHM = acceptance_angle
const float GAUSS_CONSTANT_K = 2.77258872224;

struct Material {
    uint texture_idx;       // 0xFFFFFFFF means no texture (use base_color)
    uint base_color;        // RGBA8 packed into uint32
    uint pad0, pad1;
};

// Ommatidium static  (read only)
struct OmmatidiumStatic {
    vec3  position;
    float chi;
    vec3  forward;
    float focal_um;
    vec3  right;
    float aperture_um;
    vec3  up;
    float ioa_tilt;
    vec2 saccade_dxdy;
    float ampl_lateral;
    float ampl_axial;
    float tau_rise;
    float tau_relax;
    float tau_fast;
    float tau_adapt;
    vec2  ioa_angles;
    vec2  retina_dxdy;
}; // 112 bytes

// Ommatidium dynamic
struct OmmatidiumDynamic {
    float curr_lum_fast;
    float curr_lum_slow;
    float curr_lateral_disp;
    float curr_axial_disp;
}; // 16 bytes

// Rhabdomere static (read only)
struct RhabdomereStatic {
    vec3  sensitivity;
    float wavelength_um;
    vec2  rest_acc_angles;
    vec2  rest_offset;
    float tau_membrane;
    uint  cartridge_src;
    float diameter_um;
    uint  metadata;
}; // 48 bytes

// Rhabdomere dynamic
struct RhabdomereDynamic {
    vec3  curr_direction;
    float curr_adaptation;
    vec2  curr_acc_angles;
    float optical_scale;
    float pad;
}; // 32 bytes


// Metadata Unpacking
uint  unpack_eye_id(uint m)          { return m & 15u; }                          // bits 0-3
uint  unpack_rhab_type(uint m)       { return (m >> 4u)  & 15u; }                 // bits 4-7
uint  unpack_neighbour_count(uint m) { return (m >> 8u)  & 15u; }                 // bits 8-11
uint  unpack_omm_id(uint m)          { return (m >> 12u) & 65535u; }              // bits 12-27
float unpack_chirality(uint m)       { return ((m >> 28u) & 1u) == 1u ? -1.0 : 1.0; }  // bit 28
bool  unpack_binocularity(uint m)    { return ((m >> 29u) & 1u) == 1u; }          // bit 29
bool  unpack_wiring_valid(uint m)    { return ((m >> 30u) & 1u) == 1u; }          // bit 30
bool  unpack_is_edge(uint m)         { return ((m >> 31u) & 1u) == 1u; }          // bit 31


vec4 unpack_color(uint packed_color) {
    return vec4(float(packed_color & 255u) / 255.0,
                float((packed_color >> 8u) & 255u) / 255.0,
                float((packed_color >> 16u) & 255u) / 255.0,
                float((packed_color >> 24u) & 255u) / 255.0);
}

struct Triangle {
    vec4 v0, v1, v2;       // offsets 0, 16 and 32, size 16. w is unused
    vec2 uv0, uv1, uv2;    // offsets 48, 56 and 64, size 8
    uint material_idx;     // offset 72, size 4
}; // 80 bytes

struct Point {
    vec3 position;
    float radius;
    vec3 normal;
    vec3 color;
    float pad0, pad1;
};

uniform float airy_lut[256];

// =====================================================================================================================

// Simple RNG with temporal dithering
float rand(vec2 co, float dither){
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233)) + dither) * 43758.5453);
}

// Radical-inverse Halton sequence (low-discrepancy quasi-random)
// (as in https://en.wikipedia.org/wiki/Halton_sequence#Implementation)
float halton_sequence(uint index, uint base) {
    float f = 1.0;
    float r = 0.0;
    uint i = index;
    while (i > 0u) {
        f /= float(base);
        r += f * float(i % base);
        i /= base;
    }
    return r;
}

uint pcg_hash(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float random_float(inout uint rng_state) {
    rng_state = pcg_hash(rng_state);
    return float(rng_state) / 4294967295.0;
}

struct Sampler {
    float u1;
    float u2;
};

Sampler get_samples(int mode, uint sample_idx, uint nb_samples, uint rhab_idx, uint dither_counter) {
    Sampler s;
    uint seed = pcg_hash(rhab_idx * 1973u + dither_counter * 26699u + sample_idx * 749u);

    if (mode == RNG_HALTON) {
        uint halton_idx = dither_counter * nb_samples + sample_idx + 1u;
        uint hash = pcg_hash(rhab_idx * 1973u);
        s.u1 = clamp(fract(halton_sequence(halton_idx, 2u) + float(hash & 0xFFFFu)/65535.0), 1e-6, 1.0);
        s.u2 = fract(halton_sequence(halton_idx, 3u) + float((hash >> 16u) & 0xFFFFu)/65535.0);
    }
    else if (mode == RNG_STRATIFIED) {
        float grid_size = ceil(sqrt(float(nb_samples)));
        float cell_x = float(sample_idx % uint(grid_size));
        float cell_y = float(sample_idx / uint(grid_size));

        uint rng_state = seed;
        s.u1 = (cell_x + random_float(rng_state)) / grid_size;
        s.u2 = (cell_y + random_float(rng_state)) / grid_size;
    }
    else { // RNG_PSEUDO
        uint rng_state = seed;
        s.u1 = random_float(rng_state);
        s.u2 = random_float(rng_state);
    }
    return s;
}

// =====================================================================================================================

float get_sensitivity(int mode, float dx, float dy,
                      RhabdomereStatic rs, RhabdomereDynamic rd, OmmatidiumStatic os) {

    // Radial distance in 'elliptical space' (to handle anisotropy)
    float g_min = dx / max(rd.curr_acc_angles.x, 1e-15);
    float g_maj = dy / max(rd.curr_acc_angles.y, 1e-15);
    float radial_dist = sqrt(g_min*g_min + g_maj*g_maj);

     if (mode == MODE_AIRY) {
        // The LUT goes from 0.0 to 4.0 FWHM
        float lut_index = radial_dist * (255.0 / 4.0);

        // Clamp and return
        return airy_lut[int(clamp(lut_index, 0.0, 255.0))];
    }
    else { // MODE_GAUSSIAN
        return exp(-GAUSS_CONSTANT_K * (radial_dist * radial_dist));
    }
}

vec3 sampledir_importance(RhabdomereStatic rs, RhabdomereDynamic rd, OmmatidiumStatic os, vec3 T, vec3 B, vec3 F, float u1, float u2, out float weight) {

    float phi = TWOPI * u2;
    float angle_min = rd.curr_acc_angles.x * sqrt(-log(u1) / GAUSS_CONSTANT_K);
    float angle_maj = rd.curr_acc_angles.y * sqrt(-log(u1) / GAUSS_CONSTANT_K);

    vec2 p = vec2(tan(angle_min) * cos(phi), tan(angle_maj) * sin(phi));

    float s = sin(os.ioa_tilt), c = cos(os.ioa_tilt);
    vec2 tp = mat2(c, -s, s, c) * p;

    weight = 1.0; // pure importance sampling: weight is uniform
    return normalize(mat3(T, B, F) * normalize(vec3(tp, 1.0)));
}

vec3 sampledir_hybrid(int mode, RhabdomereStatic rs, RhabdomereDynamic rd, OmmatidiumStatic os, vec3 T, vec3 B, vec3 F, float u1, float u2, out float weight) {
    float phi = TWOPI * u2;

    // Sample a 'proposal' distribution that is wider than the actual acceptance
    // -> ensures it samples the tails / Airy rings, wide-angle lights, etc
    float spread_mult = 2.0;
    float sample_sigma_min = rd.curr_acc_angles.x * spread_mult;
    float sample_sigma_maj = rd.curr_acc_angles.y * spread_mult;

    // These are the displacement angles to test (raw elliptical radii)
    float r_min = sample_sigma_min * sqrt(-log(u1) / GAUSS_CONSTANT_K);
    float r_maj = sample_sigma_maj * sqrt(-log(u1) / GAUSS_CONSTANT_K);

    // Cartesian angles (dx, dy) for this specific ray
    float dx = r_min * cos(phi);
    float dy = r_maj * sin(phi);

    // Rhabdomere sensitivity (at this specific sampled angle) -> 'Physical truth'
    float rhab_sensitivity = get_sensitivity(mode, dx, dy, rs, rd, os);

    // Sampling probability Density (PDF) of the proposal distribution -> likelihood that this ray was picked
    float p_min = dx / sample_sigma_min;
    float p_maj = dy / sample_sigma_maj;
    float pdf = exp(-GAUSS_CONSTANT_K * (p_min*p_min + p_maj*p_maj));

    // weight is truth / sampling
    weight = rhab_sensitivity / max(pdf, 1e-6);  // avoid /0 in the extreme tails

    // and convert angles to direction vector
    vec2 p = vec2(tan(dx), tan(dy));
    float s = sin(os.ioa_tilt), c = cos(os.ioa_tilt);
    vec2 tp = mat2(c, -s, s, c) * p;

    return normalize(mat3(T, B, F) * normalize(vec3(tp, 1.0)));
}

#endif // COMMONS_GLSL