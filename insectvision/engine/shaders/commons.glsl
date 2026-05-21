#ifndef COMMONS_GLSL
#define COMMONS_GLSL

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

// Lens static (read only)
struct LensStatic {
    vec3  right;
    float sacc_local_x;
    vec3  up;
    float sacc_local_y;
    vec3  forward;
    float ioa_tilt;
    vec2  ioa_axes;
    float nodal_distance_um;
    float lens_diameter_um;
}; // 64 bytes

// Lens dynamic
struct LensDynamic {
    float adapted_lum;
    float fast_lum;
    float lateral_um;
    float axial_um;
}; // 16 bytes

// Receptor static (read only)
struct ReceptorStatic {
    vec3  position;
    uint  metadata;
    vec2  rest_acc;
    vec2  rot_offset;
    vec3  sensitivity;
    float acc_tilt;
    float tau_membrane;
    uint  cartridge_src;
    float rhab_diameter_um;
    float wavelength_um;
}; // 64 bytes

// Receptor dynamic
struct ReceptorDynamic {
    vec3  direction;    float adaptation_state;
    vec2  acc_axes;     vec2  pad;
}; // 32 bytes


// Metadata Unpacking
uint unpack_eye_id(uint m)        { return m & 7u; }
uint unpack_receptor_type(uint m) { return (m >> 3u) & 15u; }
uint unpack_lens_id(uint m)       { return (m >> 11u) & 65535u; }
float unpack_chirality(uint m)    { return ((m >> 27u) & 1u) == 1u ? -1.0 : 1.0; }

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

vec3 sampledir(in ReceptorStatic rs, in ReceptorDynamic rd, in vec3 T, in vec3 B, in vec3 F, in float u1, in float u2) {
    float phi = TWOPI * u2;
    float angle_min = rd.acc_axes.x * sqrt(-log(u1) / GAUSS_CONSTANT_K);
    float angle_maj = rd.acc_axes.y * sqrt(-log(u1) / GAUSS_CONSTANT_K);

    vec2 p = vec2(tan(angle_min) * cos(phi), tan(angle_maj) * sin(phi));
    float s = sin(rs.acc_tilt), c = cos(rs.acc_tilt);
    vec2 tp = mat2(c, -s, s, c) * p;

    return normalize(mat3(T, B, F) * normalize(vec3(tp, 1.0)));
}

#endif // COMMONS_GLSL