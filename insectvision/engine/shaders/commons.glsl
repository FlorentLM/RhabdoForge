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

struct ReceptorData {
    vec3 position;      // receptor position x, y, z
    uint metadata;      // eye_id, R_type, neighbour_count, lens_id, pad
    vec3 direction;     // receptor direction x, y, z
    float acc_tilt;     // acceptance angle ellipse tilt
    vec2 acc_axes;      // acceptance angle ellipse minor, major axes
    float sensitivity;  // photometric response multiplier
    float tau;          // temporal accumulation
}; // 48 bytes

struct LensData {
    vec2 ioa_axes;      // lattice geometry axes (ellipse minor, major)
    float tilt;         // lattice geometry orientation (ellipse tilt)
    uint padding;
}; // 16 bytes

// Helper functions for unpacking

vec4 unpack_color(uint packed_color) {
    float r = float(packed_color & 255u) / 255.0;
    float g = float((packed_color >> 8u) & 255u) / 255.0;
    float b = float((packed_color >> 16u) & 255u) / 255.0;
    float a = float((packed_color >> 24u) & 255u) / 255.0;
    return vec4(r, g, b, a);
}

uint unpack_eye_id(ReceptorData rcpt) {
    return rcpt.metadata & 7u; // bits 0-2
}

uint unpack_receptor_type(ReceptorData rcpt) {
    return (rcpt.metadata >> 3u) & 15u; // bits 3-6
}

uint unpack_neighbours_count(ReceptorData rcpt) {
    return (rcpt.metadata >> 7u) & 15u; // bits 7-10
}

uint unpack_lens_id(ReceptorData rcpt) {
    return (rcpt.metadata >> 11u) & 65535u; // bits 11-26
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

// Generates a sample direction using 'true' Gaussian importance sampling
// (as in, the distribution of samples directly matches the Gaussian acceptance function)
//      - rcpt: The receptor data containing H and V acceptance angles
//      - tangent, bitangent, forward: The basis vectors of the receptor's local frame
//      - u1, u2: Two uniform random numbers in the range [0, 1]
vec3 sampledir(
    in ReceptorData rcpt,
    in vec3 tangent,
    in vec3 bitangent,
    in vec3 forward,
    in float u1,
    in float u2
) {
    // Azimuthal angle phi (uniform)
    float phi = TWOPI * u2;

    // Importance sample the polar angle theta for each axis (minor and major)
    // using the inverse CDF of the Gaussian distribution
    float angle_minor = rcpt.acc_axes.x * sqrt(-log(u1) / GAUSS_CONSTANT_K);
    float angle_major = rcpt.acc_axes.y * sqrt(-log(u1) / GAUSS_CONSTANT_K);

    // Using the same random number u1 for both maintains correlation and correctly
    // forms an elliptical distribution from a circular one

    // Convert these angles on the tangent plane to a 2D point on an axis-aligned ellipse
    vec2 point_on_ellipse;
    point_on_ellipse.x = tan(angle_minor) * cos(phi);
    point_on_ellipse.y = tan(angle_major) * sin(phi);

    // Rotate this 2D point by the receptor's elliptic tilt
    float s = sin(rcpt.acc_tilt);
    float c = cos(rcpt.acc_tilt);
    mat2 rotation_matrix = mat2(c, -s, s, c);
    vec2 tilted_point = rotation_matrix * point_on_ellipse;

    // Project from tangent plane to unit sphere using the now-tilted point
    vec3 sample_local = normalize(vec3(tilted_point, 1.0));

    // Transform from local to world coordinates and return
    return normalize(mat3(tangent, bitangent, forward) * sample_local);
}


#endif // COMMONS_GLSL