#ifndef COMMONS_GLSL
#define COMMONS_GLSL

const float PI = 3.14159265359;
const float HPI = PI * 0.5;
const float TWOPI = 2.0 * PI;

// Constant k = 4 * log(2) for a Gaussian with FWHM = acceptance_angle
const float GAUSS_CONSTANT_K = 2.77258872224;

struct Material {
    uint texture_idx;       // 0xFFFFFFFF means no texture, use base_color
    uint base_color;        // RGBA8 packed into uint32
    uint pad0, pad1;
};

struct Ommatidium {
    vec4 origin;
    vec4 direction;
    vec2 acceptance_angles;       // .x = minor axis, .y = major axis
    vec2 interommatidial_angles;  // .x = minor axis, .y = major axis
    float tilt;
    float sensitivity;
    uint packed_data;   // bits 0-2 = eye ID, bits 3-6 = receptor type, bits 7-10 = mumber of neighbours, bits 11-26 = custom ID, rest is padding
    uint padding;
};

// Helper functions for unpacking

vec4 unpack_color(uint packed_color) {
    float r = float(packed_color & 255u) / 255.0;
    float g = float((packed_color >> 8u) & 255u) / 255.0;
    float b = float((packed_color >> 16u) & 255u) / 255.0;
    float a = float((packed_color >> 24u) & 255u) / 255.0;
    return vec4(r, g, b, a);
}

uint unpack_eye_id(Ommatidium om) {
    return om.packed_data & 7u;
}

uint unpack_receptor_type(Ommatidium om) {
    return om.packed_data & 0x0Fu;
}

uint unpack_neighbours_count(Ommatidium om) {
    return (om.packed_data >> 4) & 0x0Fu;
}

uint unpack_custom_id(Ommatidium om) {
    return (om.packed_data >> 8) & 0xFFFFu;
}

struct Triangle {
    vec4 v0, v1, v2;       // offsets 0, 16 and 32, size 16. w is unused
    vec2 uv0, uv1, uv2;    // offsets 48, 56 and 64, size 8
    uint material_idx;     // offset 72, size 4
}; // total size = 80 bytes

// Single point in the point cloud
struct Point {
    vec3 position;
    float radius;
    vec3 normal;
    vec3 color;
    float pad0, pad1;  // two more floats unused
};

// Simple RNG with temporal dithering
float rand(vec2 co, float dither){
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233)) + dither) * 43758.5453);
}

// Generates a sample direction using "true" Gaussian importance sampling
// (as in, the distribution of samples directly matches the Gaussian acceptance function)
//      - om: The ommatidium data containing H and V acceptance angles
//      - tangent, bitangent, forward: The basis vectors of the ommatidium's local frame
//      - u1, u2: Two uniform random numbers in the range [0, 1]
vec3 sampledir(
    in Ommatidium om,
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
    float angle_minor = om.acceptance_angles.x * sqrt(-log(u1) / GAUSS_CONSTANT_K);
    float angle_major = om.acceptance_angles.y * sqrt(-log(u1) / GAUSS_CONSTANT_K);

    // Using the same random number u1 for both maintains correlation and correctly
    // forms an elliptical distribution from a circular one

    // Convert these angles on the tangent plane to a 2D point on an axis-aligned ellipse
    vec2 point_on_ellipse;
    point_on_ellipse.x = tan(angle_minor) * cos(phi);
    point_on_ellipse.y = tan(angle_major) * sin(phi);

    // Rotate this 2D point by the ommatidium's elliptic tilt
    float s = sin(om.tilt);
    float c = cos(om.tilt);
    mat2 rotation_matrix = mat2(c, -s, s, c);
    vec2 tilted_point = rotation_matrix * point_on_ellipse;

    // Project from tangent plane to unit sphere using the now-tilted point
    vec3 sample_local = normalize(vec3(tilted_point, 1.0));

    // Transform from local to world coordinates and return
    return normalize(mat3(tangent, bitangent, forward) * sample_local);
}


#endif // COMMONS_GLSL