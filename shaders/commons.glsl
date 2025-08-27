#ifndef COMMONS_GLSL
#define COMMONS_GLSL

const float PI = 3.14159265359;
const float HPI = PI * 0.5;
const float TWOPI = 2.0 * PI;

// Constant k = 4 * log(2) for a Gaussian with FWHM = acceptance_angle
const float GAUSS_CONSTANT_K = 2.77258872224;

struct Material {
    uint texture_idx;
    uint pad0, pad1, pad2;
};

struct Ommatidium {
    // Using vec4 for explicit 16-byte alignment
    vec4 origin;            // offset 0,  size 16. w is unused
    vec4 direction;         // offset 16, size 16. w is unused
    vec2 acceptance_angles; // offset 32, size 8
    // Explicit padding to make total size a multiple of 16 (vec4 alignment)
    float pad0, pad1;       // offset 40, size 8
}; // total size = 48 bytes

struct Triangle {
    // Using vec4 for explicit 16-byte alignment
    vec4 v0, v1, v2;       // offsets 0, 16 and 32, size 16. w is unused
    vec2 uv0, uv1, uv2;    // offsets 48, 56 and 64, size 8
    uint material_idx;     // offset 72, size 4
    // Explicit padding to make total size a multiple of 16 (vec4 alignment)
    float pad0;            // offset 76, size 4
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

    // Importance sample the polar angle theta for each axis (H and V)
    // using the inverse CDF of the Gaussian distribution
    // The random variable here is the angle itself, scaled by the acceptance angle
    float angle_h = om.acceptance_angles.x * sqrt(-log(u1) / GAUSS_CONSTANT_K);
    float angle_v = om.acceptance_angles.y * sqrt(-log(u1) / GAUSS_CONSTANT_K);

    // Using the same random number u1 for both maintains correlation and correctly
    // forms an elliptical distribution from a circular one

    // Convert these angles on the tangent plane to a 3D direction in the ommatidium's local coordinate system
    vec2 point_on_ellipse;
    point_on_ellipse.x = tan(angle_h) * cos(phi);
    point_on_ellipse.y = tan(angle_v) * sin(phi);

    // Project from tangent plane to unit sphere
    vec3 sample_local = normalize(vec3(point_on_ellipse, 1.0));

    // Transform from local to world coordinates and return
    return normalize(mat3(tangent, bitangent, forward) * sample_local);
}

#endif // COMMONS_GLSL