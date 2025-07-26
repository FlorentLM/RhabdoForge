#ifndef COMMONS_GLSL
#define COMMONS_GLSL

struct Material {
    uint texture_idx;
    uint pad0, pad1, pad2;
};

struct Ommatidium {
    vec3 origin;            // 12 bytes, offset 0 + 4 bytes pad
    // but vec3 are treated as vec4 for std430 alignment so 16 bytes, so the compiler adds 4 bytes of padding
    vec3 direction;         // 12 bytes, offset 16
    float acceptance_angle; // 4 bytes, offset 28
}; // 32 bytes total per ommatidium

struct Triangle {
    vec3 v0, v1, v2;       // 12 bytes, but offsets at 0, 16 and 32 (because of padding to 16 bytes per vec3, see above)
    vec2 uv0, uv1, uv2;    // 8 bytes, offsets at 48, 56 and 64 (no padding needed)
    uint material_idx;     // 4 bytes, offset at 72 (still no padding needed)
//    uint pad0, pad1, pad2, pad3; // 6 pads of 4 bytes
};  // so 76 bytes total... BUT:
// the total size an array in std430 alignment must be a multiple of its base alignment (which corresponds
// to the largest type present in the array (here vec3, 16 bytes)
// The next multiple of 16 after 76 is 80

struct Ray {
    vec3 origin;
    vec3 direction;
};

struct HitInfo {
    bool found;
    float t; // distance along ray
    vec3 barycentric_coords;
    uint triangle_idx;
};

#endif // COMMONS_GLSL