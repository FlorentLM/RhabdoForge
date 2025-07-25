#ifndef COMMONS_GLSL
#define COMMONS_GLSL

struct Material {
    uint texture_idx;
    uint pad0, pad1, pad2;
};

struct Ommatidium {
    vec3 origin;            // 12 bytes, offset 0
    // std430 alignment for vec3 is 16 bytes, so the compiler adds 4 bytes of padding here
    vec3 direction;         // 12 bytes, offset 16
    float acceptance_angle; // 4 bytes, offset 28
}; // 32 bytes total per ommatidium

struct Triangle {
    vec3 v0, v1, v2;
    vec2 uv0, uv1, uv2;
    uint material_idx;
    uint pad0, pad1, pad2; // align to vec4
};

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