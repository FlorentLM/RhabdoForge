#version 430 core

#include "commons.glsl"

layout(location = 0) in vec3 cone_vertex;

layout(std430, binding = 0) readonly buffer OmmatidiaInputBlock {
   Ommatidium ommatidia_data[];
};

layout(std430, binding = 1) readonly buffer ColorDataBlock {
    vec4 final_rgba[];
};

// Uniforms
uniform mat4 view;
uniform mat4 projection;
uniform mat4 eye_to_world;
uniform float cone_length = 0.015;
uniform float radius_scale = 1.0;
uniform bool is_degrees = false;

// Output to fragment shader
out flat int v_index;

// Helper to build a rotation matrix from a direction vector
mat3 rmatFromDir(vec3 dir) {
    vec3 f = normalize(dir);
    vec3 up = (abs(f.y) > 0.999) ? vec3(1,0,0) : vec3(0,1,0);
    vec3 r = normalize(cross(up, f));
    vec3 u = cross(f, r);
    return mat3(r, u, f);
}

void main() {
    int i = gl_InstanceID;
    Ommatidium om = ommatidia_data[i];

    // Read eye-local origin and direction
    vec3 O_eye = ommatidia_data[i].origin.xyz;
    vec3 D_eye = ommatidia_data[i].direction.xyz;

    // Transform to world
    vec3 O = (eye_to_world * vec4(O_eye, 1.0)).xyz;
    vec3 D = normalize((eye_to_world * vec4(D_eye, 0.0)).xyz);

    // Calculate cone base radius from acceptance angle
    vec2 acc = om.acceptance_angles;
    if (is_degrees) acc = radians(acc);
    float halfAngle = 0.5 * max(acc.x, acc.y);

    float L = cone_length;
    float baseR = L * tan(halfAngle) * radius_scale;

    // The cone primitive has its apex at (0, 0, -1) and base at z=0
    // We shift it so the apex is at the origin (0, 0, 0) before scaling and rotating
    vec3 p = cone_vertex + vec3(0.0, 0.0, 1.0);

    // Scale to the correct length and radius
    mat3 S = mat3(baseR, 0, 0,
                  0, baseR, 0,
                  0, 0, L);

    // Rotate to align with the ommatidium's direction, then translate to its origin
    mat3 R = rmatFromDir(D);
    vec3 posWS = R * (S * p) + O;

    gl_Position = projection * view * vec4(posWS, 1.0);
    v_index = i;
}