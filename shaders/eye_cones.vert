#version 430 core

#include "commons.glsl"

layout(location = 0) in vec3 cone_vertex;

layout(std430, binding = 0) readonly buffer OmmatidiaInputBlock {
   Ommatidium ommatidia_data[];
};

uniform mat4 view;
uniform mat4 projection;
uniform mat4 eye_to_world;
uniform float cone_length = 0.015;
uniform float radius_scale = 1.0;
uniform bool is_degrees = false;
uniform float viz_scale = 100.0;

out vec3 dbg_data;
out flat int v_index;

mat3 rmatFromDir(vec3 z) {
    vec3 x = normalize(cross( (abs(z.y) > 0.999) ? vec3(1,0,0) : vec3(0,1,0), z));
    vec3 y = cross(z, x);
    return mat3(x, y, z);
}

void main() {
    int i = gl_InstanceID;
    Ommatidium om = ommatidia_data[i];

    vec3 O_world = (eye_to_world * vec4(om.origin.xyz * viz_scale, 1.0)).xyz;

    vec3 D_world = normalize((eye_to_world * vec4(om.direction.xyz, 0.0)).xyz);
    float L = cone_length;
    vec2 acc = om.acceptance_angles;

    if (is_degrees) acc = radians(acc);

    float halfAngle = 0.5 * max(acc.x, acc.y);
    float baseR = L * tan(halfAngle) * radius_scale;
    baseR = max(baseR, 0.001);

    vec3 p_model = cone_vertex + vec3(0.0, 0.0, 1.0);

    mat3 S = mat3(baseR, 0, 0, 0, baseR, 0, 0, 0, L);
    mat3 R = rmatFromDir(D_world);

    vec3 offset = R * S * p_model;

    vec3 pos_world = O_world + offset;

    gl_Position = projection * view * vec4(pos_world, 1.0);

    dbg_data = pos_world;
    v_index = i;
}