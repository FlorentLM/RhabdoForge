#version 430 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
layout (location = 2) in float radius;

uniform mat4 camera;
uniform mat4 model;
uniform mat4 light_space_matrix;   // light's P * V (for shadow mapping)
uniform float radius_scale;

out vec3 vertColor;
out vec4 fragLightSpacePos;

void main() {
    vec4 world_pos = model * vec4(position, 1.0);

    gl_Position = camera * world_pos;
    vertColor = color;
    gl_PointSize = radius * radius_scale;

    fragLightSpacePos = light_space_matrix * world_pos;
}
