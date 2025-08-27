#version 430 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
layout (location = 2) in float radius;

uniform mat4 camera;
uniform mat4 model;
uniform float radius_scale;

out vec3 vertColor;

void main() {
    gl_Position = camera * model * vec4(position, 1.0);
    vertColor = color;
    gl_PointSize = radius * radius_scale;
}