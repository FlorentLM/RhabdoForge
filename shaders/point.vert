#version 430 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;

uniform mat4 camera;
uniform mat4 model;
uniform float point_size;

out vec3 v_color;

void main() {
    gl_Position = camera * model * vec4(position, 1.0);
    v_color = color;
    gl_PointSize = point_size;
}