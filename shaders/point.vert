#version 430 core

layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_color;

uniform mat4 camera;
uniform mat4 model;
uniform float u_point_size;

out vec3 v_color;

void main() {
    gl_Position = camera * model * vec4(a_pos, 1.0);
    v_color = a_color;
    gl_PointSize = u_point_size;
}