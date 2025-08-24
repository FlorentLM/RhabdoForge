#version 430 core

in vec2 v_tex_coord;
out vec4 f_color;

uniform sampler2D texture_sampler;

void main() {
    f_color = texture(texture_sampler, v_tex_coord);
}