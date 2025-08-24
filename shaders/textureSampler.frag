#version 430 core

in vec2 v_tex_coord;
out vec4 f_color;

uniform sampler2D texture;

void main() {
    f_color = texture(texture, v_tex_coord);
}