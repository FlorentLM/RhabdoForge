#version 430 core

in vec2 v_tex_coord;
out vec4 f_color;

uniform sampler2D texture_sampler;
uniform bool false_colors;
uniform bool uv_encoding;

void main() {
    vec4 tex_color = texture(texture_sampler, v_tex_coord);

    if (uv_encoding || false_colors) {
        // If UV encoded, humans can't see UV (Channel 0)
        // If simulating insect vision on normal textures, insects can't see Red (Channel 0)
        // In both cases, Channel 0 is made invisible in the background render
        tex_color.r = 0.0;
    }

    f_color = tex_color;
}