#version 430 core

// A single triangle that covers the whole screen
const vec2 positions[3] = vec2[](
    vec2(-1, -1),
    vec2( 3, -1),
    vec2(-1,  3)
);

// Output: Varying to fragment shader
layout (location = 0) out vec2 v_tex_coord;

void main() {
    vec2 pos = positions[gl_VertexID];
    gl_Position = vec4(pos, 0.0, 1.0);
    v_tex_coord = pos * 0.5 + 0.5; // UVs for texture sampling
}