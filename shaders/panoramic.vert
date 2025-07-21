#version 430 core

// A single triangle that covers the whole screen
const vec2 positions[3] = vec2[](
    vec2(-1, -1),
    vec2( 3, -1),
    vec2(-1,  3)
);

// Screen position needs to be passed to the fragment shader
// so it can calculate the 3D direction vector
out vec2 v_screen_pos;

void main()
{
    v_screen_pos = positions[gl_VertexID];
    gl_Position = vec4(v_screen_pos, 0.0, 1.0);
}