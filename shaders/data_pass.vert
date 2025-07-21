#version 430 core

// No inputs
// No outputs

// The vertex shader simply generates a full-screen quad and the fragment shader
// will figure out which ommatidium to calculate based on its pixel coordinate

// A single triangle that covers the whole screen.
// Slightly more efficient than a quad.
const vec2 positions[3] = vec2[](
    vec2(-1, -1),
    vec2( 3, -1),
    vec2(-1,  3)
);

void main()
{
    gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
}