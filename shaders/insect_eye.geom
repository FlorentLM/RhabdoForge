#version 430 core

// Receive single points
layout (points) in;
// Output a triangle strip (a 6-vertex strip makes a 4-triangle hexagon)
layout (triangle_strip, max_vertices = 6) out;

// The per-vertex input is the per-vertex output from the vertex shader (gl_Position)
in gl_PerVertex {
    vec4 gl_Position;
} gl_in[];

// Custom input variable from the vertex shader
flat in int v_primitive_id[];

// Output to the Fragment Shader
flat out int v_ommatidium_id;

// Define the 6 vertices of a hexagon in screen space
const float HEX_RADIUS = 0.2;

// The vertices are ordered to form a hexagon using a TRIANGLE_STRIP
// The order v0, v1, v5, v2, v4, v3 (from a standard counter-clockwise set) creates a strip of 4 triangles that tile the shape
const vec2 hex_verts[6] = vec2[](
    vec2(HEX_RADIUS, 0.0), // v0
    vec2(HEX_RADIUS * 0.5, HEX_RADIUS * 0.866), // v1
    vec2(HEX_RADIUS * 0.5, HEX_RADIUS * -0.866), // v5
    vec2(HEX_RADIUS * -0.5, HEX_RADIUS * 0.866), // v2
    vec2(HEX_RADIUS * -0.5, HEX_RADIUS * -0.866), // v4
    vec2(HEX_RADIUS * -1.0, 0.0) // v3
);

void main() {

    // The center of the hexagon is the position of the incoming point primitive
    vec4 center_pos = gl_in[0].gl_Position;

    // The ID of an ommatidium is the ID passed from the first (and only) vertex of the incoming point primitive
    int ommatidium_id = v_primitive_id[0];

    // Emit 6 vertices to form the hexagon
    for (int i = 0; i < 6; i++) {
        // Pass the ommatidium ID to the fragment shader for all 6 new vertices
        v_ommatidium_id = ommatidium_id;

        // Calculate the position of this new vertex
        gl_Position = center_pos + vec4(hex_verts[i], 0.0, 0.0);

        EmitVertex();
    }

    // Finish the new primitive (the hexagon)
    EndPrimitive();
}