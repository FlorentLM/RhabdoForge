#version 430 core

layout (location = 0) in vec3 a_ommatidium_dir;

uniform float u_vis_scale = 0.9;

// Interface block for the Geometry Shader
out gl_PerVertex {
    vec4 gl_Position;
};

// Custom output variable for the Geometry Shader
flat out int v_primitive_id;

void main()
{
    // Pass the ID of the vertex to the next stage
    v_primitive_id = gl_VertexID;

    // Visualization layout math
    vec2 xy_dir = a_ommatidium_dir.xy;
    float len = length(xy_dir);
    vec2 flat_pos;

    // Handle the pole case to avoid division by zero
    if (len < 0.0001) {
        flat_pos = vec2(0.0, 0.0);
    } else {
        // And project the rest onto 2d
        flat_pos = (xy_dir / len) * (1.0 - abs(a_ommatidium_dir.z));
    }

    gl_Position = vec4(flat_pos * u_vis_scale, 0.0, 1.0);
}