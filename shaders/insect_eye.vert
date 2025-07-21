#version 430 core

// just need the pre-calculated coordinates
layout (location = 0) in vec2 a_ommatidia_coords; // Input is (longitude, latitude) in radians

// Interface block for the Geometry Shader
out gl_PerVertex {
    vec4 gl_Position;
};

// Custom output variable for the Geometry Shader
flat out int v_primitive_id;

const float PI = 3.14159265359;

void main()
{
    // Pass the ID of the vertex to the next stage
    v_primitive_id = gl_VertexID;

    // Map longitude and latitude (already in radians) to screen/clip space [-1, 1]
    float screen_x = a_ommatidia_coords.x / PI; // a_pano_coords.x is longitude
    float screen_y = a_ommatidia_coords.y / (PI / 2.0); // a_pano_coords.y is latitude

    // Set the final position
    gl_Position = vec4(screen_x, screen_y, 0.0, 1.0);
}