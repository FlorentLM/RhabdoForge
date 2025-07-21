#version 430 core

// Receive the ommatidium ID from the geometry shader
flat in int v_ommatidium_id;
out vec4 FragColor;

// SSBO containing the pre-calculated colors for all ommatidia
layout(std430, binding = 1) readonly buffer ColorDataBlock {
   vec4 u_ommatidia_colors[];
};

void main()
{
    // Simply look up the color for this ommatidium and output it
    FragColor = u_ommatidia_colors[v_ommatidium_id];
}