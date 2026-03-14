#version 430 core

// Ultra simple shadow depth pass fragment shader
// For meshes this is essentially a no-op (depth is written automatically)
// and for point clouds this just discards fragments outside a circle to get round splats

uniform bool is_point_cloud;

void main()
{
    if (is_point_cloud) {
        vec2 coord = gl_PointCoord - vec2(0.5);
        if (dot(coord, coord) > 0.25) {
            discard;
        }
    }
    // Depth is written by the fixed-function pipeline
    // gl_FragDepth = gl_FragCoord.z;  (implicit)
}
