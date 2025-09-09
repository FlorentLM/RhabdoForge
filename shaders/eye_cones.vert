#version 430 core

#include "commons.glsl"

layout(location = 0) in vec3 cone_vertex;

layout(std430, binding = 0) readonly buffer OmmatidiaInputBlock {
   Ommatidium ommatidia_data[];
};

uniform mat4 view;
uniform mat4 projection;
uniform mat4 eye_to_world;

uniform int projection_mode;  // 0 = Physical Layout, 1 = Acceptance angle
uniform float cone_length;
uniform float visualisation_scale;

out flat int v_index;
flat out int v_mode;

mat3 rmatFromDir(vec3 z) {
    vec3 x = normalize(cross( (abs(z.y) > 0.999) ? vec3(1,0,0) : vec3(0,1,0), z));
    vec3 y = cross(z, x);
    return mat3(x, y, z);
}

void main() {
    int i = gl_InstanceID;
    Ommatidium om = ommatidia_data[i];

    v_mode = projection_mode;

    vec3 O_world = (eye_to_world * vec4(om.origin.xyz * visualisation_scale, 1.0)).xyz;
    vec3 D_world = normalize((eye_to_world * vec4(om.direction.xyz, 0.0)).xyz);

    mat3 R = rmatFromDir(D_world);
    mat3 S;
    vec3 pos_world;

    if (projection_mode == 1) {
        // Cones represent the field of view, originating from the ommatidium's origin
        // they will overlap if p > 1.0

        // Calculate the horizontal and vertical half-angles independently
        float half_acceptance_angle_h = 0.5 * om.acceptance_angles.x; // H angle
        float half_acceptance_angle_v = 0.5 * om.acceptance_angles.y; // V angle
        float radius_h = cone_length * tan(half_acceptance_angle_h);
        float radius_v = cone_length * tan(half_acceptance_angle_v);

        // non-uniform scaling matrix to form the elliptical cone base
        S = mat3(radius_h, 0, 0, 0, radius_v, 0, 0, 0, cone_length);

        vec3 p_model = cone_vertex + vec3(0.0, 0.0, 1.0);
        vec3 offset = R * S * p_model;
        pos_world = O_world + offset;

    } else {
        // Cones are surfels on the eye's surface, sized to tile perfectly
        // The cone base is at the ommatidium's origin, and it points inwards

        float eye_radius_world = length(om.origin.xyz) * visualisation_scale;
        if (eye_radius_world < 0.001) eye_radius_world = 0.01 * visualisation_scale;

        float half_inter_angle_h = 0.5 * om.interommatidial_angles.x; // H angle
        float half_inter_angle_v = 0.5 * om.interommatidial_angles.y; // V angle
        float radius_h = eye_radius_world * sin(half_inter_angle_h);
        float radius_v = eye_radius_world * sin(half_inter_angle_v);

        float surfel_thickness = 0.0001;
        S = mat3(radius_h, 0, 0, 0, radius_v, 0, 0, 0, surfel_thickness);

        // The canonical cone's base is at z=0 and it extends to z=-1.
        // Placing it at O_world puts its base on the surface, pointing inwards.
        vec3 offset = R * S * cone_vertex;

        pos_world = O_world + offset;
    }

    gl_Position = projection * view * vec4(pos_world, 1.0);
    v_index = i;
}