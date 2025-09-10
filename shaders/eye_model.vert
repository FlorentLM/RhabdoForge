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

out flat uint v_index;
flat out uint v_mode;
flat out uint v_eye_id;

mat3 rmatFromDir(vec3 z) {
    vec3 x = normalize(cross( (abs(z.y) > 0.999) ? vec3(1,0,0) : vec3(0,1,0), z));
    vec3 y = cross(z, x);
    return mat3(x, y, z);
}

void main() {
    int i = gl_InstanceID;
    Ommatidium om = ommatidia_data[i];

    v_mode = projection_mode;

    v_eye_id = unpack_eye_id(om);

    vec3 O_world = (eye_to_world * vec4(om.origin.xyz * visualisation_scale, 1.0)).xyz;
    vec3 D_world = normalize((eye_to_world * vec4(om.direction.xyz, 0.0)).xyz);

    // This matrix orients the cone to point in the ommatidium's direction
    mat3 R_world = rmatFromDir(D_world);

    float tilt = om.tilt;
    float s = sin(tilt);
    float c = cos(tilt);

    mat3 R_tilt = mat3(c, -s,  0,
                       s,  c,  0,
                       0,  0,  1);
    mat3 S; // scaling matrix
    vec3 pos_world;

    if (projection_mode == 1) {

        // Cones represent the field of view, based on acceptance angles
        float half_acceptance_minor = 0.5 * om.acceptance_angles.x;
        float half_acceptance_major = 0.5 * om.acceptance_angles.y;
        float radius_minor = cone_length * tan(half_acceptance_minor);
        float radius_major = cone_length * tan(half_acceptance_major);

        // Create non-uniform scaling matrix for ellipse base
        S = mat3(radius_minor, 0, 0, 0, radius_major, 0, 0, 0, cone_length);

        vec3 p_model = cone_vertex + vec3(0.0, 0.0, 1.0);

        // scale -> tilt -> world orientation
        vec3 offset = R_world * R_tilt * S * p_model;
        pos_world = O_world + offset;

    } else {

        // Cones are surfels on the eye's surface, based on interommatidial angles
        float eye_radius_world = length(om.origin.xyz) * visualisation_scale;
        if (eye_radius_world < 0.001) eye_radius_world = 0.01 * visualisation_scale;

        float half_inter_angle_minor = 0.5 * om.interommatidial_angles.x;
        float half_inter_angle_major = 0.5 * om.interommatidial_angles.y;
        float radius_minor = eye_radius_world * sin(half_inter_angle_minor);
        float radius_major = eye_radius_world * sin(half_inter_angle_major);

        float surfel_thickness = 0.0001;

        // Create non-uniform scaling matrix for ellipse base
        S = mat3(radius_minor, 0, 0, 0, radius_major, 0, 0, 0, surfel_thickness);

        // scale -> tilt -> world orientation
        vec3 offset = R_world * R_tilt * S * cone_vertex;
        pos_world = O_world + offset;
    }

    gl_Position = projection * view * vec4(pos_world, 1.0);
    v_index = i;
}