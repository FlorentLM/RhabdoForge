#version 430 core

#include "commons.glsl"

layout(location = 0) in vec3 model_vertex;

layout(std430, binding = 0) readonly buffer ReceptorsInputBlock { ReceptorData receptor_data[]; };
layout(std430, binding = 1) readonly buffer LensDataBlock { LensData lenses_data[]; };

// Uniforms
uniform mat4 view;
uniform mat4 projection;
uniform mat4 eye_to_world;

uniform int projection_mode;  // 0 = Physical Layout, 1 = Acceptance angle
uniform float cone_length;
uniform float visualisation_scale;

out flat uint v_index;
flat out uint v_mode;
flat out uint v_eye_id;
out vec3 v_world_normal;

mat3 rmatFromDir(vec3 z) {
    vec3 x = normalize(cross( (abs(z.y) > 0.999) ? vec3(1,0,0) : vec3(0,1,0), z));
    vec3 y = cross(z, x);
    return mat3(x, y, z);
}

void main() {
    int i = gl_InstanceID;
    ReceptorData rcpt = receptor_data[i];

    uint lens_id = unpack_lens_id(rcpt);
    LensData lens = lenses_data[lens_id];

    v_mode = projection_mode;
    v_eye_id = unpack_eye_id(rcpt);

    vec3 P_world = (eye_to_world * vec4(rcpt.position * visualisation_scale, 1.0)).xyz;
    vec3 D_world = normalize((eye_to_world * vec4(rcpt.direction, 0.0)).xyz);

    mat3 R_world = rmatFromDir(D_world);
    mat3 R_tilt;
    mat3 S;

    if (projection_mode == 1) {
        float tilt = rcpt.acc_tilt;
        R_tilt = mat3(cos(tilt), -sin(tilt), 0,
                      sin(tilt),  cos(tilt), 0,
                      0,          0,         1);

        float half_acc_minor = 0.5 * rcpt.acc_axes.x;
        float half_acc_major = 0.5 * rcpt.acc_axes.y;
        float radius_minor = cone_length * tan(half_acc_minor);
        float radius_major = cone_length * tan(half_acc_major);

        S = mat3(radius_minor, 0, 0, 0, radius_major, 0, 0, 0, cone_length);
        vec3 p_model = model_vertex + vec3(0.0, 0.0, 1.0);
        vec3 offset = R_world * R_tilt * S * p_model;
    }
    else {
        float tilt = lens.tilt;
        R_tilt = mat3(cos(tilt), -sin(tilt), 0,
                      sin(tilt),  cos(tilt), 0,
                      0,          0,         1);

        float eye_radius_world = length(rcpt.position) * visualisation_scale;
        if (eye_radius_world < 0.001) eye_radius_world = 0.01 * visualisation_scale;

        float half_ioa_minor = 0.5 * lens.ioa_axes.x;
        float half_ioa_major = 0.5 * lens.ioa_axes.y;
        float radius_minor = eye_radius_world * sin(half_ioa_minor);
        float radius_major = eye_radius_world * sin(half_ioa_major);

        float ovoid_height = (radius_minor + radius_major) * 0.15;

        S = mat3(radius_minor, 0, 0, 0, radius_major, 0, 0, 0, ovoid_height);
        vec3 offset = R_world * R_tilt * S * model_vertex;
    }

    mat3 model_transform = R_world * R_tilt * S;
    vec3 pos_world = P_world + (model_transform * model_vertex);

    mat3 normal_matrix = transpose(inverse(model_transform));
    v_world_normal = normalize(normal_matrix * model_vertex);

    gl_Position = projection * view * vec4(pos_world, 1.0);
    v_index = i;
}