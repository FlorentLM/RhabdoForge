#version 430 core

#include "commons.glsl"

layout(location = 0) in vec3 model_vertex;

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
out vec3 v_world_normal;

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

    mat3 R_world = rmatFromDir(D_world);
    float tilt = om.tilt;
    mat3 R_tilt = mat3(cos(tilt), -sin(tilt), 0,
                       sin(tilt),  cos(tilt), 0,
                       0,          0,         1);
    mat3 S;

    if (projection_mode == 1) {
        float half_acceptance_minor = 0.5 * om.acceptance_angles.x;
        float half_acceptance_major = 0.5 * om.acceptance_angles.y;
        float radius_minor = cone_length * tan(half_acceptance_minor);
        float radius_major = cone_length * tan(half_acceptance_major);

        S = mat3(radius_minor, 0, 0, 0, radius_major, 0, 0, 0, cone_length);

        vec3 p_model = model_vertex + vec3(0.0, 0.0, 1.0);

        vec3 offset = R_world * R_tilt * S * p_model;

    }
    else {
        float eye_radius_world = length(om.origin.xyz) * visualisation_scale;
        if (eye_radius_world < 0.001) eye_radius_world = 0.01 * visualisation_scale;

        float half_inter_angle_minor = 0.5 * om.interommatidial_angles.x;
        float half_inter_angle_major = 0.5 * om.interommatidial_angles.y;
        float radius_minor = eye_radius_world * sin(half_inter_angle_minor);
        float radius_major = eye_radius_world * sin(half_inter_angle_major);

        // Give the ovoid a nice height proportional to its base radius
        float ovoid_height = (radius_minor + radius_major) * 0.15;   // visual size factor

        S = mat3(radius_minor, 0, 0, 0, radius_major, 0, 0, 0, ovoid_height);

        // The hemisphere model is a unit hemisphere along +Z so to scale, tilt, and orient it
        vec3 offset = R_world * R_tilt * S * model_vertex;
    }

    mat3 model_transform = R_world * R_tilt * S;

    vec3 pos_world = O_world + (model_transform * model_vertex);
    // normals are just the model vertices for a unit sphere
    vec3 model_normal = model_vertex;

    // inverse transpose still needed for the transformation to handle the non-uniform scale
    mat3 normal_matrix = transpose(inverse(model_transform));
    v_world_normal = normalize(normal_matrix * model_normal);

    gl_Position = projection * view * vec4(pos_world, 1.0);
    v_index = i;
}