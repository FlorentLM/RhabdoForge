#version 430 core

layout(std430, binding = 1) readonly buffer ColorDataBlock {
    vec4 final_rgba[];
};

in flat uint v_index;
flat in uint v_mode;
flat in uint v_eye_id;
in vec3 v_world_normal;

uniform vec3 light_dir;

out vec4 FragColor;

uniform float albedo_boost = 1.0;

const vec4 EYE_COLORS[8] = vec4[](
    vec4(1.0, 0.2, 0.2, 0.25), // 0: Red
    vec4(0.2, 0.5, 1.0, 0.25), // 1: Blue
    vec4(0.2, 1.0, 0.2, 0.25), // 2: Green
    vec4(1.0, 1.0, 0.2, 0.25), // 3: Yellow
    vec4(0.2, 1.0, 1.0, 0.25), // 4: Cyan
    vec4(1.0, 0.2, 1.0, 0.25), // 5: Magenta
    vec4(1.0, 0.6, 0.2, 0.25), // 6: Orange
    vec4(0.6, 0.2, 1.0, 0.25)  // 7: Purple
);

void main() {
    // Acceptance Mode: Colour according to eye ID
    if (v_mode == 1u) {
        FragColor = EYE_COLORS[v_eye_id];
    }
    // Physical Layout Mode: Ommatidium color + shading
    else {
        // Base color from the simulation buffer
        vec3 base_rgb = final_rgba[v_index].rgb * albedo_boost;
//
//        // Lighting parameters
//        float ambient = 0.35; // ambient light intensity (lower = darker shadows)
//        vec3 light_dir = normalize(light_dir);
//        vec3 normal = normalize(v_world_normal); // ensure normal is unit length
//        float diffuse_intensity = max(0.0, dot(normal, light_dir));
//
//        // Combine base color with lighting
//        vec3 lit_rgb = base_rgb * (ambient + diffuse_intensity * (1.0 - ambient));
        vec3 lit_rgb = base_rgb;
        FragColor = vec4(clamp(lit_rgb, 0.0, 1.0), 1.0);
    }
}