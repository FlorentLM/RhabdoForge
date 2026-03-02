#version 430 core

// Input: Vertex attributes from VBO
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 vertTexCoord;
layout (location = 2) in vec3 vertNormal;

// Output: Varyings to fragment shader
layout (location = 0) out vec2 fragTexCoord;
layout (location = 1) out vec3 fragWorldPos;
layout (location = 2) out vec3 fragWorldNormal;
layout (location = 3) out vec4 fragLightSpacePos;

// Uniforms
uniform mat4 camera;              // pre-combined P * V matrix
uniform mat4 model;               // model-to-world transform matrix
uniform mat4 light_space_matrix;  // light's P * V matrix (for shadow mapping)
uniform mat3 normal_matrix;       // transpose(inverse(mat3(model)))

void main()
{
    fragTexCoord = vertTexCoord;

    vec4 world_pos = model * vec4(position, 1.0);
    fragWorldPos = world_pos.xyz;
    fragWorldNormal = normalize(normal_matrix * vertNormal);
    fragLightSpacePos = light_space_matrix * world_pos;

    // Transform for column-major vertex is: P * V * M * v
    gl_Position = camera * world_pos;
}
