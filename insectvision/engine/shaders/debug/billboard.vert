#version 330 core

layout(location = 0) in vec3 aOffset;     // quad corner offset (in screen px)
layout(location = 1) in vec3 aColor;

uniform mat4 uView;
uniform mat4 uProj;
uniform vec3 uWorldPos;                    // anchor in world space
uniform float uScale;                      // pixel -> NDC factor

out vec3 vColor;

void main() {
    vColor = aColor;

    // Project anchor to clip space
    vec4 clipPos = uProj * uView * vec4(uWorldPos, 1.0);

    // Offset in NDC (screen-aligned)
    vec2 ndcOffset = aOffset.xy * uScale;
    clipPos.xy += ndcOffset * clipPos.w;

    gl_Position = clipPos;
}
