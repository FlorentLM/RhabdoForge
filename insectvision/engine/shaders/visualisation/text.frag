#version 330 core

in vec2 TexCoords;
out vec4 color;

uniform sampler2D fontAtlas;
uniform vec4 textColor;

void main() {
    // Sample the fonts atlas (which we stored in the red channel)
    float alpha = texture(fontAtlas, TexCoords).a;

    // Discard fragment if it's fully transparent to avoid artifacts
    if(alpha < 0.01)
        discard;

    // Combine the uniform color with the sampled alpha
    color = vec4(textColor.rgb, textColor.a * alpha);
}