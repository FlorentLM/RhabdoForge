#version 430 core

uniform sampler2D texture1;

in vec2 fragTexCoord;

out vec4 finalColor;

void main() {
    finalColor = texture(texture1, fragTexCoord);
}