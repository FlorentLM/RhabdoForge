#version 150

uniform sampler2D texture1;
uniform sampler2D texture2;

in vec2 fragTexCoord;

out vec4 finalColor;

void main() {
    finalColor = mix(texture(texture1, fragTexCoord), texture(texture2, fragTexCoord), 0.2);
}