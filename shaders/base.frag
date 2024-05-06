#version 150

uniform sampler2D tex;
uniform sampler2D tex2;

in vec2 fragTexCoord;

out vec4 finalColor;

void main() {
    finalColor = texture(tex2, fragTexCoord);
//    finalColor = mix(texture(tex, fragTexCoord), texture(tex2, fragTexCoord), 0.2);
}