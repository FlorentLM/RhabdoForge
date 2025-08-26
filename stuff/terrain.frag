#version 330

uniform sampler2D Heightmap;

uniform sampler2D Color1;
uniform sampler2D Color2;

uniform sampler2D Cracks;
uniform sampler2D Darken;

in vec2 v_text;

out vec4 finalColor;

void main() {
	float height = texture(Heightmap, v_text).r;
	float border = smoothstep(0.5, 0.7, height);

	vec3 color1 = texture(Color1, v_text * 7.0).rgb;
	vec3 color2 = texture(Color2, v_text * 6.0).rgb;

	vec3 color = color1 * (1.0 - border) + color2 * border;

	color *= 0.8 + 0.2 * texture(Darken, v_text * 3.0).r;
	color *= 0.5 + 0.5 * texture(Cracks, v_text * 5.0).r;
	color *= 0.5 + 0.5 * height;

	finalColor = vec4(color, 1.0);
}