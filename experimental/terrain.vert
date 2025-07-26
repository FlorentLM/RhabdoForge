#version 330

uniform mat4 camera;
uniform sampler2D Heightmap;

in vec2 pos;

out vec2 v_text;

void main() {
	vec4 vertex = vec4(pos - 0.5, texture(Heightmap, pos).r * 0.2, 1.0);
	gl_Position = camera * vertex;
	v_text = pos;
}