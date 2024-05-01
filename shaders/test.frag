#version 150

#define H 0.032
#define S ((3./2.) * H/sqrt(3.))

uniform sampler2D tex;
in vec2 fragTexCoord;

out vec4 finalColor;

vec2 hexCoord(vec2 hexIndex) {
	float i = hexIndex.x;
	float j = hexIndex.y;
	vec2 r;
	r.x = i * S;
	r.y = j * H + (mod(i,2.0)) * H/2.;
	return r;
}

vec2 hexIndex(vec2 coord) {
	vec2 r;
	float x = coord.x;
	float y = coord.y;
	float it = float(floor(x/S));
	float yts = y - (mod(it,2.0)) * H/2.;
	float jt = float(floor((1./H) * yts));
	float xt = x - it * S;
	float yt = yts - jt * H;
	float deltaj = (yt > H/2.)? 1.0:0.0;
	float fcond = S * (2./3.) * abs(0.5 - yt/H);

	if (xt > fcond) {
		r.x = it;
		r.y = jt;
	}
	else {
		r.x = it - 1.0;
		r.y = jt - (mod(r.x,2.0)) + deltaj;
	}

	return r;
}

void main() {
    vec2 uv = fragTexCoord.xy;
	vec2 hexIx = hexIndex(uv);
	vec2 hexXy = hexCoord(hexIx);

    finalColor = texture(tex, hexXy);
}