uniform float t;
uniform float toggleMode;
varying float u, v;

varying float vertexX;
varying float vertexY;

uniform sampler2D sky;
uniform sampler2D leaf;

const float pi = 3.1415926535;

void main()
{
    // This is for calculating the normals from the vertices
    if (toggleMode == 0.0)
    {
        vec4 leafC = texture2D(leaf, gl_TexCoord[1].st);
        vec4 skyC = texture2D(sky, gl_TexCoord[0].st);

        float a = leafC.a;

        float b = gl_FragCoord.x;

        float sinTerm = 0.1 * sin(gl_FragCoord.x * gl_FragCoord.x + gl_FragCoord.y * gl_FragCoord.y + 5.0 * t);
        float normX = 2.0 * gl_FragCoord.x * sinTerm;
        float normY = 2.0 * gl_FragCoord.y * sinTerm;

        float normalizedNormX = normX/(normX*normX + normY*normY);
        float normalizedNormY = normY/(normX*normX + normY*normY);

        vec4 leafC1 = texture2D(leaf, vec2(u/(2.0*pi), v/(2.0*pi)));
        vec4 skyC1 = texture2D(sky, vec2((normalizedNormX + 1.0)/2.0, (normalizedNormY + 1.0)/2.0));
        float a1 = leafC1.a;

        vec4 result = (a * leafC + (1.0-a) * skyC);

        vec4 result1 = (a1 * leafC1 + (1.0-a1) * skyC1);
        gl_FragColor = result;
    }

    // This is for calculating the normals from the fragments
	if (toggleMode == 1.0)
	{

	    float U = vertexX;
	    float V = vertexY;

	
	    float sinTerm = 0.1 * sin(U * U + V * V + 5.0 * t);
	    float normX = 2.0 * U * sinTerm;
	    float normY = 2.0 * V * sinTerm;

	    float normalizedNormX = normX/sqrt(normX*normX + normY*normY + 1);
	    float normalizedNormY = normY/sqrt(normX*normX + normY*normY + 1);

	    //texture_coordinates = vec2(gl_MultiTexCoord0);

       vec4 leafC1 = texture2D(leaf, vec2(u/(2.0*pi), v/(2.0*pi)));

       vec4 skyC1 = texture2D(sky, vec2((normalizedNormX + 1.0)/2.0, (normalizedNormY + 1.0)/2.0));

       float a1 = leafC1.a;

       vec4 result = (a1 * leafC1 + (1.0-a1) * skyC1);
       gl_FragColor = result;
	   
    }
}
