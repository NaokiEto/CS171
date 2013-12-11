uniform float t;
uniform float toggleMode;

const float pi = 3.1415926535;

varying float u, v;
varying float vertexX;
varying float vertexY;

void main()
{
    // This is the  mode for calculating normals from vertices
    if (toggleMode == 0.0)
    {
        // Convert from the [-5,5]x[-5,5] range provided into radians
        // between 0 and 2*theta
        u = (gl_Vertex.x + 5.0) / 10.0 * 2.0 * pi;
        v = (gl_Vertex.y + 5.0) / 10.0 * 2.0 * pi;

        float x = 4.50 * sqrt(gl_Vertex.x * gl_Vertex.x + gl_Vertex.y * gl_Vertex.y);
        float height = cos(gl_Vertex.x * gl_Vertex.x + gl_Vertex.y * gl_Vertex.y + 5.0*t);
        gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.x, 0.1 * height,gl_Vertex.y,1.0);


        float sinTerm = 0.1 * sin(gl_Vertex.x * gl_Vertex.x + gl_Vertex.y * gl_Vertex.y + 5.0 * t);
        float normX = 2.0 * gl_Vertex.x * sinTerm;
        float normY = 2.0 * gl_Vertex.y * sinTerm;

        float normalizedNormX = normX/sqrt(normX*normX + normY*normY + 1.0);
        float normalizedNormY = normY/sqrt(normX*normX + normY*normY + 1.0);
        gl_TexCoord[0].s = (normalizedNormX + 1.0)/2.0;
        gl_TexCoord[0].t = (normalizedNormY + 1.0)/2.0;
        gl_TexCoord[0].p = (1.0/(normX*normX + normY*normY + 1.0) + 1.0)/2.0;

        // modify GL_TexCoord by mapping u,v coords to [0,1]
        gl_TexCoord[1].s = u/(2.0 * pi);
        gl_TexCoord[1].t = v/(2.0 * pi);
    }

    // This mode is for the calculating of normals from fragments
	if (toggleMode == 1.0)
	{
        // Convert from the [-5,5]x[-5,5] range provided into radians
        // between 0 and 2*theta
        // convert the coordinates such that u and v are between 0 and 2pi
        u = (gl_Vertex.x + 5.0) / 10.0 * 2.0 * pi;
        v = (gl_Vertex.y + 5.0) / 10.0 * 2.0 * pi;

        // height function (for the waves)
        // let height equal cos(x^2 + y^2 + 5t)
        float height = cos(gl_Vertex.x * gl_Vertex.x + gl_Vertex.y * gl_Vertex.y + 5.0*t);
        gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.x, 0.1 * height,gl_Vertex.y,1.0);

        vertexX = gl_Vertex.x;
        vertexY = gl_Vertex.y;

        // 
        float sinTerm = 0.1 * sin(gl_Vertex.x * gl_Vertex.x + gl_Vertex.y * gl_Vertex.y + 5.0 * t);
        float normX = 2.0 * gl_Vertex.x * sinTerm;
        float normY = 2.0 * gl_Vertex.y * sinTerm;

        // 
        float normalizedNormX = normX/(normX*normX + normY*normY + 1.0);
        float normalizedNormY = normY/(normX*normX + normY*normY + 1.0);
        gl_TexCoord[0].s = (normalizedNormX + 1.0)/2.0;
        gl_TexCoord[0].t = (normalizedNormY + 1.0)/2.0;
        gl_TexCoord[0].p = (1.0/(normX*normX + normY*normY + 1.0) + 1.0)/2.0;

        // modify GL_TexCoord by mapping u,v coords to [0,1]
        gl_TexCoord[1].s = u/(2.0 * pi);
        gl_TexCoord[1].t = v;
    }

}
