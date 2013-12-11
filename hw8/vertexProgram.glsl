uniform float t;
const float pi = 3.1415926535;

varying float u, v;

void main()
{
   // Convert from the [-5,5]x[-5,5] range provided into radians
   // between 0 and 2*theta
   u = (gl_Vertex.x + 5.0) / 10.0 * 2 * pi;
   v = (gl_Vertex.y + 5.0) / 10.0 * 2 * pi;
   float r = sin(4*u+t)/4+1;
   float R = cos(6*cos(u)-t)/4+3;

   float a = R+r*cos(v);
   vec3 world = vec3(a*cos(u), a*sin(u), r*sin(v));

   gl_Position = gl_ModelViewProjectionMatrix * vec4(world,1.0);
}
