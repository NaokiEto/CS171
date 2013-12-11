uniform float t;
varying float u, v;

void main()
{
   //u += sin(t);
   //v += t;
   gl_FragColor = vec4(sin(40*u),sin(40*v),0.0,1.0);
}
