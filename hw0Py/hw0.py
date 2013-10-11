from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys

# This function gets called every time the window needs to be updated
# i.e. after being hidden by another window and brought back into view,
# or when the window's been resized.
# You should never call this directly, but use glutPostRedisply() to tell
# GLUT to do it instead.

def redraw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    gluSphere(quad, 2.0, 256, 256)
    glutSwapBuffers()
    
# GLUT calls this function when the windows is resized.
# All we do here is change the OpenGL viewport so it will always draw in the
# largest square that can fit the window

def resize(w, h):
    if h == 0:
        h = 1
    
    # ensuring our windows is a square
    if w > h:
        w = h
    else:
        h = w
        
    
    # reset the current viewport and perspective transformation
    glViewport(0, 0, w, h)
    
    # tell GLUT to call the redrawing function, in this case redraw()
    glutPostRedisplay()

# GLUT calls this function when a key is pressed. Here we just quit when ESC or
# 'q' is pressed.
def keyfunc(key, x, y):
    if key == 27 or key == 'q' or key == 'Q':
        exit(0)

#
# Sets up an OpenGL light.  This only needs to be called once
# and the light will be used during all renders.
#
def initLights():
    amb      = [ 1.0, 1.0, 1.0, 1.0 ]
    diff     = [ 1.0, 1.0, 1.0, 1.0 ]
    spec     = [ 1.0, 1.0, 1.0, 1.0 ]
    lightpos = [ 2.0, 2.0, 5.0, 1.0 ]

    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, amb)
    glLightfv(GL_LIGHT0, GL_AMBIENT, amb)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diff)
    glLightfv(GL_LIGHT0, GL_SPECULAR, spec)
    glLightfv(GL_LIGHT0, GL_POSITION, lightpos)
    glEnable(GL_LIGHT0)
    
    # Turn on lighting.  You can turn it off with a similar call to
    # glDisable().
    glEnable(GL_LIGHTING);


#
# Sets the OpenGL material state.  This is remembered so we only need to
# do this once.  If you want to use different materials, you'd need to do this
# before every different one you wanted to use.
#

def initMaterial():
    emit = [0.0, 0.0, 0.0, 1.0]
    amb  = [0.0, 0.0, 0.0, 1.0]
    diff = [0.0, 0.0, 1.0, 1.0]
    spec = [1.0, 1.0, 1.0, 1.0]
    shiny = 20.0

    glMaterialfv(GL_FRONT, GL_AMBIENT, amb)
    glMaterialfv(GL_FRONT, GL_DIFFUSE, diff)
    glMaterialfv(GL_FRONT, GL_SPECULAR, spec)
    glMaterialfv(GL_FRONT, GL_EMISSION, emit)
    glMaterialfv(GL_FRONT, GL_SHININESS, shiny)


#
# Set up OpenGL state.  This does everything so when we draw we only need to
# actually draw the sphere, and OpenGL remembers all of our other settings.
#

def initGL():
    # Tell openGL to use Gouraud shading:
    glShadeModel(GL_SMOOTH)
    
    # Enable back-face culling:
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

    # Enable depth-buffer test.
    glEnable(GL_DEPTH_TEST)
    
    # Set up projection and modelview matrices ("camera" settings) 
    # Look up these functions to see what they're doing.
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glFrustum(-0.5, 0.5, -0.5, 0.5, 1, 10)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(5, 5, 5, 0, 0, 0, 1, 0, 0)

    # set light parameters
    initLights()

    # set material parameters
    initMaterial()

    # initialize the "quadric" used by GLU to render high-level objects.
    global quad
    quad = gluNewQuadric()
    gluQuadricOrientation(quad, GLU_OUTSIDE)


#
# Main entrance point, obviously.
# Sets up some stuff then passes control to glutMainLoop() which never
# returns.
#

def main():
    glutInit(sys.argv)
    # Get a double-buffered, depth-buffer-enabled window, with an
    # alpha channel.
    # These options aren't really necessary but are here for examples.
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

    glutInitWindowSize(800, 800)
    glutInitWindowPosition(300, 100)

    glutCreateWindow("CS171 HW0")
    
    initGL()

    # set up GLUT callbacks.
    glutDisplayFunc(redraw)
    glutReshapeFunc(resize)
    glutKeyboardFunc(keyfunc)

    glutMainLoop()

    return 1
    
# run the script
if __name__ == "__main__":
    main()
