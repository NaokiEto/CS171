from OpenGL.GL import *
from OpenGL.GL.shaders import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys

def display():

    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()

    glColor3f(0.0, 0.0, 1.0)

    #gluLookAt(2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    glPushMatrix()

    glTranslatef(0.0,0.0,0.0)

    glRotatef(90.0,1.0,0.0,0.0)

    qobj = gluNewQuadric()

    #gluQuadricDrawStyle(qobj, GLU_LINE)

    gluCylinder(qobj, 0.2, 0.2, 0.5, 200, 150)

    glPopMatrix()
    #glFlush()
    glutSwapBuffers()

if __name__ == "__main__":

    glutInit(sys.argv)

    glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE | GLUT_DEPTH)

    glutInitWindowSize(500, 500)
    glutInitWindowPosition(300, 100)

    glutCreateWindow("This should work")

    # Enable depth-buffer test.
    #glEnable(GL_DEPTH_TEST)

    glClearColor(1.0, 1.0, 1.0, 1.0)



    glutDisplayFunc(display) 

    glutMainLoop()
