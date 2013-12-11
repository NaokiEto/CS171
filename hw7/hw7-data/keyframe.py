from OpenGL.GL import *
from OpenGL.GL.shaders import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys

def display():

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glColor3f(1.0, 0.0, 1.0)
    glBegin(GL_LINES)

    #qobj = gluNewQuadric()

    #gluCylinder(qobj, 1.0, 1.0, 0.4, 1, 16)

    #glutSwapBuffers()
      

if __name__ == "__main__":

    glutInit(sys.argv)

    # x dimension size
    xRes = int(sys.argv[1])

    # y dimension size
    yRes = int(sys.argv[2])

    # iv file name to input
    ivFile = sys.argv[3]

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

    glutInitWindowSize(xRes, yRes)
    glutInitWindowPosition(300, 100)

    glutCreateWindow("CS171 HW6")

    glutDisplayFunc(display) 

    glutMainLoop()
