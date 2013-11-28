#!/usr/bin/python

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import pyparsing as pp
import sys
import os
import numpy as np
from math import*

# GLUT calls this function when a key is pressed. Here we just quit when ESC or
# 'q' is pressed.
def keyfunc(key, x, y):
    if key == 27 or key == 'q' or key == 'Q':
        exit(0)

def redraw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glEnable(GL_POINT_SMOOTH)

    glPointSize(5.0)
    glColorPointer(3, GL_FLOAT, 0, colorpoints)
    glVertexPointer(3, GL_FLOAT, 0, origpoints)
    glDrawArrays(GL_POINTS, 0, len(origpoints))
    glDrawArrays(GL_LINE_STRIP, 0, len(origpoints))

    glPointSize(1.0)
    glDisable(GL_POINT_SMOOTH)

    glColorPointer(3, GL_FLOAT, 0, pointscurve)
    glVertexPointer(3, GL_FLOAT, 0, colorscurve)
    glDrawArrays(GL_LINE_STRIP, 0, len(pointscurve))

    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)

    glutSwapBuffers()

if __name__ == "__main__":

    glutInit(sys.argv)

    # x dimension size
    xRes = int(sys.argv[1])

    # y dimension size
    yRes = int(sys.argv[2])

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

    glutInitWindowSize(xRes, yRes)
    glutInitWindowPosition(300, 100)

    glutCreateWindow("CS171 HW5")
	# Dark blue background
    glClearColor(0.0, 0.0, 0.0, 0.0)

    # this array is for rendering the 4 big original control points
    origpoints = np.array([[0.9, 0.9, 0.0],
                           [0.9, -0.9, 0.0],
                           [-0.9, -0.9, 0.0],
                           [-0.9, 0.9, 0.0]])

    # this array is for the adding of additional control points
    ctrlpoints = np.array([[0.9, 0.9, 0.0],
                           [0.9, -0.9, 0.0],
                           [-0.9, -0.9, 0.0],
                           [-0.9, 0.9, 0.0]])

    # this array is for the coloring of the 4 big original control points
    # (all white color)
    colorpoints = np.array([[1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0]])

    colorToRender = np.array([[1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0]])

    pointscurve = np.zeros((100, 3))

    colorscurve = np.zeros((100, 3))

    # add 0's and 1's
    knotvector = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # initialize the B to all 0's (the amount follow the for-loops)
    B = np.zeros((4, 5))


    # the 2 is for debugging
    for iteration in range(3):
        u = float(iteration)/float(100)
        print u

        #B0 = (1 - u**3)/6.0
        #B1 = (3*u**3 - 6*u**2 + 4)/6.0
        #B2 = (-3*u**3 + 3*u**2 + 3*u + 1)/6.0
        #B3 = u**3 / 6.0

        #totalpts = 4 + iteration
        for k in range(3, -1, -1):
            # Initialization of Cox-deBoor's recursion formula
            if (u >= knotvector[k] and u <= knotvector[k + 1]):
                B[k][1] = 1.0
                print "yeah! ", k
            else:
                B[k][1] = 0.0

        for k in range(3, -1, -1):

            # fixed to up to 4 for cubic
            for d in range(2, 5):

                # Cox-deBoor's Recursion (page 443 of Computer Graphics book)
                if (float(knotvector[k + d - 1] - knotvector[k]) != 0.0):
                    B[k][d] += (float(u) - float(knotvector[k]))/float(knotvector[k + d - 1] - knotvector[k]) * float(B[k][d-1])
                    print "the multiplied recursive is: ", float(B[k][d-1])
                    print "the numerator is: ", float(u) - knotvector[k]
                    print "first turn: ", B[k][d]

                if (float(knotvector[k + d] - knotvector[k + 1]) != 0.0):
                    B[k][d] += (float(knotvector[k + d]) - float(u))/float(knotvector[k + d] - knotvector[k + 1]) * float(B[k + 1][d - 1])
                    print "the multiplied recursive is: ", float(B[k+1][d-1])
                    print "second turn: ", B[k][d]

                print "the new B at k: ", k, ", and d: ", d, " is: ", B[k][d]
        P = 0

        # i (j in this case) only ranges from 0 to 4 (the number of control points, which is n) in the 
        # summation formula
        for j in range(4):
            #print B[j][e]
            #print ctrlpoints[j]
            P += float(B[j][4]) * ctrlpoints[j]
            print "this is: ", B[j][4]
        print P

        pointscurve[iteration][0] = P[0]
        pointscurve[iteration][1] = P[1]
        pointscurve[iteration][2] = P[2]

        colorscurve[iteration][0] = 1.0
        colorscurve[iteration][1] = 1.0
        colorscurve[iteration][2] = 1.0
        #ctrlpoints = np.append(ctrlpoints, [P], axis = 0)
        #colorToRender = np.append(colorToRender, [[1.0, 1.0, 1.0]], axis = 0)
        #print ctrlpoints
        #ctrlpoints = np.append(ctrlpoints, [P], axis = 0)
        #print knotvector
        #for j in range(len(knotvector)):
        #    if (j >= 4 and j <= totalpts):
        #        knotvector = np.insert(knotvector, j, j - 4 + 1)
        #print ctrlpoints
        print knotvector

    #print pointsToRender
    # set up GLUT callbacks.
    glutDisplayFunc(redraw)
    glutKeyboardFunc(keyfunc)

    glutMainLoop()
