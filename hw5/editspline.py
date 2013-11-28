#!/usr/bin/python

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import pyparsing as pp
import sys
import os
import numpy as np
from math import*

def mouseCB(button, state, x, y):
    mouseX = x
    mouseY = y
    global mouseMiddleDown 
    mouseMiddleDown = False
    global mouseLeftDown 
    mouseLeftDown = False
    global mouseRightDown 
    mouseRightDown = False

    global shiftPressed
    shiftPressed = False

    if(button == GLUT_LEFT_BUTTON):
        if(state == GLUT_DOWN):
            #print "the x-coordinate pixel is: ", x
            #print "the y-coordinate pixel is: ", y
            mouseLeftDown = True
            global lastRotX
            lastRotX = (x/float(xRes) - 0.5)*2.0

            global lastRotY
            lastRotY = (1.0 - y/float(yRes) - 0.5)*2.0

        elif(state == GLUT_UP):
            mouseLeftDown = False

    elif(button == GLUT_RIGHT_BUTTON):
        if(state == GLUT_DOWN):
            mouseRightDown = True
            print "mouse Right is down!"
        elif(state == GLUT_UP):
            mouseRightDown = False
            print "wut, this should not be happening"

def mouseMotionCB(x, y):

    mouseX = (x/float(xRes) - 0.5)*2.0
    mouseY = (1.0 - y/float(yRes) - 0.5)*2.0

    print "the point the mouse clicks on is: ", mouseX, ", ", mouseY

    # for left button down, be able to move the control points around
    if(mouseLeftDown):
        # find the index of the newpoint in ctrlpoints
        index = 2*len(ctrlpoints)
        i = 0

        while (index == 2*len(ctrlpoints) and i < len(ctrlpoints)):
            print "whoa, ", i
            if (abs(mouseX - ctrlpoints[i][0]) < 0.01 and abs(mouseY - ctrlpoints[i][1]) < 0.01):
                index = i
                print "hit occurred"
                print index
                ctrlpoints[index][0] = mouseX
                ctrlpoints[index][1] = mouseY

                origpoints[index][0] = mouseX
                origpoints[index][1] = mouseY

                print "the ctrlpoints are: ", ctrlpoints

                print "the origpoints are: ", origpoints
                #pointscurve = np.zeros((101, 3))

                # knot-vector of 4 0's and 4 1's
                knotvector = np.array([0, 0, 0, 0, 1, 1, 1, 1])


                # the 2 is for debugging
                for iteration in range(101):

                    # initialize the B to all 0's (the amount follow the for-loops)
                    B = np.zeros((4, 5))

                    u = float(iteration)/float(100)
                    #print u

                    # amount of control points
                    for k in range(4):
                        # Initialization of Cox-deBoor's recursion formula
                        if (u >= knotvector[k] and u <= knotvector[k + 1]):
                            B[k][1] = 1.0
                            #print "yeah! ", k
                        else:
                            B[k][1] = 0.0

                    # summation from 3 to 0, decrement 1, control points
                    for k in range(3, -1, -1):

                        # fixed to up to 4 for cubic (2, 3, 4), degree
                        for d in range(2, 5):

                            #print "the initial B[k][d] is: ", B[k][d]
                            # Cox-deBoor's Recursion (page 443 of Computer Graphics book)
                            if (float(knotvector[k + d - 1] - knotvector[k]) != 0.0):
                                B[k][d] += (float(u) - float(knotvector[k]))/\
                                           float(knotvector[k + d - 1] - knotvector[k]) * float(B[k][d-1])
                                #print "the multiplied recursive is: ", float(B[k][d-1])
                                #print "the numerator is: ", float(u) - knotvector[k]
                                #print "first turn: ", B[k][d]

                            if (float(knotvector[k + d] - knotvector[k + 1]) != 0.0):
                                #print "the knotvector is: ", knotvector[k+d]
                                #print "V2: ", knotvector[k+1]
                                #print knotvector
                                B[k][d] += (float(knotvector[k + d]) - float(u))/float(knotvector[k + d]\
                                            - knotvector[k + 1]) * float(B[k + 1][d - 1])
                                #print "the numerator is: ", float(knotvector[k + d]) - float(u)
                                #print "the multiplied recursive is: ", float(B[k+1][d-1])
                                #print "second turn: ", B[k][d]

                            #print "the new B at k: ", k, ", and d: ", d, " is: ", B[k][d]
                    newP = 0

                    # i (j in this case) only ranges from 0 to 4 (the number of control points, which is n) in the 
                    # summation formula
                    for j in range(4):
                        #print B[j][e]
                        #print ctrlpoints[j]
                        newP += float(B[j][4]) * ctrlpoints[j]
                        #print "this is: ", B[j][4]
                    #print P

                    pointscurve[iteration][0] = newP[0]
                    pointscurve[iteration][1] = newP[1]
                    pointscurve[iteration][2] = 0.0

            i += 1
    print "wut", pointscurve[0]
    #print "the allegedly changed ctrlpoints are: ", ctrlpoints
    # tell GLUT to call the redrawing function, in this case redraw()
    glutPostRedisplay()

# GLUT calls this function when a key is pressed. Here we just quit when ESC or
# 'q' is pressed.
def keyfunc(key, x, y):
    if key == 27 or key == 'q' or key == 'Q':
        exit(0)

def redraw():
    print "done!"
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glEnable(GL_POINT_SMOOTH)

    glPointSize(5.0)
    glColorPointer(3, GL_FLOAT, 0, colorpoints)
    glVertexPointer(3, GL_FLOAT, 0, origpoints)
    glDrawArrays(GL_POINTS, 0, len(origpoints))

    print "the 4 points are: ", origpoints
    glDrawArrays(GL_LINE_STRIP, 0, len(origpoints))

    glPointSize(1.0)

    glColorPointer(3, GL_FLOAT, 0, colorscurve)
    glVertexPointer(3, GL_FLOAT, 0, pointscurve)
    glDrawArrays(GL_POINTS, 0, len(pointscurve))

    print "the first point of the allegedly changed pointscurve is: ", pointscurve[0]
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
    origpoints = np.array([[-0.9, 0.9, 0.0],
                           [-0.9, -0.9, 0.0],
                           [0.9, -0.9, 0.0],
                           [0.9, 0.9, 0.0]])

    # this array is for the adding of additional control points
    ctrlpoints = np.array([[-0.9, 0.9, 0.0],
                           [-0.9, -0.9, 0.0],
                           [0.9, -0.9, 0.0],
                           [0.9, 0.9, 0.0]])

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

    pointscurve = np.zeros((101, 3))

    colorscurve = np.zeros((101, 3))

    # knot-vector of 4 0's and 4 1's
    knotvector = np.array([0, 0, 0, 0, 1, 1, 1, 1])


    # the 2 is for debugging
    for iteration in range(101):

        # initialize the B to all 0's (the amount follow the for-loops)
        B = np.zeros((4, 5))

        u = float(iteration)/float(100)
        print u

        # amount of control points
        for k in range(4):
            # Initialization of Cox-deBoor's recursion formula
            if (u >= knotvector[k] and u <= knotvector[k + 1]):
                B[k][1] = 1.0
                #print "yeah! ", k
            else:
                B[k][1] = 0.0

        # summation from 3 to 0, decrement 1, control points
        for k in range(3, -1, -1):

            # fixed to up to 4 for cubic (2, 3, 4), degree
            for d in range(2, 5):

                #print "the initial B[k][d] is: ", B[k][d]
                # Cox-deBoor's Recursion (page 443 of Computer Graphics book)
                if (float(knotvector[k + d - 1] - knotvector[k]) != 0.0):
                    B[k][d] += (float(u) - float(knotvector[k]))/float(knotvector[k + d - 1] - knotvector[k]) * float(B[k][d-1])
                    #print "the multiplied recursive is: ", float(B[k][d-1])
                    #print "the numerator is: ", float(u) - knotvector[k]
                    #print "first turn: ", B[k][d]

                if (float(knotvector[k + d] - knotvector[k + 1]) != 0.0):
                    #print "the knotvector is: ", knotvector[k+d]
                    #print "V2: ", knotvector[k+1]
                    #print knotvector
                    B[k][d] += (float(knotvector[k + d]) - float(u))/float(knotvector[k + d] - knotvector[k + 1]) * float(B[k + 1][d - 1])
                    #print "the numerator is: ", float(knotvector[k + d]) - float(u)
                    #print "the multiplied recursive is: ", float(B[k+1][d-1])
                    #print "second turn: ", B[k][d]

                #print "the new B at k: ", k, ", and d: ", d, " is: ", B[k][d]
        P = 0

        # i (j in this case) only ranges from 0 to 4 (the number of control points, which is n) in the 
        # summation formula
        for j in range(4):
            #print B[j][e]
            #print ctrlpoints[j]
            P += float(B[j][4]) * ctrlpoints[j]
            #print "this is: ", B[j][4]
        #print P

        pointscurve[iteration][0] = P[0]
        pointscurve[iteration][1] = P[1]
        pointscurve[iteration][2] = 0.0

        colorscurve[iteration][0] = 1.0
        colorscurve[iteration][1] = 1.0
        colorscurve[iteration][2] = 1.0

        #print knotvector

    #print origpoints
    #print pointscurve
    #print colorscurve

    #print len(pointscurve)

    #print pointsToRender
    # set up GLUT callbacks.
    glutDisplayFunc(redraw)
    glutKeyboardFunc(keyfunc)
    glutMouseFunc(mouseCB)
    glutMotionFunc(mouseMotionCB)

    glutMainLoop()
