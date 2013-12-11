#!/usr/bin/python

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import pyparsing as pp
import sys
import numpy as np
from math import*

def mouseCB(button, state, x, y):
    global mouseLeftDown 
    mouseLeftDown = False
    global mouseRightDown 
    mouseRightDown = False

    mouseCBx = (x/float(xRes) - 0.5)*2.0
    mouseCBy = (1.0 - y/float(yRes) - 0.5)*2.0

    global shiftPressed
    shiftPressed = False

    if(button == GLUT_LEFT_BUTTON):
        if(state == GLUT_DOWN):
            #print "the x-coordinate pixel is: ", x
            #print "the y-coordinate pixel is: ", y
            mouseLeftDown = True

        elif(state == GLUT_UP):
            mouseLeftDown = False

    elif(button == GLUT_RIGHT_BUTTON):
        # if the right mouse button is pressed down, add a new knot
        if(state == GLUT_DOWN):
            global knotvector
            global ctrlpoints
            global origpoints
            global colorpoints

            print "mouse Right is down! ", mouseCBx, ", ", mouseCBy

            i = 0
            newknot = 0
            while (i < len(pointscurve) and newknot == 0):
                print "it should go here ", i
                # approximate the x and y because we cannot get exactly the x and y
                if (abs(mouseCBx - pointscurve[i][0]) < 0.01 and abs(mouseCBy - pointscurve[i][1]) < 0.01):
                    newknot = 1.0 - float(i)/len(pointscurve)
                    print "the new knot is: ", newknot
                    # find where to insert the knot into the knot vector
                    correctidx = 2*len(knotvector)

                    knotidx = 0
                    # figure out where to put the new knot
                    while (knotidx < len(knotvector) and correctidx == 2*len(knotvector)):
                        if knotvector[knotidx] > newknot:
                            correctidx = knotidx
                        knotidx += 1
                    #print knotvector
                    # insert the knot into the correct place in the knot vector (in order, from least to biggest)\
                    knotvector = np.insert(knotvector, int(correctidx), newknot)

                    # create a vertical stack of the first part of ctrlpoints with the new control point added
                    #ctrlInter = np.vstack((ctrlpoints[:correctidx], np.array([mouseCBx, mouseCBy, 0.0])))

                    # create a vertical stack of the previous with the second part of ctrlpoints
                    #ctrlpoints = np.vstack((ctrlInter, ctrlpoints[correctidx:]))

                    # create a vertical stack of the first part of origpoints with the new control point added
                    #origInter = np.vstack((ctrlpoints[:correctidx], np.array([mouseCBx, mouseCBy, 0.0])))

                    # create a vertical stack of the previous with the second part of origpoints
                    #origpoints = np.vstack((origInter, origpoints[correctidx:]))
                    # initialize the B to all 0's (the amount follow the for-loops)
                    oldlen = len(ctrlpoints)
                    newctrl = np.zeros((oldlen + 1, 3))

                    # Calculate the a values specified in Nurbs.ps, page 4
                    j = correctidx - 1
                    a = [0]*(oldlen)
                    for b in range(1, j - 4 + 1):
                        a[b] = 1.0
                    for c in range(j - 4 + 1, j + 1):
                        a[c] = (newknot - knotvector[c])/(knotvector[c + 4] - knotvector[c])
                    
                    # Calculate the new control points
                    newctrl[0][0] = ctrlpoints[0][0]
                    newctrl[0][1] = ctrlpoints[0][1]
                    newctrl[0][2] = ctrlpoints[0][2]

                    newctrl[oldlen][0] = ctrlpoints[oldlen - 1][0]
                    newctrl[oldlen][1] = ctrlpoints[oldlen - 1][1]
                    newctrl[oldlen][2] = ctrlpoints[oldlen - 1][2]

                    for d in range(1, oldlen):
                        newctrl[d][0] = (1 - a[d]) * ctrlpoints[d - 1][0] + a[d] * ctrlpoints[d][0]
                        newctrl[d][1] = (1 - a[d]) * ctrlpoints[d - 1][1] + a[d] * ctrlpoints[d][1]
                        newctrl[d][2] = (1 - a[d]) * ctrlpoints[d - 1][2] + a[d] * ctrlpoints[d][2]

                    ctrlpoints = np.copy(newctrl)
                    origpoints = np.copy(newctrl)
                i += 1

            print "the new knotvector is: ", knotvector
            print "the new control points: ", ctrlpoints
            print "the new origpoints are: ", origpoints

            colorpoints = np.append(colorpoints, np.array([1.0, 1.0, 1.0]))

            print "the new colors are: ", colorpoints

            # the 2 is for debugging
            for iteration in range(1001):

                # initialize the B to all 0's (the amount follow the for-loops)
                B = np.zeros((len(ctrlpoints), 5))

                u = float(iteration)/float(1000)
                #print u

                # amount of control points
                for k in range(len(ctrlpoints)):
                    # Initialization of Cox-deBoor's recursion formula
                    if (u >= knotvector[k] and u <= knotvector[k + 1]):
                        B[k][1] = 1.0
                        #print "yeah! ", k
                    else:
                        B[k][1] = 0.0

                # summation from 3 to 0, decrement 1, control points
                # here, generically, it is len(ctrlpoints) - 1 to 0 decrement by 1
                for k in range(len(ctrlpoints) - 1, -1, -1):

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
                for j in range(len(ctrlpoints)):
                    #print B[j][e]
                    #print ctrlpoints[j]
                    newP += float(B[j][4]) * ctrlpoints[j]
                    #print "this is: ", B[j][4]
                #print P

                #print "after moving, the ctrl points are: ", ctrlpoints

                pointscurve[iteration][0] = newP[0]
                pointscurve[iteration][1] = newP[1]
                pointscurve[iteration][2] = 0.0


            glutPostRedisplay()
        elif(state == GLUT_UP):
            mouseRightDown = False
            #print "wut, this should not be happening"

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
                # knotvector = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])


                # the 2 is for debugging
                for iteration in range(1001):

                    # initialize the B to all 0's (the amount follow the for-loops)
                    B = np.zeros((len(ctrlpoints), 5))

                    u = float(iteration)/float(1000)
                    #print u

                    # amount of control points
                    for k in range(len(ctrlpoints)):
                        # Initialization of Cox-deBoor's recursion formula
                        if (u >= knotvector[k] and u <= knotvector[k + 1]):
                            B[k][1] = 1.0
                            #print "yeah! ", k
                        else:
                            B[k][1] = 0.0

                    # summation from 3 to 0, decrement 1, control points
                    # here, generically, it is len(ctrlpoints) - 1 to 0 decrement by 1
                    for k in range(len(ctrlpoints) - 1, -1, -1):

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
                    for j in range(len(ctrlpoints)):
                        #print B[j][e]
                        #print ctrlpoints[j]
                        newP += float(B[j][4]) * ctrlpoints[j]
                        #print "this is: ", B[j][4]
                    #print P

                    #print "after moving, the ctrl points are: ", ctrlpoints

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

    pointscurve = np.zeros((1001, 3))

    colorscurve = np.zeros((1001, 3))

    # knot-vector of 4 0's and 4 1's
    knotvector = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])


    # the 2 is for debugging
    for iteration in range(1001):

        # initialize the B to all 0's (the amount follow the for-loops)
        B = np.zeros((4, 5))

        u = float(iteration)/float(1000)
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
