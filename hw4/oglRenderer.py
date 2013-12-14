#!/usr/bin/python

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import pyparsing as pp
import sys
import os
import numpy as np
from math import*

def get_arcball_vector(x, y):
    p = np.array([x, y, 0.0])
    OP_squared = x * float(x) + float(y) * y
    if (OP_squared <= 1.0*1.0):
        p[2] = 1.0*sqrt(1.0*1.0 - OP_squared)  # Pythagore
    else:
        p = p/np.linalg.norm(p) # nearest point
    return p

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
            print "the x-coordinate pixel is: ", x
            print "the y-coordinate pixel is: ", y
            mouseLeftDown = True
            global lastRotX
            lastRotX = (x/float(xRes) - 0.5)*2.0

            global lastRotY
            lastRotY = (1.0 - y/float(yRes) - 0.5)*2.0

            #global currRotX
            #currRotX = (x/float(xRes) - 0.5)*2.0

            #global currRotY
            #currRotY = (1.0 - y/float(yRes) - 0.5)*2.0

        elif(state == GLUT_UP):
            mouseLeftDown = False

    elif(button == GLUT_RIGHT_BUTTON):
        if(state == GLUT_DOWN):
            mouseRightDown = True
            print "mouse Middle is down!"
        elif(state == GLUT_UP):
            mouseRightDown = False
            print "wut, this should not be happening"
    print "the state of mouseMiddleDown is: ", mouseMiddleDown

    mod = glutGetModifiers()
    if (mod == GLUT_ACTIVE_SHIFT):
        print "shift is activated!"
        shiftPressed = True


def mouseMotionCB(x, y):
    mvm = glGetFloatv(GL_MODELVIEW_MATRIX)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # if only the right mouse button is down, then the 
    # translation feature is enabled
    if (mouseRightDown and shiftPressed == False):
        #print "right right right activated!"
        mouseX = x
        mouseY = y

        global lastX
        dx = mouseX - lastX
        global lastY
        dy = mouseY - lastY
        lastX = mouseX
        lastY = mouseY

        glTranslatef(dx/float(100), -1*dy/float(100), 0)
        glRotatef(0, 0, 0, 0)
        glScalef(1.0, 1.0, 1.0)

        intermedmvm = glGetFloatv(GL_MODELVIEW_MATRIX)

    # If the right button on the mouse is down while 'shift' is pressed simultaneously
    # then the zooming feature is enabled.
    if (mouseRightDown and shiftPressed):
        mouseX = x
        mouseY = y
        global lastZoom
        dy = mouseY - lastZoom
        lastZoom = mouseY
        glTranslatef(0, 0, -1*dy/float(100))
        glRotatef(0, 0, 0, 0)
        glScalef(1.0, 1.0, 1.0)

    glMultMatrixf(mvm)

    newmvm = glGetFloatv(GL_MODELVIEW_MATRIX)
    #print "new mvm is: ", newmvm
    
    # tell GLUT to call the redrawing function, in this case redraw()
    glutPostRedisplay()

    if(mouseLeftDown):
        global currRotX
        global currRotY
        global lastRotX
        global lastRotY

        #print "the currRotX is: ", currRotX
        #print "the lastRotX is: ", lastRotX

        #cameraAngleY += (x - mouseX)
        #cameraAngleX += (y - mouseY)
        #print "the x-coordinate pixel2 is: ", x
        #print "the y-coordinate pixel2 is: ", y
        mouseX = (x/float(xRes) - 0.5)*2.0
        mouseY = (1.0 - y/float(yRes) - 0.5)*2.0

        currRotX = mouseX
        currRotY = mouseY

        va = get_arcball_vector(lastRotX, lastRotY)
        vb = get_arcball_vector(currRotX, currRotY)

        if (va[0] != vb[0] or va[1] != vb[1]):

            angle = acos(min(1.0, np.inner(va, vb)))

            axis_normalized = np.cross(va, vb)

            glRotatef(angle*180.0/pi,
                      axis_normalized[0],
                      axis_normalized[1],
                      axis_normalized[2])

            lastRotX = currRotX
            lastRotY = currRotY

def draw1():

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    print len(verticesAccum)

    for i in range(len(translateAccum)):
        glPushMatrix()
        lenOfSep = len(translateAccum[i])/3
        for j in range(lenOfSep):
            translX = translateAccum[i][3*j]
            translY = translateAccum[i][3*j + 1]
            translZ = translateAccum[i][3*j + 2]

            rotateX = rotateAccum[i][4*j]
            rotateY = rotateAccum[i][4*j + 1]
            rotateZ = rotateAccum[i][4*j + 2]
            rotateAngle = rotateAccum[i][4*j + 3]

            scaleX = scalefAccum[i][3*j]
            scaleY = scalefAccum[i][3*j + 1]
            scaleZ = scalefAccum[i][3*j + 2]

            glTranslatef(translX, translY, translZ)
            glRotatef(rotateAngle*180.0/pi, rotateX, rotateY, rotateZ)
            glScalef(scaleX, scaleY, scaleZ)


        ambien = ambientAccum[i]
        diffus = diffuseAccum[i]
        specul = specularAccum[i]
        shinin = shininess[i]

        emissi = [0.0, 0.0, 0.0, 1.0]

        glMaterialfv(GL_FRONT, GL_AMBIENT, ambien)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, diffus)
        glMaterialfv(GL_FRONT, GL_SPECULAR, specul)
        glMaterialfv(GL_FRONT, GL_EMISSION, emissi)
        glMaterialfv(GL_FRONT, GL_SHININESS, shinin)

        IPT = verticesAccum[i]
        INT = normsAccum[i]

        # activate and specify pointer to vertex array
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_VERTEX_ARRAY)

        glNormalPointer(GL_FLOAT, 0, INT)
        glVertexPointer(3, GL_FLOAT, 0, IPT)

        glDrawArrays(GL_TRIANGLES, 0, len(IPT))

        glPopMatrix()
        glutSwapBuffers() 

        # deactivate vertex arrays after drawing
        glDisableClientState(GL_VERTEX_ARRAY)
        #glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)

# run the script
if __name__ == "__main__":

    cameraX = 0
    cameraY = 0
    cameraZ = 0

    glutInit(sys.argv)

    # whether to pick flat, gourad, or phong shading
    # 0 is wireframe, 1 is flat, 2 is gouraud
    shade = int(sys.argv[1])

    # x dimension size
    xRes = int(sys.argv[2])

    # y dimension size
    yRes = int(sys.argv[3])

    # iv file name to input
    ivFile = sys.argv[4]

    # Get a double-buffered, depth-buffer-enabled window, with an
    # alpha channel.
    # These options aren't really necessary but are here for examples.
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

    glutInitWindowSize(xRes, yRes)
    glutInitWindowPosition(300, 100)

    glutCreateWindow("CS171 HW4")

    # Tell openGL to use Gouraud shading:
    if (shade == 2):
        glShadeModel(GL_SMOOTH)

    if (shade == 1):
        glShadeModel(GL_FLAT)

    # Tell opennGL to use WireFrame
    if (shade == 0):
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    
    # Enable back-face culling:
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

    # Enable depth-buffer test.
    glEnable(GL_DEPTH_TEST)

    lightidx = 0

    GL_LIGHTC = [GL_LIGHT0, GL_LIGHT1, GL_LIGHT2, GL_LIGHT3, GL_LIGHT4, GL_LIGHT5, GL_LIGHT6, GL_LIGHT7]

    # define grammar
    # number is float form
    # does +/-, 0., .2, and exponentials
    number = pp.Regex(r"[-+]?([0-9]*\.[0-9]*|[0-9]+)([Ee][+-]?[0-9]+)?")
    number.setParseAction(lambda toks:float(toks[0]))

    leftBrace = pp.Literal("{")
    rightBrace = pp.Literal("}")
    leftBracket = pp.Literal("[")
    rightBracket = pp.Literal("]")
    comma = pp.Literal(",")

    # Optional added for the additional number for rotation
    parameter = pp.Optional(pp.Word( pp.alphas )) + pp.Optional(leftBracket) + \
                pp.Optional(leftBrace) + pp.Optional(rightBracket) + \
                pp.Optional(rightBrace) + pp.ZeroOrMore(number + pp.Optional(comma))

    # Open a file
    fo = open(ivFile, "r")

    first = fo.readline()

    lights = np.zeros([1, 3])

    # these are the numbers for translation, rotation, and scale factor for all
    # transform blocks in all separators. Each separator will have one list
    # containing all the transform blocks. To determine which part in the list is
    # which transform block, we will go by 3's for translate and rotate, and 1's for
    # scale factor
    translateAccum = []
    rotateAccum = []
    scalefAccum = []
    
    # these are the ambient, diffuse, specular, and shininess values accumulated for
    # all separators. Since each separator has 1 material block, we don't need to make
    # each separator have a list
    ambientAccum = []
    diffuseAccum = []
    specularAccum = []
    shininess = []

    # these are vertices and norms accumulated for all separators.
    verticesAccum = []
    normsAccum = []
    
    global lastRotX
    lastRotX = 0
    global lastRotY
    lastRotY = 0
    global currRotX
    currRotX = 0
    global currRotY
    currRotY = 0

    # as long as we don't reach the end of the file
    while (first != ''):
        firstparse = parameter.parseString(first)

        # if we reach PerspectiveCamera parameter
        if (len(firstparse) != 0 and (firstparse[0] == 'PerspectiveCamera')):
            first = fo.readline()
            # if there is a blank line, read another main parameter
            while (first.strip() != ''):
                firstparse = parameter.parseString(first)
                # position parameter
                if (firstparse[0] == 'position'):
                    #global cameraX
                    cameraX = firstparse[1]
                    #global cameraY
                    cameraY = firstparse[2]
                    #global cameraZ
                    cameraZ = firstparse[3]

                    global lastZoom
                    lastZoom = cameraZ
                    campos = np.array([cameraX, cameraY, cameraZ])

                # orientation paramter
                elif (firstparse[0] == 'orientation'):
                    x = firstparse[1]
                    y = firstparse[2]
                    z = firstparse[3]
                    angle = firstparse[4]

                # near distance parameter
                elif (firstparse[0] == 'nearDistance'):
                    n = firstparse[1]

                # far distance paramter
                elif (firstparse[0] == 'farDistance'):
                    f = firstparse[1]
        
                # left parameter
                elif (firstparse[0] == 'left'):
                    l = firstparse[1]

                # right parameter
                elif (firstparse[0] == 'right'):
                    r = firstparse[1]
            
                # bottom parameter
                elif (firstparse[0] == 'bottom'):
                    b = firstparse[1]

                # top parameter
                elif (firstparse[0] == 'top'):
                    t = firstparse[1]
                first = fo.readline()

            # translation matrix (for position of camera)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            # inverse rotation matrix (for orientation of camera) (angle is in degrees)
            glRotatef(-1*angle*180.0/pi, x, y, z)

            # multiply with inverse translation matrix
            glTranslatef(-1.0*cameraX, -1.0*cameraY, -1.0*cameraZ)

            # save the inverse camera matrix
            glPushMatrix()

            global cameraMat
            cameraMat = glGetFloatv(GL_MODELVIEW_MATRIX)

            # calculate the camera matrix
            # World space to camera space

            # calculate the Perspective Projection matrix
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glFrustum(l, r, b, t, n, f)

            global projMat
            projMat = glGetFloatv(GL_PROJECTION_MATRIX)

        glMatrixMode(GL_MODELVIEW)

        # if we reach PointLight parameter
        while (len(firstparse) != 0 and (firstparse[0] == 'PointLight')):
            if (shade == 1 or shade == 2):
                first = fo.readline()
                lightidx += 1
                amb = [0.0]*4
                lightpos = [0.0]*4
                diff = [0.0]*4
                spec = [0.0]*4
                # default position and color
                lightpos[0] = 0.0
                lightpos[1] = 0.0
                lightpos[2] = 1.0
                lightpos[3] = 1.0
                amb[0] = 0.0
                amb[1] = 0.0
                amb[2] = 0.0
                amb[3] = 1.0
                diff[0] = 1.0
                diff[1] = 1.0
                diff[2] = 1.0
                diff[3] = 1.0
                spec[0] = 1.0
                spec[1] = 1.0
                spec[2] = 1.0
                spec[3] = 1.0
                # if there is a blank line, read another main parameter
                while (first.strip() != ''):
                    firstparse = parameter.parseString(first)
                    # location parameter
                    if (firstparse[0] == 'location'):
                        lightpos[0] = firstparse[1]
                        lightpos[1] = firstparse[2]
                        lightpos[2] = firstparse[3]
                    # color parameter
                    elif (firstparse[0] == 'color'):
                        diff[0] = firstparse[1]
                        diff[1] = firstparse[2]
                        diff[2] = firstparse[3]

                        spec[0] = firstparse[1]
                        spec[1] = firstparse[2]
                        spec[2] = firstparse[3]
                    
                    first = fo.readline()

                glEnable(GL_LIGHTING)
                glLightModelfv(GL_LIGHT_MODEL_AMBIENT, amb)
                glLightfv(GL_LIGHTC[lightidx], GL_AMBIENT, amb)
                glLightfv(GL_LIGHTC[lightidx], GL_DIFFUSE, diff)
                glLightfv(GL_LIGHTC[lightidx], GL_SPECULAR, spec)
                glLightfv(GL_LIGHTC[lightidx], GL_POSITION, lightpos)
                glEnable(GL_LIGHTC[lightidx])

            first = fo.readline()
            firstparse = parameter.parseString(first)

        # Turn on lighting.  You can turn it off with a similar call to
        # glDisable().

        # if we reach the Separator parameter
        while (len(firstparse) != 0 and (firstparse[0] == 'Separator')):
            first = fo.readline()
            firstparse = parameter.parseString(first)
            
            # transform multiplication, initialized to identity matrix
            totaltransform = np.identity(4)

            totalNorm = np.identity(4)

            translateSep = []
            rotateSep = []
            scalefSep = []

            # if we reach the Transform sub-parameter
            while (len(firstparse) != 0 and (firstparse[0] == 'Transform')):
                first = fo.readline()
                firstparse = parameter.parseString(first)
                
                translate = rotate = scaleFactor = ''

                # default values for translate
                tX = 0
                tY = 0
                tZ = 0

                # default values for rotation
                normalizedrX = 0
                normalizedrY = 0
                normalizedrZ = 0
                rAngle = 0

                # default values for scale factor
                sfX = 1
                sfY = 1
                sfZ = 1

                # as long as we aren't at the end of the Transform parameter
                while (firstparse[0] != '}'):

                    # translation
                    if (firstparse[0] == 'translation'):
                        translate = firstparse[0]
                        tX = firstparse[1]
                        tY = firstparse[2]
                        tZ = firstparse[3]

                    # rotation
                    elif (firstparse[0] == 'rotation'):
                        rotate = firstparse[0]
                        rX = float(firstparse[1])
                        rY = float(firstparse[2])
                        rZ = float(firstparse[3])

                        normalizedrX = rX/(sqrt(rX**2 + rY**2 + rZ**2))
                        normalizedrY = rY/sqrt(rX**2 + rY**2 + rZ**2)
                        normalizedrZ = rZ/sqrt(rX**2 + rY**2 + rZ**2)
                        rAngle = float(firstparse[4])
                        
                    # scale factor
                    elif (firstparse[0] == 'scaleFactor'):
                        scaleFactor = firstparse[0]
                        sfX = firstparse[1]
                        sfY = firstparse[2]
                        sfZ = firstparse[3]
                    first = fo.readline()
                    firstparse = parameter.parseString(first)

                # converting to pixel coordinates
                # keep in mind that the top left corner is (0, 0)
                global lastX
                lastX = (tX + 1.0)*xRes/2
                global lastY
                lastY = yRes - (tY + 1.0)*yRes/2

                translateSep.append(tX)
                translateSep.append(tY)
                translateSep.append(tZ)

                rotateSep.append(normalizedrX)
                rotateSep.append(normalizedrY)
                rotateSep.append(normalizedrZ)
                rotateSep.append(rAngle)

                scalefSep.append(sfX)
                scalefSep.append(sfY)
                scalefSep.append(sfZ) 

                # end of Transform block parameter
                if (len(firstparse) != 0 and (firstparse[0] == '}')):
                    first = fo.readline()
                    firstparse = parameter.parseString(first)

            # entering the Material subparameter
            if (len(firstparse) != 0 and (firstparse[0] == 'Material')):
                first = fo.readline()
                # the -20 is determine if the parameter is included or not
                amb = [-20, 0.0, 0.0, 1.0]
                diff = [-20, 0.0, 0.0, 1.0]
                spec = [-20, 0.0, 0.0, 1.0]
                shiny = -20
                # read lines until we reach the ending curly brace
                while (first.strip() != '}'):
                    firstparse = parameter.parseString(first)
                    # ambient color parameter
                    if (firstparse[0] == 'ambientColor'):
                        amb[0] = firstparse[1]
                        amb[1] = firstparse[2]
                        amb[2] = firstparse[3]
                    # diffuse color parameter
                    elif (firstparse[0] == 'diffuseColor'):
                        diff[0] = firstparse[1]
                        diff[1] = firstparse[2]
                        diff[2] = firstparse[3]
                    # specular color parameter
                    elif (firstparse[0] == 'specularColor'):
                        spec[0] = firstparse[1]
                        spec[1] = firstparse[2]
                        spec[2] = firstparse[3]
                    # diffuse color parameter
                    elif (firstparse[0] == 'shininess'):
                        shiny = firstparse[1]
                    first = fo.readline()
                # if the parameters are not there, set to default colors
                if (amb[0] == -20):
                    amb[0] = 0.2
                    amb[1] = 0.2
                    amb[2] = 0.2
                if (diff[0] == -20):
                    diff[0] = 0.8
                    diff[1] = 0.8
                    diff[2] = 0.8
                if (spec[0] == -20):
                    spec[0] = 0.0
                    spec[1] = 0.0
                    spec[2] = 0.0
                if (shiny == -20):
                    shiny = 0.2
                #glEnable(GL_COLOR_MATERIAL);

                ambientAccum.append(amb)
                diffuseAccum.append(diff)
                specularAccum.append(spec)
                shininess.append(shiny)
                
                first = fo.readline()
                firstparse = parameter.parseString(first)

            while (first.strip() == ''):
                first = fo.readline()
                firstparse = parameter.parseString(first)

            # entering the Coordinate subparameter
            if (len(firstparse) != 0 and (firstparse[0] == 'Coordinate')):
                first = fo.readline()
                firstparse = parameter.parseString(first)
                if (len(firstparse) != 0 and (firstparse[0] == 'point')):
                    # Create a list of the coordinates
                    coordsList = []
                    # to compensate for the point
                    f = 2
                    while (firstparse[f] != ']' and firstparse[f] != '}' and first.strip() != '}'):
                        xArr = float(firstparse[f])
                        yArr = float(firstparse[f+1])
                        zArr = float(firstparse[f+2])

                        #coordsNoTuple.append(xArr)
                        #coordsNoTuple.append(yArr)
                        #coordsNoTuple.append(zArr)

                        tuplecoord = [xArr, yArr, zArr]
                        coordsList.append(tuplecoord)

                        first = fo.readline()
                        firstparse = parameter.parseString(first)
                        f = 0
                first = fo.readline()
                firstparse = parameter.parseString(first)
            while (first.strip() == ''):
                first = fo.readline()
                firstparse = parameter.parseString(first)

            # end of Coordinates block parameter
            if (len(firstparse) != 0 and (firstparse[0] == '}')):
                first = fo.readline()
                firstparse = parameter.parseString(first)

            print "wut"

            # entering the Normal subparameter
            if (len(firstparse) != 0 and (firstparse[0] == 'Normal')):
                first = fo.readline()
                firstparse = parameter.parseString(first)
                if (len(firstparse) != 0 and (firstparse[0] == 'vector')):
                    # Create a list of the normals
                    vectorsList = []
                    # to compensate for the point
                    f = 2
                    while (firstparse[f] != ']' and firstparse[f] != '}' and first.strip() != '}'):
                        xVec = float(firstparse[f])
                        yVec = float(firstparse[f+1])
                        zVec = float(firstparse[f+2])

                        tuplepoint = [xVec, yVec, zVec]
                        vectorsList.append(tuplepoint)
                        first = fo.readline()
                        firstparse = parameter.parseString(first)
                        f = 0
                first = fo.readline()
                firstparse = parameter.parseString(first)
            while (first.strip() == ''):
                first = fo.readline()
                firstparse = parameter.parseString(first)

            # end of Coordinates block parameter
            if (len(firstparse) != 0 and (firstparse[0] == '}')):
                first = fo.readline()
                firstparse = parameter.parseString(first)

            # start into the IndexedFaceSet block parameter
            if (len(firstparse) != 0 and (firstparse[0] == 'IndexedFaceSet')):
                first = fo.readline()

                # to determine which vertices to render
                polygonvertices = []
                normvertices = []
                # Create a list of the points in space for the lighting function
                worldPoints = []
                # Create a list of norms in world space for the lighting function
                worldNorms = []
                # Create a list of both vertices and norms indices
                both = []
                RealIndices = []
                NormIndices = []

                index = -1
                # for both the coordinates and the normals
                indexforboth = []
                tempindex = []
                firstpoint = -100
                firstparse = parameter.parseString(first)

                # for the first row, with the coordIndex as firstparse[0]
                i = 0
                # Go through the line
                space = 0
                while (i < len(firstparse) and firstparse[0] != 'normalIndex'):
                    k = firstparse[i]
                    # if the element is a comma, bracket, or coordIndex, then move on to next element
                    while ((k == ',') or (k == '[') or (k == ']') or (k == 'coordIndex')):
                        if (i < len(firstparse) - 1):
                            i += 1
                            k = firstparse[i]
                            print "the firstparse first is: ", firstparse
                            print k
                        else:
                            first = fo.readline()
                            firstparse = parameter.parseString(first)
                            i = 0
                            k = firstparse[i]
                            print "the firstparse is: ", firstparse
                    # if  we have 3 points (whether we reach 1 or -1), add the triangle
                    # to the big list, and reset the 3point list to have the first point
                    if (len(polygonvertices) == 3):
                        print "never reach here?"
                        worldPoints.append(polygonvertices)
                        if (firstpoint == -100):
                            firstpoint = polygonvertices[0]
                        RealIndices.append(polygonvertices[0])
                        RealIndices.append(polygonvertices[1])
                        RealIndices.append(polygonvertices[2])
                        lastpoint = polygonvertices[2]
                        polygonvertices = []
                        print polygonvertices
                        if (int(k) != -1):
                            # add the very first point
                            polygonvertices.append(firstpoint)
                            # add the point before current point
                            polygonvertices.append(lastpoint)
                            # add the current point
                            polygonvertices.append(int(k))

                    # if there are less than 3 points in the list, add another point
                    elif (len(polygonvertices) < 3):
                        polygonvertices.append(int(k))

                    # if reach the end of a face
                    if (k == -1):
                        # reset the list of 3 points to an empty list
                        polygonvertices = []
                        # if the next index is in the line
                        # we have to add 2 to accomodate that comma
                        if (i+2 < len(firstparse)):
                            firstpoint = int(firstparse[i+2])
                            print "the new first point is: ", firstpoint
                            #print "the first point is: ", firstpoint
                            polygonvertices.append(firstpoint)
                            # move to next comma after first point
                            i += 2
                        
                        else:
                            first = fo.readline()
                            firstparse = parameter.parseString(first)
                            if (firstparse[0] != 'normalIndex'):
                                firstpoint = int(firstparse[0])
                                polygonvertices.append(firstpoint)
                            i = 0
                        
                        print "hey"
                    i += 1
                    print "dum dum dum", polygonvertices

                print "life sucks"
          
                if (firstparse[0] == 'normalIndex'):
                    firstpoint = -100
                    j = 0
                    while(j < len(firstparse) and first.strip() != '}'):
                        k = firstparse[j]
                        # if the element is a comma, bracket, or coordIndex, then move on to next element
                        while ((k == ',') or (k == '[') or (k == ']') or (k == 'normalIndex')):
                            if (j < len(firstparse) - 1):
                                j += 1
                                k = firstparse[j]
                            else:
                                first = fo.readline()
                                firstparse = parameter.parseString(first)
                                j = 0
                                print "mhm, ", firstparse
                                k = firstparse[j]

                        # if  we have 3 points (whether we reach 1 or -1), add the triangle
                        # to the big list, and reset the 3point list to have the first point
                        if (len(normvertices) == 3):
                            print "2never reach here?"
                            worldNorms.append(normvertices)
                            if (firstpoint == -100):
                                firstpoint = normvertices[0]
                            NormIndices.append(normvertices[0])
                            NormIndices.append(normvertices[1])
                            NormIndices.append(normvertices[2])
                            lastpoint = normvertices[2]
                            normvertices = []
                            print normvertices
                            if (int(k) != -1):
                                # add the very first point
                                normvertices.append(firstpoint)
                                # add the point before current point
                                normvertices.append(lastpoint)
                                # add the current point
                                normvertices.append(int(k))
                            print "2the index is at : ", int(k)
                            print "norms will roll"

                        # if there are less than 3 points in the list, add another point
                        elif (len(normvertices) < 3):
                            normvertices.append(int(k))
                            print "nooooooo2 ", int(k)
                        # if reach the end of a face
                        if (k == -1):
                            print "we go here!2"
                            # reset the list of 3 points to an empty list
                            normvertices = []
                            # if the next index is in the line
                            # we have to add 2 to accomodate that comma
                            if (j+2 < len(firstparse)):
                                firstpoint = int(firstparse[j+2])
                                normvertices.append(firstpoint)
                                # move to next comma after first point
                                j += 2
                            else:
                                first = fo.readline()
                                firstparse = parameter.parseString(first)
                                if (firstparse[0] != '}'):
                                    firstpoint = int(firstparse[0])
                                    normvertices.append(firstpoint)
                                j = 0
                            print "hey2 ", normvertices
                        j += 1
                        print "ho ho ho ho ho ho ho heheheheheheheh ", normvertices
                                
                    first = fo.readline()
                    firstparse = parameter.parseString(first)

            print "wut3"

            for i in range(len(worldPoints)):
                both.append(worldPoints[i])
            
            first = fo.readline()

            print "The real indices of vertices are: ", RealIndices
            print len(RealIndices)

            IndexedPointsTuple = []
            IndexedPoints = []
            for i in range(len(RealIndices)):
                indextobe = RealIndices[i]
                if (indextobe != -1):
                    tuplecoord = coordsList[int(indextobe)]
                    IndexedPointsTuple.append(tuplecoord)
                    IndexedPoints.append(tuplecoord[0])
                    IndexedPoints.append(tuplecoord[1])
                    IndexedPoints.append(tuplecoord[2])

            IndexedNormsTuple = []
            IndexedNorms = []
            for i in range(len(NormIndices)):
                indextobe = NormIndices[i]
                if (indextobe != -1):
                    tupleNorm = vectorsList[int(indextobe)]
                    IndexedNormsTuple.append(tupleNorm)
                    IndexedNorms.append(tupleNorm[0])
                    IndexedNorms.append(tupleNorm[1])
                    IndexedNorms.append(tupleNorm[2])

            print "the RealIndices are: "
            print len(RealIndices)
            print RealIndices

            print "the NormIndices are: "
            print len(NormIndices)
            print NormIndices

            print "hi there!"
            translateAccum.append(translateSep)
            rotateAccum.append(rotateSep)
            scalefAccum.append(scalefSep)

            verticesAccum.append(IndexedPointsTuple)
            normsAccum.append(IndexedNormsTuple)
        first = fo.readline()
        firstparse = parameter.parseString(first)
        print "the next line1 is: ", firstparse
        if (first == ''):
            print "hey! it ended!"
            print "the translateAccum is: ", translateAccum
            print "the rotateAccum is: ", rotateAccum
            print "the scalefAccum is: ", scalefAccum

            print "the ambientAccum: ", ambientAccum
            print "the diffuseAccum: ", diffuseAccum
            print "the specularAccum: ", specularAccum
            print "the shininess is: ", shininess

            print "the verticesAccum is: ", verticesAccum
            print "the normsAccum is: ", normsAccum
    fo.close()

    glutDisplayFunc(draw1)

    glutReshapeFunc(resize)

    glutKeyboardFunc(keyfunc)
    glutMouseFunc(mouseCB)
    glutMotionFunc(mouseMotionCB)

    glutMainLoop()
    glPopMatrix()
