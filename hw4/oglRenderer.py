#!/usr/bin/python

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import pyparsing as pp
import sys
import os
import numpy as np
from math import*

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
    # Initialize all the mouse presses to not happening
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
            mouseLeftDown = True
        elif(state == GLUT_UP):
            mouseLeftDown = False

    elif(button == GLUT_MIDDLE_BUTTON):
        if(state == GLUT_DOWN):
            mouseRightDown = True
        elif(state == GLUT_UP):
            mouseRightDown = False

    elif(button == GLUT_RIGHT_BUTTON):
        if(state == GLUT_DOWN):
            mouseMiddleDown = True
        elif(state == GLUT_UP):
            mouseMiddleDown = False

    # this is for the key push of 'Shift'
    mod = glutGetModifiers();
    if (mod == GLUT_ACTIVE_SHIFT):
        shiftPressed = True

def mouseMotionCB(x, y):
    # Save the original modelview matrix - we'll need it later
    mvm = glGetFloatv(GL_MODELVIEW_MATRIX)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    if(mouseLeftDown):
        mouseX = x
        mouseY = y
    if(mouseRightDown):
        mouseY = y
    if (mouseMiddleDown and shiftPressed == False):
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

    glMultMatrixf(mvm)

    newmvm = glGetFloatv(GL_MODELVIEW_MATRIX)
    #print "new mvm is: ", newmvm

    if (mouseMiddleDown and shiftPressed):
        mouseX = x
        mouseY = y
        global lastY
        dy = mouseY - lastY
        lastY = mouseY
        glTranslatef(0, 0, -1*dy/float(100))
        glRotatef(0, 0, 0, 0)
        glScalef(1.0, 1.0, 1.0)
    
    # tell GLUT to call the redrawing function, in this case redraw()
    glutPostRedisplay()

#def redraw(worldArray):  

def draw1():

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    #glPushMatrix()

    print len(verticesAccum)

    #print "the scale factor list is: ", scalefAccum

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

            # calculate the separator matrix (S = TRS)
            # it is for Object Space to World Space
            # also for transformed normal matrix

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

        #print "the verticesAccum is: ", IPT
        #print "the normsAccum is: ", INT

        # activate and specify pointer to vertex array
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_VERTEX_ARRAY)

        glNormalPointer(GL_FLOAT, 0, INT)

        glVertexPointer(3, GL_FLOAT, 0, IPT)

        # IMPORTANT: make sure it is the length of the array
        # (number of vertices)
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
    glShadeModel(GL_SMOOTH)
    
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
            glTranslatef(-1*cameraX, -1*cameraY, -1*cameraZ)

            # save the inverse camera matrix
            glPushMatrix()

            # calculate the camera matrix
            # World space to camera space

            # calculate the Perspective Projection matrix
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glFrustum(l, r, b, t, n, f)
            # Save the perspective projection matrix

        glMatrixMode(GL_MODELVIEW)

        # if we reach PointLight parameter
        while (len(firstparse) != 0 and (firstparse[0] == 'PointLight')):
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

                        global lastX
                        lastX = (tX + 1)*500.0/2
                        global lastY
                        lastY = (tY + 1)*500.0/2
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
                        #rotationSep = rotationMatrix(normalizedrX, normalizedrY, normalizedrZ, rAngle)
                        
                    # scale factor
                    elif (firstparse[0] == 'scaleFactor'):
                        scaleFactor = firstparse[0]
                        sfX = firstparse[1]
                        sfY = firstparse[2]
                        sfZ = firstparse[3]
                    first = fo.readline()
                    firstparse = parameter.parseString(first)

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

            # entering the Normal subparameter
            if (len(firstparse) != 0 and (firstparse[0] == 'Normal')):
                first = fo.readline()
                firstparse = parameter.parseString(first)
                if (len(firstparse) != 0 and (firstparse[0] == 'vector')):
                    #print "wut wut wut wtu wut"
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
                        else:
                            first = fo.readline()
                            firstparse = parameter.parseString(first)
                            i = 0
                            k = firstparse[i]
                    # if  we have 3 points (whether we reach 1 or -1), add the triangle
                    # to the big list, and reset the 3point list to have the first point
                    if (len(polygonvertices) == 3):
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
                    i += 1
          
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
                                k = firstparse[j]
                        
                        #print "for normalindex: ", firstparse

                        # if  we have 3 points (whether we reach 1 or -1), add the triangle
                        # to the big list, and reset the 3point list to have the first point
                        if (len(normvertices) == 3):
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

                        # if there are less than 3 points in the list, add another point
                        elif (len(normvertices) < 3):
                            normvertices.append(int(k))
                        # if reach the end of a face
                        if (k == -1):
                            # reset the list of 3 points to an empty list
                            normvertices = []
                            # if the next index is in the line
                            # we have to add 2 to accomodate that comma
                            if (j+2 < len(firstparse)):
                                firstpoint = int(firstparse[j+2])
                                #print "the first point is: ", firstpoint
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
                        j += 1
                                
                    first = fo.readline()
                    firstparse = parameter.parseString(first)

            for i in range(len(worldPoints)):
                both.append(worldPoints[i])
            
            first = fo.readline()

            # Accumulate the vertices such that they are in order of the indices
            IndexedPointsTuple = []
            for i in range(len(RealIndices)):
                indextobe = RealIndices[i]
                if (indextobe != -1):
                    tuplecoord = coordsList[int(indextobe)]
                    IndexedPointsTuple.append(tuplecoord)

            # Accumulate the norms such that they are in the order of the indices
            IndexedNormsTuple = []
            for i in range(len(NormIndices)):
                indextobe = NormIndices[i]
                if (indextobe != -1):
                    tupleNorm = vectorsList[int(indextobe)]
                    IndexedNormsTuple.append(tupleNorm)

            # These are for the drawing function
            translateAccum.append(translateSep)
            rotateAccum.append(rotateSep)
            scalefAccum.append(scalefSep)

            # Accumulate the accumulation of vertices (so like for the fourCubes)
            verticesAccum.append(IndexedPointsTuple)

            # Accumulate the accumulation of the norms (so like for the fourCubes)
            normsAccum.append(IndexedNormsTuple)
        first = fo.readline()
        firstparse = parameter.parseString(first)
    fo.close()

    # Call the drawing function
    glutDisplayFunc(draw1)

    # Ability to resize the window
    glutReshapeFunc(resize)

    # This takes care of the keys (so like the exiting is 'q')
    glutKeyboardFunc(keyfunc)

    # This takes care of detecting which parts of the mouse are pressed
    glutMouseFunc(mouseCB)

    # This takes care of what happens to the object when
    # certain parts of the mouse are pressed
    glutMotionFunc(mouseMotionCB)

    glutMainLoop()
    glPopMatrix()
