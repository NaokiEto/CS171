#!/usr/bin/python

from OpenGL.GL import *
from OpenGL.GL.shaders import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import pyparsing as pp
import sys
import os
import numpy as np
from math import*
import png
import itertools

import PIL
import Image
import array

def idle():
    global time
    time += 0.01

    print "the time is now: ", time

    skyvertices = []
    skynorms = []

    #skytexture = []

    # go through the rows
    for srow in range(numvertpts):
        skyrow = []
        skynormsrow = []
        #skytextrow = []
        # go through the columns
        for scol in range(numhorizpts):
            x = scol/float(numhorizpts) * 2.0 - 1.0
            y = srow/float(numvertpts) * 2.0 - 1.0

            #skyrow.append([x, -1.0, y])

            skyrow.append([x, 0.03*cos(20.0*(x*x + y*y + 4.0*time)) - 1.0, y])

            #skyvertices[srow][scol] = np.append(skyvertices[srow][scol], [[cos(x + y)]]) 
            #skyvertices[srow][scol] = np.append(skyvertices[srow][scol], [[y]])

            dfdx = -2.0*x*0.03*20.0*sin(20.0*(x*x + y*y + 4.0*time))
            dfdy = -2.0*y*0.03*20.0*sin(20.0*(x*x + y*y + 4.0*time))

            normalizingfactor = sqrt(dfdx*dfdx + dfdy*dfdy + 1.0)

            #skynormsrow.append([dfdx/normalizingfactor, dfdy/normalizingfactor, 1.0/normalizingfactor])

            skynormsrow.append([dfdx/float(normalizingfactor), -1.0/normalizingfactor, dfdy/float(normalizingfactor)])
            #skynorms[srow][scol] = np.append(skynorms[srow][scol], [[1.0]])
            #skynorms[srow][scol] = np.append(skynorms[srow][scol], [[-1.0*dfdy]])

            #skytextrow.append([scol/float(numhorizpts), srow/float(numvertpts)])

        skyvertices.append(skyrow)
        skynorms.append(skynormsrow)
        #skytexture.append(skytextrow)

    #print "the skyvertices are: ", skyvertices

    global indexedskyvertex
    indexedskyvertex = []

    global indexedskynorms
    indexedskynorms = []

    '''
    global indexedskytexture
    indexedskytexture = []
    '''
    # now index these vertices
    # we use 99 instead of 100 to avoid overflow
    for rowidx in range(numvertpts - 1):
        bottom = skyvertices[rowidx + 1]
        top = skyvertices[rowidx]

        bottomnorm = skynorms[rowidx + 1]
        topnorm = skynorms[rowidx]
        '''
        bottomtext = skytexture[rowidx + 1]
        toptext = skytexture[rowidx]
        '''
        for colidx in range(numhorizpts - 1):

            # make sure it is counter-clockwise (for backculling or something like that)
            #  *--*
            #  | /
            #  |/
            #  *
            # this is for the top triangle
            # this is for the vertices
            topleft = top[colidx]
            topright = top[colidx + 1]
            bottomleft = bottom[colidx]
    
            indexedskyvertex.append(topright)
            indexedskyvertex.append(topleft)
            indexedskyvertex.append(bottomleft)

            # this is for the norms
            topleftnorm = topnorm[colidx]
            toprightnorm = topnorm[colidx + 1]
            bottomleftnorm = bottomnorm[colidx]

            indexedskynorms.append(toprightnorm)
            indexedskynorms.append(topleftnorm)
            indexedskynorms.append(bottomleftnorm)

            '''
            # this is for the textures
            toplefttext = toptext[colidx]
            toprighttext = toptext[colidx + 1]
            bottomlefttext = bottomtext[colidx]

            indexedskytexture.append(toprighttext)
            indexedskytexture.append(toplefttext)
            indexedskytexture.append(bottomlefttext)
            '''

            # make it counterclockwise for backculling or something like that
            #     *
            #    /|
            #   / |
            #  *--*
            # this is for the bottom triangle
            # this is for the vertices
            bottomright = bottom[colidx + 1]

            indexedskyvertex.append(bottomleft)
            indexedskyvertex.append(bottomright)
            indexedskyvertex.append(topright)

            # this is for the norms
            bottomrightnorm = bottomnorm[colidx]

            indexedskynorms.append(bottomleftnorm)
            indexedskynorms.append(bottomrightnorm)
            indexedskynorms.append(toprightnorm)

            '''
            # this is for the textures
            bottomrighttext = bottomtext[colidx]
            
            indexedskytexture.append(bottomlefttext)
            indexedskytexture.append(bottomrighttext)
            indexedskytexture.append(toprighttext)
            '''

    #print "does it change in the idlefunc? the first two terms are: ", indexedskyvertex[0], " and ", indexedskyvertex[1]

    #drawsky()
    #glutPostRedisplay(drawsky)
    #glutDisplayFunc(drawsky)
    glutPostRedisplay()

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

    # if press left mouse button, do rotation
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

    mod = glutGetModifiers();
    if (mod == GLUT_ACTIVE_SHIFT):
        print "shift is activated!"
        shiftPressed = True


def mouseMotionCB(x, y):
    mvm = glGetFloatv(GL_MODELVIEW_MATRIX)
    #print "mvm is: ", mvm
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    #print "the mouseMiddleDown is: ", mouseMiddleDown

    # if only the right mouse button is down, then the 
    # translation feature is enabled
    if (mouseRightDown and shiftPressed == False):
        #print "right right right activated!"
        mouseX = x
        mouseY = y

        #print "the xRes is: ", xRes
        #print "the yRes is: ", yRes
        #print "the x-coord is: ", mouseX
        #print "the y-coord is: ", mouseY

        global lastX
        dx = mouseX - lastX
        global lastY
        dy = mouseY - lastY
        lastX = mouseX
        lastY = mouseY

        glTranslatef(dx/float(100), -1*dy/float(100), 0)
        glRotatef(0, 0, 0, 0)
        glScalef(1.0, 1.0, 1.0)
        #glRotatef(rotateAngle*180.0/pi, rotateX, rotateY, rotateZ)
        #glScalef(scaleX, scaleY, scaleZ)

        intermedmvm = glGetFloatv(GL_MODELVIEW_MATRIX)
        #print "intermediate mvm is: ", intermedmvm

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

        mouseX = (x/float(xRes) - 0.5)*2.0
        mouseY = (1.0 - y/float(yRes) - 0.5)*2.0

        currRotX = mouseX
        currRotY = mouseY

        va = get_arcball_vector(lastRotX, lastRotY)
        vb = get_arcball_vector(currRotX, currRotY)

        #print "the va matrix is: ", va
        #print "the vb matrix is: ", vb

        if (va[0] != vb[0] or va[1] != vb[1]):

            angle = acos(min(1.0, np.inner(va, vb)))

            #angle = ((va[0] - vb[0])**2 + (va[1] - vb[1])**2)*0.2

            #print "the angle is: ", angle

            axis_normalized = np.cross(va, vb)

            #glTranslatef(0, 0, 0)
            glRotatef(angle*180.0/pi,
                      axis_normalized[0],
                      axis_normalized[1],
                      axis_normalized[2])
            #glScalef(1.0, 1.0, 1.0)

            #print "the camera matrix is: ", cameraMat

            lastRotX = currRotX
            lastRotY = currRotY

def drawsky():

    #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    #print "the index in drawsky is: ", len(texturestodraw) - 1
    glBindTexture(GL_TEXTURE_2D, textureBinds[len(texturestodraw) - 1])

    # Enable use of textures
    glEnable(GL_TEXTURE_2D)


    #glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, 2)

    glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP)
    glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP)
    glEnable(GL_TEXTURE_GEN_S)
    glEnable(GL_TEXTURE_GEN_T)

    #print "the number of vertices passed through the draw function: ", len(indexedskyvertex), " and the index is ", i

    #print "OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO"

    #print "the new first two terms of indexedskyvertex are: ", indexedskyvertex[0], " and ", indexedskyvertex[1]

    #glEnableClientState( GL_TEXTURE_COORD_ARRAY )

    glEnableClientState(GL_NORMAL_ARRAY)
    # activate and specify pointer to vertex array
    glEnableClientState(GL_VERTEX_ARRAY)

    glNormalPointer(GL_FLOAT, 0, indexedskynorms)
    #glTexCoordPointer(2, GL_FLOAT, 0, indexedskytexture)

    glVertexPointer(3, GL_FLOAT, 0, indexedskyvertex)

    #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glDrawArrays(GL_TRIANGLES, 0, len(indexedskyvertex))

    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)
    #glDisableClientState(GL_TEXTURE_COORD_ARRAY)

    glDisable(GL_TEXTURE_GEN_S)
    glDisable(GL_TEXTURE_GEN_T)

    glDisable(GL_TEXTURE_2D)
    glutSwapBuffers() 

def draw1():


    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    #glPushMatrix()

    print len(verticesAccum)

    print "the drawing occurs for: ", filename

    #print "the scale factor list is: ", scalefAccum

    #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    for i in range(len(translateAccum) + 1):
        if (i < len(translateAccum)):
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

                #print "the translation is: ", translX, ", ", translY, ", ", translZ
                #print "the rotation is: ", rotateX, ", ", rotateY, ", ", rotateZ, ", ", rotateAngle
                #print "the scale factor is: ", scaleX, scaleY, scaleZ

                glTranslatef(translX, translY, translZ)
                glRotatef(rotateAngle*180.0/pi, rotateX, rotateY, rotateZ)
                glScalef(scaleX, scaleY, scaleZ)

            # only work with the material if materialsActivate is 1 (in other words,
            # if we have the materials subparameter)
            if (materialsActivate == 1):
                ambien = ambientAccum[i]
                diffus = diffuseAccum[i]
                specul = specularAccum[i]
                shinin = shininess[i]

                emissi = [0.0, 0.0, 0.0, 1.0]

                glMaterialfv(GL_FRONT, GL_AMBIENT, ambien)
                #print "the ambient2 is: ", ambien
                glMaterialfv(GL_FRONT, GL_DIFFUSE, diffus)
                #print "the diffuse2 is: ", diffus
                glMaterialfv(GL_FRONT, GL_SPECULAR, specul)
                #print "the specular2 is: ", specul
                #print "the emission2 is: ", emissi
                glMaterialfv(GL_FRONT, GL_EMISSION, emissi)
                #print "the shiny2 is: ", shinin
                glMaterialfv(GL_FRONT, GL_SHININESS, shinin)

            #print "verticesAccum[i] = ", verticesAccum[i]

            #print "texturesAccum[i] = ", texturesAccum[i]

            IPT = verticesAccum[i]
            INT = np.array(texturesAccum[i], dtype=np.float32)

            #print "the texture indices to render are: ", texturestodraw

            print "the number of vertices passed through the draw function: ", len(IPT), " and the index is ", i

            #print "the verticesAccum is: ", IPT
            #print "the normsAccum is: ", INT

            #glActiveTexture(GL_TEXTURE0 + totaltextures - 1)
            #glActiveTexture(GL_TEXTURE1)

            #print "the length of the textureBinds in the draw function is: ", totaltextures

            # to avoid the sky.png at the end of the list
            for fdx in range(len(texturestodraw) - 1):
                if i in texturestodraw[fdx]:
                    glBindTexture(GL_TEXTURE_2D, textureBinds[fdx])
                    print "Success! the fdx is: ", fdx

            print "the number index to look for: ", i

            glEnableClientState( GL_TEXTURE_COORD_ARRAY )

            # activate and specify pointer to vertex array
            glEnableClientState(GL_VERTEX_ARRAY)

            # Enable use of textures
            glEnable(GL_TEXTURE_2D)

            glVertexPointer(3, GL_FLOAT, 0, IPT)
            glTexCoordPointer(2, GL_FLOAT, 0, INT)

            glDrawArrays(GL_TRIANGLES, 0, len(IPT))

            # deactivate vertex arrays after drawing
            glDisableClientState(GL_VERTEX_ARRAY)

            glDisableClientState(GL_TEXTURE_COORD_ARRAY)

            glDisable(GL_TEXTURE_2D)

            #glutSwapBuffers() 
            # has reached the end

            #print "does this work? DDDDDDDDDDDDDDDDDDDDDD ", texturestodraw[len(texturestodraw) - 1][0]
            #print "the total length of this loop is: ", len(translateAccum)
            glPopMatrix()
        
        elif (i == len(translateAccum)):
            drawsky()
        #print "pop successful"

# run the script
if __name__ == "__main__":
    # initializing the mouse presses to no presses
    #global mouseLeftDown 
    #mouseLeftDown = False
    #global mouseRightDown 
    #mouseRightDown = False
    #global mouseMiddleDown 
    #mouseMiddleDown = False

    #global mouseX 
    #mouseX = 0
    #global mouseY 
    #mouseY = 0

    global time
    time = 0.0

    cameraX = 0
    cameraY = 0
    cameraZ = 0

    glutInit(sys.argv)

    # x dimension size
    xRes = int(sys.argv[1])

    # y dimension size
    yRes = int(sys.argv[2])

    # iv file name to input
    ivFile = sys.argv[3]

    # Create a list of the points in space for the lighting function
    worldPoints = []
    # Create a list of textures containing the rgb values
    worldTextures = []

    RealIndices = []
    TextureIndices = []

    # Get a double-buffered, depth-buffer-enabled window, with an
    # alpha channel.
    # These options aren't really necessary but are here for examples.
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

    glutInitWindowSize(xRes, yRes)
    glutInitWindowPosition(300, 100)

    glutCreateWindow("CS171 HW6")

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
    period = pp.Literal(".")
    sharp = pp.Literal("#")

    # Optional added for the additional number for rotation
    parameter = pp.Optional(sharp) + pp.Optional(pp.Word( pp.alphas )) + \
                pp.Optional(pp.Word( pp.alphas ) + period + pp.Word(pp.alphas)) + \
                pp.Optional(leftBracket) + pp.Optional(leftBrace) + \
                pp.Optional(rightBracket) + pp.Optional(rightBrace) + \
                pp.ZeroOrMore(number + pp.Optional(comma))

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

    # if we have a materials sub-block in the the separator block, set materialsActivate
    # to 1. If not, leave at 0.
    materialsActivate = 0

    # if we have not yet read the png file, leave pngActivate at 0.
    # Once we read the png file, set pngActivate to 1
    pngActivate = 0

    # This is if we skip a parameter in the separator block but later get to it
    specialActivate = 0

    # these are vertices and textures accumulated for all separators.
    verticesAccum = []
    texturesAccum = []
    
    global lastRotX
    lastRotX = 0
    global lastRotY
    lastRotY = 0
    global currRotX
    currRotX = 0
    global currRotY
    currRotY = 0

    global numhorizpts
    numhorizpts = 50
   
    global numvertpts
    numvertpts = 50

    # list of texture png names (non-repeating)
    texturesList = []

    # list of texture binding thingies
    textureBinds = []

    # list of textures to draw
    texturestodraw= []

    # as long as we don't reach the end of the file
    while (first != ''):
        firstparse = parameter.parseString(first)

        #print "at the beginning of while loop: ", firstparse

        # if we reach PerspectiveCamera parameter
        if (len(firstparse) != 0 and (firstparse[0] == 'PerspectiveCamera')):
            print "hooray"
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

                    #print "the position of the camera is: ", campos

                # orientation paramter
                elif (firstparse[0] == 'orientation'):
                    x = firstparse[1]
                    y = firstparse[2]
                    z = firstparse[3]
                    angle = firstparse[4]
                    #print "the orientation is: ", x, ", ", y, ", ", z, ", ", angle
                    #print "the params are: ", y

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
            print "the angle is: ", -1*angle*180.0/pi

            # multiply with inverse translation matrix
            glTranslatef(-1.0*cameraX, -1.0*cameraY, -1.0*cameraZ)

            # save the inverse camera matrix
            glPushMatrix()

            global cameraMat
            cameraMat = glGetFloatv(GL_MODELVIEW_MATRIX)

            # calculate the camera matrix
            # World space to camera space
            #cameraMat = np.dot(translateCam, rotationCam)

            # calculate the Perspective Projection matrix
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glFrustum(l, r, b, t, n, f)
            # Save the perspective projection matrix
            #glPushMatrix()
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
                        #totalred += red
                        #totalgreen += green
                        #totalblue += blue
                    
                    first = fo.readline()
                print "light position"
                print lightpos
                print "diffusion"
                print diff
                print "specular"
                print spec
                print "ambient"
                print amb

                glEnable(GL_LIGHTING)
                glLightModelfv(GL_LIGHT_MODEL_AMBIENT, amb)
                glLightfv(GL_LIGHTC[lightidx], GL_AMBIENT, amb)
                #print "the light ambient is: ", amb
                glLightfv(GL_LIGHTC[lightidx], GL_DIFFUSE, diff)
                #print "the light diffuse is: ", diff
                glLightfv(GL_LIGHTC[lightidx], GL_SPECULAR, spec)
                #print "the light specular is: ", spec
                glLightfv(GL_LIGHTC[lightidx], GL_POSITION, lightpos)
                #print "the light position is: ", lightpos
                glEnable(GL_LIGHTC[lightidx])

                #glPushMatrix()

            first = fo.readline()
            firstparse = parameter.parseString(first)

        # Turn on lighting.  You can turn it off with a similar call to
        # glDisable().

        # print "the light matrix is: "
        # print lights
        # if we reach the Separator parameter
        while (len(firstparse) != 0 and ((firstparse[0] == 'Separator') or specialActivate == 1)):
            if (specialActivate == 0):
                first = fo.readline()
                firstparse = parameter.parseString(first)
                
                # transform multiplication, initialized to identity matrix
                totaltransform = np.identity(4)

                totalNorm = np.identity(4)

                translateSep = []
                rotateSep = []
                scalefSep = []

                # to determine which vertices to render
                polygonvertices = []
                texturevertices = []

                uvvertices = []
                # Create a list of the points in space for the lighting function
                worldPoints = []

                # Create a list of textures in terms of UV-coordinates
                UVCoords = []
                
                # Create a list of both vertices and norms indices
                both = []
                # indices of the coordinates
                RealIndices = []

                # Create a list of textures in terms of pixel data (255, 255, 255) style
                # this has been indexed
                TextureIndices = []

            specialActivate = 0
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
                        #rotationSep = rotationMatrix(normalizedrX, normalizedrY, normalizedrZ, rAngle)
                        
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
                    
                #glTranslatef(tX, tY, tZ)
                #glRotatef(rAngle*180.0/pi, normalizedrX, normalizedrY, normalizedrZ)
                #glScalef(sfX, sfY, sfZ)

                # end of Transform block parameter
                if (len(firstparse) != 0 and (firstparse[0] == '}')):
                    first = fo.readline()
                    firstparse = parameter.parseString(first)
                
            while (len(firstparse) != 0 and (firstparse[0] == '#')):
                print "sharp time"
                first = fo.readline()
                firstparse = parameter.parseString(first)

            # entering the Material subparameter
            if (len(firstparse) != 0 and (firstparse[0] == 'Material')):
                # activate since we reached the Material sub-block
                materialsActivate = 1
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
            #print "so so so so "
            #print "amb"
            #print amb
            #print "diff"
            #print diff
            #print "spec"
            #print spec
            #print "emit"
            #print emit
            #print "shiny"
            #print shiny

            while (first.strip() == ''):
                first = fo.readline()
                firstparse = parameter.parseString(first)

            if (len(firstparse) != 0 and (firstparse[0] == 'Texture')):
                print "yay"
                pngActivate = 1
                first = fo.readline()
                firstparse = parameter.parseString(first)
                f = 0
                while (firstparse[f] != '}' and firstparse[f] != ']' and first.strip() != '}'):
                    if (firstparse[f] == 'filename'):

                        # construct the filename from the 3 parsed parts ("file prefix" + "." + "png")
                        filename = firstparse[1] + firstparse[2] + firstparse[3]
                        print "the texture file name is: ", filename

                        # if the file name is unique, add it to the texturesList
                        if str(filename) not in texturesList:
                            newfile = [len(translateAccum)]
                            texturesList.append(str(filename))
                            print "why hello"
                            print texturesList

                            # open the png file for reading and in binary form
                            #openpng = open(filename, 'rb')

                            # read the png data
                            im = PIL.Image.open(filename)

                            im = im.convert("RGBA") 
                            try:
                                row_count, column_count, img_data = im.size[0], im.size[1], im.tostring("raw", "RGBA", 0, -1)
                            except SystemError:
                                row_count, column_count, img_data = im.size[0], im.size[1], im.tostring("raw", "RGBX", 0, -1)


                            # use that fancy module png
                            #pngfile = png.Reader(file=openpng)

                            #img = PIL.Image.open(openpng) # .jpg, .bmp, etc. also work
                            #img_data = np.array(list(img.getdata()), np.int8)

                            #print img_data

                            #row_count, column_count = img_data.shape

                            #row_count, column_count, p, meta = pngfile.read()
                            #print "the contents of the png file is: ", pngfile.read()

                            #pngdata = list(p)

                            #print "the size of the pngdata is: ", len(pngdata)


                            #print "the size of the first element is: ", len(pngdata[0])

                            #print "the list in png file is: ", p

                            # get the data, including the amount of rows and amount of columns
                            #row_count, column_count, pngdata, meta = pngfile.asDirect()

                            #print "the pngdata is: ", pngdata

                            #image_2d = np.vstack(itertools.imap(np.uint16, pngdata))

                            #print "the numpy version is: ", image_2d

                            #image_2d = np.transpose(image_2d)

                            print "The size of the non-numpy version is: ", row_count, " by ", column_count

                            print texturesList
                            print len(texturesList)

                            # initialize generation of texture
                            # It creates a blank texture in GL memory
                            # the lenght of the texturesList is a good indicator of what the id of the texture png is

                            totaltextures = len(texturesList)

                            texture = glGenTextures(1)

                            print "the texture generated from glGenTextures is: ", texture

                            textureBinds.append(texture)

                            print "the textureBinds are: ", textureBinds

                            # Bind texture into OpenGL memory
                            glBindTexture(GL_TEXTURE_2D, textureBinds[totaltextures - 1])

                            #glPixelStorei(GL_UNPACK_ALIGNMENT,1)

                            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

                            #"GL_TEXTURE" + str(len(textureBinds) - 1)

                            # append a new set of indices for a new texture
                            # these indices will be important for glBindTexture
                            # the position of newfile in texturestodraw corresponds to the the draw id for
                            # glBindTexture
                            texturestodraw.append(newfile)

                        else:

                            # if a particular texture is repeated, add the index. These indices are based on
                            # the length of translateAccum
                            texturestodraw[totaltextures - 1].append(len(translateAccum))

                    first = fo.readline()
                    firstparse = parameter.parseString(first)
                # after reading the } character, move on to the next line
                first = fo.readline()
                firstparse = parameter.parseString(first)
                print "after I am done with getting the png file, line says: ", first
                print firstparse

            # entering the Coordinate subparameter
            if (len(firstparse) != 0 and (firstparse[0] == 'Coordinate')):
                first = fo.readline()
                firstparse = parameter.parseString(first)
                if (len(firstparse) != 0 and (firstparse[0] == 'point')):
                    print "point time!"
                    # Create a list of the coordinates
                    coordsList = []
                    #coordsNoTuple = []
                    # to compensate for the point
                    # if the length of the line is greater than 2, then set f (which is the index)
                    # to 2 since "point" and "[" would take the first two indices 0 and 1
                    f = 2

                    # if the length of the line is less than 2 (so if it is point [
                    if (len(firstparse) < 3):
                        first = fo.readline()
                        firstparse = parameter.parseString(first)
                        f = 0

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
                #print "the coordsList is: ", coordsList
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

            #print firstparse

            while (len(firstparse) != 0 and (firstparse[0] == '#')):
                print "sharp time"
                first = fo.readline()
                firstparse = parameter.parseString(first)

            # entering the TextureCoordinate subparameter
            if (len(firstparse) != 0 and (firstparse[0] == 'TextureCoordinate')):
                first = fo.readline()
                firstparse = parameter.parseString(first)
                if (len(firstparse) != 0 and (firstparse[0] == 'point')):
                    #print "wut wut wut wtu wut"
                    # Create a list of the texture points
                    TexturesList = []
                    priorList = []
                    UVList = []

                    # if the length of the line is greater than 2, then set f (which is the index)
                    # to 2 since "point" and "[" would take the first two indices 0 and 1
                    f = 2

                    # if the length of the line is less than 2 (so if it is point [
                    if (len(firstparse) < 3):
                        first = fo.readline()
                        firstparse = parameter.parseString(first)
                        f = 0
                    while (firstparse[f] != ']' and firstparse[f] != '}' and first.strip() != '}'):
                        # read the decimal that is between 0 and 1 for the row and column of the texture
                        xTex = float(firstparse[f])
                        yTex = float(firstparse[f+1])
                        openglyTex = 1.0 - yTex
                        #zVec = float(firstparse[f+2])

                        # the row and column indices
                        # I add 0.01 so for the 12.9999, it goes over 13 and rounds down to 13 like it should
                        #rowTex = int(xTex*row_count + 0.01)
                        #colTex = column_count - int(yTex*column_count + 0.01)

                        # subtract 1 from the row and 1 from the column
                        # this is because we start at 0, and so if we have rowTex = 36, that means the 36th row
                        # out of 256 rows. That would be row # 35 if we start at row 0
                        #numpyrow = rowTex - 1
                        #numpycol = colTex - 1

                        #textureindex = numpyrow*column_count + numpycol

                        # get the coordinates at that particular row and col from the numpy array
                        # we have to take into account that the numpy array is 256x768 instead of
                        # 256 by 256 (so the coordinates are not separate from each other)
                        #textpoint = [1000]*3

                        #textpoint[0] = image_2d[numpyrow][3*numpycol]
                        #textpoint[1] = image_2d[numpyrow][3*numpycol + 1]
                        #textpoint[2] = image_2d[numpyrow][3*numpycol + 2]

                        # we have to flip the numpycol and numpyrow for the array to work with the png pixels
                        # I am guessing that the array was set up strangely via png and itertools modules
                        #trialtext = [1000]*3
                        #trialtext[0] = pngdata[numpyrow][3*numpycol]
                        #trialtext[1] = pngdata[numpyrow][3*numpycol + 1]
                        #trialtext[2] = pngdata[numpyrow][3*numpycol + 2]

                        #tupleText = [xTex, yTex] #[xVec, yVec, zVec]
                        #priorList.append([textpoint[0], textpoint[1], textpoint[2]])    
                        #priorList.append([trialtext[0], trialtext[1], trialtext[2]])


                        # Create a list containing the UV coordinates
                        #UVList.append([xTex, openglyTex])

                        UVList.append([xTex, yTex])

                        #TexturesList.append([textpoint[0], textpoint[1], textpoint[2]])
                        #TexturesList.append([numpyrow, numpycol])
                        first = fo.readline()
                        firstparse = parameter.parseString(first)
                        f = 0
                #print "the UV indices are: ", UVList
                #print "the textures list is: ", priorList
                #print "the indices of the textures list is: ", TexturesList
                first = fo.readline()
                firstparse = parameter.parseString(first)
            while (first.strip() == ''):
                first = fo.readline()
                firstparse = parameter.parseString(first)

            # end of Coordinates block parameter
            if (len(firstparse) != 0 and (firstparse[0] == '}')):
                first = fo.readline()
                firstparse = parameter.parseString(first)
            #print "howd"
            #print TexturesList

            print "wut2"
            #print first
            # start into the IndexedFaceSet block parameter
            if (len(firstparse) != 0 and (firstparse[0] == 'IndexedFaceSet')):
                first = fo.readline()

                # to determine which vertices to render
                polygonvertices = []
                texturevertices = []

                uvvertices = []
                # Create a list of the points in space for the lighting function
                worldPoints = []

                # Create a list of textures in terms of UV-coordinates
                UVCoords = []
                
                # Create a list of both vertices and norms indices
                both = []
                # indices of the coordinates
                RealIndices = []

                # Create a list of textures in terms of pixel data (255, 255, 255) style
                # this has been indexed
                TextureIndices = []

                index = -1
                # for both the coordinates and the normals
                indexforboth = []
                tempindex = []
                firstpoint = -100
                firstparse = parameter.parseString(first)

                while (len(firstparse) != 0 and (firstparse[0] == '#')):
                    print "sharp time"
                    first = fo.readline()
                    firstparse = parameter.parseString(first)

                # for the first row, with the coordIndex as firstparse[0]
                i = 0
                #print "wtf"
                # Go through the line
                space = 0
                while (i < len(firstparse) and firstparse[0] != 'textureCoordIndex'):
                    k = firstparse[i]
                    #print "the term is: ", k
                    #print "the index is: ", i
                    # if the element is a comma, bracket, or coordIndex, then move on to next element
                    while ((k == ',') or (k == '[') or (k == ']') or (k == 'coordIndex')):
                        if (i < len(firstparse) - 1):
                            i += 1
                            k = firstparse[i]
                            #print "the firstparse first is: ", firstparse
                            #print k
                        else:
                            first = fo.readline()
                            firstparse = parameter.parseString(first)
                            i = 0
                            k = firstparse[i]
                            #print "the firstparse is: ", firstparse
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
                        #print polygonvertices
                        if (int(k) != -1):
                            # add the very first point
                            polygonvertices.append(firstpoint)
                            # add the point before current point
                            polygonvertices.append(lastpoint)
                            # add the current point
                            polygonvertices.append(int(k))
                        #print "the index is at : ", int(k)
                        #print "heads will roll"

                    # if there are less than 3 points in the list, add another point
                    elif (len(polygonvertices) < 3):
                        polygonvertices.append(int(k))
                        #print "nooooooo ", int(k)

                    # if reach the end of a face
                    if (k == -1):
                        #print "we go here! ", firstparse
                        # Put the list of 3 points in the big list
                        #worldPoints.append(polygonvertices)
                        #RealIndices.append(polygonvertices[0])
                        #RealIndices.append(polygonvertices[1])
                        #RealIndices.append(polygonvertices[2])
                        #both.append(polygonvertices)
                        # reset the list of 3 points to an empty list
                        polygonvertices = []
                        # if the next index is in the line
                        # we have to add 2 to accomodate that comma
                        if (i+2 < len(firstparse)):
                            firstpoint = int(firstparse[i+2])
                            #print "the new first point is: ", firstpoint
                            #print "the first point is: ", firstpoint
                            polygonvertices.append(firstpoint)
                            # move to next comma after first point
                            i += 2
                        
                        else:
                            first = fo.readline()
                            firstparse = parameter.parseString(first)
                            # if we reach the end bracket as the first element (which means it is by itself)
                            # then move to next line
                            if (firstparse[0] == ']'):
                                first = fo.readline()
                                firstparse = parameter.parseString(first)
                            if (firstparse[0] != 'textureCoordIndex'):
                                firstpoint = int(firstparse[0])
                                polygonvertices.append(firstpoint)
                            i = 0
                        
                        print "hey"
                    i += 1
                    #print "dum dum dum", polygonvertices
                #print "the points indices are: ", worldPoints

                print "life sucks"
          
                #first = fo.readline()
                #firstparse = parameter.parseString(first)
                if (firstparse[0] == 'textureCoordIndex'):
                    firstpoint = -100
                    j = 0
                    while(j < len(firstparse) and first.strip() != '}'):
                        k = firstparse[j]
                        # if the element is a comma, bracket, or coordIndex, then move on to next element
                        while ((k == ',') or (k == '[') or (k == ']') or (k == 'textureCoordIndex')):
                            if (j < len(firstparse) - 1):
                                j += 1
                                k = firstparse[j]
                            else:
                                first = fo.readline()
                                firstparse = parameter.parseString(first)
                                j = 0
                                #print "mhm, ", firstparse
                                k = firstparse[j]
 
                        #print "for normalindex: ", firstparse
                        '''                        
                        # if we don't reach the end of a face
                        if (k != -1):
                            # append the pixel data
                            texturevertices.append(TexturesList[int(k)])
                            # append the uv coordinate
                            uvvertices.append(UVList[int(k)])

                        # if we do reach the end of a face
                        else:
                            # append the group of pixel data
                            worldTextures.append(texturevertices)

                            # append the group of UV coordinates
                            #UVCoords.append(uvvertices)

                            # reset
                            texturevertices = []
                            uvvertices = []
                        j += 1
                        '''
                        
                        # if  we have 3 points (whether we reach 1 or -1), add the triangle
                        # to the big list, and reset the 3point list to have the first point
                        if (len(texturevertices) == 3):
                            print "2never reach here?"
                            worldTextures.append(texturevertices)
                            if (firstpoint == -100):
                                firstpoint = texturevertices[0]
                            TextureIndices.append(texturevertices[0])
                            TextureIndices.append(texturevertices[1])
                            TextureIndices.append(texturevertices[2])
                            lastpoint = texturevertices[2]
                            texturevertices = []
                            #print texturevertices
                            if (int(k) != -1):
                                # add the very first point
                                texturevertices.append(firstpoint)
                                # add the point before current point
                                texturevertices.append(lastpoint)
                                # add the current point
                                texturevertices.append(int(k))
                            print "2the index is at : ", int(k)
                            print "textures will roll"

                        # if there are less than 3 points in the list, add another point
                        elif (len(texturevertices) < 3):
                            texturevertices.append(int(k))
                            print "nooooooo2 ", int(k)
                        # if reach the end of a face
                        if (k == -1):
                            print "we go here!2"
                            # Put the list of 3 points in the big list
                            #worldPoints.append(polygonvertices)
                            #RealIndices.append(polygonvertices[0])
                            #RealIndices.append(polygonvertices[1])
                            #RealIndices.append(polygonvertices[2])
                            #both.append(polygonvertices)
                            # reset the list of 3 points to an empty list
                            texturevertices = []
                            # if the next index is in the line
                            # we have to add 2 to accomodate that comma
                            if (j+2 < len(firstparse)):
                                firstpoint = int(firstparse[j+2])
                                #print "the first point is: ", firstpoint
                                texturevertices.append(firstpoint)
                                # move to next comma after first point
                                j += 2
                            else:
                                first = fo.readline()
                                firstparse = parameter.parseString(first)
                                # if we reach the end bracket as the first element (which means it is by itself)
                                # then move to next line
                                if (firstparse[0] == ']'):
                                    first = fo.readline()
                                    firstparse = parameter.parseString(first)
                                if (firstparse[0] != '}'):
                                    firstpoint = int(firstparse[0])
                                    texturevertices.append(firstpoint)
                                j = 0
                            #print "hey2 ", texturevertices
                        j += 1
                        #print "ho ho ho ho ho ho ho heheheheheheheh ", texturevertices
                        #first = fo.readline()
                        #print "hahahahahahahahahahahahahah
                        #print worldNorms
                                
                    first = fo.readline()
                    firstparse = parameter.parseString(first)
                #print worldPoints

            #print "the world textures are: ", TextureIndices
            #print "wut3"


            # Assign Image data to texture
            # The parameters are:
            # type of texture (GL_TEXTURE_2D)
            # used for mipmapping, 0 for us
            # number of color channels
            # number of pixels across
            # number of pixels up and down
            # border: must be 0
            # format of pixel data
            # type of pixel data
            # data
            if (pngActivate == 1):
                glTexImage2D(GL_TEXTURE_2D, 0, 3, row_count, column_count, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

            # Optional, will implement later
            #glTexParameteri(GL_TEXTURE_2D, )

            # go through the vertices of the textures in worldTextures
            # create a list of all the points and colors in the texture.

            if (len(worldPoints) != 0):
                for i in range(len(worldPoints)):
                    both.append(worldPoints[i])
                #print "both is: "
                #print both
            
            first = fo.readline()

            #print "the coordsList is: "
            #print coordsList
            #print "the vectorsList is: "
            #print vectorsList

            #print "The real indices of vertices are: ", RealIndices
            #print len(RealIndices)

            #print "The real indices of the normals are: ", NormIndices
            #print len(NormIndices)


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

            
            IndexedTexturesTuple = []
            IndexedTextures = []
            for i in range(len(TextureIndices)):
                indextobe = TextureIndices[i]
                if (indextobe != -1):
                    tupleTexture = UVList[int(indextobe)]
                    IndexedTexturesTuple.append(tupleTexture)
                    IndexedTextures.append(tupleTexture[0])
                    IndexedTextures.append(tupleTexture[1])
            
            '''
            print "the RealIndices are: "
            print len(RealIndices)
            print RealIndices
            '''

            #print  "the IndexedPointsTuples are: "
            #print len(IndexedPointsTuple)
            #print IndexedPointsTuple
    
            #print "the TextureIndices are: "
            #print len(TextureIndices)
            #print TextureIndices

            #print "The IndexedTexturesTuple are: "
            #print len(IndexedTexturesTuple)
            #print IndexedTexturesTuple

            print "hi there!"
            #draw1()
            #glutPostRedisplay()
            #first = fo.readline()
            #print "the next line2 is: ", first
            translateAccum.append(translateSep)
            rotateAccum.append(rotateSep)
            scalefAccum.append(scalefSep)

            verticesAccum.append(IndexedPointsTuple)
            texturesAccum.append(IndexedTexturesTuple)

        '''
        if (len(texturesAccum) != 0):
            print "the length of the texturesAccum is nonzero! : ", len(texturesAccum)
            glutDisplayFunc(draw1)

            glutReshapeFunc(resize)

            glutKeyboardFunc(keyfunc)
            glutMouseFunc(mouseCB)
            glutMotionFunc(mouseMotionCB)
        '''
        print "so something goes missing not really just some random message: ", first
        firstparse = parameter.parseString(first)

        print firstparse

        print "the length of firstparse is: ", len(firstparse)
        
        if (len(firstparse) != 0 and firstparse[0] != 'PerspectiveCamera' and firstparse[0] != 'PointLight'
            and firstparse[0] != 'Separator' and firstparse[0] != 'Transform'
            and firstparse[0] != 'Material' and firstparse[0] != 'Texture'
            and firstparse[0] != 'Coordinate' and firstparse[0] != 'point'
            and firstparse[0] != 'TextureCoordinate' and firstparse[0] != 'IndexedFaceSet'
            and firstparse[0] != 'coordIndex' and firstparse[0] != 'textureCoordIndex'):
            first = fo.readline()
            firstparse = parameter.parseString(first)
        if (len(firstparse) == 0 or firstparse[0] == '#'):
            first = fo.readline()
            firstparse = parameter.parseString(first)

        if (len(firstparse) != 0 and (firstparse[0] == 'Transform' or firstparse[0] == 'Material'
            or firstparse[0] == 'Texture' or firstparse[0] == 'Coordinate'
            or firstparse[0] == 'TextureCoordinate' or firstparse[0] == 'IndexedFaceSet')):
            specialActivate = 1
        '''
        first = fo.readline()
        firstparse = parameter.parseString(first)
        '''
        print "the next line1 is: ", firstparse
        if (first == ''):
            print "hey! it ended!"
            #print "the translateAccum is: ", translateAccum
            #print "the rotateAccum is: ", rotateAccum
            #print "the scalefAccum is: ", scalefAccum

            if (materialsActivate == 1):
                print "the ambientAccum: ", ambientAccum
                print "the diffuseAccum: ", diffuseAccum
                print "the specularAccum: ", specularAccum
                print "the shininess is: ", shininess

            #print "the verticesAccum is: ", verticesAccum
            #print "the texturesAccum is: ", texturesAccum
    fo.close()

    #skyvertices = np.zeros([100, 100])
    #skynorms = np.zeros([100, 100])

    skyvertices = []
    skynorms = []

    #skytexture = []

    # go through the rows
    for srow in range(numvertpts):
        skyrow = []
        skynormsrow = []
        #skytextrow = []
        # go through the columns
        for scol in range(numhorizpts):
            x = scol/float(numhorizpts) * 2.0 - 1.0
            y = srow/float(numvertpts) * 2.0 - 1.0

            #skyrow.append([x, -1.0, y])

            skyrow.append([x, 0.03*cos(20.0*(x*x + y*y + 4*time)) - 1.0, y])

            #skyvertices[srow][scol] = np.append(skyvertices[srow][scol], [[cos(x + y)]]) 
            #skyvertices[srow][scol] = np.append(skyvertices[srow][scol], [[y]])

            dfdx = -2.0*x*0.03*20.0*sin(20.0*(x*x + y*y + 4.0*time))
            dfdy = -2.0*y*0.03*20.0*sin(20.0*(x*x + y*y + 4.0*time))

            normalizingfactor = sqrt(dfdx*dfdx + dfdy*dfdy + 1.0)
            #skynormsrow.append([dfdx/normalizingfactor, dfdy/normalizingfactor, 1.0/normalizingfactor])

            skynormsrow.append([dfdx/float(normalizingfactor), -1.0/normalizingfactor, dfdy/float(normalizingfactor)])
            #skynorms[srow][scol] = np.append(skynorms[srow][scol], [[1.0]])
            #skynorms[srow][scol] = np.append(skynorms[srow][scol], [[-1.0*dfdy]])

            #skytextrow.append([scol/float(numhorizpts), srow/float(numvertpts)])

        skyvertices.append(skyrow)
        skynorms.append(skynormsrow)
        #skytexture.append(skytextrow)

    #print "the skyvertices are: ", skyvertices

    global indexedskyvertex
    indexedskyvertex = []

    global indexedskynorms
    indexedskynorms = []

    #global indexedskytexture
    #indexedskytexture = []

    # now index these vertices
    # we use 99 instead of 100 to avoid overflow
    for rowidx in range(numvertpts - 1):
        bottom = skyvertices[rowidx + 1]
        top = skyvertices[rowidx]

        bottomnorm = skynorms[rowidx + 1]
        topnorm = skynorms[rowidx]

        '''
        bottomtext = skytexture[rowidx + 1]
        toptext = skytexture[rowidx]
        '''

        for colidx in range(numhorizpts - 1):

            # make sure it is counterclockwise for backculling
            #  *--*
            #  | /
            #  |/
            #  *
            # this is for the top triangle
            # this is for the vertices
            topleft = top[colidx]
            topright = top[colidx + 1]
            bottomleft = bottom[colidx]
    
            indexedskyvertex.append(topright)
            indexedskyvertex.append(topleft)
            indexedskyvertex.append(bottomleft)

            # this is for the norms
            topleftnorm = topnorm[colidx]
            toprightnorm = topnorm[colidx + 1]
            bottomleftnorm = bottomnorm[colidx]

            indexedskynorms.append(toprightnorm)
            indexedskynorms.append(topleftnorm)
            indexedskynorms.append(bottomleftnorm)

            '''
            # this is for the textures
            toplefttext = toptext[colidx]
            toprighttext = toptext[colidx + 1]
            bottomlefttext = bottomtext[colidx]

            indexedskytexture.append(toprighttext)
            indexedskytexture.append(toplefttext)
            indexedskytexture.append(bottomlefttext)
            '''

            # make sure it is counter-clockwise for backculling
            #     *
            #    /|
            #   / |
            #  *--*
            # this is for the bottom triangle
            # this is for the vertices
            bottomright = bottom[colidx + 1]

            indexedskyvertex.append(bottomleft)
            indexedskyvertex.append(bottomright)
            indexedskyvertex.append(topright)

            # this is for the norms
            bottomrightnorm = bottomnorm[colidx]

            indexedskynorms.append(bottomleftnorm)
            indexedskynorms.append(bottomrightnorm)
            indexedskynorms.append(toprightnorm)

            '''
            # this is for the textures
            bottomrighttext = bottomtext[colidx]
            
            indexedskytexture.append(bottomlefttext)
            indexedskytexture.append(bottomrighttext)
            indexedskytexture.append(toprighttext)
            '''

    #print "the vertices indexed for sky are: ", indexedskyvertex 

    newfile = [len(translateAccum) + 1]

    # takes care of the sky data
    sky = PIL.Image.open("sky.png")

    sky = sky.convert("RGBA") 
    try:
        row_sky, column_sky, sky_data = sky.size[0], sky.size[1], sky.tostring("raw", "RGBA", 0, -1)
    except SystemError:
        row_sky, column_sky, sky_data = sky.size[0], sky.size[1], sky.tostring("raw", "RGBX", 0, -1)

    print "the row count and column count for the sky png is: ", row_sky, " by ", column_sky

    texturesList.append("sky.png")

    totaltextures = len(texturesList)

    texture = glGenTextures(1)

    print "the texture generated from glGenTextures is: ", texture

    textureBinds.append(texture)

    # Bind texture into OpenGL memory
    glBindTexture(GL_TEXTURE_2D, textureBinds[totaltextures - 1])

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)    

    texturestodraw.append(newfile)

    glTexImage2D(GL_TEXTURE_2D, 0, 3, row_sky, column_sky, 0, GL_RGBA, GL_UNSIGNED_BYTE, sky_data)

    #print "the sky data is: ", sky_data

    glutDisplayFunc(draw1)

    glutIdleFunc(idle)

    #glutPostRedisplay()

    glutReshapeFunc(resize)

    glutKeyboardFunc(keyfunc)
    glutMouseFunc(mouseCB)
    glutMotionFunc(mouseMotionCB)
    #mvm = (GLfloat * 16)()

    glutMainLoop()
    glPopMatrix()
