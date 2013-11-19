#!/usr/bin/python

import pyparsing as pp
import sys
import os
import numpy as np
from math import*
import getopt
#from lightingfunction import lightfunc
#from raster import *

opts,operands = getopt.getopt(sys.argv[4:], "e", ["eyelight"])

print opts
print operands

dimlight = 0

for option in opts:
    print option[0]
    if option[0] == '--eyelight':
        print "hello"
        dimlight = 1

def zeroclip(X):
    #print "the dimensions are: ", X.shape
    zeroclist = np.zeros(X.shape[0])
    for i in range(len(X)):
        if (X[i] > 0.0):
            zeroclist[i] = X[i]
    return zeroclist

def oneclip(X):
    oneclist = np.zeros(X.shape[0])
    for i in range(len(X)):
        if (X[i] < 1.0):
            oneclist[i] = X[i]
        else:
            oneclist[i] = 1.0
    return oneclist

def unit(x):
    if np.linalg.norm(x) != 0.0:
        return x/np.linalg.norm(x)
    else:
        return 0.0

#unit(x) = x / |x| when |x| != 0.0 else 0.0

def lightfunc(n, v, material, lights, camerapos):
    # let n = surface normal (nx,ny,nz)
    # let v = point in space (x,y,z)
    # let lights = [light0, light1, ... ]
    # let camerapos = (x,y,z)
    
    scolor = material[2]  # specular (r,g,b)
    dcolor = material[1]  # diffuse (r,g,b)
    acolor = material[0]  # ambient (r,g,b)
    shiny =  material[3][0]  # shiny (a scalar, an exponent >= 0)

    #print "the shininess is: ", material[3]
    #print "the ambient color is: ", acolor
    #print "the diffuse color is: ", dcolor
    #print "the specular color is: ", scolor

    # start off the diffuse and specular at pitch black
    diffuse = np.array([0.0, 0.0, 0.0])
    specular = np.array([0.0, 0.0, 0.0])
    # copy the ambient color (for the eyelight ex/cred code, you 
    # can change it here to rely on distance from the camera)
    ambient = acolor

    #print lights.shape[0]

    for i in range(1, (lights.shape[0] - 1)/2 + 1):

        # get the light position and color from the light
        # let lx = light position (x,y,z)
        # let lc = light color (r,g,b)
        lx = lights[2*i - 1]
        lc = lights[2*i]

        #print "the lx is: ", lx

        #print "the lc is: "
        #print lc

        # first calculate the addition this light makes
        # to the diffuse part

        #print "the matrices are: ", lc
        #print np.inner(n, unit(lx -v))
        ddiffuse = zeroclip(np.dot(lc, (np.inner(n, unit(lx - v)))))
        # print "the ddiffuse is: ", ddiffuse
        # accumulate that
        diffuse += ddiffuse

        # calculate the specular exponent
        k = np.inner(n, unit(unit(camerapos - v) + unit(lx - v)))
        if k <= 0:
            k = 0
        # calculate the addition to the specular highlight
        # k^shiny is a scalar, lc is (r,g,b)
        dspecular = zeroclip(pow(k, shiny) * lc)

        # print "the dspecular is: ", dspecular
        # acumulate that
        specular += dspecular

    #print "the diffuse is: ", diffuse

    # after working on all the lights, clamp the diffuse value to 1
    d = oneclip(diffuse)
    # note that d,dcolor,specular and scolor are all (r,g,b).
    # * here represents component-wise multiplication
    # print d
    
    #print dcolor
    #print specular
    #print scolor
    #print "The specular is: ", specular
    rgb = oneclip(ambient + d*dcolor + specular*scolor)
    return rgb

def rotationMatrix(a, b, c, theta):
    return np.array([[a**2 + (1 - a**2)*cos(theta), a*b*(1-cos(theta)) - c*sin(theta), a*c*(1-cos(theta)) + b*sin(theta), 0],
                    [a*b*(1-cos(theta)) + c*sin(theta), b**2 + (1-b**2)*cos(theta), b*c*(1-cos(theta)) - a*sin(theta), 0],
                    [a*c*(1-cos(theta)) - b*sin(theta), b*c*(1-cos(theta)) + a*sin(theta), c**2 + (1-c**2)*cos(theta), 0],
                    [0, 0, 0, 1]])

def translateMatrix(a, b, c):
    return np.array([[1, 0, 0, a], 
                    [0, 1, 0, b], 
                    [0, 0, 1, c], 
                    [0, 0, 0, 1]])

def scalefactorMatrix(a, b, c):
    return np.array([[a, 0, 0, 0], 
                   [0, b, 0, 0], 
                   [0, 0, c, 0], 
                   [0, 0, 0, 1]])

fo = sys.stdin

# whether to pick flat, gourad, or phong shading
# 0 is flat, 1 is gouraud, 2 is phong
shade = int(sys.argv[1])

# x dimension size
xRes = int(sys.argv[2])

# y dimension size
yRes = int(sys.argv[3])

#if (len(sys.argv) == 5):
# eyelight parameter that adds a dim light
    

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

# make a list of all the pixels of the window
pixel = [0]*xRes*yRes

# make a list of the z-buffers of the pixels
zed = [float(1000)]*xRes*yRes

# Must be called before any drawing occurs.  Initializes window variables for
#  proper integer pixel coordinates.
def initDraw(xMin, xMax, yMin, yMax, xr, yr):
    global windowXMin, windowXMax, windowYMin, windowYMax, xRes, yRes
    windowXMin = xMin
    windowXMax = xMax
    windowYMin = yMin
    windowYMax = yMax
    xRes = xr
    yRes = yr

# Helper function to convert a real x position into an integer pixel coord.
def getPixelX(x):
    return int((x - windowXMin) / (windowXMax - windowXMin) * xRes)

# Helper function to convert a real y position into an integer pixel coord.
def getPixelY(y):
    return int((y - windowYMin) / (windowYMax - windowYMin) * yRes)

# Helper function f as defined in the course text.  Not sure exactly what
#  it means, on a conceptual level.
def fhelp(vert0, vert1, x, y):
    x0 = vert0[0]
    y0 = vert0[1]
    x1 = vert1[0]
    y1 = vert1[1]
    return float((y0 - y1) * x + (x1 - x0) * y + x0 * y1 - x1 * y0)


# The meat of the package.  Takes three vertices in arbitrary order, with each
#  vertex consisting of an x and y value in the first two data positions, and
#  any arbitrary amount of extra data, and calls the passed in function on
#  every resulting pixel, with all data values magically interpolated.
# The function, drawPixel, should have the form
#    drawPixel(x, y, data)
# where x and y are integer pixel coordinates, and data is the interpolated
# data in the form of a tuple. 
# verts should be a (3-element) list of tuples, the first 2 elements in each
# tuple being X and Y, and the remainder whatever other data you want
# interpolated such as normal coordinates or rgb values
def raster(verts, shadingType, material, lights, campos, matrix):
    xMin = xRes + 1
    yMin = yRes + 1
    xMax = yMax = -1

    coords = [ (getPixelX(vert[0]), getPixelY(vert[1])) for vert in verts ]

    coordsX = verts[0]
    coordsY = verts[1]
    coordsZ = verts[2]

    lenX = len(coordsX)

    # find the bounding box
    for c in coords:
        if c[0] < xMin: xMin = c[0]
        if c[1] < yMin: yMin = c[1]
        if c[0] > xMax: xMax = c[0]
        if c[1] > yMax: yMax = c[1]

    # normalizing values for the barycentric coordinates
    # not sure exactly what's going on here, so read the textbook
    fAlpha = fhelp(coords[1], coords[2], coords[0][0], coords[0][1])
    fBeta = fhelp(coords[2], coords[0], coords[1][0], coords[1][1])
    fGamma = fhelp(coords[0], coords[1], coords[2][0], coords[2][1])

    if abs(fAlpha) < .0001 or abs(fBeta) < .0001 or abs(fGamma) < .0001:
        return

    print "checkpoint"

    # go over every pixel in the bounding box
    for y in range(max(yMin, 0), min(yMax, yRes)):
        for x in range(max(xMin, 0), min(xMax, xRes)):
            # calculate the pixel's barycentric coordinates
            alpha = fhelp(coords[1], coords[2], x, y) / fAlpha
            beta = fhelp(coords[2], coords[0], x, y) / fBeta
            gamma = fhelp(coords[0], coords[1], x, y) / fGamma

            #print "the alpha, beta and gamma are: ", alpha, ", ", beta, ", ", gamma

            data = [-1000]*3

            # if the coordinates are positive, do the next check
            if alpha >= 0 and beta >= 0 and gamma >= 0:
                # interpolate the data for either gouraud or phong
                if (shadingType != 0):
                    data = [0]*lenX
                    for i in range(lenX):
                        data[i] = alpha * coordsX[i] + beta * coordsY[i] + gamma * coordsZ[i]
                # interpolate the data for flat
                elif (shadingType == 0):
                    data = [0]*6
                    for i in range(3):
                        data[i] = alpha * coordsX[i] + beta * coordsY[i] + gamma * coordsZ[i]
                    for i in range(3, 6):
                        data[i] = coordsX[i]

            # and finally, draw the pixel
            if data[2] >= -1:

                # in the case where we have phong shading, use the light function
                if (shadingType == 2 and zed[(yRes - y - 1)*xRes + x - 1] > data[2]):
                    
                    ndcX = float(x)/float(xRes) * (windowXMax - windowXMin) + windowXMin
                    ndcY = float(y)/float(yRes) * (windowYMax - windowYMin) + windowYMin
                    newpoint = np.dot(matrix, np.array([[ndcX], 
                                                        [ndcY], 
                                                        [data[2]], 
                                                        [1.0]]))
                    #print newpoint
                    # this is for the norms
                    newnorm = np.array([data[3], data[4], data[5]])
                    # this is for the point in light function
                    realpt = np.array([newpoint.item(0)/newpoint.item(3), 
                                       newpoint.item(1)/newpoint.item(3), 
                                       newpoint.item(2)/newpoint.item(3)])
                    #print realpt
                    rgb = lightfunc(newnorm/np.linalg.norm(newnorm), realpt, material, lights, campos)

                    #print "the rgb is: ", rgb

                    pixel[(yRes - y - 1)*xRes + x - 1] = [rgb[0], rgb[1], rgb[2]]
                    #print (yRes - y - 1)*xRes + x - 1
                    zed[(yRes - y - 1)*xRes + x - 1] = data[2]
                # print "the index in question is: ", (yRes - y)*xRes + x
                # the format of colortuplewithZ is (X, Y, Z, R, G, B), where R, G, B is the color
                elif (shadingType != 2 and zed[(yRes - y - 1)*xRes + x - 1] > data[2]):
                    pixel[(yRes - y - 1)*xRes + x - 1] = [data[3], data[4], data[5]]
                    zed[(yRes - y - 1)*xRes + x - 1] = data[2]

# split the text between file name and file extension
fileName, fileExtension = os.path.splitext(os.readlink('/proc/self/fd/0'))

# create the ppm file
ppm = open(fileName + ".ppm", "w")
ppm.write("P3 \n")
ppm.write(str(xRes) + " " + str(yRes) + "\n")
ppm.write(str(255) + "\n")

first = fo.readline()

lights = np.zeros([1, 3])

cameraX = 0
cameraY = 0
cameraZ = 0

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
                cameraX = firstparse[1]
                cameraY = firstparse[2]
                cameraZ = firstparse[3]
                campos = np.array([cameraX, cameraY, cameraZ])

            # orientation paramter
            elif (firstparse[0] == 'orientation'):
                x = firstparse[1]
                y = firstparse[2]
                z = firstparse[3]
                angle = firstparse[4]
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
        translateCam = translateMatrix(cameraX, cameraY, cameraZ)

        # rotation matrix (for orientation of camera)
        rotationCam = rotationMatrix(x, y, z, angle)

        # calculate the camera matrix
        # World space to camera space
        cameraMat = np.dot(translateCam, rotationCam)

        # calculate the Perspective Projection matrix
        perspectiveProj = np.array([[2.0*n / (r - l), 0, float(r + l)/(r - l), 0], 
                                    [0, 2.0*n / (t - b), float(t + b) / (t - b), 0],
                                    [0, 0, -1.0*(f + n)/(f - n), -2.0*(f*n)/(f - n)],
                                    [0, 0, -1, 0]])

    totalred = 0
    totalgreen = 0
    totalblue = 0

    # if we reach PointLight parameter
    while (len(firstparse) != 0 and (firstparse[0] == 'PointLight')):
        first = fo.readline()
        # default position and color
        lightX = 0
        lightY = 0
        lightZ = 1
        red = 1
        green = 1
        blue = 1
        # if there is a blank line, read another main parameter
        while (first.strip() != ''):
            firstparse = parameter.parseString(first)
            # location parameter
            if (firstparse[0] == 'location'):
                lightX = firstparse[1]
                lightY = firstparse[2]
                lightZ = firstparse[3]
            # color parameter
            elif (firstparse[0] == 'color'):
                red = firstparse[1]
                green = firstparse[2]
                blue = firstparse[3]
                totalred += red
                totalgreen += green
                totalblue += blue
            first = fo.readline()
        # print "light location is: ", lightX, ", ", lightY, ", ", lightZ
        lights = np.append(lights,[[lightX, lightY, lightZ]], axis = 0)
        lights = np.append(lights,[[red, green, blue]], axis = 0)
        first = fo.readline()
        firstparse = parameter.parseString(first)

    # print "the light matrix is: "
    # print lights
    # if we reach the Separator parameter
    while (len(firstparse) != 0 and (firstparse[0] == 'Separator')):
        first = fo.readline()
        firstparse = parameter.parseString(first)

        # add the dimlights if enabled
        if (dimlight == 1):
            lights = np.append(lights, [[cameraX - 1, cameraY, cameraZ]], axis = 0)
            lights = np.append(lights, [[0.2*totalred, 0.2*totalgreen, 0.2*totalblue]], axis = 0)
            dimlight = 0
        
        # transform multiplication, initialized to identity matrix
        totaltransform = np.identity(4)

        totalNorm = np.identity(4)

        # if we reach the Transform sub-parameter
        while (len(firstparse) != 0 and (firstparse[0] == 'Transform')):
            first = fo.readline()
            firstparse = parameter.parseString(first)
            
            translate = rotate = scaleFactor = ''

            # as long as we aren't at the end of the Transform parameter
            while (firstparse[0] != '}'):

                # translation
                if (firstparse[0] == 'translation'):
                    translate = firstparse[0]
                    tX = firstparse[1]
                    tY = firstparse[2]
                    tZ = firstparse[3]
                    translateSep = translateMatrix(tX, tY, tZ)
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
                    rotationSep = rotationMatrix(normalizedrX, normalizedrY, normalizedrZ, rAngle)
                # scale factor
                elif (firstparse[0] == 'scaleFactor'):
                    scaleFactor = firstparse[0]
                    sfX = firstparse[1]
                    sfY = firstparse[2]
                    sfZ = firstparse[3]
                    scalefactorSep = scalefactorMatrix(sfX, sfY, sfZ)
                first = fo.readline()
                firstparse = parameter.parseString(first)

            # calculate the separator matrix (S = TRS)
            # it is for Object Space to World Space
            # also for transformed normal matrix

            if (translate == '' and rotate == 'rotation' and scaleFactor == ''):
                S = rotationSep
                normS = rotationSep
            elif (translate == 'translation' and rotate == '' and scaleFactor == ''):
                S = translateSep
                normS = np.array([[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])
            elif (translate == '' and rotate == '' and scaleFactor == 'scaleFactor'):
                S = scalefactorSep
                normS = scalefactorSep
            elif (translate == 'translation' and rotate == 'rotation' and scaleFactor == ''):
                S = np.dot(translateSep, scalefactorSep)
                normS = rotationSep
            elif (translate == 'translation' and rotate == '' and scaleFactor == 'scaleFactor'):
                S = np.dot(translateSep, scalefactorSep)
                normS = scalefactorSep
            elif (translate == '' and rotate == 'rotation' and scaleFactor == 'scaleFactor'):
                S = np.dot(rotationSep, scalefactorSep)
                normS = np.dot(rotationSep, scalefactorSep)
            elif (translate == 'translation' and rotate == 'rotation' and scaleFactor == 'scaleFactor'):
                SIntermediate = np.dot(rotationSep, scalefactorSep)
                S = np.dot(translateSep, SIntermediate)
                normS = np.dot(rotationSep, scalefactorSep)

            #print "the normS is: ", normS

            # Multiply up the transform matrices
            totaltransform = np.dot(totaltransform, S)

            # multiply up the transform matrices (and take the the inverse and transpose) for normals
            totalNorm = np.dot(totalNorm, np.transpose(np.linalg.inv(normS)))

            # end of Transform block parameter
            if (len(firstparse) != 0 and (firstparse[0] == '}')):
                first = fo.readline()
                firstparse = parameter.parseString(first)

            # calculate camera space to NDC (Normalized Device Coordinate) Space
            transformInter = np.dot(perspectiveProj, np.linalg.inv(cameraMat))
            transformMat = np.dot(transformInter, totaltransform)

            # calculate normal transformation matrix
            # normTransform = np.transpose(np.linalg.inv(totalNorm))

        # entering the Material subparameter
        if (len(firstparse) != 0 and (firstparse[0] == 'Material')):
            first = fo.readline()
            # this is determine if the parameter is included or not
            ambX = -20
            diffX = -20
            specX = -20
            shiny = -20
            # read lines until we reach the ending curly brace
            while (first.strip() != '}'):
                firstparse = parameter.parseString(first)
                # ambient color parameter
                if (firstparse[0] == 'ambientColor'):
                    ambX = firstparse[1]
                    ambY = firstparse[2]
                    ambZ = firstparse[3]
                # diffuse color parameter
                elif (firstparse[0] == 'diffuseColor'):
                    diffX = firstparse[1]
                    diffY = firstparse[2]
                    diffZ = firstparse[3]
                # specular color parameter
                elif (firstparse[0] == 'specularColor'):
                    specX = firstparse[1]
                    specY = firstparse[2]
                    specZ = firstparse[3]
                # diffuse color parameter
                elif (firstparse[0] == 'shininess'):
                    shiny = firstparse[1]
                first = fo.readline()
            # if the parameters are not there, set to default colors
            if (ambX == -20):
                ambX = 0.2
                ambY = 0.2
                ambZ = 0.2
            if (diffX == -20):
                diffX = 0.8
                diffY = 0.8
                diffZ = 0.8
            if (specX == -20):
                specX = 0.0
                specY = 0.0
                specZ = 0.0
            if (shiny == -20):
                shiny = 0.2
            material = np.array([[ambX, ambY, ambZ],
                                 [diffX, diffY, diffZ],
                                 [specX, specY, specZ],
                                 [shiny, 0.0, 0.0]])
            
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
                # Create a list of world space coordinates
                worldcoords = []
                # to compensate for the point
                f = 2
                while (firstparse[f] != ']' and firstparse[f] != '}' and first.strip() != '}'):
                    xArr = float(firstparse[f])
                    yArr = float(firstparse[f+1])
                    zArr = float(firstparse[f+2])

                    newmat = np.dot(transformMat, np.array( [[xArr], 
                                                             [yArr], 
                                                             [zArr], 
                                                             [1.0]] ))
                    newworld = np.dot(totaltransform, np.array( [[xArr], [yArr], [zArr], [1.0]] ))

                    #print "the total transform is: ", totaltransform

                    newX = newmat[0,0]/newmat[3,0]
                    newY = newmat[1,0]/newmat[3,0]
                    newZ = newmat[2,0]/newmat[3,0]

                    worldX = newworld[0,0]/newworld[3,0]
                    worldY = newworld[1,0]/newworld[3,0]
                    worldZ = newworld[2,0]/newworld[3,0]

                    coordsList.append(newX)
                    coordsList.append(newY)
                    coordsList.append(newZ)

                    worldcoords.append(worldX)
                    worldcoords.append(worldY)
                    worldcoords.append(worldZ)

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

                    normNew = np.dot(totalNorm, np.array( [[xVec], 
                                                           [yVec], 
                                                           [zVec], 
                                                           [1.0]] ))
                    normX = normNew[0,0]/normNew[3,0]
                    normY = normNew[1,0]/normNew[3,0]
                    normZ = normNew[2,0]/normNew[3,0]
                    vectorsList.append(normX)
                    vectorsList.append(normY)
                    vectorsList.append(normZ)
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
            toRender = []
            # Create a list of the points in space for the lighting function
            worldPoints = []
            # Create a list of norms in world space for the lighting function
            worldNorms = []
            # indices of the coordinates
            #indices = []
            #indicestoRend = []

            index = -1
            # for both the coordinates and the normals
            indexforboth = []
            tempindex = []
            # read until the end of the IndexedFaceSet block parameter
            while(first.strip() != '}'):
                firstparse = parameter.parseString(first)

                # for the first row, with the coordIndex as firstparse[0]
                i = 0
                #print "wtf"
                # Go through the line
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

                    worldPoints.append(k)
                    index += 1
                        
                    # if k does not equal to 1, append the coordinates
                    if (k != -1):
                        x1 = coordsList[int(3*k)]
                        y1 = coordsList[int(3*k) + 1]
                        z1 = coordsList[int(3*k) + 2]
                        polygonvertices.append(x1)
                        polygonvertices.append(y1)
                        polygonvertices.append(z1)
                        #indices.append(k)
                        tempindex.append(index)
                    else:
                        #print polygonvertices
                        #print "the number of points is: ", len(polygonvertices)/2
                        j = 3
                        while (j < len(polygonvertices) - 5):
                            # actual point
                            rbackcull = np.array([polygonvertices[0], polygonvertices[1], polygonvertices[2]])
                            # index of the coord point
                            #bcullIndices1 = indices[0]
                            # index of indices
                            tempindex1 = tempindex[0]
                            sbackcull = np.array([polygonvertices[j], polygonvertices[j+1], polygonvertices[j+2]])
                            #bcullIndices2 = indices[j/3]
                            tempindex2 = tempindex[j/3]
                            tbackcull = np.array([polygonvertices[j+3], polygonvertices[j+4], polygonvertices[j+5]])
                            #bcullIndices3 = indices[j/3 + 1]
                            tempindex3 = tempindex[j/3 + 1]

                            a = np.cross(rbackcull - sbackcull, sbackcull - tbackcull)
                            #print rbackcull
                            #print a
                            # if first 3 points, and the z value is greater than 0
                            if (j == 3 and a[2] > 0):
                                toRender.append(rbackcull[0])
                                toRender.append(rbackcull[1]) 
                                toRender.append(sbackcull[0])
                                toRender.append(sbackcull[1])
                                toRender.append(tbackcull[0])
                                toRender.append(tbackcull[1])
                                #indicestoRend.append(bcullIndices1)
                                #indicestoRend.append(bcullIndices2)
                                #indicestoRend.append(bcullIndices3)

                                indexforboth.append(tempindex1)
                                indexforboth.append(tempindex2)
                                indexforboth.append(tempindex3)
                            # if past first 3 points, and the z value is greater than 0,
                            # then render last point of second or third or fourth or so on triangle
                            elif (j > 3 and a[2] > 0):
                                toRender.append(tbackcull[0])
                                toRender.append(tbackcull[1])
                                #indicestoRend.append(bcullIndices3)

                                indexforboth.append(tempindex3)
                            j += 3
                        # if we have a few points in toRender and our last element is not -1
                        if (len(toRender) > 0 and toRender[-1] != -1):
                            toRender.append(-1)
                            #indicestoRend.append(-1)
                            indexforboth.append(index)
                        # once do the backcull, reset polygonvertices to empty
                        polygonvertices = []
                        # also result the indices
                        indices = []
                        # also the temp indices
                        tempindex = []
                    i += 1
                #print toRender
                #print indicestoRend
                #print indexforboth
                #print len(indicestoRend)
                #print len(indexforboth)
      
                first = fo.readline()
                firstparse = parameter.parseString(first)
                if (firstparse[0] == 'normalIndex'):
                    i = 0
                    while(i < len(firstparse)):
                        k = firstparse[i]
                        # if the element is a comma, bracket, or coordIndex, then move on to next element
                        while ((k == ',') or (k == '[') or (k == ']') or (k == 'normalIndex')):
                            if (i < len(firstparse) - 1):
                                i += 1
                                k = firstparse[i]
                            else:
                                first = fo.readline()
                                firstparse = parameter.parseString(first)
                                i = 0
                                k = firstparse[i]
                        
                        worldNorms.append(k)

                        # put the 1st point in x1, y1
                        # multiply by 1.0/2.0 because the origin is in the center of the window
                        x1 = k
                        y1 = k

                        i += 1
                        #print "for normalindex: ", firstparse
                            
                first = fo.readline()
                firstparse = parameter.parseString(first)
            #print worldNorms

            # initialize the window stuff
            initDraw(-1, 1, -1, 1, xRes, yRes)

            # let's try out the lighting function
            # list of the rgb values at each point
            rgbList = [0]

            j = 0

            while (j < len(indexforboth)):
                oneface = []

                # create a list of the points that make the face (does not include -1)
                while (worldNorms[indexforboth[j]] != -1):
                    oneface.append(indexforboth[j])
                    j += 1

                # for the mainstay point - we get the location of the first index
                normIdx1 = worldNorms[oneface[0]]
                ptsIdx1 = worldPoints[oneface[0]]

                normX1 = vectorsList[int(3*normIdx1)]
                normY1 = vectorsList[int(3*normIdx1) + 1]
                normZ1 = vectorsList[int(3*normIdx1) + 2]

                #normalizedX1 = normX1/(sqrt(pow(normX1, 2) + pow(normY1, 2) + pow(normZ1, 2)))
                #normalizedY1 = normY1/(sqrt(pow(normX1, 2) + pow(normY1, 2) + pow(normZ1, 2)))
                #normalizedZ1 = normZ1/(sqrt(pow(normX1, 2) + pow(normY1, 2) + pow(normZ1, 2)))

                pointX1 = worldcoords[int(3*ptsIdx1)]
                pointY1 = worldcoords[int(3*ptsIdx1) + 1]
                pointZ1 = worldcoords[int(3*ptsIdx1) + 2]

                # get the point that are in camera space to raster
                rasterX1 = coordsList[int(3*ptsIdx1)]
                rasterY1 = coordsList[int(3*ptsIdx1) + 1]
                rasterZ1 = coordsList[int(3*ptsIdx1) + 2]

                i = 1
                while (i < len(oneface) - 1):

                    # 2nd point
                    idxboth2 = oneface[i]
                    normIdx2 = worldNorms[idxboth2]
                    ptsIdx2 = worldPoints[idxboth2]

                    normX2 = vectorsList[int(3*normIdx2)]
                    normY2 = vectorsList[int(3*normIdx2) + 1]
                    normZ2 = vectorsList[int(3*normIdx2) + 2]

                    pointX2 = worldcoords[int(3*ptsIdx2)]
                    pointY2 = worldcoords[int(3*ptsIdx2) + 1]
                    pointZ2 = worldcoords[int(3*ptsIdx2) + 2]

                    # get the 2nd point that are in camera space to raster
                    rasterX2 = coordsList[int(3*ptsIdx2)]
                    rasterY2 = coordsList[int(3*ptsIdx2) + 1]
                    rasterZ2 = coordsList[int(3*ptsIdx2) + 2]

                    # 3rd point
                    idxboth3 = oneface[i+1]
                    normIdx3 = worldNorms[idxboth3]
                    ptsIdx3 = worldPoints[idxboth3]

                    normX3 = vectorsList[int(3*normIdx3)]
                    normY3 = vectorsList[int(3*normIdx3) + 1]
                    normZ3 = vectorsList[int(3*normIdx3) + 2]

                    pointX3 = worldcoords[int(3*ptsIdx3)]
                    pointY3 = worldcoords[int(3*ptsIdx3) + 1]
                    pointZ3 = worldcoords[int(3*ptsIdx3) + 2]

                    # get the 3rd point that are in camera space to raster
                    rasterX3 = coordsList[int(3*ptsIdx3)]
                    rasterY3 = coordsList[int(3*ptsIdx3) + 1]
                    rasterZ3 = coordsList[int(3*ptsIdx3) + 2]

                    #print " the norm and point are: "
                    #print norm
                    #print point

                    #print "the norm is: ", norm
                    #print "the point is: ", point

                    NDCtoWorld = np.identity(4)

                    if (shade == 2):

                        listpts = [[rasterX1, rasterY1, rasterZ1, normX1, normY1, normZ1], 
                                   [rasterX2, rasterY2, rasterZ2, normX2, normY2, normZ2], 
                                   [rasterX3, rasterY3, rasterZ3, normX3, normY3, normZ3]]

                        #print "the first norm is: ", normX1, ", ", normY1, ", ", normZ1
                        #print "the second norm is: ", normX2, ", ", normY2, ", ", normZ2
                        #print "the third norm is: ", normX3, ", ", normY3, ", ", normZ3

                        # matrix to convert from NDC to world space
                        NDCtoWorld = np.dot(cameraMat, np.linalg.inv(perspectiveProj))
                        print "ta-da"
                        raster(listpts, shade, material, lights, campos, NDCtoWorld)
                    
                    elif (shade == 1):
                        #print "the camera position: ", campos
                        norm1 = np.array([normX1, normY1, normZ1])
                        point1 = np.array([pointX1, pointY1, pointZ1])
                        rgbpoint1 = lightfunc(norm1/np.linalg.norm(norm1), point1, material, lights, campos)
                        #print "the rgbpoint1 is: ", rgbpoint1


                        norm2 = np.array([normX2, normY2, normZ2])
                        point2 = np.array([pointX2, pointY2, pointZ2])
                        rgbpoint2 = lightfunc(norm2/np.linalg.norm(norm2), point2, material, lights, campos)
                        #print "the rgbpoint2 is: ", rgbpoint2

                        norm3 = np.array([normX3, normY3, normZ3])
                        point3 = np.array([pointX3, pointY3, pointZ3])
                        rgbpoint3 = lightfunc(norm3/np.linalg.norm(norm3), point3, material, lights, campos)
                        #print "the rgbpoint3 is: ", rgbpoint3

                        listpts = [[rasterX1, rasterY1, rasterZ1, rgbpoint1[0], rgbpoint1[1], rgbpoint1[2]], 
                                   [rasterX2, rasterY2, rasterZ2, rgbpoint2[0], rgbpoint2[1], rgbpoint2[2]], 
                                   [rasterX3, rasterY3, rasterZ3, rgbpoint3[0], rgbpoint3[1], rgbpoint3[2]]]

                        raster(listpts, shade, material, lights, campos, NDCtoWorld)              

                    #print "the normal avg normalized is: ", normAvg/np.linalg.norm(normAvg)
                    #print "the point average is: ", pointAvg      

                    #print "the world coordinates first three: ", worldcoords[0], ", ", worldcoords[1], ", ", worldcoords[2]

                    #print "the world coordinates second three: ", worldcoords[3], ", ", worldcoords[4], ", ", worldcoords[5]

                    # flat shading
                    elif (shade == 0):
                        normAvg = np.array([(normX1 + normX2 + normX3)/3.0, 
                                         (normY1 + normY2 + normY3)/3.0, 
                                         (normZ1 + normZ2 + normZ3)/3.0])
                        pointAvg = np.array([(pointX1 + pointX2 + pointX3)/3.0, 
                                          (pointY1 + pointY2 + pointY3)/3.0, 
                                          (pointZ1 + pointZ2 + pointZ3)/3.0])

                
                        rgb = lightfunc(normAvg/np.linalg.norm(normAvg), pointAvg, material, lights, campos)
                        #rgbList.append(rgb)
                        #print rgb

                        listpts = [[rasterX1, rasterY1, rasterZ1, rgb[0], rgb[1], rgb[2]], 
                                   [rasterX2, rasterY2, rasterZ2, rgb[0], rgb[1], rgb[2]], 
                                   [rasterX3, rasterY3, rasterZ3, rgb[0], rgb[1], rgb[2]]]
                        raster(listpts, shade, material, lights, campos, NDCtoWorld)
                    # move to the next point
                    i += 1
                    print "NEXT POINT"
                # pass through the -1, which we left at earlier in the while loop
                j += 1
            #print rgbList
                
        first = fo.readline()
    first = fo.readline()

for l in range(xRes*yRes):
    onepixel = pixel[l]
    if (onepixel != 0):
        ppm.write(str(int(onepixel[0] * 255)) + " " + str(int(onepixel[1] * 255)) + " " + str(int(onepixel[2] * 255)) + "\n")
    else:
        ppm.write(str(0) + " " + str(0) + " " + str(0) + "\n")

ppm.close()
