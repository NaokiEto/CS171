#!/usr/bin/python

import pyparsing as pp
import sys
import os
import numpy as np
from math import*

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

# regular bresenham, for positive slopes between 0 and 1.
def bresenham(a1, b1, a2, b2, xlen, ylen):
    y = b1
    dy = b2 - b1
    dxdy = b2 - b1 + a1 - a2
    F = b2 - b1 + a1 - a2
    for i in range(a1, a2 + 1):
        pixel[(ylen/2 - y - 1)*xlen + i + xlen/2 - 1] = 1
        if(F < 0):
            F += dy
        else:
            y += 1
            F += dxdy
    return

# using the algorithm for pixel that (xlen - x)*xlen + (ylen - y)
# for positive slopes greater than 1
def bresenhamG1(a1, b1, a2, b2, xlen, ylen):
    y = a1
    dy = a2 - a1
    dxdy = a2 - a1 + b1 - b2
    F = a2 - a1 + b1 - b2
    for i in range(b1, b2 + 1):
        pixel[(xlen/2 - i - 1)*xlen + (y + ylen/2) - 1] = 1
        if(F < 0):
            F += dy
        else:
            y += 1
            F += dxdy
    return

# using the algorithm to flip over the y-axis, and do the bresenham algorithm
# then, flip back to correct pixel form by subtracting xlen - (i + xlen/2)
# for negative slopes between -1 and 0
def bresenhamNeg(a1, b1, a2, b2, xlen, ylen):
    y = b2
    dy = b1 - b2
    dxdy = b1 - b2 + a1 - a2
    F = b1 - b2 + a1 - a2
    for i in range(-1*a2, -1*a1 + 1):
        # minus 1 because of it starts at 0
        pixel[(ylen/2 - y - 1)*xlen + xlen/2 - i - 1] = 1
        if(F < 0):
            F += dy
        else:
            y += 1
            F += dxdy
    return

# using the algorithm for pixel that (xlen - x)*xlen + (ylen - y)
# for negative slopes less than -1
def bresenhamNegG1(a1, b1, a2, b2, xlen, ylen):
    y = -1*a2
    dy = a2 - a1
    dxdy = a2 - a1 + b2 - b1
    F = a2 - a1 + b2 - b1
    for i in range(b2, b1 + 1):
        pixel[(xlen/2 - i - 1)*xlen + ylen/2 - y - 1] = 1
        if(F < 0):
            F += dy
        else:
            y += 1
            F += dxdy
    return

fo = sys.stdin

# x dimension size
xRes = int(sys.argv[1])

# y dimension size
yRes = int(sys.argv[2])

# define grammar
# number is float form
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

# split the text between file name and file extension
fileName, fileExtension = os.path.splitext(os.readlink('/proc/self/fd/0'))

# create the ppm file
ppm = open(fileName + ".ppm", "w")
ppm.write("P3 \n")
ppm.write(str(xRes) + " " + str(yRes) + "\n")
ppm.write(str(255) + "\n")

first = fo.readline()

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
                translateX = firstparse[1]
                translateY = firstparse[2]
                translateZ = firstparse[3]

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
        translateCam = translateMatrix(translateX, translateY, translateZ)

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

    # if we reach the Separator parameter
    while (len(firstparse) != 0 and (firstparse[0] == 'Separator')):
        first = fo.readline()
        firstparse = parameter.parseString(first)
        
        # transform multiplication, initialized to identity matrix
        totaltransform = np.array([[1.0, 0.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0, 0.0],
                                   [0.0, 0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0]])

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

            if (translate == '' and rotate == 'rotation' and scaleFactor == ''):
                S = rotationSep
            elif (translate == 'translation' and rotate == '' and scaleFactor == ''):
                S = translateSep
            elif (translate == '' and rotate == '' and scaleFactor == 'scaleFactor'):
                S = scalefactorSep
            elif (translate == 'translation' and rotate == 'rotation' and scaleFactor == ''):
                S = np.dot(translateSep, scalefactorSep)
            elif (translate == 'translation' and rotate == '' and scaleFactor == 'scaleFactor'):
                S = np.dot(translateSep, scalefactorSep)
            elif (translate == '' and rotate == 'rotation' and scaleFactor == 'scaleFactor'):
                S = np.dot(rotationSep, scalefactorSep)
            elif (translate == 'translation' and rotate == 'rotation' and scaleFactor == 'scaleFactor'):
                SIntermediate = np.dot(rotationSep, scalefactorSep)
                S = np.dot(translateSep, SIntermediate)

            # Multiply up the transform matrices
            totaltransform = np.dot(totaltransform, S)

            # end of Transform block parameter
            if (len(firstparse) != 0 and (firstparse[0] == '}')):
                first = fo.readline()
                firstparse = parameter.parseString(first)

            # calculate camera space to NDC (Normalized Device Coordinate) Space
            transformInter = np.dot(perspectiveProj, np.linalg.inv(cameraMat))
            transformMat = np.dot(transformInter, totaltransform)

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
                    # multiply the transformation matrix by 4x1 matrix (the coordinates with 1.0 as w-coordinate)
                    newmat = np.dot(transformMat, np.array( [[xArr], 
                                                             [yArr], 
                                                             [zArr], 
                                                             [1.0]] ))
                    newX = newmat[0,0]/newmat[3,0]
                    newY = newmat[1,0]/newmat[3,0]
                    newZ = newmat[2,0]/newmat[3,0]
                    coordsList.append(newX)
                    coordsList.append(newY)
                    coordsList.append(newZ)
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

            # read until the end of the IndexedFaceSet block parameter
            while(first.strip() != '}'):
                firstparse = parameter.parseString(first)
                # for the first row, with the coordIndex as firstparse[0]
                i = 0

                # Go through the line
                while (i < len(firstparse)):
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
                        
                    # put the 1st point in x1, y1
                    # multiply by 1.0/2.0 because the origin is in the center of the window
                    x1 = int(coordsList[int(3*k)] * xRes * 1.0/2.0)
                    y1 = int(coordsList[int(3*k) + 1] * yRes * 1.0/2.0)

                    # attempt to move to next element in the coordIndex list
                    i += 1
                    if (i >= len(firstparse)):
                        first = fo.readline()
                        firstparse = parameter.parseString(first)
                        i = 0
                    j = firstparse[i]
                    while (j == ',' or j == '[' or j == ']' or j == 'coordIndex'):
                        if (i < len(firstparse) - 1):
                            i += 1
                            j = firstparse[i]
                        else:
                            first = fo.readline()
                            firstparse = parameter.parseString(first)
                            i = 0
                            j = firstparse[i]

                    # put the 2nd point in x2, y2
                    x2 = int(coordsList[int(3*j)] * xRes * 1.0/2.0)
                    y2 = int(coordsList[int(3*j) + 1] * yRes * 1.0/2.0)

                    a1 = x1
                    b1 = y1
                    a2 = x2
                    b2 = y2

                    # Apply Bresenham's line algorithm
                    if (a2 <= a1):            
                        m1 = a2
                        n1 = b2
                        m2 = a1
                        n2 = b1
                    else:
                        m1 = a1
                        n1 = b1
                        m2 = a2
                        n2 = b2
                    # for the case where the positive slope <= 1 
                    if((n2 - n1 >= 0) and (m2 - m1 >= n2 - n1)):
                        bresenham(m1, n1, m2, n2, xRes, yRes)

                    # for the case where the positive slope > 1
                    elif((n2 - n1 >= 0) and (m2 - m1 < n2 - n1)):
                        bresenhamG1(m1, n1, m2, n2, xRes, yRes)

                    # for the case where the negative slope >= -1
                    elif((n2 - n1 < 0) and (m2 - m1 >= n1 - n2)):
                        bresenhamNeg(m1, n1, m2, n2, xRes, yRes)

                    # for the case where the negative slope < -1
                    elif((n2 - n1 < 0) and (m2 - m1 < n1 - n2)):
                        bresenhamNegG1(m1, n1, m2, n2, xRes, yRes)
                    
                    i += 1
                    if (i >= len(firstparse)):
                        first = fo.readline()
                        firstparse = parameter.parseString(first)
                        i = 0
                    t = firstparse[i]
                    while (t == ',' or t == '[' or t == ']' or t == 'coordIndex'):
                        i += 1
                        if (i >= len(firstparse)):
                            first = fo.readline()
                            firstparse = parameter.parseString(first)
                            i = 0
                        t = firstparse[i]

                    # for more than two points
                    while (i < len(firstparse) and firstparse[i] != -1):
                        l = firstparse[i]
                        while (l == ',' or l == '[' or l == ']' or l == 'coordIndex'):
                            i += 1
                            l = firstparse[i]

                            if (i >= len(firstparse)):
                                first = fo.readline()
                                firstparse = parameter.parseString(first)
                                i = 0

                        # put the 3rd point in x3, y3
                        x3 = int(coordsList[int(3*l)] * xRes * 1.0/2.0)
                        y3 = int(coordsList[int(3*l) + 1] * yRes * 1.0/2.0)

                        a1 = a2
                        b1 = b2
                        a2 = x3
                        b2 = y3

                        # Apply Bresenham's line algorithm
                        if (a2 <= a1):            
                            m1 = a2
                            n1 = b2
                            m2 = a1
                            n2 = b1
                        else:
                            m1 = a1
                            n1 = b1
                            m2 = a2
                            n2 = b2
                        # for the case where the positive slope <= 1 
                        if((n2 - n1 >= 0) and (m2 - m1 >= n2 - n1)):
                            bresenham(m1, n1, m2, n2, xRes, yRes)

                        # for the case where the positive slope > 1
                        elif((n2 - n1 >= 0) and (m2 - m1 < n2 - n1)):
                            bresenhamG1(m1, n1, m2, n2, xRes, yRes)

                        # for the case where the negative slope >= -1
                        elif((n2 - n1 < 0) and (m2 - m1 >= n1 - n2)):
                            bresenhamNeg(m1, n1, m2, n2, xRes, yRes)

                        # for the case where the negative slope < -1
                        elif((n2 - n1 < 0) and (m2 - m1 < n1 - n2)):
                            bresenhamNegG1(m1, n1, m2, n2, xRes, yRes)
                        i += 1
                        if (i >= len(firstparse)):
                            first = fo.readline()
                            firstparse = parameter.parseString(first)
                            i = 0
                        g = firstparse[i]
                        while (g == ',' or g == '[' or g == ']' or g == 'coordIndex'):
                            if (i < len(firstparse) - 1):
                                i += 1
                                g = firstparse[i]
                            else:
                                first = fo.readline()
                                firstparse = parameter.parseString(first)
                                i = 0
                                g = firstparse[i]
                    a1 = x1
                    b1 = y1
                    a2 = x3
                    b2 = y3

                    # Apply Bresenham's line algorithm
                    if (a2 <= a1):            
                        m1 = a2
                        n1 = b2
                        m2 = a1
                        n2 = b1
                    else:
                        m1 = a1
                        n1 = b1
                        m2 = a2
                        n2 = b2
                    # for the case where the positive slope <= 1 
                    if((n2 - n1 >= 0) and (m2 - m1 >= n2 - n1)):
                        bresenham(m1, n1, m2, n2, xRes, yRes)

                    # for the case where the positive slope > 1
                    elif((n2 - n1 >= 0) and (m2 - m1 < n2 - n1)):
                        bresenhamG1(m1, n1, m2, n2, xRes, yRes)

                    # for the case where the negative slope >= -1
                    elif((n2 - n1 < 0) and (m2 - m1 >= n1 - n2)):
                        bresenhamNeg(m1, n1, m2, n2, xRes, yRes)

                    # for the case where the negative slope < -1
                    elif((n2 - n1 < 0) and (m2 - m1 < n1 - n2)):
                        bresenhamNegG1(m1, n1, m2, n2, xRes, yRes)

                    if (i < len(firstparse)):
                        # if -1 is element, move on
                        if (firstparse[i] == -1):
                            i += 1
                first = fo.readline()
        first = fo.readline()
    first = fo.readline()

for l in range(xRes*yRes):
    if(pixel[l] == 1):
        ppm.write(str(255) + " " + str(255) + " " + str(255) + "\n")
    else:
        ppm.write(str(0) + " " + str(0) + " " + str(0) + "\n")

ppm.close()
