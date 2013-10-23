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
    print "the x-coords are: ", a1, a2
    print "the y-coords are: ", b1, b2
    for i in range(a1, a2 + 1):
        print "the index is: ", (ylen/2 - y) * xlen + i + xlen/2
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
    print "the coordinates are: ", a1, b1, "and ", a2, b2 
    for i in range(b2, b1 + 1):
        print "i is: ", i
        print "y is: ", y
        print "the index is: ", (xlen/2 - i - 1)*xlen + ylen/2 - y
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
number = pp.Regex(r"[-+]?([0-9]*\.[0-9]+|[0-9]+)")
number.setParseAction(lambda toks:float(toks[0]))

leftBrace = pp.Literal("{")
rightBrace = pp.Literal("}")
leftBracket = pp.Literal("[")
rightBracket = pp.Literal("]")
comma = pp.Literal(",")

# Optional added for the additional number for rotation
parameter = pp.Optional(pp.Word( pp.alphas )) + pp.Optional(leftBracket) + \
            pp.Optional(leftBrace) + pp.Optional(rightBracket) + \
	        pp.Optional(rightBrace) + pp.Optional(number) + pp.Optional(comma) + \
            pp.Optional(number) + pp.Optional(comma) + pp.Optional(number) + \
            pp.Optional(comma) + pp.Optional(number) + pp.Optional(comma) + \
            pp.Optional(number) + pp.Optional(comma) + pp.Optional(number) + \
            pp.Optional(comma) + pp.Optional(number) + pp.Optional(comma) + \
            pp.Optional(number)

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
    print firstparse

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

        # translation matrix
        translateCam = translateMatrix(translateX, translateY, translateZ)

        # rotation matrix
        rotationCam = rotationMatrix(x, y, z, angle)

        # calculate the camera matrix
        # World space to camera space
        cameraMat = np.dot(translateCam, rotationCam)

        # calculate the Perspective Projection matrix
        perspectiveProj = np.array([[2.0*n / (r - l), 0, float(r + l)/(r - l), 0], 
                                    [0, 2.0*n / (t - b), float(t + b) / (t - b), 0],
                                    [0, 0, -1.0*(f + n)/(f - n), -2.0*(f*n)/(f - n)],
                                    [0, 0, -1, 0]])

        print perspectiveProj

    # if we reach the Separator parameter
    if (len(firstparse) != 0 and (firstparse[0] == 'Separator')):
        first = fo.readline()
        firstparse = parameter.parseString(first)
        
        # if we reach the Transform sub-parameter
        if (len(firstparse) != 0 and (firstparse[0] == 'Transform')):
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
                    rX = firstparse[1]
                    rY = firstparse[2]
                    rZ = firstparse[3]
                    rAngle = firstparse[4]
                    rotationSep = rotationMatrix(rX, rY, rZ, rAngle)
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
            

            # calculate camera space to NDC (Normalized Device Coordinate) Space

            transformInter = np.dot(perspectiveProj, np.linalg.inv(cameraMat))
            transformMat = np.dot(transformInter, S)
        print transformMat
        print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
        print firstparse
        print "wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww"

        # end of Transform block parameter
        if (len(firstparse) != 0 and (firstparse[0] == '}')):
            first = fo.readline()
            firstparse = parameter.parseString(first)

        # entering the Coordinate subparameter
        if (len(firstparse) != 0 and (firstparse[0] == 'Coordinate')):
            first = fo.readline()
            firstparse = parameter.parseString(first)
            if (len(firstparse) != 0 and (firstparse[0] == 'point')):
                print "hi there"
                first = fo.readline()
                firstparse = parameter.parseString(first)

                # Create a list of the coordinates
                coordsList = []
                while (firstparse[0] != ']' and firstparse[0] != '}'):
                    newmat = np.dot(transformMat, np.array( [[float(firstparse[0])], [float(firstparse[1])], [float(firstparse[2])], [1.0]] ))
                    print newmat
                    newX = newmat[0,0]/newmat[3,0]
                    newY = newmat[1,0]/newmat[3,0]
                    newZ = newmat[2,0]/newmat[3,0]
                    coordsList.append(newX)
                    coordsList.append(newY)
                    coordsList.append(newZ)
                    first = fo.readline()
                    firstparse = parameter.parseString(first)
                print "the coords list is ", coordsList
            first = fo.readline()
            firstparse = parameter.parseString(first)

        # end of Coordinates block parameter
        if (len(firstparse) != 0 and (firstparse[0] == '}')):
            first = fo.readline()
            firstparse = parameter.parseString(first)
            print "wutttt"
        print firstparse

        if (len(firstparse) != 0 and (firstparse[0] == 'IndexedFaceSet')):
            first = fo.readline()
            firstparse = parameter.parseString(first)
            print "the parsed stuff is: ", firstparse
            # for the first row, with the coordIndex as firstparse[0]

            i = 0
            while (i < len(firstparse)):
                k = firstparse[i]
                while ((k == ',') or (k == '[') or (k == ']') or (k == 'coordIndex')):
                    print "i never get here huh"
                    i += 1
                    k = firstparse[i]
                print "the coordinates are ", k
                x1 = int(coordsList[int(3*k)] * xRes)
                y1 = int(coordsList[int(3*k) + 1] * yRes)

                # attempt to move to next element in the coordIndex list
                i += 1
                j = firstparse[i]
                while (j == ',' or j == '[' or j == ']' or j == 'coordIndex'):
                    i += 1
                    j = firstparse[i]
                print "the element is: ", j
                x2 = int(coordsList[int(3*j)] * xRes)
                y2 = int(coordsList[int(3*j) + 1] * yRes)

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
                    print "HHHHHHHHHHHHHHHHHHHHHHHHHHHHH"
                    bresenhamNegG1(m1, n1, m2, n2, xRes, yRes)
                
                # make sure the coordinates are not empty string or polyline
                while (firstparse[i] != -1 and (subset.strip() != '') and (str(subset) != "")):
                    point3 = parameter.parseString(subset) 
                    x3 = int(point3[0] * xRes/(int(xmax) - int(xmin)))
                    y3 = int(point3[1] * yRes/(int(ymax) - int(ymin)))

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
                    # move to the next line
                    subset = fo.readline()
                # skip over the -1
                i += 2
            

    first = fo.readline()

ppm.close()


