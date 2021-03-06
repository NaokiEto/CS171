#!/usr/bin/python

import numpy as np
import pyparsing as pp
from math import*
import sys
from PIL import Image
import os

# regular bresenham
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
def bresenhamNeg(a1, b1, a2, b2, xlen, ylen):
    y = b2
    dy = b1 - b2
    dxdy = b1 - b2 + a1 - a2
    F = b1 - b2 + a1 - a2
    for i in range(-1*a2, -1*a1 + 1):
        pixel[(ylen/2 - y - 1)*xlen + xlen/2 - i - 1] = 1
        if(F < 0):
            F += dy
        else:
            y += 1
            F += dxdy
    return

# using the algorithm for pixel that (xlen - x)*xlen + (ylen - y)
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

xmin = sys.argv[1]
xmax = sys.argv[2]
ymin = sys.argv[3]
ymax = sys.argv[4]
xRes = int(sys.argv[5])
yRes = int(sys.argv[6])

# make a list of all the pixels of the window
pixel = [0]*xRes*yRes

fo = sys.stdin

# define grammar
# number is float form
number = pp.Regex(r"-?\d+(\.\d*)?([Ee][+-]?\d+)?")
number.setParseAction(lambda toks:float(toks[0]))

# Optional added for the additional number for rotation
parameter = pp.Optional(pp.Word( pp.alphas )) + pp.Optional(number) + pp.Optional(number)

first = fo.readline()

fileName, fileExtension = os.path.splitext(os.readlink('/proc/self/fd/0'))

ppm = open(fileName + ".ppm", "w")
ppm.write("P3 \n")
ppm.write(str(xRes) + " " + str(yRes) + "\n")
ppm.write(str(255) + "\n")

# not the end of the file
while (first != ''):

    line1 = parameter.parseString(first)

    if(str(first) == "\n"):
        first = fo.readline()

    # if the line says polyline
    if ("polyline" in first):
        # read the coordinates
        subset = fo.readline()

        point1 = parameter.parseString(subset)
        x1 = int(point1[0] * xRes/(int(xmax) - int(xmin)))
        y1 = int(point1[1] * yRes/(int(ymax) - int(ymin)))

        # read the next set of coordinates
        subset = fo.readline()
        point2 = parameter.parseString(subset)

        x2 = int(point2[0] * xRes/(int(xmax) - int(xmin)))
        y2 = int(point2[1] * yRes/(int(ymax) - int(ymin)))

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
        # move to the next line
        subset = fo.readline()

        # make sure the coordinates are not empty string or polyline
        while (("polyline" not in subset) and (subset.strip() != '') and (str(subset) != "")):
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
    else:
        subset = fo.readline()
    first = subset

for l in range(xRes*yRes):
    if(pixel[l] == 1):
        ppm.write(str(255) + " " + str(255) + " " + str(255) + "\n")
    else:
        ppm.write(str(0) + " " + str(0) + " " + str(0) + "\n")

ppm.close()

