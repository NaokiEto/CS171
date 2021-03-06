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
def f(vert0, vert1, x, y):
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
def raster(verts, drawPixel, shadingType, material, lights, campos, matrix):
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

    print "the yMin is: ", yMin
    print "the yMax is: ", yMax
    print "the xMin is: ", xMin

    # normalizing values for the barycentric coordinates
    # not sure exactly what's going on here, so read the textbook
    fAlpha = f(coords[1], coords[2], coords[0][0], coords[0][1])
    fBeta = f(coords[2], coords[0], coords[1][0], coords[1][1])
    fGamma = f(coords[0], coords[1], coords[2][0], coords[2][1])

    if abs(fAlpha) < .0001 or abs(fBeta) < .0001 or abs(fGamma) < .0001:
        return

    print "checkpoint"

    # go over every pixel in the bounding box
    for y in range(max(yMin, 0), min(yMax, yRes)):
        for x in range(max(xMin, 0), min(xMax, xRes)):
            # calculate the pixel's barycentric coordinates
            alpha = f(coords[1], coords[2], x, y) / fAlpha
            beta = f(coords[2], coords[0], x, y) / fBeta
            gamma = f(coords[0], coords[1], x, y) / fGamma

            #print "the alpha, beta and gamma are: ", alpha, ", ", beta, ", ", gamma

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
                    drawPixel(x, y, data, shadingType, material, lights, campos, windowXMin, windowXMax, windowYMin, windowYMax, matrix)
