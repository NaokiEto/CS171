#!/usr/bin/python

import numpy as np

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
