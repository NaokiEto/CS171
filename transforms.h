/*
 * Steven Mueller
 * diffusor@ugcs.caltech.edu
 * cs/cns 171  Spring 2004
 * this code is in the process of being released under the GNU GPLed
 */

/*
 * Specializations of matrices (row/column-wise) and their associated
 * operations wrt graphics.
 */


#ifndef __TRANSFORMS_H_GUARD__
#define __TRANSFORMS_H_GUARD__

#include <iosfwd>
#include <cstdio>
#include <cassert>
#include "minmatrix.h"


/******************************************************************************
 * 
 * Transformation operations
 *
 ******************************************************************************/


/******************************************************************************
  * TODO:  All the transformations!
  ******************************************************************************/

// An example of plugging stuff into a 4x4 matrix
template<typename T>
Matrix<T,4,4> somekindof_transform(T a, T b, T c) {
    Matrix<T,4,4> res;
    res.clear(0);
    res(0,0) = a;
    res(0,3) = 1;
    res(1,1) = 1;
    res(1,3) = b;
    res(2,3) = -1;
    res(3,2) = -c;
    res(3,3) = 1;
    return res;
}

// Convenience function to construct a 3x1 vector out of 3 arguments
template<typename T>
Matrix<T,3,1> makeVector(T e1, T e2, T e3) {
    Matrix<T,3,1> m;
    T a[] = {e1,e2,e3};
    m.wipe_copy(a);
    return m;
}

#endif // __TRANSFORMS_H_GUARD__
