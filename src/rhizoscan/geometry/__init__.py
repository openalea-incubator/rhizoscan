"""
Module containing some homogenous geometry functionalities
"""

import numpy as _np
import scipy as _sp
import scipy.ndimage as _nd

from scipy.linalg import inv, pinv
from scipy.linalg import svd  as _svd

from numpy.lib.stride_tricks import broadcast_arrays as _broadcast

from rhizoscan.ndarray     import as_vector as _vector
from rhizoscan.ndarray     import lookup    as _lookup

from rhizoscan.workflow import node as _node # to declare workflow nodes
# quick access
dot = _np.dot

@_node('product')
def mdot(*args):
    """ Apply multiple dot product: mdot(T1, T2,...,Tn) = T1 * T2 * ... * Tn """
    return reduce(dot,args)

@_node('homogeneous_coordinates')
def homogeneous(coordinates, N=None):
    """
    Assert `coordinates` are NxK homogeneous coordinates in (N-1)-dimension of K vectors

    Assert `coordinates` is numpy array and, if the first array dimension is 
    less than N, add one-valued rows at the bottom of `coordinates`.
    
    If N is None, add one "homogeneous coordinate": `N=coordinates.shape[0]+1`
    """
    c = _np.asanyarray(coordinates)
    
    if N is None:      N = c.shape[0]+1
    if c.shape[0]<N: c = _np.concatenate((c,_np.ones((N-c.shape[0],) + c.shape[1:])),axis=0)
        
    return c

@_node('normalized_data')
def normalize(data, istransform=False):
    """
    Normalize data by its "projective coordinates" - ie. set w=1 of vector [x,y,w] 
    
    By default, `data` should be a vector -shape=(k,)- or a shpaed (k,N) array 
    of N vectors. Each vector is then normalized its last value.
    
    If `istransform` is True, data is considered a transformation matrix and is
    normalized by its lower right value.
        
    If data is a (k,) vector, returns a (k,1) array 
    """
    data = _np.asanyarray(data)
    if data.ndim==1: data.shape = (data.size,1)  # convert vector to array
    
    if data.shape[0]<=2: 
        return _np.vstack((data,_np.ones((1,)+data.shape[1:])))
    elif istransform:
        return data / data.flat[-1]
    else:
        return data / data[-1][_np.newaxis]
        
        
@_node('H')
def fit_homography(src,dst):
    """ 
    Find 3x3 homography matrix, such that 'src' points are mapped to 'dst'
    using the linear DLT method. Input data are normalized automatically.
    
    src and dst should be 3xN arrays of N (homogeneous) 2d point correspondances 
    
    return array H, such that (with . the matrix dot product):
        dst = H . src    (up to fitting errors)
   
    Code taken from:
    http://www.janeriksolem.net/search/label/homography
    of Jan Erik Solem
    """
    if src.shape != dst.shape:
        raise RuntimeError('number of points do not match')

    # condition data (important for numerical reasons)
    # -- source --
    src   /= src[[2]]                            # normalize by "projective" coordinates
    m      = _np.mean(src[:2], axis=1)
    maxstd = _np.max(_np.std(src[:2], axis=1)) + 1e-9
    C1     = _np.diag([1/maxstd, 1/maxstd, 1])
    C1[0:2,2] = -m/maxstd
    src    = dot(C1,src)

    # -- destination --
    dst   /= dst[[2]]                            # normalize by "projective" coordinates
    m      = _np.mean(dst[:2], axis=1)
    maxstd = _np.max(_np.std(dst[:2], axis=1)) + 1e-9
    C2     = _np.diag([1/maxstd, 1/maxstd, 1])
    C2[0:2,2] = -m/maxstd
    dst    = dot(C2,dst)
                           
    
    # create matrix for linear method, 2 rows for each correspondence pair
    nbr_correspondences = src.shape[1]
    A = _np.zeros((2*nbr_correspondences,9))
    
    for i in range(nbr_correspondences):
        x,y = src[:2,i]
        u,v = dst[:2,i]            
        A[2*i]   = [-x,-y,-1,  0, 0, 0,  u*x,x*y,  u]
        A[2*i+1] = [ 0, 0, 0, -x,-y,-1,  v*x,v*y,  v]
        
    ## for line based homography fitting this requires specific normalization
    #  due to possible projective coordinates==0
    #for i in range(nbr_correspondences):
    #    x,y = src[:2,i]
    #    u,v = dst[:2,i]            
    #    A[2*i]   = [-u, 0, u*x, -v, 0,v*x, -1, 0, x]
    #    A[2*i+1] = [ 0,-u, u*y,  0,-v,v*y,  0,-1, y]
                   
    U,S,V = _svd(A)
    H = V[8].reshape((3,3))
    
    # decondition
    H = mdot(inv(C2),H,C1)
    
    # normalize and return
    return H / H[2,2]
    
    
@_node('T')
def fit_affine(src,dst):
    """ 
    Find 3x3 affine transformation matrix, mapping 'src' points to 'dst'.
    Input data are normalized automatically.
    
    src and dst should be 3xN arrays of N (homogeneous) 2d point correspondances 
    
    return array H, such that (with . the matrix dot product):
        dst = H . src    (up to fitting errors)
   
    Code taken from:
    http://www.janeriksolem.net/2009/06/affine-transformations-and-warping.html
    of Jan Erik Solem
    """

    if src.shape != dst.shape:
        raise RuntimeError, 'number of points do not match'

    # condition data
    #-from points-
    m      = _np.mean(src[:2], axis=1)
    maxstd = _np.max(_np.std(src[:2], axis=1))
    C1     = _np.diag([1/maxstd, 1/maxstd, 1]) 
    C1[0:2,2] = -m/maxstd
    src_cond  = dot(C1,src)

    #-to points-
    m  = _np.mean(dst[:2], axis=1)
    C2 = C1.copy() #must use same scaling for both point sets
    C2[0:2,2] = -m/maxstd
    dst_cond  = dot(C2,dst)

    # conditioned points have mean zero, so translation is zero
    A = _np.concatenate((src_cond[:2],dst_cond[:2]), axis=0)
    U,S,V = _svd(A.T)

    #create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
    tmp = V[:2].T
    B   = tmp[:2]
    C   = tmp[2:4]

    tmp2 = _np.concatenate((dot(C,pinv(B)),_np.zeros((2,1))), axis=1) 
    H    = _np.vstack((tmp2,[0,0,1]))

    #decondition
    H = mdot(inv(C2),H,C1)

    return H / H[2][2]
    
@_node('array')
def transform(data=None, T=None, coordinates=None, grid=None, order=1, mode='constant',cval=0.0):
    """
    Apply general homogeneous transformation T.
    
    Compute an array where each point value is taken in `data` at position 
    determined by transformation `T` (with spline interpolation)::
    
        out[coord] = data[ T * coord ]
    
    :Inputs:
      - data       
            a N-dimensional array-like object.
            if None, return the transformed coordinates (T * coord)
      - T
            a (N+1)x(M+1) projection matrix
      - coordinates
            a Mx[K] or homogeneous (M+1)xK coordinates of K M-dimensional points.
            [K] might be of any shape. 
            If coordinates is None, use "grid" arguments
      - grid
            can be used to generate coordinates. Grid should be a tuple of
            length M indicating the indices in each dimension. 
            Any of the following is allowed:
            
              - an integer equivalent to range from 0 to this number
              - a slice object              Eg: slice(100,200,5)  
              - a set of pixel indices      Eg: [4,2,3]
              - or any combinaison          Eg: (3,slice(2,5),[7,2])
        
      - order
            the order of spline interpolation
      - mode and cval
            how interpolation treats points transformed out of input data
        
    :See Also: 
        - scipy.ndimage.affine_transform
        - ndarray.lookup
    """
    
    if T is None: raise TypeError('Transformation T is required')
    if coordinates is None and grid is None: raise TypeError('Either of coodinates or grid should be given')
    
    # manage coordinates
    if coordinates is not None:
        coord = homogeneous(_np.asanyarray(coordinates), T.shape[0])
    else:
        # parse possible grid values s
        grid = list(grid)
        for i,g in enumerate(grid):
            if isinstance(g,slice):             # slice
                grid[i] = _np.arange(max(g.start,0),g.stop,max(g.step,1))
            else:
                g = _np.asanyarray(g)            # seq
                if g.ndim==0:                   # or integer
                    grid[i] = _np.arange(g)
            # make it
            grid[i] = _np.reshape(grid[i],[1 if j!=i else len(grid[i]) for j in xrange(len(grid))])
                
        coord = homogeneous(_broadcast(*grid))
        
    # reshape to coordinates set MxK
    shape = coord.shape[1:]
    coord.shape = coord.shape[0], coord[0].size
    
    c = normalize(dot(T,coord))[:-1]
    
    if data is None:
        output = c
        output.shape = (output.shape[0],) + shape
    else:
        output = _lookup(data, c, order=order,mode=mode,cval=cval)
        output.shape = shape
    
    return output
