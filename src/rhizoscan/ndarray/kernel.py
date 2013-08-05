import numpy as _np
from scipy import ndimage as _nd
from .     import norm    as _norm

from rhizoscan.workflow import node as _node # to declare workflow nodes
@_node({'name':'kernel'})
def coordinates(shape):
    """
    Compute an array containing for each axis the coordinates arrays of given shape:
        coord = coordinates( shape )
    
    
    :Input:
        shape: a list/tuple/vector of the kernel sizes of each dimension
              
    :Output:
        A numpy array of shape (N, [shape]) where N is the length of given 'shape'
        and returned coord[i,:] is the centered coordinates over the ith dimension
        
    :Example:
        coordinates((3,4))
            array([[[-1, -1, -1, -1],
                    [ 0,  0,  0,  0],
                    [ 1,  1,  1,  1]],
            
                   [[-1,  0,  1,  2],
                    [-1,  0,  1,  2],
                    [-1,  0,  1,  2]]])
    """
    if _np.isscalar(shape): shape = [shape]
    else:                   shape = _np.asarray(shape).tolist()
    
    return _np.mgrid[map(slice,[-((s-1)/2) for s in shape],[s/2+1 for s in shape])]
    
@_node({'name':'kernel'})
def distance(shape, metric=2):
    """
    return a distance kernel of given shape:
        d = distance(shape, metric='euclidian')
        
    :Input:
        shape:  a scalar (for 1d) or list/tuple/vector of the kernel shape
        metric: the distance function used. Same as the 'method' argument of array.norm()
        
    :Output:
        an array of given shape, where the center cell is zero, and all others
        have values equal to there distance to this center
        
    :Example:
        distance((3,4))
            array([[ 1.41,  1. ,  1.4,  2.23],
                   [ 1.  ,  0. ,  1. ,  2.  ],
                   [ 1.41,  1. ,  1.4,  2.23]])

    """
    coord = coordinates(shape)
    return _norm(coord,method=metric,axis=0)
    
@_node({'name':'kernel'})
def ellipse(radius,shape=None):
    """
    return a boolean array an ellipse kernel
        circle = ellipse(shape, radius)
        
    :Input:
        radius: a tuple the ellipse radius for each dimension. 
        shape:  a scalar (for 1d) or list/tuple/vector of the kernel shape
                *** It must have same length as 'shape' ***
                By default (if None), the maximum ellipse embedable in 'shape'
        
    :Output:
        an array of given shape, where the pixel inside the ellipse have True value
        
    :Example:
        ellipse((5,9),(2,3)).astype(int)
            array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 1, 1, 1, 1, 1, 0, 0],
                   [0, 1, 1, 1, 1, 1, 1, 1, 0],
                   [0, 0, 1, 1, 1, 1, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0]])

    """
    radius = _np.asarray([radius],dtype='float32').ravel().tolist()
    if shape is None: shape = [int(2*r+1) for r in radius]

    coord = map(_np.divide,tuple(coordinates(shape=shape)), radius)
    return _np.sqrt(reduce(_np.add,map(_np.square,coord)))<=1

@_node({'name':'kernel'})
def gaussian(sigma, shape=[]):
    """
    return a gaussian kernel of given shape:
        d = gaussian(sigma, shape=None)
        
    :Input:
        sigma:  a scalar or list/tuple of the sigma parameter for each dimension
        shape:  a scalar or list/tuple of the kernel shape
                if shape size is less than sigma, missing dimension are set to None 
                all None value are replaced to a size determined by sigma
        
    :Output:
        A gaussian kernel of suitable shape. 
        The total sum of all kernel values is equal to 1.
        
    :Example:
        np.round(gaussian((2,3),shape=(4,8)),3)
        array([[ 0.014,  0.032,  0.053,  0.063,  0.053,  0.032,  0.014,  0.004],
               [ 0.018,  0.041,  0.068,  0.081,  0.068,  0.041,  0.018,  0.006],
               [ 0.014,  0.032,  0.053,  0.063,  0.053,  0.032,  0.014,  0.004],
               [ 0.007,  0.015,  0.025,  0.03 ,  0.025,  0.015,  0.007,  0.002]])
    """
    sigma = _np.asarray([sigma]).ravel()
    shape = tuple(_np.asarray([shape]).ravel())

    auto  = tuple([8*s+1 for s in sigma])
    shape = auto[0:(sigma.size-len(shape))] + shape[(len(shape)-sigma.size):]
    
    coord = coordinates(shape)                              # (ndim,[shape])
    sigma.shape = (sigma.size,) + (1,)*len(shape)           # (ndim, [ones])
    kernel = _np.exp(-0.5 * _np.sum(coord**2 * (1./sigma), axis=0))

    return kernel / kernel.sum()
    
