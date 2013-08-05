import numpy as _np
from ..ndarray import second_derivatives as _2nd_derivatives

from rhizoscan.workflow import node as _node # to declare workflow nodes
@_node('uu','uv','vv')
def gradient_covariance_2d(image, size=3):
    """
    Compute the uu,uv,vv covariances of the gradient over a neighborhood of each image pixel
    u (resp. v) is the image gradient over x (resp. y)-coordinate.
    
    Use eigen_gradient_2d to compute the eigenvalues of the covariance matrix
    
    Input:
        - image: a 2d array, or a tuple containing the precomputed (u,v) gradient (in this order)
        - size:  the size of the neighborhood, either a scalar, or a 2-values tuple
        
    Output:
        - uu,uv,vv: the u-by-u, u-by-v, v-by-v correlation array
        
    The x-by-y covariance is computed as the sum of the x_i by the y_i (with i the
    index of a neighbor) divided by the total number of neighbor pixels.
    Note that the gradient are not centered (i.e. the mean value is removed) as it
    is not relevant to gradient diffusion.
    
    The vu covariance is equal to the uv
    """
    
    if isinstance(image, tuple): u,v = image
    else:                        v,u = _np.gradient(image)
    c = lambda x: _nd.uniform_filter(x,size=size)
    
    return c(u*u), c(u*v), c(v*v)
    
def eigen_gradient_2d(image, size=3, sort='descend'):
    """
    Compute the eigenvalues of the gradient (absolute) covariance for each pixel
    of the image, over its neighborhood
    
    Input:
        - image: a 2d array, or a tuple containing the precomputed (u,v) gradient (in this order)
        - size:  the size of the neighborhood, either a scalar, or a 2-values tuple
        - sort:  Order of returned eigenvalues. It can be either:
                  'ascend':  sort eigenvalue in  ascending order
                  'descend': sort eigenvalue in descending order (default)
                   or any function that takes 2 input arrays: (L1,L2), 
                   and return a boolean array of the same dimension, with True
                   value where L1 is correctly before L2
                   [see eigenvalue_2d()]
        
    Output:
        - an array of shape ([S],2)  where [S] is the shape of input arrays
    
    See also: gradient_covariance_2d, eigenvalue_2d
    """
    #uu,uv,vv = map(_np.abs,gradient_covariance_2d(image,size=size))
    uu,uv,vv = gradient_covariance_2d(image,size=size)
    return eigenvalue_2d(uu,uv,uv,vv, sort=sort)

@_node('eigenvalues')
def eigenvalue_2d(a,b,c,d, sort='descend'):
    """
    For arrays a,b,c and d, compute the eigenvalues of all matrices 
        [[a, b],
         [c, d]]
    
    
    :Input:
        a,b,c,d: arrays of the same size 
        sort:    optional argument which can be either:
                  'ascend':  sort eigenvalue in  ascending order
                  'descend': sort eigenvalue in descending order (default)
                   or any a function that takes 2 input arrays: (L1,L2), 
                   and return a boolean array of the same dimension, with True
                   value where L1 is correctly before L2
                   Example:
                   sort=lambda L1,L2: L1<L2    is equivalent to   sort='ascend'
                   (read:  "L1 should be less than L2")
                     
    :Output:
        an array of shape ([S],2)  where [S] is the shape of input arrays
    """
    ##todo: return 2 matrices, one for each eigenvalue, or make eigv indices at start ?
    shape = a.shape
    a = a.ravel()
    b = b.ravel()
    c = c.ravel()
    d = d.ravel()
    
    # compute trace and eigenvalue difference
    trace = a+d
    delta = _np.sqrt( trace**2 - 4*(a*d - b*c) )/2
    
    # compute eigenvalues 
    eigval = _np.tile( (trace/2)[:,_np.newaxis], (1,2) )
    eigval[:,0] += delta
    eigval[:,1] -= delta
    
    eigval = _np.abs(eigval)
    
    # sort eigenvalues
    if not isinstance(sort,type(lambda:1)):
        if   sort=='ascend':  sort = lambda L1,L2: L1>L2
        elif sort=='descend': sort = lambda L1,L2: L1<L2
        else:
            raise TypeError("sort arguments should be 'ascend', descend' or a function (See doc)")
            
    order  = sort(eigval[:,[1]],eigval[:,[0]])     # check if eigv[0] and eigv[1] should be inverted
    eigval = _np.choose(_np.hstack((-order,order)), (eigval[:,[0]],eigval[:,[1]]))
    
    eigval.shape = shape + (2,)
    return eigval    
    

def ridge_2d(img):
    """
    :todo: develop, make doc and OA wrapper,... or maybe delete
    """
    x_prev = img[2:-1,2:-1] > img[2:-1,1:-2]
    x_next = img[2:-1,2:-1] > img[2:-1,3:]
    y_prev = img[2:-1,2:-1] > img[1:-2,2:-1]
    y_next = img[2:-1,2:-1] > img[3:  ,2:-1]
    ridge = _np.zeros(img.shape,dtype='bool')
    ridge[2:-1,2:-1] = (x_prev & x_next) | (y_prev & y_next)
    
    return ridge


@_node({'name':'eigenvalues'})
def hessian_value_2d(array,sort='descend'):
    """
    Compute and return the eigenvalues of the hessian marix 
        [[dI_00,dI_01],
         [dI_01,dI_11]]
    
    where the dI_ij are the second derivatives of input array over the i and j coordinates.
    (they can be computed using function: array.second_derivatives(...))
    
    :Input:
        array: A 2d array, for which the second derivatives will be computed
               Otherwise, it can be a tuple or list containing the second
               derivatives (dI_00, dI_01, dI_11) -- in this order --, where
               dI_ij is the 2nd derivative of array over the ith and jth coordinates
                  
        sort:  Optional argument which can be either:
                 'ascend':  sort eigenvalue in  ascending order
                 'descend': sort eigenvalue in descending order (default)
                  or any a function that takes 2 input arrays: (L1,L2), 
                  and return a boolean array of the same dimension, with True
                  value where L1 is correctly before L2
                  (see function eigenvalue_2d)
                     
    :Output:
        An array of shape ([S],2)  where [S] is the shape of the input
                     
    :Example:
           H_val = hessian_value_2d(image,sort=lambda L1,L2: L1<L2) 
           (which is equivalent to  sort='ascend',  read "L1 should be less than L2")
    """
    # if input is not tuple or list (i.e. should be a 2d matrix, not tested)
    if not isinstance(array,(tuple,list)):
        array = _2nd_derivatives(array)

    Ixx,Ixy,Iyy = array
    
    return eigenvalue_2d( Ixx, Ixy, Ixy, Iyy, sort=sort)
    

@_node({'name':'vesselness'})
def frangi_2d(image, b=1, c=1):
    """
    Compute the frangi Vesselness value of all pixels of input image
    
        ... exp(- 1/2 * L1**2 / (b * L2)**2 )*(1 - exp(- 0.5 * (L1**2+L2**2) / c ))
    
    :Input:
        image: a 2d array
               or a tuple/list containing the image second derivatives (see array.second_derivatives)
        b,c:   the parameter of frangi vesselness  
        
    ##.... to develop / redo ... 
    see: http://www.mathworks.com/matlabcentral/fileexchange/24409-hessian-based-frangi-vesselness-filter/content/FrangiFilter2D.m
    """
    
    # get the seconde derivative
    if isinstance(image, _np.ndarray):
        # compute them if input is the an image
        Ixx,Iyy,Ixy = _2nd_derivatives(image)
    else:
        # retrieve them if input is a tuple containing seconde derivatives
        Ixx,Iyy,Ixy = image
    
    # compute the eigenvalue of the hessian
    eigval = hessian_value_2d((Ixx,Ixy,Iyy), sort=lambda L1,L2: _np.abs(L1) < _np.abs(L2))
    

    # compute the Frangi vesselness measure (stored in array V)
    shape = eigval.shape[0:-1]
    eigval.shape = _np.prod(shape) ,2
    
    i  = eigval[:,1]<=0    # non zeros vesselness
    L1 = eigval[i,0]       # lambda 1
    L2 = eigval[i,1]       # lambda 2
    
    V       = _np.zeros(_np.prod(shape))
    V[i]    = _np.exp(- 0.5 * L1**2 / (b * L2)**2 )
    if c is not None:
        V[i] *=  (1 - _np.exp(- 0.5 * (L1**2+L2**2) / c ))
    
    V.shape = shape
    
    return V


