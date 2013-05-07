import numpy as _np
from ..ndarray import second_derivatives as _2nd_derivatives

from rhizoscan.workflow.openalea import aleanode as _aleanode # decorator to declare openalea nodes

##todo:
##   frangi_2d: multi scale ?   (see: matlab fileexchange frangi2D)
##   eig_2d : eigenvalues and vectors ?  (see same fileexchange, function eig2image)
##   eigenvalue_3d_vec ?   hessian_value_3d ?
def ridge_2d(img):
    ##todo: develop, make doc and OA wrapper,... or maybe delete
    x_prev = img[2:-1,2:-1] > img[2:-1,1:-2]
    x_next = img[2:-1,2:-1] > img[2:-1,3:]
    y_prev = img[2:-1,2:-1] > img[1:-2,2:-1]
    y_next = img[2:-1,2:-1] > img[3:  ,2:-1]
    ridge = _np.zeros(img.shape,dtype='bool')
    ridge[2:-1,2:-1] = (x_prev & x_next) | (y_prev & y_next)
    
    return ridge


@_aleanode({'name':'eigenvalues'})
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
                  (see function eigenvalue_2d_vec
                     
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
    

@_aleanode({'name':'vesselness'})
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


