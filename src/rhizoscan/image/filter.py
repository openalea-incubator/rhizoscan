import numpy as _np
from scipy import ndimage as _nd

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
   
   
@_node('vesselness','vesselness_image_list')
def vesselness(image,sigmas=[2.0],beta=2.0,c=200.0):
    """
    Compute vesselness of given `image`
    
    ##TODO doc...
    
    Author: Guillaume Cerutti
    """
    vesselness = []
        
    for s,sigma in enumerate(sigmas):
        # --> Computing Gradient"
        gaussian_img = _nd.gaussian_filter(_np.array(image,_np.float32),sigma)
        gradient_x_img = _nd.sobel(gaussian_img,axis=0)
        gradient_y_img = _nd.sobel(gaussian_img,axis=1)
    
        # --> Computing Hessian
        hessian_xx_img = _nd.sobel(gradient_x_img,axis=0)
        hessian_xy_img = _nd.sobel(gradient_x_img,axis=1)
        hessian_yx_img = _nd.sobel(gradient_y_img,axis=0)
        hessian_yy_img = _nd.sobel(gradient_y_img,axis=1)
    
        # Computing Vesselness
        # --------------------
        #   Computing Hessian Eigenvalues"
        hessian_xx = hessian_xx_img.ravel()
        hessian_xy = hessian_xy_img.ravel()
        hessian_yx = hessian_yx_img.ravel()
        hessian_yy = hessian_yy_img.ravel()
    
        hessian_trace = hessian_xx+hessian_yy
        delta = _np.sqrt(_np.power(hessian_yy-hessian_xx,2.0) + 4.0*hessian_xy*hessian_yx)  
        
        eigval = _np.tile((hessian_trace/2.0)[:,_np.newaxis],(1,2))
        eigval[:,0] += delta/2.0
        eigval[:,1] -= delta/2.0
    
        lambdas_img = eigval.reshape(image.shape+(2,))
        hessian_norm_img = _np.sqrt(_np.power(lambdas_img[...,0],2.0) + _np.power(lambdas_img[...,1],2.0))
    
        #   Computing Vesselness
        sorted_lambdas_img = _np.sort(abs(eigval)).reshape(image.shape+(2,))
        max_lambdas = _np.argmax(abs(eigval),axis=1)
        max_lambdas_img = _np.select([max_lambdas,1-max_lambdas],[eigval[:,1],eigval[:,0]]).reshape(image.shape)
    
        lambdas_1 = sorted_lambdas_img[...,0]
        lambdas_2 = sorted_lambdas_img[...,1]
    
        if isinstance(c,list) or isinstance(c,_np.ndarray):
            vesselness_img = _np.exp(-(_np.power(lambdas_1/lambdas_2,2.0))/(2.0*_np.power(beta,2.0)))*(1-_np.exp(-_np.power(hessian_norm_img,2.0)/(2.0*(_np.power(c[s],2.0)))))
        else:
            vesselness_img = _np.exp(-(_np.power(lambdas_1/lambdas_2,2.0))/(2.0*_np.power(beta,2.0)))*(1-_np.exp(-_np.power(hessian_norm_img,2.0)/(2.0*_np.power(c,2.0))))
    
        vesselness_img[_np.where(_np.isnan(vesselness_img))] = 0
        vesselness_img[_np.where(max_lambdas_img>0)] = 0
    
        vesselness.append(vesselness_img)
    
    fused_vesselness_img = _np.max(_np.array(vesselness),axis=0)
    
    return fused_vesselness_img, vesselness

def ridge_2d(image):
    """
    :todo: develop, make doc and OA wrapper,... or maybe delete
    """
    x_prev = image[2:-1,2:-1] > image[2:-1,1:-2]
    x_next = image[2:-1,2:-1] > image[2:-1,3:]
    y_prev = image[2:-1,2:-1] > image[1:-2,2:-1]
    y_next = image[2:-1,2:-1] > image[3:  ,2:-1]
    ridge = _np.zeros(image.shape,dtype='bool')
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


