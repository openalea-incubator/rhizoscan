import numpy as _np
from scipy import ndimage as _nd
from .     import norm    as _norm, add_dim as _add_dim
from .     import kernel  as _kernel

from rhizoscan.workflow.openalea import aleanode as _aleanode # decorator to declare openalea nodes


@_aleanode({'name':'filtered_array'})
def apply(array, **kwargs):
    """
    Apply a set of standard filter to array data: 
    
    Call: apply(array-data, <list of key=value arguments>)

    The list of key-value define the filtering to be done and should be given in
    the order to be process. Possible key-value are:
    
      * smooth:  gaussian filtering, value is the sigma parameter (scalar or tuple)
      * uniform: uniform  filtering (2)
      * max:     maximum  filtering (1)
      * min:     minimum  filtering (1)
      * median:  median   filtering (1)
      
      * dilate: grey dilatation (1)
      * erode:  grey erosion    (1)
      * close:  grey closing    (1)
      * open:   grey opening    (1)
      
      * linear_map: call linear_map(), value is the tuple (min,max)   (3)
      * normalize:  call normalize(),  value is the method            (3)
      * adaptive:   call adaptive(),   value is the sigma             (3)
      * adaptive_:  call adaptive(),   with uniform kernel            (3)
          
    The filtering is done using standard scipy.ndimage functions.
    
    (1) The value given (to the key) is the width of the the filter: 
        the distance from the center pixel (the size of the filter is thus 2*value+1)
        The neighborhood is an (approximated) boolean circle (up to discretization)
    (2) Same as (*) but the neighborhood is a complete square
    (3) See doc of respective function
    """
    for key in kwargs:
        value = kwargs[key]
        if key not in ('smooth','uniform'):
            fp = _kernel.distance(array.ndim*(2*value+1,))<=value  # circular filter
            
        if   key=='smooth' : array = _nd.gaussian_filter(array, sigma=value)
        elif key=='uniform': array = _nd.uniform_filter( array, size=2*value+1)
        elif key=='max'    : array = _nd.maximum_filter( array, footprint=fp)
        elif key=='min'    : array = _nd.minimum_filter( array, footprint=fp)
        elif key=='median' : array = _nd.median_filter(  array, footprint=fp)

        elif key=='dilate' : array = _nd.grey_dilation(  array, footprint=fp)
        elif key=='erode'  : array = _nd.grey_erosion(   array, footprint=fp)
        elif key=='open'   : array = _nd.grey_opening(   array, footprint=fp)
        elif key=='close'  : array = _nd.grey_closing(   array, footprint=fp)
        
        elif key=='linear_map': array = linear_map(array, min=value[0], max=value[1])
        elif key=='normalize' : array = normalize( array, method = value)
        elif key=='adaptive'  : array = adaptive(  array, sigma  = value, kernel='gaussian')
        elif key=='adaptive_' : array = adaptive(  array, sigma  = value, kernel='uniform')
        else: 
            print '\033[031mUnrecognized filter :', key
            
    return array

@_aleanode({'name':'mapped_array'})
def linear_map(array, min=0, max=1, axis=None):
    """
    mapped_array = linear_map( array, min=0, max=1, axis=None ) 
    
    Map array values from [array.min(),array.max()] to given [min,max] arguments
    over the specified axis
    
    :Input:
        array:   array to map from
        min&max: value to map to
                 if min (or max) is None, use the array min (or max) 
        axis:   Axis over which the array minimum and maximum are computed. 
                By default `axis` is None, and taken values are for the whole array.
    
    :Output:
        the array with mapped values
    """
    amin = _np.nanmin(array, axis=axis)
    amax = _np.nanmax(array, axis=axis)
    
    # reshape amin and amax if necessary
    if axis is not None:
        amin = _add_dim(amin,axis=axis,size=array.shape[axis])
        amax = _add_dim(amax,axis=axis,size=array.shape[axis])

    if min is None: min = amin
    if max is None: max = amax
    
    return (max-min) * (array-amin)/(amax-amin) + min  

@_aleanode({'name':'normalized_array'})
def normalize(array, method='euclidian', axis=None):
    """
    normalized_array = normalize(array, method='minmax')
    
    normalize array values, following given methods
    
    :Input:
        array:  array to normalize
        method: either
                  'minmax':          values are map from array min & max to (0,1)   
                  'euclidian' or 2:  divide by the euclidian norm   *** default ***
                  'taxicab'   or 1:  divide by the taxicab   norm
                   any number p:     divide by the p-norm, i.e. (sum_i(abs(x_i**p))**(1/p)
        axis:   Axis over which the normalization is done. 
                By default `axis` is None, and it is normalized over the whole array.
                
    :Output:
        the normalized array
        
    :Note:
        For all method but 'minmax', an epsilon value is added to the norm to
        avoid division by zero.
      
    """
    if method=='minmax': 
        return linear_map(array,min=0,max=1,axis=axis)
    else:                
        return array / (_norm(array, method=method, axis=axis,squeeze=False) + 2**-1000)

@_aleanode({'name':'filtered_data'})
def adaptive(data, sigma=None, kernel='gaussian'):
    """
    Compute adaptive filtering on data:     data - mu(data)
    
    Where mu is one of the local-average fonctions (from scipy.ndimage):
        - gaussian_filter() with sigma as parameter if kernel is 'gaussian'
        - uniform_filter()  with sigma use as size parameter if kernel is 'uniform'

    This is a second-derivative-like filtering that gives a positive values for 
    local maxima (mount) and ridges (crest), and gives negative values for local
    minima and valley
    
    The name "adaptive" is taken from "adaptive thresholding" that apply
    thresholding on the returned values
    """
    if kernel=='gaussian': mu = _nd.gaussian_filter(data,sigma=sigma)
    else:                  mu = _nd.uniform_filter( data,size =sigma)
    
    return data - mu

@_aleanode({'name':'threshold'})
def otsu(array,step='all',stat=None):
    """
    Compute the threshold value (with otsu's method) that maximize intra-class inertia 
    
    Use:    threshold = otsu(array, step='all', stat=None)
    
    :Input:
      array: an nd array
      step:  either 'all', meaning all possible value
              or a number, meaning to evenly sample possible values in this number of steps
              or a 1D list,tuple,array containing the list of values to test
              ***  if step is not 'all', cells value are considered to be  ***
              ***     equal to the closest lower value in selected step    *** 
                  
      stat: If not None, it should be a python list to which is append,
              the list of tested threshold
              the intra-class inertia for each threshold values
              the omega value of the first, then of the second, class
              the mean (mu) value of the 1st, then 2nd, class (see algorithm description)
    
    :Output:
        threshold: the selected threshold
    """
    # manage step list    
    if step=='all':
        val = _np.unique(array)
    else:
        val = _np.asarray(step).ravel()
        if val.size==1: 
            val = _np.linspace(array.min(),array.max(),val)
        else:
            # remove values out of range
            min = array.min()
            max = array.max()
            val[val<min] = min
            val[val>max] = max
            val = _np.unique(_np.hstack((val,min,max)))
    
    # compute histogram
    n, step = _np.histogram(array,_np.hstack((val,_np.inf)))
    
    # omega: number of elements for lower (w1) and higher (w2) class
    w1 = _np.cumsum(n)
    w2 = _np.cumsum(n[::-1])[::-1]
    
    # u (mu): mean value of both classes
    u1 = _np.cumsum(n*val) / w1
    u2 = _np.cumsum((n*val)[::-1])[::-1] / w2
    
    # find maximum intra-class variance
    var = w1*w2*(u1-u2)**2
    threshold = val[_np.argmax(var)]
    
    # if stat is a list, append variance, omega 1&2, mean 1&2
    if isinstance(stat,list): 
        stat.append(val); stat.append(var)
        stat.append(w1);  stat.append(w2)
        stat.append(u1);  stat.append(u2)

    return threshold

@_aleanode({'name':'mask'})
def threshold(array, value='otsu'):
    """
    Basic array thresholding: return array >= value
    
    If value is 'otsu' (default), use the threshold computed with otsu's method.
    """
    if value=='otsu': value = otsu(array)
    return array >= value

@_aleanode({'name':'mask'})
def adaptive_threshold(data, sigma=None, size=3, threshold=0):
    """
    Compute adaptive thresholding on data
        data >= mu(data) * (1 + threshold)
    
    Where mu is one of the fonctions (from scipy.ndimage):
        - gaussian_filter() with sigma as parameter    -  if sigma is not None (1st tested option)
        - uniform_filter()  with size  as parameter    -  otherwise 
        
    'threshold' can be view as a percentage of strengthening (for positve value) or
    relaxation (negative) of the inquality thresholding over the local average value (mu)
    """
    if sigma is not None: mu = _nd.gaussian_filter(data,sigma=sigma)
    else:                 mu = _nd.uniform_filter( data,size=size)
    
    return data >= mu * (1+threshold)


