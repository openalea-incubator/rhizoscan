import numpy as _np
import scipy.ndimage as _nd
from numpy.lib.stride_tricks import as_strided     as _as_strided
from scipy.ndimage.filters   import minimum_filter as _minimum_filter

from rhizoscan.workflow.openalea import aleanode as _aleanode # decorator to declare openalea nodes

# icon of openalea package
__icon__ = 'cube.png'

def virtual_array(shape,value=1):
    """
    return an ndarray of virtual shape containing a single 'value' element
    efficient way of having a constant "readonly" array
    
    See also: reshape
    """
    return reshape(value,-_np.atleast_1d(shape))

def as_vector(array,dim=0, size=None,fvalue=0,fside=False):
    """
    Make a vector out of input, adding cells if necessary.
    
    Input
    -----
    array:  an array-like object to convert to a vector
    dim:    The dimension for storing the vector. By default (0) return a
            vector of shape (size,)
    size:   if not None, enforce the size of the return vector adding or 
            removing cells if necessary. See 'fvalue' and 'fside'
    fvalue: value to fill the created cells with
            if 'nearest, use the value of the closest existing cell
    fside:  if True, new cells are added at end of vector
            if False, they are added at begining
    """
    array  = _np.asanyarray(array).ravel()
    
    if size is None:
        vector = array
    else:
        if fside: array  = array[::-1]
        if fvalue=='nearest': fvalue = array[-1]
        
        vector = _np.ones(size) * fvalue
        vector[:array.size] = array
        
        if fside: vector = vector[::-1]
        
    if dim>0: vector.shape = (1,)*dim + (vector.size,)
    
    return vector

@_aleanode('integer_array')
def asint(array, default='int', minbyte=1, u=True, i=True):
    """
    return the array unchanged if it is an ndarray of integer dtype, or convert 
    it to 'default' dtype. If array is not an ndarray, convert it.
    
    default: the default integer dtype if convertion is needed
    minbyte: impose a minimum bytes precision
    u: if True accept unsigned integer dtype
    i: if True accept   signed integer dtype
    """
    return array if isint(array,u=u,i=i)>=minbyte else _np.array(array,dtype=default)

@_aleanode('is_integer_array')
def isint(array, u=True, i=True):
    """
    Test weither array is of integer dtype.
    
    Return False if not integer, of the byte precision (integer) if it is
    Return False if array is not an ndarray
    """
    dt = ()+ (('u',) if u else ()) + (('i',) if i else ())
    if isinstance(array,_np.ndarray) and array.dtype.kind in dt: 
          return array.dtype.itemsize
    else: return False
    
@_aleanode('float_array')
def asfloat(array, default='float', minbyte=2):
    """
    return the array unchanged if it is an ndarray of float dtype, or convert 
    it to 'default' dtype. If array is not an ndarray, convert it.
    
    default: the default float dtype if convertion is needed
    minbyte: impose a minimum bytes precision (exist in 2,4,8 or 16 bytes precision)
    """
    return array if isfloat(array)>=minbyte else _np.array(array,dtype=default)

@_aleanode('is_float_array')
def isfloat(array):
    """
    Test weither array is of float dtype.
    
    Return False if not float, of the byte precision (integer) if it is
    Return False if array is not an ndarray
    """
    if isinstance(array,_np.ndarray) and array.dtype.kind=='f': 
          return array.dtype.itemsize
    else: return False
    
@_aleanode('axis','start','stop','step')
def aslice(axis, *args):
    """
    same as slice but return a tuple with None-slice for all preceding axis
        aslice(axis, [start], stop, [step])
    """
    return (slice(None),)*axis + (slice(*args),)
    

@_aleanode({'name':'norm'})
def norm(array, method=2, axis=None, squeeze=True):
    """
    Compute the norm of the input array, following either method
    
        larrayl = norm(array, method='euclidian', axis=None, squeeze=True)
    
    :Input:
        array:    elements to compute the norm
        method:   either
                    'euclidian' or 2:  the euclidian norm
                    'taxicab' or 1:    the taxicab   norm
                     any number p:     the p-norm, i.e. (sum(abs(array**p))**(1/p)
                  
        axis:     Axis over which the sum is done. By default `axis` is None,
                  and all the array elements are summed.
        squeezed: if axis is not None,
                  if True, the respective axis of returned array is removed
                  otherwise, it is it is kept and boradcast
                
    :Output:
        if axis is None, return the total norm of the array values
        otherwise, return an array with shape equal to the original array shape
        with the respective axis removed

    """
    if   method=='euclidian': method = 2
    elif method=='taxicab':   method = 1
    p = float(method)
    
    norm = _np.sum(_np.abs(_np.asanyarray(array))**p,axis=axis)**(1/p)
    
    if axis is not None and squeeze is False:
        norm.shape = norm.shape[:axis] + (1,) + norm.shape[axis:]  # unsqueeze

    return norm

@_aleanode({'name':'gradient_norm'})
def gradient_norm(array):
    """
    Compute the norm of gradient of the array: ( sum of squared derivatives )**1/2
    """
    return reduce(_np.add,map(_np.square,_np.gradient(array)))**0.5


@_aleanode({'name':'sec_derivatives_list'})
def second_derivatives(array, smooth=2):
    """
    Compute the second derivatives of all dimensions pairs of the input array
    
    :Inputs:
        array:  any ndarray
        smooth: the second derivative are computed by convolving the input array
                by [1,-2,1]. The smooth parameter set how many times this basic
                filter is convoluted with [1,2,1]/2, which smooth it.
    
    :Output:
        Return a tuple of the second derivative arrays in the order (where dd_ij 
        is the the second derivative d2(array)/didj for a N-dimensional array):
        (dd_00, dd_01, ..., dd_0N, dd_11, dd_12, ..., dd_1N, ..., dd_N-1N, dd_NN)

    :Example:
       for 3d array 'volume_array'
       dv_00, dv_01, dv_02, dv_11, dv_12, dv_22 = second_derivatives(volume_array)
       
    See also:
      numpy.gradient    
    """
    # compute the derivative filter
    dd  = [1,-1]
    for i in xrange(smooth):
        dd = _np.convolve(dd,[1,2,1])/2. 
    
    # compute the second derivatives
    res = ()
    for i in xrange(array.ndim):
        tmp = _nd.convolve1d(array,dd,axis=i)
        for j in xrange(i,array.ndim):
            res += _nd.convolve1d(tmp,dd,axis=j),
    
    return res


# managing nD indices
# -------------------
@_aleanode({'name':'flat_indices'})
def ravel_indices(indices, shape):
    """
    Convert nD to 1D indices for an array of given shape.
        flat_indices = ravel_indices(indices, size)
    
    :Input:
        indices: array of indices. Should be integer and have shape=([S],D), 
                 for S the "subshape" of indices array, pointing to a D dimensional array.
        shape:   shape of the nd-array these indices are pointing to (a tuple/list/ of length D)
        
    :Output: 
        flat_indices: an array of shape S
    
    :Note: 
       This is the opposite of unravel_indices: for any tuple 'shape'
          ind is equal to    ravel_indices(unravel_indices(ind,shape),shape)
                   and to  unravel_indices(  ravel_indices(ind,shape),shape)
    """
    dim_prod = _np.cumprod([1] + list(shape)[:0:-1])[_np.newaxis,::-1]
    ind = _np.asanyarray(indices)
    S   = ind.shape[:-1]
    K   = _np.asanyarray(shape).size
    ind.shape = S + (K,)
    
    return _np.sum(ind*dim_prod,-1)
    

@_aleanode({'name':'nD_indices'})
def unravel_indices(indices,shape):
    """
    Convert indices in a flatten array to nD indices of the array with given shape.
        nd_indices = unravel_indices(indices, shape)
    
    :Input:
        indices: array/list/tuple of flat indices. Should be integer, of any shape S
        shape:   nD shape of the array these indices are pointing to
        
    :Output: 
        nd_indices: a nd-array of shape [S]xK, where 
                    [S] is the shape of indices input argument
                    and K the size (number of element) of shape     
    
    :Note:
        The algorithm has been inspired from numpy.unravel_index 
        and can be seen as a generalization that manage set of indices
        However, it does not return tuples and no assertion is done on 
        the input indices before convertion:
        The output indices might be negative or bigger than the array size
        
        This is the opposite of ravel_indices:  for any tuple 'shape'
          ind is equal to    ravel_indices(unravel_indices(ind,shape),shape)
                   and to  unravel_indices(  ravel_indices(ind,shape),shape)
    """

    dim_prod = _np.cumprod([1] + list(shape)[:0:-1])[::-1]
    ind = _np.asanyarray(indices)
    S   = ind.shape
    K   = _np.asanyarray(shape).size
    
    ndInd = ind.ravel()[:,_np.newaxis]/dim_prod % shape
    ndInd.shape = S + (K,)
    return ndInd
    
    

# virtual 1-dimentional tiling that use stride tricks
@_aleanode({'name':'plus1D_array'})
def add_dim(array, axis=-1, size=1, shift=0):
    """
    Insert a virtual dimension using stride tricks  (i.e. broadcasting)
    *** The appended dimension is a repeated view over the same data ***
    
    call:    new_array = add_dim(array, axis=-1, size=1, shift=0)
    
    :Input:
        array: a numpy ndarray
        axis:  the index of the virtual axis to insert
               the axis number can be negative. If axis=-n: 
                  the added axis is the (array.ndim-n)th dimension of output array
                  (if n=-1, add an axis at the end of input array)
        size:  the number of element this new axis will have
        shift: (optional) if given, the added dimension becomes a shifted view
               in the input array. The ith element along the shift dimension 
               start at element i*shift, which should be the index of an element
               in the given array.
                                   *** warning *** 
                with shift, some valid indices point out of given array memory. 
                     Using it might CRASH PYTHON. Use at your own risk
                                   ***************
                                   
        With default arguments (axis, size and shift), add_dim add a
        singleton axis at the end of input array
    
    :Output:
        if input array shape is S, the returned array has shape (S[:axis], size, S[axis:])
        
    :Example:
      if A is a has 2x3x4 array, then B = add_dim(A,axis,5) will have shape:
          (5,2,3,4)   if   axis= 0
          (2,3,5,4)   if   axis= 2
          (2,3,4,5)   if   axis= 3 or -1
      
      B = add_dim(A,axis=-1,size=1) is the same as B = A[:,:,:,newaxis]
      B = add_dim(A,axis= 0,size=1) is the same as B = A[newaxis]
      
    :Note:
        The returned array is a (broadcasted) view on the input array. 
        Changing its elements value will affect the original data.
    """
    A  = _np.asanyarray(array)
    sh = _np.array(A.shape)
    st = _np.array(A.strides)
    
    # assert type of shift array and pad it with zeros
    if shift:
        shift = as_vector(shift,size=A.ndim,fvalue=0,fside=0)
        shift = _np.dot(shift, st)
    
    axis = _np.mod(axis,A.ndim+1)
    
    sh = _np.hstack((sh[:axis], size,   sh[axis:]))
    st = _np.hstack((st[:axis], shift,  st[axis:]))
    
    return _as_strided(A,shape=sh, strides=st)


@_aleanode('reshaped_array')
def reshape(array, newshape, order='A'):
    """
    Similar as numpy.reshape but allow to had virtual dimension of size > 1
    
    Virtual dimension are indicated by negative value in argument 'newshape'
    
    Parameters
    ----------
    a : array_like
        Array to be reshaped.
    newshape : int or tuple of ints
        The new shape should be compatible with the original shape. 
        One shape dimension can be None. In this case, the value is inferred
        from the length of the array and remaining dimensions.
        Any negative integer means a new dimension with this size. 
    order : {'C', 'F', 'A'}, optional
        Determines whether the array data should be viewed as in C
        (row-major) order, FORTRAN (column-major) order, or the C/FORTRAN
        order should be preserved.
    
    Returns
    -------
    reshaped_array : a view on input array if possible, otherwise a copy.
    
    See Also
    --------
    numpy.reshape
    
    Example:
    --------
        A = np.arange(12).reshape((3,4))
        print A
            [[ 0,  1,  2,  3],
             [ 4,  5,  6,  7],
             [ 8,  9, 10, 11]]
        B = reshape(A,(2,-3,None))
        print B.shape
            (2, 3, 6)
        # second dimension of B is virtual:
        B[1,1,1] = 42
        print B
            [[[ 0  1  2  3  4  5]
              [ 0  1  2  3  4  5]
              [ 0  1  2  3  4  5]]
            
             [[ 6 42  8  9 10 11]
              [ 6 42  8  9 10 11]
              [ 6 42  8  9 10 11]]]
              
    *** Warning: the order option has not been tested *** 
    """
    array = _np.reshape(array,[-1 if s is None else 1 if s<0 else s for s in newshape], order=order)
    shape = [abs(s) if s is not None else s2 for s,s2 in zip(newshape,array.shape)]
    strid = [st if s>=0 or s is None else 0  for st,s in zip(array.strides,newshape)]
    return _as_strided(array,shape=shape, strides=strid)
    
# make a bigger array by adding elements around it
@_aleanode({'name':'padded_array'})
def pad_array(array, low_pad, high_pad, fill_value = 0):
    """
    Returned a copy of input array with increased shape
    
    :Input:
      array:      input data
      low_pad:    the number of elements to add before each dimension  (*)
      high_pad:   the number of elements to add after  each dimension  (*)
      fill_value: either the value to put in added array elements (default=0)
                  or 'nearest' to fill with the value of the nearest input data
      
    (*) can be either a single number, in which case it applies to all dimensions, 
        or a tuple/list/array with length less or equal to array dimension. 
        If less, the (preceeding) missing value are set to 0
        In all cases, the values are rounded to the nearest integers
        
    :Output:
      The returned array has shape = input array shape + |low_pad| + |high_pad|
      and same dtype as input array
      
    :Note:
      the input array data is contained in the returned array, 
      starting at position |low_pad|
      
    :Example:
      In:   pad_array(np.arange(6).reshape(2,3)+10, 1, (1,3), fill_value=-1)
      Out:  array([[-1, -1, -1, -1, -1, -1, -1],
                   [-1, 10, 11, 12, -1, -1, -1],
                   [-1, 13, 14, 15, -1, -1, -1],
                   [-1, -1, -1, -1, -1, -1, -1]])
    """
    array = _np.asanyarray(array)
    
    # convert padding to suitable vectors
    low_pad  = as_vector( low_pad,size=array.ndim,fvalue='nearest' if _np.isscalar( low_pad) else 0,fside=0)
    high_pad = as_vector(high_pad,size=array.ndim,fvalue='nearest' if _np.isscalar(high_pad) else 0,fside=0)
    
    # assert pad values are in Z+
    low_pad  = _np.maximum(_np.abs(_np.round(_np.asarray( low_pad))),0)
    high_pad = _np.maximum(        _np.round(_np.asarray(high_pad)) ,0)
    
    # construct padded array
    in_indices = map(slice,low_pad,low_pad+array.shape)
    
    if not isinstance(fill_value,basestring):
        arr = _np.ones(low_pad + array.shape + high_pad, dtype=array.dtype) \
              * _np.asarray(fill_value,dtype=array.dtype)
        # copy initial array data in the right position
        arr[in_indices] = array
        
    else: # fill_value == 'nearest'
        arr = _np.empty(low_pad + array.shape + high_pad, dtype=array.dtype)

        # copy initial array data in the right position
        arr[in_indices] = array
        
        # fill with nearest 
        all = slice(None)
        for dim in xrange(arr.ndim):
            L = low_pad[dim]
            H = L + array.shape[dim]
            
            # fill lower pad
            pad  = [all if d!=dim else slice(0,L)   for d in xrange(arr.ndim)]
            face = [all if d!=dim else slice(L,L+1) for d in xrange(arr.ndim)]
            arr[pad] = arr[face]

            # fill higher pad
            pad  = [all if d!=dim else slice(H,None) for d in xrange(arr.ndim)]
            face = [all if d!=dim else slice(H-1,H)  for d in xrange(arr.ndim)]
            arr[pad] = arr[face]

    return arr.view(type(array))


@_aleanode({'name':'local_min'})
def local_min(array, footprint=3, mask = None, strict=False, return_indices=False):
    """
    Detects the local minima of the given array using the local minimum scipy filter.

    'footprint' defined the neighborhood of each array cell. 
    If it is a scalar, use a window of this size in all dimension
    if it is a list or tuple of length equal to the array rank, each element 
    define the window size in each dimension
    Otherwise it must be a numpy array of same rank as input array
    
    If 'mask' is not None, its zero (or False) valued entries are considered invalid  

    if 'strict' is False (default) an array element is considered a local minima 
    if it is less or equal to all its neighbors. It can result in the selection
    of whole minimal plateau (connected area where pixels have the same value).
    Otherwise, some noise is added to select unconnected local minima.
    
    If 'return_indices' is True,  it returns a list of 2 lists:
    the 1st containing the x et the 2nd the y coordinates of all local minima.
    => Use zip(*local_min(...)) to get list of (x,y) tuples 
    
    This function has been strongly inspired by
      http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    # define the neighborhood
    array = _np.asanyarray(array)
    if _np.isscalar(footprint): footprint = (footprint,)*array.ndim
    if isinstance(footprint, (tuple,list)) and len(footprint)==array.ndim:
        neighborhood = _np.ones(footprint)
    else:
        neighborhood = _np.asanyarray(footprint)
    
    if strict:
        # remove neighboring local minima that occurs when neighboring cell
        # have exact same value.
        # This is done by adding a noise lower than the minimum difference `
        # between all values of input array
        min_step = _np.diff(_np.unique(array.flat)).min()
        grid  = _np.mgrid[map(slice,array.shape)]
        array = array + _np.random.random(array.shape)*min_step
        ##todo: replace random by generated map => no value should be the same
        #array = array + _np.add(*map(lambda x:_np.abs(_np.mod(x,4)-2), grid ))*min_step/4
       
       
    # apply the local minimum filter:
    # all locations that have minimum value in their neighborhood are set to 1
    loc_min = (_minimum_filter(array, footprint=neighborhood)==array)
    
    
    # remove local minima that are out of mask
    if mask is not None:
        loc_min = loc_min & mask
    
    if return_indices:
        return _np.where(loc_min)
    else:
        return loc_min
    
@_aleanode({'name':'filled_array'})
def fill(data, invalid=None, max_distance=-1):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell
    
    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. 
                 True cells indicate where data value should be filled (replaced)
                  => If None (default), use: invalid  = np.isnan(data)
        max_distance: distance up to which cells are filled. If negative, no limit. 
               
    Output: 
        Return a filled array. 
    """
    if invalid is None: invalid = _np.isnan(data)

    if max_distance<0:
        ind = _nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
        return data[tuple(ind)]
    else:
        d,ind = _nd.distance_transform_edt(invalid, return_indices=True)
        res   = data[tuple(ind)]
        res[d>=max_distance] = data[d>=max_distance]
        return res


def lookup(array,indices, order=1, mode='constant', cval=0):
    """ 
    Read values in 'array' at position 'indices' with interpolation of given 'order'
    
    Input:
    ------
        array:   ndarray
                 the input array of dimension N
        indices: array like
                 the coordinates of the position to lookup. It should have shape
                 NxK for K points in the N dimensional input array
        order:   int, optional (default 1)
                 The order of the spline interpolation. 
                 0 is nearest & 1 is linear
        mode:    'constant' (default), 'nearest', 'reflect' or 'wrap' - optional
                 method of evaluating position out of given array. 
    
    This function is a direct call to ndimage.interpolation.map_coordinates
    but with different default order, no prefilter, and different meaning
    """ 
    return _nd.interpolation.map_coordinates(array,indices,order=order, mode=mode, cval=cval)
    
def diagonal(array):
    """ Return a **view** of the diagonal elements of 'array' """
    from numpy.lib.stride_tricks import as_strided
    return as_strided(array,shape=(min(array.shape),),strides=(sum(array.strides),))
    
## todo neighbor-array
#def neighborArray(array, neighbor=None, border='pad', footprint=None, mode='constant', cval=0):
#    """
#    create a N+1 dimensional array from N-d input array.
#    
#    nborArray = neighborArray( in-argument )
#    
#    nborArray[..., i] is the ith neighbor of array element '...' (the indices in input array
#    
#    either neighbor liste of indices, or footprint Nd array with non zeros 
#    elements indicating neighbor, should be provided
#    
#    border = 'crop' consider cropped array as main array
#             'pad'  reallocate a bigger array by padding input one
#             'none' don't manage border, but side neighbors will be out of array memory
#    ##TODO: use pad array instead, or something else ???
#    """
#    from scipy import ndimage as nd
#    
#    # K is the list of neighbor
#    if neighbor is None:
#        K  = _np.transpose(footprint.nonzero())   # neighbors are nonzero value of footprint
#        K -= (_np.array(footprint.shape)/2)       # center footprint
#    else:
#        K = _np.asarray(neighbor)
#        
#    KNum = K.shape[0]
#    
#    nArray = _np.zeros((array.size,KNum), dtype=array.dtype)
#    
#    for i in range(KNum):
#        nArray[:,i] = nd.interpolation.shift(array,tuple(-K[i]),order=0,mode=mode,cval=cval).ravel()
# 
#    nArray.shape = array.shape + (KNum,)
#    
#    return nArray



# add a quick pointer to ArrayGraph here
#from graph import ArrayGraph as Graph


