import numpy as _np
import scipy.ndimage as _nd

from . import ravel_indices      as _ravel
from . import second_derivatives as _2nd_derivatives
from . import gradient_norm      as _grad_norm
from . import local_min          as _local_min
from . import fill               as _fill
from .filter import linear_map   as _linear_map
from .filter import apply        as _filter

from rhizoscan.workflow.openalea import aleanode as _aleanode # decorator to declare openalea nodes


@_aleanode('label','N')
def cluster(array, method='watershed', seed='local_min', seed_min_d=1, **filter_args):
    """ Segment image in area around selected seeds
    
    :Input:
        array:  The input array to label
               
        method: The clustering method. One of:
                'gradient' - call shortest_path_cluster(...) => require skimage 
                'nearest'  - cluster pixels to closest seed
                'watershed' (default) - use scipy.ndimage.watershed_ift
                
        seed:   The seed to cluster image around. Can be either an interger 
                array where seeds are none zeros cells, or a string indicating 
                which method to use to select seeds. 
                In the later case, the value must be one of the valid 'method' 
                of the find_seed() function
                - See find_seed() documentation -
                  
        seed_min_d: Optional arguments passed to find_seed(...) 
        
        Other key-value arguments can be given for initial filtering, which are
        passed to ndarray.filter.apply(...) function. 
        Note that the order in which filters are processed is not garantied
    
    :Output:
        the labeled image, interger array of values in [0,N] 
        the number of segmented area (N)
       
       
    :Example of postprocessing: 
        #To have the minimum, maximum and mean value of label area

        import numpy as np
        import scipy.ndimage as nd
    
        label,N  = label(img) 
        
        lab_max  = np.array(nd.maximum(img,label,index=np.arange(0,N+1)))
        lab_min  = np.array(nd.minimum(img,label,index=np.arange(0,N+1)))
        lab_mean = np.array(nd.mean   (img,label,index=np.arange(0,N+1)))
        
        # to get a map of these values onto the labeled area 
        # (eg. for visualisation or further image processing) 
        min_map  = lab_min [label] 
        max_map  = lab_max [label] 
        mean_map = lab_mean[label] 
        
    See scipy.ndimage for other processing of label map
    """
    # filter image
    image = _filter(array,**filter_args)

    # find seeds
    if isinstance(seed,basestring):
        seed,N = find_seed(image,method=seed,min_distance=seed_min_d)
    else:
        N = seed.max()
    
    # label the image using selected clustering method
    if cluster=='gradient':
        # cluster pixels to seed such that connecting path minize sum of gradient norm 
        try:
            label = shortest_path_cluster(_grad_norm(image),seed,geometric=True)
        except ImportError:
            print 'Error importing skimage, switch to "watershed" method'
            label,N = label(image,cluster='ift',seed=seed,seed_min_d=1)
        
    elif cluster=='nearest':    # indices map to the closest seed
        label = _fill(seed,seed!=0)
        
    else:  # cluster=='watershed'  # use scipy.ndimage.watershed_ift
        image = _np.asarray(_linear_map(image,min=0,max=1)*255,dtype='uint8') ## image auto-conversion tool
        label = _nd.watershed_ift(input=image, markers=seed) 
    
    return label,N

@_aleanode('seed','N')
def find_seed(image, method='local_min', min_distance=1, **filter_args):
    """
    Select seed pixels for clustering
    
    :Input:
        image: the nd array to find seed in
        method: the method use to select seeds. It can be:
                - local_min: find local minima (default)
                - extremum:  find local extrema
                - dist2border: find pixels the further from thresholded gradient norm
        min_distance: the minimum number of pixel seperating 2 seeds
        
        optional key-arguments that is passed to array.filter.apply()
        
    :Output:
        seed_map: an array of same shape as input image, of integer dtype
                  where background are 0, and seeds are positive integer. A seed
                  can be several connected pixels. All seeds have different values
        N: the number of seed found
        
    :todo:  
        extremum is not done yet => mixed of null gradient, null 2nd derivatives
        pb:     
    """
    # filter image
    image = _filter(image,**filter_args)

    # make height map for seed detection (highest pixels)
    if method=='extremum':
        # seeds are local minimum of the absolute value of the norm of the
        # second derivatives of the image 
        seed = reduce(_np.add,map(_np.square,_2nd_derivatives(image)))**0.5
    elif method=='dist2border':
        # seeds are pixels with maximum distance to "main" gradient 
        # (i.e. gradient norm thresholded using otsu's method) 
        seed = _grad_norm(image)
        seed = threshold(seed,'otsu')
        seed = -_nd.distance_transform_edt(-seed)
    else:
        # method=='local_min':
        seed = image
        
    # find and label seeds
    seed   = _local_min(seed,footprint=(2*min_distance+1))  # find local maximum
    seed,N = _nd.label(seed, structure=_np.ones(image.ndim*(3,)))     # label them        
    
    return seed,N
        
@_aleanode('labeled_cluster')
def shortest_path_cluster(array,seed,geometric=True):
    """
    cluster array cells around seeds such that the connecting path has minimum cost
    The cost of a path is the sum of the array cells value on the path 

    *** This function require the skimage (scikits image) module ***

    :Input:
        array: float   array containing the cost of each pixels
        seed:  integer array where 0 are background cells, to cluster around seeds
               and positive value are clustering seed (the value is the label)
        geometric: if True, weight diagonal edges by 1/sqrt(2)
                   - see skimage.graph.MCP and MCP_geometric for details -
                   
    :Output:
        labeled array, i.e. an integer array where each cell has the value
        of the closest seed
    """
    import skimage.graph as graph
    
    # create graph object
    if geometric: g = graph.MCP_Geometric(array)  
    else:         g = graph.MCP          (array)
    
    c,t = g.find_costs(zip( *seed.nonzero() ))   # compute minimum path
    
    # convert skimage.graph trace to parenting index-map (with flat indices)
    offsets = _np.concatenate((-g.offsets,[[0]*seed.ndim]))         # add null shift at end of offsets
    p = _np.arange(seed.size) + _ravel(offsets[t.ravel()],array.shape) #_np.dot(offsets[t.ravel()],[array.shape[1],1])

    # find tree roots (top ancestor of all array elements)
    # iteratively replace parent indices by grand parent, until there is no change
    gp  = p[p]   # grand-parent
    ggp = p[gp]  # grand-grand-parent
    while _np.any(gp!=ggp):
        gp  = ggp
        ggp = p[gp]
    
    gp.shape = seed.shape

    return seed.ravel()[gp]



@_aleanode('dilated_label')
def label_dilation(label, distance, metric='chessboard'):
    """
    Dilate labeled area in label array by given distance (in pixels)
    A label cannot dilate over any other
        
    Input:
        label:    a label array
        distance: a scalar indicating the euclidian distance
        metric:   if 'chessbord', use chessbord distance
                  if 'taxicab',   use taxicap   distance
                  otherwise,      use euclidian distance (slower)
    """
    if metric in ('taxicab','chessboard'):
        dist, ind = _nd.distance_transform_cdt(label==0, return_indices=True,metric=metric)
    else:
        dist, ind = _nd.distance_transform_edt(label==0, return_indices=True)
    
    dil_label = label[tuple(ind)]
    dil_label[dist>distance] = 0

    return dil_label

@_aleanode('closed_label')
def label_erosion(label, distance, metric='chessboard'):
    """
    Erode labeled area in label array by given distance (in pixels)
    
    The erosion is done with respect to background only, not other label. Thus 
    connected label stay connected, if the touching pixels are far enough from
    the background.
        
    Input:
        label:    a label array
        distance: a scalar indicating the euclidian distance
        metric:   if 'chessbord', use chessbord distance
                  if 'taxicab',   use taxicap   distance
                  otherwise,      use euclidian distance (slower)
    """
    ##todo: as fast, use bin erosion (grey erosion is no use)?
    # what about distance ? use big (ellipse) structure, or low with iterations ? 
    if metric in ('taxicab','chessboard'):
        dist, ind = _nd.distance_transform_cdt(label!=0, return_indices=True,metric=metric)
    else:
        dist, ind = _nd.distance_transform_edt(label!=0, return_indices=True)
    
    ero_label = label.copy()
    ero_label[dist<=distance] = 0

    return ero_label

@_aleanode('label_size')
def label_size(label, min_label=None):
    """
    Number of pixels in each label
    
    min_label is the minimum number of label to compute the size of, i.e. the 
    minimum size of the returned array .
    """
    #D return _np.array(_nd.sum(rar.virtual_array(label.shape),labels=label, index=range(0,label.max()+1)))
    return _np.bincount(label.ravel(), minlength=min_label)

@_aleanode('cleaned_label')
def clean_label(label, min_size = 1, min_dim=0):
    """
    remove labeled area that have 
      - a number of pixels less than min_size
      - both dimension of bounding box less than than min_dim
    
    Return:
      updated label map with contiguous numbering
    """
    valid = label_size(label)>=min_size
    valid[0] = True  # always keep background
    
    if min_dim:
        label_dim  = [False if o is None else max(map(lambda sl: sl.stop-sl.start,o))>=min_dim for o in _nd.find_objects(label)]
        valid[1:] &= label_dim

    valid[0] = 0
    maplab   = _np.cumsum(valid) * valid
    label = maplab[label]
        
    return label


## in development
#----------------
def fill_label_holes(label):
    # in development (working): fill holes in labeled area
    ##todo: doc, add max_width option ?
    bg = label==0
    bg_lab,bg_n = _nd.label(bg)
    flabel = _fill(label,label==0)
    
    bg_std = _nd.standard_deviation(flabel,bg_lab,range(1,bg_n+1))
    bg_std.insert(0,0) 
    bg_std = _np.array(bg_std)
    flabel[(bg_std>0)[bg_lab]] = 0
    
    return flabel
    
def interpixel_watershed(img,min_dist=3,max_iter=100000, **filter_args):
    ##don't work... stop dev at sort_label
    img = _filter(img, **filter_args)
    
    # make marker map
    size = (2*min_dist+1)
    marker = _local_min(-img,footprint=size).astype('int16') # find markers
    marker,N = _nd.label(marker, structure=_np.ones(img.ndim*(3,)))
    
    # sort label (marker)
    mean  = _nd.mean(img,marker,index=_np.arange(1,N+1))
    order = _np.argsort(mean)
    # here order[i] = position of marker==i+1 in the sorted list of all marker values
    #order[marker[marker!=0]-1]
    
    # => knowing that all pixels of a label have same image value
    #mask = marker!=0
    #label  = marker[mask]
    
    #order = _np.argsort(img[mask])
   
    
    #marker[mask] = label[order]
    
    return marker, N, order

def sort_label(label,value,N):
    """don't work..."""
    mask = label!=0
    lab  = label[mask]
    ord  = _np.argsort(lab)
    lab  = lab[ord]
    val  = value[mask][ord]
    
    order = _np.sort(_nd.mean(val,lab,index=_np.arange(1,N+1)))
    
    lab = label.astype('float32')
    lab[mask] = order[label[mask]-1]+1
    
    return lab
    

def hysteresis(img, threshold='otsu', step=0.1):
    """
    :todo: finish => choose suitable arguments, and procedure
    """
    if threshold == 'otsu':
        threshold = otsu(img)
    else: threshold = _np.asarray(threshold)
    
    if threshold.size==1:
        threshold = threshold - _np.arange(0,threshold,step)
        
    mask = img >= threshold[0]
    for t in threshold[1:]:
        mask = mask | (_nd.binary_dilation(mask) & (img > t))
    
    return mask
    
