import numpy as _np
import scipy.ndimage as _nd
                
from rhizoscan.ndarray              import virtual_array  as _virtual_arr 
from rhizoscan.ndarray.measurements import clean_label    as _clean_label
from rhizoscan.ndarray.measurements import label_size     as _label_size

from rhizoscan.workflow import node as _node # to declare workflow nodes

@_node('root_cluster','transform', 'bbox')
def segment_root(image, n=4, pixel_size=1, min_dimension=5, is_segmented=False):
    """
    Segment root image and find 'n' circles 
    
    Once segmented (using segment_root_image), the reference frame is detected 
    as made of circular shapes.
    
    Input:
    ------
      - image:         an image of root system in a petri plate
      - n:             the number of circles to be found
      - pixel_size:    size of a pixel in the unit of your choice
                         e.g. 1/(25.4*scaned-dpi) for millimeter unit
      - min_dimension: remove pixel area that have less that this number of 
                       pixels in width or height.
      - is_segmented:  if True, the input image is already segmented.
    Output:
    -------
      - The segmented (labeled) root mask cropped around the circle frame.
        The frame are removed.
      - The 3x3 transformation matrix that represents the mapping of image 
        coordinates into the detected frame: the origin is the top-left circle, 
        x-axis pointing toward the top-right circle, and the size (scale) is 
        computed based on the given 'pixel_size'. 
      - The bounding box containing all detected circles w.r.t the original mask 
        shape. Given as a tuple pair of slices (the cropping used)
        *** If n<=2, do not crop images ***
        
    ##Warning: currently it only crop vertically
    """
    if is_segmented:
        mask = image
    else:
        from . import segment_root_image
        mask = segment_root_image(image)
    
    d = _nd.distance_transform_edt(mask)
    cluster = _nd.label(d>0)[0]
        
    # remove not big enough clusters
    cluster[:,0] = 0; cluster[:,-1] = 0   # just in case, 
    cluster[0,:] = 0; cluster[-1,:] = 0   # remove border
    if min_dimension>=0:
        cluster  = _clean_label(cluster,min_dim=min_dimension)

    # detect frame circles
    area1 = _np.pi*_nd.maximum(d,cluster,index=_np.arange(cluster.max()+1))**2
    area2 = _label_size(cluster)
    fitv  = 2*area1 - area2 # area1 - abs(area1 - area2) ##?
    fitv[0] = 0
    index = _np.argsort(fitv)[-n:]
    
    if _np.sum(fitv[index]>0)<n:
        index = index[fitv[index]>0]
        print '  Warning, only %d reference circles detected, instead of %d' % (index.size,n)
        
    # find circles position and bbox in image
    obj = _np.asarray(_nd.find_objects(cluster))[index-1,:] 
    start = _np.vectorize(lambda o: o.start)(obj).min(axis=0)
    stop  = _np.vectorize(lambda o: o.stop )(obj).max(axis=0)
    pos   = _np.asarray(_nd.center_of_mass(_virtual_arr(shape=cluster.shape), labels=cluster, index=index))
    
    # remove circle mask from cluster
    for o,i in enumerate(index):
        subm = cluster[obj[o][0], obj[o][1]]
        subm[subm==i] = 0
        
    # crop cluster map, if possible
    if index.size>2:
        circle_box = map(slice,start,stop)
        circle_box[1] = slice(0,cluster.shape[1]) ## only crop vertically
        cluster    = cluster[circle_box]
    
        # detect x-coordinates: top circles are on the x-axis
        order = _np.argsort(pos[:,0])
        pos   = pos[order][:2]           # keep the top two circles
        order = _np.argsort(pos[:,1])
        y,x   = pos[order].T             # sort by x-coordinates
        angle = _np.arctan2(y[1]-y[0], x[1]-x[0]) # angle to horizontal
        
        # create affine transorm  - coord as in [y,x,1] order !!
        sa = _np.sin(angle)
        ca = _np.cos(angle)
        R  = _np.array([[ca,-sa, 0],[sa, ca, 0],[0,0,1]])
        T  = _np.array([[1,0,-y[0]],[0,1,-x[0]],[0,0,1]])
        T  = _np.dot(R,T)*pixel_size
        T[-1,-1] = 1
    else:
        T = _np.eye(3)
        circle_box = map(slice,cluster.shape)
    

    return cluster, T, circle_box

def detect_circles(n, mask=None, cluster=None, dmap=None, min_dimension=None, min_quality=0):
    """
    Find 'n' circles in mask (or cluster)
    
    For all connected components of mask (or cluster), this function compute a 
    "quality" coefficient as the area of the biggest inscribed circles minus 
    the area of the cluster out of that circle. 
    It then returns the indices of the n clusters with highest coefficient 
    
    The quality coefficient can be seen as the size of the circles, minus the 
    error to being a circle. The min_quality parameter would mean the minimum 
    radius of detected circles, subestimated by accepted error area.
    
    :Input:
        n:  
            Number of circles to be found
        mask:    
            Binary array to find the circles in (not used if cluster is given)
        cluster:
            (optional) Label map of mask. If given, mask is  not used. Otherwise 
            it is computed from mask            
        dmap:
            (optional) Distance map of mask
        min_dimension: 
            Does not process clusters that don't have at least one dimension
            bigger than 'min_dimension'
        min_quality:
            If not None, filter out cluster with quality less than this.
            If less than n cluster respect that rule, and n>0, print a warning.
            
    :Output:
        - indices of the 'n' detected "best" circles in clsuter map
        - estimated "quality" of the circles
        - cluster map  (updated in-place if given)
        - distance map (same as input if given)
    """
    if cluster is None:
        if mask is None:
            raise TypeError('either mask of cluster should be given')
        cluster = _nd.label(mask)[0]
    
    # remove not big enough clusters
    cluster[:,0] = 0; cluster[:,-1] = 0   # remove borders
    cluster[0,:] = 0; cluster[-1,:] = 0   # --------------
    if min_dimension>=0:
        cluster[:]  = _clean_label(cluster,min_dim=min_dimension)

    if dmap is None:
        dmap = _nd.distance_transform_edt(cluster>0)
        
    # detect frame circles
    area1 = _np.pi*_nd.maximum(dmap,cluster,index=_np.arange(cluster.max()+1))**2
    area2 = _label_size(cluster)
    fitv  = 2*area1 - area2 # area1 - abs(area1 - area2) ##?
    fitv[0] = 0
    index = _np.argsort(fitv)[-n:][::-1]
    
    if min_quality is not None:
        index = index[fitv[index]>min_quality]
        if n>0 and index.size<n:
            print '  Warning, only %d circles detected, instead of %d' % (index.size,n)
    
    return index, fitv[index], cluster, dmap
