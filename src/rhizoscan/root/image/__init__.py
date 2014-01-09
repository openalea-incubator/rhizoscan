import numpy as _np
import scipy.ndimage as _nd

from rhizoscan.ndarray              import virtual_array  as _virtual_arr 
from rhizoscan.ndarray.measurements import clean_label    as _clean_label
from rhizoscan.ndarray.measurements import label_size     as _label_size
from rhizoscan.image.measurements   import skeleton_label as _skeleton_label # used by linear_label
from rhizoscan.image                import Image          as _Image

from rhizoscan.workflow import node as _node # to declare workflow nodes

@_node('image', inputs=[dict(name='filename',interface='IFileStr'),dict(name='normalize',interface='IBool',value=True)])
def load_image(filename, normalize=True):
    img = _Image(filename, dtype='f', color='gray')
    if normalize:
        img,op = normalize_image(img)
        img.__serializer__.post_op = op
    return img

@_node('normalized_image')
def normalize_image(img):
    """
    Set image [min,max] to [0,1] and root pixel lighter than background
     --- computation are done *in place* ---
     
    :Outputs:
     - the normilized image
     - a list of operator string such as used by rhizoscan.image.PILSerializer
    """
    imin = img.min()
    imax = img.max()
    img -= imin
    img /= imax
    op = ['add(image,-%s)'%str(imin),'div(image,-%s)'%str(imax)]
    if img.mean() > 0.5:
        img[:] = 1-img
        op.append('sub(1,image)')
        
    return img, op


@_node('filtered_image')
def remove_background(image, distance, smooth=1):
    """
    Simple background removal method
    
    Method:
    -------
    compute    image - background
    where the background is estimated as by interpolation of local minima
    
    The local minima are computed to be spaced by at least 'distance' pixels
    
    The interpolation is nearrest if smooth is 0, and an average between closest
    minima weighted by a gaussian distribution with sigma = smooth*distance pixels
    This is not a real interpolation, but a fast approximation. In particular, 
    pixels which are to far away from any minima (more than 3 sigma) won't be
    interpolated and get value 0.
    """
    lmin = _nd.minimum_filter(image,size=distance)
    if smooth: return image - _nd.gaussian_filter(lmin,sigma=distance*smooth)
    else:      return image - lmin
    

@_node('mask')
def segment_root_image(image, mask=None):
    """
    Segment root pixels from background pixels
    
    The segmentation is done using EM fitting of 2 gaussian distributions: 
    one for the background, and one for the foreground.
    
    If mask is provided, do not segment these area
    """
    from rhizoscan.stats import gmm1d, cluster_1d  ## relative import ?
    # segment image
    # -------------
    if mask is None: m = slice(None)
    else:            m = mask
    im   = image
    #n,w  = gmm1d(im[m], classes=2, bins=256)
    mask = _np.zeros(im.shape,dtype=int)
    mask[m] = cluster_1d(im[m], classes=2, bins='unique' if im.dtype=='uint8' else 256)#, distributions=n, w)
    
    return mask
        
@_node('root_mask','transform', 'bbox')
def segment_root_in_petri_frame(image, plate_border=0.06, plate_width=1, min_dimension=5, filtering=1, is_segmented=False):
    """
    Segment root image and detect a petri plate
    
    Once segmented (using segment_root_image), the petri plate is detected as 
    the biggest contiguous area and a linear homogeneous transformation is 
    fitted (homography).
    
    Input:
    ------
        image:         an image of root system in a petri plate
        plate_border:  the size of the plate border (overlap of the top part on 
                       the bottom), it should be given as a percentage of the 
                       plate width
        plate_width:   the plate size in your unit of choice (see output) 
        min_dimension: remove pixel area that have less that this number of 
                       pixels in width or height.
        is_segmented:  if True, the input image is already segmented.
        
    Output:
    -------
      - The segmented (labeled) root mask cropped around the petri plate.
        The petri plate are removed
      - The 3x3 transformation matrix that represents the mapping of image 
        coordinates into the plate frame with origin at the top-left corner, 
        y-axis pointing downward, and of size given by plate_width
      - The bounding box of the detected plate w.r.t the original mask shape
        given as a tuple pair of slices (the cropping used)
        
    ##todo: to be moved to root.image.plate (which could be renamed to petri or petri_plate)
    """
    from .plate import find_plate
    
    if is_segmented: mask = image
    else:            mask = segment_root_image(image)
    
    # cluster mask & remove border 
    cluster = _nd.label(mask, structure=_np.ones((3,3)))[0]
    cluster[:,0] = 0; cluster[:,-1] = 0
    cluster[0,:] = 0; cluster[-1,:] = 0

    # find objects
    obj   = _nd.find_objects(cluster)
    obj   = [o if o is not None else (slice(0,0),)*2 for o in obj]
    osize = _np.array([(o[0].stop-o[0].start)*(o[1].stop-o[1].start) for o in obj])
    # slightly more robust than argmax only (?)
    pid   = (osize>0.9*osize.max()).nonzero()[0]+1  
    ##old: pid   = _np.argmax([(o[0].stop-o[0].start)*(o[1].stop-o[1].start) for o in obj])+1
    
    # get slice of plate object, and dilate it by 1 pixels
    #   > always possible because cluster border have been set to zero
    ##old: plate_box = [slice(o.start-1,o.stop+1) for o in obj[pid-1]]
    obj_plate = [obj[i] for i in pid-1]  
    min0 = min([o[0].start-1 for o in obj_plate])
    max0 = max([o[0].stop +1 for o in obj_plate])
    min1 = min([o[1].start-1 for o in obj_plate])
    max1 = max([o[1].stop +1 for o in obj_plate])
    plate_box = [slice(min0,max0),slice(min1,max1)]
    cluster   = cluster[plate_box]
    plate = reduce(_np.add,[cluster==id for id in pid])
    # quicker and more robust than fill_holes
    #   => only works because plate has a convex shape
    ##old: plate = _nd.binary_fill_holes(plate)
    def cs(x,axis,reverse):
        if reverse: sl = [slice(None)]*axis + [slice(None,None,-1)]
        else:       sl = slice(None)
        return _np.cumsum(x[sl],axis=axis,dtype=bool)[sl]
    plate = cs(plate,1,1)&cs(plate,1,0)&cs(plate,0,1)&cs(plate,0,0)
    
    # remove cluster not *inside* the plate 
    d2out    = _nd.distance_transform_edt(plate)
    border   = plate_border*2*d2out.max() # d2out.max: plate radius
    in_plate = d2out >= border
    cluster  = cluster*in_plate
        
    # remove not big enough clusters
    cluster  = _clean_label(cluster,min_dim=min_dimension)
    if filtering>0:
        mask = _nd.binary_dilation(cluster>0,iterations=filtering)
        mask = _nd.binary_closing(mask,structure=_np.ones((3,3)),iterations=filtering)
    else:
        mask = cluster>0
        
    plate = find_plate(plate.astype('f'),border=border, plate_width=plate_width)

    return mask, plate, plate_box

@_node('root_cluster','transform', 'bbox')
def segment_root_in_circle_frame(image, n=4, pixel_size=1, min_dimension=5, is_segmented=False):
    """
    Segment root image and detect a petri plate
    
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
    if is_segmented: mask = image
    else:            mask = segment_root_image(image)
    
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

#def neighbor_number(mask, diag=True):
#    """
#    Compute and return the number of neighbors of each pixels
#    
#    Input:             
#    ------
#        mask: should be a boolean 2d array object. If not boolean, instead of 
#              the neighbor number, this function compute the (possibly weighted)
#              sum of neighbors value
#              
#        diag: the cost of a diagonal elements
#                 - False means to count only direct neighbors (4-connected)
#                 - True  means to count    all      neighbors (8-connected)
#                 - other numerical value can be used to make weighted sum
#   
#    Output:
#    -------
#        an array of same shape as input mask, where 0 valued (input) pixels are
#        considered background: they are not counted as neighbors and their `
#        neighbor number (of non-zero neighbors) are returned as negative
#        
#          nbor = neighbor_number(mask)
#          
#          abs(nbor)        => number of foreground neighbors of all pixels
#          (nbor>0) * nbor  => neighbor number for foreground (non-0) pixels only 
#          (nbor<0) * nbor  => neighbor number for background (0)     pixels only 
#    """
#    ##move to image.measurement
#    kind  = ['b','u','i','f']
#    mask  = _np.asarray(mask)
#    diag  = _np.atleast_1d(diag)
#    mkind = (_np.argmax([k==mask.dtype.kind for k in kind]), mask.dtype.itemsize)
#    dkind = (_np.argmax([k==diag.dtype.kind for k in kind]), diag.dtype.itemsize) 
#    dtype = max(mkind,dkind,(2,4))
#    dtype = _np.dtype(kind[dtype[0]] + str(dtype[1]))
#        
#    x = mask.astype(dtype)
#    k = _np.array([[diag,1,diag],[1,0,1],[diag,1,diag]],dtype=dtype)
#    return (-1)**(x==0) * _nd.convolve(x,k)


