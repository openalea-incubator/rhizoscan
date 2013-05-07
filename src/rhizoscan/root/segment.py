"""
##OBSOLETE  => replaced by image package
"""


import numpy as _np
import scipy.ndimage as _nd

from rhizoscan  import ndarray   as rar
from rhizoscan.ndarray.filter           import otsu
from rhizoscan.ndarray.measurements     import clean_label 
from rhizoscan.image.measurements       import skeleton_label
from rhizoscan.workflow                 import Struct   as _Struct

from rhizoscan.workflow.openalea  import aleanode as _aleanode # decorator to declare openalea nodes

from . import stats  as _stats

# add reference to ndimage.label
label = _nd.label
_aleanode('label','N')(label)


@_aleanode('filtered_image')
def remove_local_min(image, distance, smooth=0, mask=None):
    """
    Simple background removal method: return   image - local_minimum(image)
    
    The local minimum is computed over a square of *radius* distance
    
    If smooth is not zeros, a simple interpolation is applied on the local
    minimum using a gaussian kernel with sigma equal to smooth*distance. 
    Note that the interpolation does not necessarily fit the local minimum, and 
    in general, it results in a returned image containing negative pixels.
    """
    lmin = _nd.minimum_filter(image,size=2*distance+1)
    if smooth: return image - _nd.gaussian_filter(lmin,sigma=distance*smooth)
    else:      return image - lmin
    

def segment_root(image, mask=None, min_cluster_dimension=5,detect_plate=1,crop=True, plate_border=0.06):
    """
    ##to develop
    compute mask (cluster label?) from image
      x compute mask using gmm
      x label
      x remove label < min_cluster_dimension
      x detect the biggest cluster (the plate)
      x remove all clusters out of plate
      x detach root connected to plate => dist from out of plate >= plate_border 
      x reorder label with plate last
      
      ! border pixels are set to zero
      return cluster map and plate bbox as a slice tuple
    """
    # segment image
    # -------------
    if mask is None: m = slice(None)
    else:            m = mask
    im   = image
    n,w  = _stats.gmm1d(im[m], classes=2, bins=256)
    mask = _np.zeros(im.shape,dtype=int)
    mask[m] = _stats.cluster_gmm1d(im[m], n, w)
    
    # cluster mask & remove cluster that are too little 
    # -------------------------------------------------
    cluster = _nd.label(mask, structure=_np.ones((3,3)))[0]
    cluster[:,0] = 0; cluster[:,-1] = 0
    cluster[0,:] = 0; cluster[-1,:] = 0

    if min_cluster_dimension:
        cluster  = clean_label(cluster,min_dim=min_cluster_dimension)
        
    plate_box = (slice(None),) * im.ndim
        
    # detect plate and filter cluster
    # -------------------------------
    if detect_plate:
        obj   = _nd.find_objects(cluster)
        pid   = _np.argmax([(o[0].stop-o[0].start)*(o[1].stop-o[1].start) for o in obj])+1
        
        if crop:
            # get slice of plate object, and dilate it by 1 pixels
            #   > always possible because cluster border have been set to zero
            plate_box = [slice(o.start-1,o.stop+1) for o in obj[pid-1]] 
            cluster   = cluster[plate_box]
        plate = cluster == pid
        
        # remove cluster not *inside* the plate 
        d2out    = _nd.distance_transform_edt(_nd.binary_fill_holes(plate))
        in_plate = d2out >= plate_border*2*d2out.max()
        cluster  = _nd.label(cluster * in_plate)[0]
        cluster  = clean_label(cluster,min_dim=min_cluster_dimension)
        
        # add plate cluster as last cluster 
        cluster[plate*(-in_plate)] = cluster.max()+1
        
        
    return cluster, plate_box


def find_seed(cluster, radius_min):
    from rhizoscan.ndarray import local_min
    from skimage.morphology import watershed
    dmap = _nd.distance_transform_edt(cluster>0)
    dmax = local_min(-dmap)*(dmap>1)
    marker,N = _nd.label(dmax)
    label = watershed(-dmap,marker,mask=cluster>0)
    lab_d = nd.maximum(dmap,labels=label,index=np.arange(N+1))
    seed = lab_d[label]>radius_min

    return seed  ## todo: label seed
    
    
@_aleanode('label', 'skeleton_structure','filter_data')
def linear_label(cluster, closing=1):
    """
    Partition cluster map as a set of linear segments
    
    Input:
        cluster: a label map of pixels clusters such as it is return by 
                 segment_root or scipy.ndimage.label
        
    Output:
        segment_map: the label map of all linear segment - cluster divided in segments
        segment_skl: the map of segments pixels of the skeleton
        node_skl   : the map of nodes    pixels of the skeleton
           
    require scikits.image
    """

    # "smooth" the mask boundary
    # --------------------------
    #nbor = neighbor_number(mask)
    #mask[nbor<-4] = 1  # add masked (0) pixels that touch at least 5 unmasked (1) pixels
    #nbor = _np.maximum(nbor,neighbor_number(mask))
    #mask[abs(nbor)<3] = 0  # remove unmask pixels that touch less than 3 masked pixels

    #maps = _Struct()
    #sklt = _Struct()

    # compute cluster map and filter those not big enough
    # ---------------------------------------------------
    # cluster = _nd.label(mask)[0]
    #if min_dim:
    #    maps.cluster  = clean_label(maps.cluster,min_dim=min_dim)
    #    mask          = maps.cluster!=0
        
    # compute labeled skeleton  
    # ------------------------
    mask = cluster!=0
    segment_skl,node_skl = skeleton_label(mask, closing=closing)[:2]
    
    # compute segment map  
    # -------------------
    segment_map = _nd.distance_transform_edt(segment_skl==0,return_indices=True,return_distances=False)
    segment_map = segment_skl[tuple(segment_map)] * (mask)

    return segment_map, segment_skl, node_skl


def _linear_label_old(mask,image=None, intensity=None, gradient=None, min_dim=0):
    """
    Label mask as a set of linear segments, the filter out selected label
    
    Input:
        mask:      the mask to be labeled
        image:     optional value array used by filtering
        intensity: filter out label with too low mean value     (*)
        gradient:  filter out label with too low mean gradient  (*)
        min_dim:   the minimum width and height of a cluster of connected labels  
        
        (*) require image to be given in argument. Value can be ont of
            None    No filtering is done
            'mean'  will threshold out component that have average intensity or
                    gradient less than the mean over the image.
            'otsu'  use otsu thresholding
            'delbg' remove an estilated background considered as the largest 
                    histogram pick
            any value that is used as threshold
        
    Output:
        the label map     (remaining after filtering)
        the skeleton structure (updated by filtering), containing 2 fields
           'segment' = the map of segments pixels
           'node'    = the map of nodes    pixels
        a 'filtering' structure containing the fields   - or None if no filter
            'label' :    the label map before filtering
            'intensity': the average intensity of labels used for filtering
            'gradient':  the average gradient  of labels used for filtering
            'linear':    the average linear coefficient of labels used for filtering
    """
    ## todo: rename variabel etc...
    #        add some boolean list of the min_dim label size (negative if removed)
        
    # "smooth" the mask boundary
    # --------------------------
    nbor = neighbor_number(mask)
    mask[nbor<-4] = 1  # add masked (0) pixels that touch at least 5 unmasked (1) pixels
    nbor = _np.maximum(nbor,neighbor_number(mask))
    mask[abs(nbor)<3] = 0  # remove unmask pixels that touch less than 3 masked pixels

    # compyte labeled skeleton then whole mask by (nearest neighbor) diffusion 
    seg,node = labeled_skeleton(mask)
    label    = _nd.distance_transform_edt(seg==0,return_indices=True,return_distances=False)
    label    = seg[tuple(label)] * (mask)


    # quit if no filtering is asked
    # -----------------------------
    if intensity is None and gradient is None and min_dim==0:
        return label,_Struct(segment=seg,node=node),None
        
    
    mask  = label!=0
    ind   = range(0,label.max()+1)
    
    lab2  = label.copy()
    defM  = _np.ones(len(ind))  # default sub-labeling  - keep all
    defM[0] = -1               # but background


    # filtering function
    def filter_label(data, threshold):
        sublab = _np.array(_nd.mean(data, labels=label, index=ind))
        sublab[0] = 0
        if threshold=='otsu':
            sublab -= otsu(data)
        elif threshold=='mean':
            sublab -= data.mean()
        elif threshold=='delbg':
            bg      = data <= (_np.mean(data) + 2*_np.std(data))
            sublab -= (_np.mean(data[bg]) + 2*_np.std(data[bg]))
        else:
            sublab -= data.max() * threshold
        return sublab
        
    # mean gradient norm of (dilated) label
    # -------------------------------------
    if gradient:
        lab_grad  = filter_label(rar.gradient_norm(image), gradient)
        lab2     *= lab_grad[label]>=0
    else:
        lab_grad  = defM
        
        
    # compute mean image value per label
    # ----------------------------------
    if intensity:
        lab_int  = filter_label(image, intensity)
        lab2    *= lab_int[label]>=0
    else:
        lab_int = defM
        
    # remove labels that is not big enough
    # ------------------------------------
    if min_dim:
        mask  = clean_label(_nd.label(lab2!=0)[0],min_dim=min_dim)!=0
        lab2 *= mask
        seg  *= mask
        node *= mask
    
    return lab2, _Struct(segment=seg,node=node), _Struct(label=label,intensity=lab_int,gradient=lab_grad)


@_aleanode('nbor_number')
def neighbor_number(mask, diag=True):
    """
    Compute and return the number of neighbors of each pixels
    
    Input:
    ------
        mask: should be a boolean 2d array object. If not boolean, instead of 
              the neighbor number, this function compute the (possibly weighted)
              sum of neighbors value
              
        diag: the cost of a diagonal elements
                 - False means to count only direct neighbors (4-connected)
                 - True  means to count    all      neighbors (8-connected)
                 - other numerical value can be used to make weighted sum
   
    Output:
    -------
        an array of same shape as input mask, where 0 valued (input) pixels are
        considered background: they are not counted as neighbors and their `
        neighbor number (of non-zero neighbors) are returned as negative
        
          nbor = neighbor_number(mask)
          
          abs(nbor)        => number of foreground neighbors of all pixels
          (nbor>0) * nbor  => neighbor number for foreground (non-0) pixels only 
          (nbor<0) * nbor  => neighbor number for background (0)     pixels only 
    """
    ## to be moved to image.measurement
    kind  = ['b','u','i','f']
    mask  = _np.asarray(mask)
    diag  = _np.atleast_1d(diag)
    mkind = (_np.argmax([k==mask.dtype.kind for k in kind]), mask.dtype.itemsize)
    dkind = (_np.argmax([k==diag.dtype.kind for k in kind]), diag.dtype.itemsize) 
    dtype = max(mkind,dkind,(2,4))
    dtype = _np.dtype(kind[dtype[0]] + str(dtype[1]))
        
    x = mask.astype(dtype)
    k = _np.array([[diag,1,diag],[1,0,1],[diag,1,diag]],dtype=dtype)
    return (-1)**(x==0) * _nd.convolve(x,k)


