import numpy as _np
import scipy.ndimage as _nd

from rhizoscan.ndarray import local_min as _local_min
from rhizoscan.ndarray.measurements import label_size as _label_size
from rhizoscan.stats import gmm1d       as _gmm1d
from rhizoscan.stats import cluster_1d  as _cluster_1d

from rhizoscan.workflow import node as _node # to declare workflow nodes
@_node('leaf_map')
def detect_leaves(mask, image, leaf_number, root_radius, leaf_bbox=None, sort=True):
    """
    Detect leaves cluster in 'mask' based on pixels luminosity of 'image'
    
    Input:
    ------
    mask: 
        A binary array
    image: 
        The luminosity of pixels. Should have the same shape as mask
    leaf_number: 
        Number of clusters to extract - The biggest detected clusters are kept
    root_radius:
        Parameter used for morphological filtering (closing). The given value 
        should typically be be higher than maximum root radius, butin the same 
        order (approx. twice the real value is a good choice).
    leaf_bbox:
        Optional bounding box in mask to look for leaves: 
        a `[xmin,ymin,xmax,ymax]` list
    sort:
        if True, sort detected leaves cluster by their x-coordinates
        
    output:
    -------
        computed seed map
    """
    
    # segment mask
    label, N, radius, dmap = blob_cluster(mask)
    
    # classify leaf area
    lab_lum = _nd.maximum(image,label,index=_np.arange(N+1))
    lab_lum[0] = 0
    
    bins = 256 if image.dtype!='uint8' else 'unique'
    n,w = _gmm1d(image[mask],classes=2, bins=bins)
    lab_leaf = _cluster_1d(lab_lum,distributions=n,weights=w, bins=bins)
    lab_leaf[0] = 0
    
    leaf = (lab_leaf>0)[label]  # leaf mask
    
    # morphological filtering: 
    #   masked closing = masked dilation + masked erosion (i.e masked dilation of -leaf)
    leaf = _nd.binary_dilation(leaf,iterations=root_radius, mask = mask)#, output=leaf)
    leaf = (_nd.binary_dilation((leaf==0)*mask,iterations=root_radius,mask=mask)==0)*mask
    
    return _cluster_seed(seed_mask=leaf, seed_number=leaf_number, seed_bbox=leaf_bbox, sort=sort)

@_node('seed_map')
def detect_seeds(mask, seed_number, radius_min, seed_bbox=None, sort=True):
    """
    Detect seed clusters in 'mask' based on local shape radius
    
    Input:
    ------
    mask: 
        A binary array
    seed_number: 
        Number of clusters to extract - The biggest detected clusters are kept
    radius_min:
        Minimum radius of seed, in pixels
    seed_bbox:
        Optional bounding box in mask to look for seeds: 
        a `[xmin,ymin,xmax,ymax]` list
    sort:
        if True, sort detected leaves cluster by their x-coordinates
        
    output:
    -------
        computed seed map
    """
    ## replace radius_min selection by gmm clustering on blob radius ?
    ## add morphological closing? using radius_min as param ?
    
    # find seed pixels
    label, N, radius, dmap = blob_cluster(mask)#, dmin=radius_min)
    #radius = _nd.maximum(dmap,labels=label,index=_np.arange(N+1))
    seed = (radius>radius_min)[label]  ##-1
    seed[label==0] = 0
    
    return _cluster_seed(seed_mask=seed, seed_number=seed_number, seed_bbox=seed_bbox, sort=sort)
    
def _cluster_seed(seed_mask, seed_number, seed_bbox=None, sort=True):
    if seed_bbox is not None:
        wbound = [seed_bbox[0],seed_bbox[2]]
        hbound = [seed_bbox[1],seed_bbox[3]]
        wbound = [int(w*seed_mask.shape[1]) for w in wbound]
        hbound = [int(h*seed_mask.shape[0]) for h in hbound]
        seed_mask[:,:wbound[0]] = 0
        seed_mask[:,wbound[1]:] = 0
        seed_mask[:hbound[0],:] = 0
        seed_mask[hbound[1]:,:] = 0
    
    # identify 'seed_number' best clusters (best=biggest)
    seed_lab,N = _nd.label(seed_mask)
    seed_size  = _label_size(seed_lab)
    seed_size[0] = 0
    seed_id = _np.argsort(seed_size)[-seed_number:]
    
    seed_id = seed_id[seed_id>0] ## just in case, but need a better algorithm
    
    # sort selected seed clusters by their x-coordinates
    if sort:
        obj  = _nd.find_objects(seed_lab)
        xmin = [obj[i-1][1].start for i in seed_id]
        seed_id = seed_id[_np.argsort(xmin)]
        
    # map detected seeds labels to range(leaf_number) 
    dt = 'uint8'
    seed_map = _np.zeros(N+1, dtype=dt)
    seed_map[seed_id] = _np.arange(1,seed_number+1, dtype=dt)

    return seed_map[seed_lab]

@_node('blob_map','blob_number','distance_map')
def blob_cluster(mask, dmin=1):
    """
    Cluster mask into the distance map local maximum
    This is some radius based shape-clustering
    
    :Output:
        label:  the label map of the mask
        N:      the number of labels
        radius: the maximum radius of all label ##**starting at label 1**
        dmap:   the distance map of the mask
    
    *** require skimage:  skimage.morphology.watershed ***
    """
    from skimage.morphology import watershed
    dmap = _nd.distance_transform_edt(mask)
    dmax = _local_min(-dmap, footprint=1+2*dmin)*(dmap>dmin)
    marker,N = _nd.label(dmax)
    #pos,r = strong_local_max(dmap)
    #N   = pos.shape[0]
    #marker = _np.zeros(dmap.shape, dtype=int)
    #marker[pos[:,0],pos[:,1]] = _np.arange(1,N+1)
    label = watershed(-dmap,marker,mask=mask)
    r = _nd.maximum(dmap, marker, _np.arange(N+1))
    return label, N, r, dmap
    
def strong_local_max(dmap, dmin=1):
    """ method to return the "strong" local maximum of a distance map
    
    Weak local maximum are local maximum that are in the cirle of influence of 
    a bigger one (i.e. that has a bigger distance to border).
    This method returns all the local maximum that are not "weak".
    
    :Input:
        dmap: a distance_map to compute the local maximum from
        dmin: minimum distance between two local maximum
        
    :Output:
        a Nx2 array of the position of the local maximum 
        a length N vector of the radius of influence of those local maximum
    """
    from scipy.spatial.distance import cdist
    from scipy.spatial import cKDTree
    
    # all local max, their position and a kdtree used below 
    dmax = _local_min(-dmap,footprint=2*dmin+1)*(dmap>1)
    pos  = _np.transpose(dmax.nonzero())
    kdt  = cKDTree(pos)
    
    # radius of influence of local max, and their descending order 
    radius = dmap[pos[:,0],pos[:,1]]  ## but for here, it seems it could work in nD
    order  = _np.argsort(radius)[::-1]
    
    # remove local max that are in the circle of influence  
    # of a local max that has a bigger radius
    i = -1
    keep = _np.ones(pos.shape[0],dtype=bool)
    while i<order.size:
        # for each local max, in descending order of their radius,
        #   remove all other local max that are in the circle of influence
        
        # find next unremoved local max
        while i<order.size and keep[order[i]]==0: 
            i+=1
        if i>=order.size: 
            break
            
        ##print '%d/%d - %f' % (i, len(order), keep.mean());
        
        # use kdtree to quickly find the local max to remove
        ##   do this directly in the image ? faster or slower ?
        p = pos[order[i],:]
        max_d = max(radius[order[i]], 1.9)  ## always remove direct neighbor
        dist, ind = kdt.query(x=p,k=(2*max_d+1)**2 /2, distance_upper_bound=max_d)
        keep[ind[-_np.isinf(dist)]] = 0
        keep[order[i]] = 1  # do not remove self !  
        i = i+1
        
    return pos[keep], radius[keep]
    

