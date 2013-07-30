import numpy as _np
import scipy.ndimage as _nd

from rhizoscan.ndarray import local_min as _local_min
from rhizoscan.ndarray.measurements import label_size as _label_size
from ..stats import gmm1d       as _gmm1d
from ..stats import cluster_1d  as _cluster_1d

from rhizoscan.workflow.openalea import aleanode as _aleanode

@_aleanode('leaf_map')
def detect_leaves(mask, image, leaf_number, root_radius, leaf_height=None, sort=True):
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
    leaf_height:
        if not None, must be a two element list of the minimal/maximal height of
        the leaves in the mask image
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
    
    return _cluster_seed(seed_mask=leaf, seed_number=leaf_number, seed_height=leaf_height, sort=sort)

@_aleanode('seed_map')
def detect_seeds(mask, seed_number, radius_min, seed_height=None, sort=True):
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
    
    return _cluster_seed(seed_mask=seed, seed_number=seed_number, seed_height=seed_height, sort=sort)
    
def _cluster_seed(seed_mask, seed_number, seed_height=None, sort=True):
    if seed_height is not None:
        bound = [int(h*seed_mask.shape[0]) for h in seed_height]
        seed_mask[:bound[0]] = 0
        seed_mask[bound[1]:] = 0
    
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

@_aleanode('blob_map','blob_number','distance_map')
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
    

## ---------- detect seed in graph *OLD* -----------
#  -------------------------------------------------

def detect_graph_seed(graph,seed_prop,seed_number):

    # find leaves segments
    ## hard-written spatial prob !!!
    #Y = _np.mean(graph.node.y[graph.segment.node],axis=1)
    #Y = _np.exp(-(Y-200)**2/150**2)
    S = _cluster_1d(graph.segment[seed_prop],classes=2,bins=256)
    
    from .graph import SegmentGraph
    nb = SegmentGraph(segment=graph.segment, node=graph.node).edges
    nb[nb>graph.segment.size+1] = 0
    
    #not_seed = (S==0)&(-(S[nb]&(nb!=0)).any(axis=1))
    #sur_seed = (S>0)&(S[nb]|(nb==0)).all(axis=1)
    
    S[0] = False#True  # do not propagate
    
    invalid = nb==0
    valid = -invalid
    not_seed = _np.sum(valid,axis=1)==0
    
    # opening: remove too little seeds
    S = (S[nb]|invalid).all(axis=1)  # erode
    S = (S[nb]&  valid).any(axis=1)  # dilate
    
    # morphological closing
    S_prev = []
    i = 0
    while _np.any(S!=S_prev) & (i<10): #for i in range(c):
        S_prev = S.copy()
        S = (S[nb]&  valid).any(axis=1)  # dilate
        S = (S[nb]|invalid).all(axis=1)  # erode
        S[not_seed] = False
        i += 1
        #graph.plot('k',sc=S+1)
        #raw_input(str(i))
        
    # id connected leaves cluster
    sid = _np.cumsum(S>0)*(S>0)
    sid0 = []
    while _np.any(sid0!=sid):
        sid0 = sid.copy()
        sid = _np.maximum(_np.max(sid[nb],axis=1)*(sid>0), sid) #maximum is for robustess, should not happen theoretically

    # keep the 'seed_number' biggest leaves area
    area = graph.segment.area
    suid = _np.unique(sid)[1:] # no zeros id
    sarea = _nd.sum(area, labels=sid, index = suid) 
    
    sarea_order = _np.argsort(sarea)
    suid = suid[sarea_order[-seed_number:]]
    
    # sort seed clusters by the x-coodinates for there bounding box
    # and renumber of leaves id: from 1 to seed_number
    sx = _np.zeros(suid.shape)
    for i,id in enumerate(suid):
        sx[i] = _np.min([s.start for s in graph.segment.x_box[sid==id]])
    rank = sx.argsort().argsort()+1
    
    idmap = _np.zeros(sid.max()+1, dtype=int)
    idmap[suid] = rank
    #idmap = _np.cumsum(idmap)*(idmap>0)
    
    graph.segment.leaves = idmap[sid]
    graph.save()

    return graph.segment.leaves

def track_graph_seed(graph_seq, seed_prop, seed_number):
    # iteratively detect seed in each graph of graph_seq 
    for g in graph_seq:
        detect_graph_seed(g, seed_prop=seed_prop, seed_number=seed_number)
        
