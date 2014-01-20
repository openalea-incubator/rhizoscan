"""
Module to track root archiecture sequences

Tracking steps:
 - node|segment(t+1) to axe(t) distance
    - distance from node(t+1) to node(t): cdist
    - distance from node(t+1) to root segment(t) - listed by axe
    - distance from node(t+1) to root axes(t) - min d(n,s) per axes
    - input: axialtree_t1, graph-t2
    - output:  distance matrix, ???
 
 - todo: match plant seeds
    - distance from seeds centers
    - match closest 1-to-1
    
 - todo: init first axial tree
 
 - todo: iterative shortest path to find all axes (lower order 1st)
    - start at parent axe (init with leaves)
    - distance sum should be (an approximation of) a suitable curve distance
    - select "best path", max-cover/min-distance
    
 - todo: root "growth":
    - use (& arrange) the graph-2-axial method?
"""
import numpy as _np
from scipy import ndimage as _nd

from rhizoscan.workflow import node  as _node
from rhizoscan.image    import Image as _Image
from rhizoscan.opencv   import descriptors as _descriptors
from rhizoscan.geometry import transform   as _transform

_NA_ = _np.newaxis


def track_root(dseq, update=False, verbose=True, plot=False):
    """
    TESTING / IN DEV
    
    track root in dataset sequence `dseq`
    
    each item in `dseq` should have attributes graph,tree,key_point,descriptor
    
    use rhizoscan.opencv stuff to estimate affine transform between image
    """
    d0 = dseq[0].load()
    d1 = dseq[1].load()
    
    # image tracking
    if not update and d1.has_key('image_transform'):
        T = d1.image_transform
    else:
        kp0   = d0.key_point
        desc0 = d0.descriptor
        kp1   = d1.key_point
        desc1 = d1.descriptor
        
        T = _descriptors.affine_match(kp0,desc0, kp1,desc1, verbose=verbose)
        d1.image_transform = T

    t = d0.tree
    g = d1.graph

    # make a copy of g with transformed node postion 
    gnpos = _transform(T=T, coordinates=g.node.position)
    g_input = g
    g = g.copy()
    g.node = g.node.copy()
    g.node.position = gnpos

    # compute g node to t segment distance
    d,s,p = node_to_axe_distance(g.node.position, t)

    if plot:
        from matplotlib import pyplot as plt
        g.plot(bg='k',sc=(g.segment.seed>0)+1)
        t.plot(bg=None)
        n = gnpos
        I = _np.arange(p.shape[1])  # list of nodes index
        k = _np.argmin(d,axis=1)    # best axe match for each node
        plt.plot([n[0,:],p[0,I,k]],[n[1,:],p[1,I,k]], 'b')
    
    
    # match seed
    seed_match, unmatch_t, unmatch_g = match_seed(t,g)
    
    # get ids of first nodes of t 1st order axe 
    
    return t,g,seed_match
    # match axe 0
    # match axe i>0
    
def match_seed(g1,g2):
    """
    Match seeds of graphs `g1` and `g2`
    
    The matching is simply matching the closest seed mean position
    """
    from rhizoscan.root.comparison import direct_matching
    
    pid1, x1,y1 = mean_seed_position(g1)
    pid2, x2,y2 = mean_seed_position(g2)
    
    # compute distance matrix
    x1 = x1.reshape(-1,1)
    y1 = y1.reshape(-1,1)
    x2 = x2.reshape(1,-1)
    y2 = y2.reshape(1,-1)
    d  = ((x1-x2)**2 + (y1-y2)**2) #**.5 not necessary for matching
    
    match,unmatch1,unmatch2 = direct_matching(d)
    
    return match, unmatch1,unmatch2
    

def mean_seed_position(g):
    """
    outputs: plant-id, seeds mean x, seed mean y
    """
    mask  = g.segment.seed>0
    nseed = g.segment.node[mask]
    lseed = g.segment.seed[mask]
    mask  = nseed.all(axis=1)  # remove bg segment
    nseed = nseed[mask]
    lseed = lseed[mask]
    
    pid = _np.unique(lseed)
    x = _nd.mean(g.node.x[nseed],labels=lseed.reshape(-1,1),index=pid)
    y = _nd.mean(g.node.y[nseed],labels=lseed.reshape(-1,1),index=pid)
    
    return pid,x,y
    

@_node('key_point','descriptor')
def detect_sift(image, verbose=True):
    kp, desc = _descriptors.detect_sift(image)
    
    if desc.max()<256:
        desc = _Image(desc)
        desc.set_serializer(pil_format='PNG',ser_dtype='uint8',ser_scale=1,extension='.png')
    elif verbose:
        print '  descriptors cannot be serialized into png'
        
    return kp, desc


def node_to_axe_distance(nodes,tree):
    """
    Compute the distance of all nodes in `graph` to all axes in `tree`
    
    :Inputs:
      - `nodes`:
          [k]x[n] array of `n` nodes in `k` dimensional coordinates such as the 
          the  `position` attribute of NodeList (i.e. RootGraph.node)
      - `tree`:
          a RootAxialTreeGraph object, containing a NodeList `node`,
          a SegmentList `segment` and an AxeList `axe` attributes
          
    :Outputs:
      - The distance array with shape (n,|ta|) where |ta| is the axe number 
      - An array of the corresponding segment ids to which the nodes are closest
        to. Its shape is (n,|ta|)
      - node projection on the axe. Its shape is (k,n,|ta|) for k coordinates 
        The norm |nodes-node_projection| is equal to the returned distance, 
        but for empty axes (usually axe 0)
        
    :Example:
    
      >>> from matplotlib import pyplot as plt
      >>>
      >>> # let g be a RootGraph object and t a RootAxialTree
      >>>
      >>> d,s,p = node_to_axe_distance(g.node.position,t)
      >>> 
      >>> t.plot(bg='k')
      >>> g.plot(bg=None)
      >>> 
      >>> k = 5                        # k: id of axe to plot distance from
      >>> n = g.node.position
      >>> plt.plot([n[0,:],p[0,:,k]],[n[1,:],p[1,:,k]])
      >>>
      >>> lnpl = ((p-g.node.position[:,:,None])**2).sum(axis=0)**.5
      >>> (lnpl==d)[:,1:].all()
      >>> # True 
    """
    from scipy.spatial.distance import cdist
    from scipy.ndimage import minimum as label_min
    
    t = tree
    
    # node-to-segment distance
    # ------------------------
    ts_list = []                      # (flat) list of segments of all tree axes  
    map(ts_list.extend,t.axe.segment)
    tsn   = t.segment.node[ts_list]   # node ids of tree segment (|ts|,node12)
    d_ns, nproj = node_to_segment_distance(nodes,t.node.position[:,tsn])


    # node-to-axe distance
    # --------------------
    #    find min distance of nodes to all axe segments
    # index of last segment of all tree axe w.r.t the 2nd axis of d_ns 
    ta_end = _np.vectorize(len)(t.axe.segment)
    ta_end = _np.cumsum(ta_end)
                                                                 
    # for all axes, find the segment with minimum distance
    Nn = d_ns.shape[0]
    Na = len(ta_end)
    s_id = _np.empty(  (Nn,Na), dtype=int)
    d_na = _np.empty(  (Nn,Na), dtype=d_ns.dtype)
    nprj = _np.empty((2,Nn,Na), dtype=nproj.dtype)
    start = 0
    I = _np.arange(Nn)
    for i,end in enumerate(ta_end):
        if start==end:
            # empty axe (no segment)
            s_id[  :,i] = 0
            d_na[  :,i] = _np.inf
            nprj[:,:,i] = 0
        else:
            s_id[  :,i] = d_ns[:,start:end].argmin(axis=1)
            d_na[  :,i] = d_ns [I,start+s_id[:,i]]
            nprj[:,:,i] = nproj[:,I,start+s_id[:,i]]

        start = end
    
    return d_na, s_id, nprj

def node_to_segment_distance(nodes,segments):
    """
    Compute the minimum distance from all `nodes` to all `segments`
    
    `nodes`: 
       a (k,n) array for the k-dimensional coordinates of n nodes
    `segments`:
       a (k,s,2) array for the k-dim coordinates of the 2 nodes of s segments
       
    :Outputs: 
       - a (n,s) array of the node-segment distances
       - a (k,n,s) array of the coordinates of the closest point to all nodes on 
         all segments.
    """
    norm = lambda x: (x**2).sum(axis=0)**.5
    
    n1    = segments[...,0]           # 1st segment node, shape (k,s)
    n2    = segments[...,1]           # 2nd segment node, shape (k,s)
    sdir  = n2-n1                     # direction vector of segment
    lsl   = norm(sdir)                # distance between n1 and n2
    lsl   = _np.maximum(lsl,2**-5)    
    sdir /= lsl                       # make sdir unit vectors
    
    # distance from n1 to the projection of nodes on segments
    #    disallow projection out of segment: values are in [0,lsl]
    on_edge = ((nodes[:,:,_NA_]-n1[:,_NA_,:])*sdir[:,_NA_,:]).sum(axis=0) # (n,s) 
    on_edge = _np.minimum(_np.maximum(on_edge,0),lsl[_NA_,:])
            
    # distance from node to "node projected on sdir"
    nproj = n1[:,_NA_,:] + on_edge[_NA_,:,:]*sdir[:,_NA_,:]   # (k,n,s)
    d = norm(nproj - nodes[:,:,_NA_])                         # (n,s)

    return d, nproj

def segment_to_projection_area(graph, node_projection):
    """
    Approximate the area between `graph` segment to their projection
    
    The return value is the area between the `graph` segments and the segment
    constructed from both their `node_projection`
    
    *** Note: for now, the resulting quads are expected not to cross edges ***
    
    :Inputs:
      - `graph`:
          a RootGraph object, containing a NodeList `node` and SegmentList 
          `segment` attributes
          
      - `node_projection`:
          Array of the projection of node from which the distance is computed
          It should have (|gn|,A,k) shape, where |gn| is the number of nodes 
          in `graph`, A is the number of object the distance is computed w.r.t
          (aka axes) and k the number of coordinates.
          
      This function is designed to be applied following `node_to_axe_distance`, 
      where `node_projection` is its 3rd output. 
    
    :Outputs:
      The area of `graph` segment to their projection, as an array of shape
      (|gs|,|a|), where |gs| is the number of segments and |a| of projections
    """
    # For detail on the algorithm, see rhizoscan-technical.pdf, section:
    #  "Computing distance from RootGraph to RootAxialTree
    
    # var used in comments on array shape
    #    k:   number of coordinates
    #   |gs|: number of graph segments
    #    n12: 2, the number of nodes per segment
    #   |a|:  number of axe, i.e. number of node projection in node_projection
    
    # base variables adn functions
    g = graph
    n = g.node.position[:,g.segment.node]    # node coord, (k,|gs|,n12)
    p = node_projection[:,g.segment.node]    # node proj,  (k,|gs|,n12,|a|)    

    cross = lambda a,b: _np.abs(a[0]*b[1] - a[1]*b[0])

    # Structuring vectors
    # -------------------
    #     dn:  vector from segment nodes n1 & n2
    #     np1: vector from n1 to its projection (p1)
    #     np2: vector from n2 to its projection (p2)
    #     dp:  vector from p1 to p2
    dn  = _np.diff(n,axis=-1)[...,0]    # shape (k,|gs|)
    np  = p-n[:,:,:,_NA_]               # shape (k,|gs|,n12,|a|)
    np1 = np[:,:,0]                     # shape (k,|gs|,|a|)
    np2 = np[:,:,1]                     # shape (k,|gs|,|a|)
    dp  = _np.diff(p,axis=2)[:,:,0]     # shape (k,|gs|,|a|)
    
    
    # process the "normal case": convex and concave in n2
    #    compute (|dn x np1| + |-dp x np2|)/2
    area = (cross(dn[:,:,_NA_],np1)+cross(-dp,np2))/2
    
    ## todo: crossed quad case
    #    find crossing that split dn in dn1 and dn2, and dp in dp1, and dp2
    #    compute (|dn1 x np1| + |dn2 x np2|)/2
    ## todo: uncrossed case withconcavity in n1
    #    find n1-concave case
    #    compute (|-dp x np2| - |dn x np1|)/2
    
    return area
    
