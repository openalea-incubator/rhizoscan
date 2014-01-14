"""
Module to track root archiecture sequences

todo:
 - segment(t+1) to axe(t) distance
    - distance from node(t+1) to node(t): cdist
    - distance from node(t+1) to root segment(t) - listed by axe
    - distance from node(t+1) to root axes(t) - min d(n,s) per axes
    - input: axialtree_t1, graph-t2
    - output:  distance matrix, ???
 
 
 - iterative shortest path to find all axes (lower order 1st)
    - start at parent axe (init with leaves)
    - distance sum should be (an approximation of) a suitable curve distance
    - select "best path", max-cover/min-distance
    
 - root "growth":
    - use (& arrange) the graph-2-axial method?
"""
import numpy as _np

def node_to_axe_distance(graph,tree):
    """
    Compute the distance of all nodes in `graph` to all axes in `tree`
    
    :Inputs:
      - `graph`:
          a RootGraph object, containing a NodeList `node` attribute
      - `tree`:
          a RootAxialTreeGraph object, containing a NodeList `node`,
          a SegmentList `segment` and an AxeList `axe` attributes
          
    :Outputs:
      - The distance array with shape |gn|x|ta| where |gn| is the number of 
        `graph` nodes and |ta| the number of `tree` axe
      - An array of the corresponding segment ids to which the nodes are closest
        to. Its shape is |gn|x|ta|
      - node projection on the axe. Its shape is [k]x|gn|x|ta| for k coordinates 
        The norm |node-node_projection| is equal to the returned distance, 
        but for empty axes (usually axe 0)
        
    :Example:
    
      >>> from matplotlib import pyplot as plt
      >>>
      >>> d,s,p = node_to_axe_distance(g,t)
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
    norm = lambda x: (x**2).sum(axis=0)**.5
    _NA_ = _np.newaxis
    
    t = tree
    g = graph                                          
    
    # extract some relevant data from axe segment
    # -------------------------------------------
    #   n1,n2: nodes 1&2 of all tree segments
    #   sdir:  unit direction vector from n1 to n2
    ts_list = []                        # (flat) list of segments of all tree axes  
    map(ts_list.extend,t.axe.segment)
    tsn   = tree.segment.node[ts_list]  # node ids of tree segment (|ts|,node12)
    pos   = tree.node.position[:,tsn]   # coordinates of ts nodes  (xy,|ts|,node12)
    n1    = pos[:,:,0]                  # position of ts 1st node  (xy,|ts|)
    n2    = pos[:,:,1]                  # position of ts 2nd node  (xy,|ts|)
    sdir  = n2-n1                       # vector from n1 to n2     (xy,|ts|)
    lsl   = norm(sdir)                  # distance between n1 and n2
    lsl   = _np.maximum(lsl,2**-5)      
    sdir /= lsl                         # make sdir unit vectors
    
    
    # node-to-segment distance matrix
    # -------------------------------
    gnpos = g.node.position         # graph node coordinates (xy,|gn|)
    
    # gnode projection on segment
    #    i.e. distance from segment n1 to the projection of gnode on segment 
    #    disallow projection out of segment: values are in [0,lsl]
    on_edge = ((gnpos[:,:,_NA_]-n1[:,_NA_,:])*sdir[:,_NA_,:]).sum(axis=0) # (|gn|,|ts|) 
    on_edge = _np.minimum(_np.maximum(on_edge,0),lsl[_NA_,:])
            
    # distance from node to "node projected on sdir"
    nproj = n1[:,_NA_,:] + on_edge[_NA_,:,:]*sdir[:,_NA_,:]   # (xy,|gn|,|ts|)
    d_ns = norm(nproj - gnpos[:,:,_NA_])                      # (|gn|,|ts|)


    # node-to-axe distance: i.e. min distance to all axe segments
    # -----------------------------------------------------------
    # index of last segment of all tree axe w.r.t the 2nd axis of d_ns 
    ta_end = _np.vectorize(len)(t.axe.segment)
    ta_end = _np.cumsum(ta_end)
                                                                 
    # for all axes, find the segment with minimum distance
    Nn = d_ns.shape[0]
    Na = len(ta_end)
    s_id = _np.empty(  (Nn,Na),  dtype=int)
    d_na = _np.empty(  (Nn,Na),  dtype=d_ns.dtype)
    nprj = _np.empty((2,Nn,Na),dtype=nproj.dtype)
    start = 0
    I = _np.arange(Nn)
    for i,end in enumerate(ta_end):
        if start==end:
            # empty axe (no segment)
            s_id[:,i] = 0
            d_na[:,i] = 0
            nprj[:,i] = 0
        else:
            s_id[  :,i] = d_ns[:,start:end].argmin(axis=1)
            d_na[  :,i] = d_ns [I,start+s_id[:,i]]
            nprj[:,:,i] = nproj[:,I,start+s_id[:,i]]

        start = end
    
    return d_na, s_id, nprj


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
          It should have |gn|x[A]x[k] shape, where |gn| is the number of nodes 
          in `graph`, [A] is the number of object the distance is computed w.r.t
          (aka axes) and [k] the number of coordinates.
          
      This function is designed to be applied following `node_to_axe_distance`, 
      where `node_projection` is its 3rd output. 
    
    :Outputs:
      The area of `graph` segment to their projection, as an array of shape
      |gs|x|a|, where |gs| is the number of segments and |a| of projections
    """
    # For detail on the algorithm, see rhizoscan-technical.pdf, section:
    #  "Computing distance from RootGraph to RootAxialTree
    
    # var used in comments on array shape
    #    xy:  number of coordinates
    #   |gs|: number of graph segments
    #    n12: 2, the number of nodes per segment
    #   |a|:  number of axe, i.e. number of node projection in node_projection
    
    # base variables adn functions
    g = graph
    n = g.node.position[:,g.segment.node]    # node coord, (xy,|gs|,n12)
    p = node_projection[:,g.segment.node]    # node proj,  (xy,|gs|,n12,|a|)    

    _NA_ = _np.newaxis
    
    cross = lambda a,b: _np.abs(a[0]*b[1] - a[1]*b[0])

    # Structuring vectors
    # -------------------
    #     dn:  vector from segment nodes n1 & n2
    #     np1: vector from n1 to its projection (p1)
    #     np2: vector from n2 to its projection (p2)
    #     dp:  vector from p1 to p2
    dn  = _np.diff(n,axis=-1)[...,0]    # shape (xy,|gs|)
    np  = p-n[:,:,:,_NA_]               # shape (xy,|gs|,n12,|a|)
    np1 = np[:,:,0]                     # shape (xy,|gs|,|a|)
    np2 = np[:,:,1]                     # shape (xy,|gs|,|a|)
    dp  = _np.diff(p,axis=2)[:,:,0]     # shape (xy,|gs|,|a|)
    
    
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
    

def track_root(dseq, update=False):
    """
    TESTING / IN DEV
    
    track root in dataset sequence `dseq`
    
    each item in `dseq` should have graph & tree computed
    
    use rhizoscan.opencv stuff to estimate affine transform between image
    """
    d0 = dseq[0].load()
    d1 = dseq[1].load()
    
    # image tracking
    if not d1.has_key('T'):
        im_desc0 = 
        
    