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

def segment_to_axe_distance(graph,tree):
    """
    Compute the distance of all segment in `graph` to all axes in `tree`
    """
    from scipy.spatial.distance import cdist
    from scipy.ndimage import minimum as label_min
    norm = lambda x: (x**2).sum(axis=0)**.5
    _AXE = _np.newaxis
    
    t = tree
    g = graph
    
    ### compute distance from graph nodes to tree nodes
    ### ===============================================
    ##gn = g.node.position  # gn: Graph Node coordinates
    ##tn = t.node.position  # tn: Tree  Node coordinates
    ##nd = cdist(gn.T,tn,.T)    # nd: Node Distance

    # (flat) list of segments of all tree axes  
    ts_list = []
    map(ts_list.extend,t.axe.segment)
    
    # flat list of axe ids of all tree segments, following ts_list order
    ts_axid = []
    map(ts_axid.extend,[[i]*len(s_list) for i,s_list in enumerate(t.axe.segment)])
    
    # extract some relevant data from axe segment
    # -------------------------------------------
    #   n1,n2: nodes 1&2 of all tree segments
    #   sdir:  unit direction vector from n1 to n2
    tsn   = tree.segment.node[ts_list]  # node ids of tree segment (|ts|,node12)
    pos   = tree.node.position[:,tsn]   # coordinates of ts nodes  (xy,|ts|,node12)
    n1    = pos[:,:,0]                  # position of ts 1st node  (xy,|ts|)
    n2    = pos[:,:,1]                  # position of ts 2nd node  (xy,|ts|)
    sdir  = n2-n1                       # vector from n1 to n2     (xy,|ts|)
    lsl   = norm(sdir)                  # distance between n1 and n2
    lsl   = _np.maximum(lsl,2**-5)      
    sdir /= lsl                         # make sdir unit vectors
    
    # compute the node-to-segment distance matrix
    # -------------------------------------------
    gnpos = g.node.position         # graph node coordinates (xy,|gn|)
    
    # gnode projection on sdir: distance from n1 to the gn projected on sdir 
    #    disallow projection out of segment (i.e not in [0,lsl])
    on_edge = ((gnpos[:,:,_AXE]-n1[:,_AXE,:])*sdir[:,_AXE,:]).sum(axis=0) # (|gn|,|ts|) 
    on_edge = _np.minimum(_np.maximum(on_edge,0),lsl[_AXE,:])
            
    # distance from node to "node projected on sdir"
    d_ns = norm(n1[:,_AXE,:] + on_edge[_AXE,:,:]*sdir[:,_AXE,:] - gnpos[:,:,_AXE]) # (|gn|,|ts|)

    ### check for node projected out of segment
    ##mask = on_edge<0
    ##d[mask] = norm(n1[:,mask]-p)
    ##mask = on_edge>lsl
    ##d[mask] = norm(n2[:,mask]-p)
    ##return d

    # distance from node to axes: i.e. min distance to all axe segment
    # ----------------------------------------------------------------
    #ts_axid = _np.array(ts_axid)[_AXE,:]
    #d_na = label_min(d_ns,labels=ts_axid, index=_np.arange(len(t.axe.segment)+1))
    
    return d_ns
