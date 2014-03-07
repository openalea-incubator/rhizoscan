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
from rhizoscan.geometry.polygon import distance_to_segment as _distance_to_segment

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

    # make a copy of g with transformed node position 
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
    

def axe_projection(tree, graph, transform):
    """
    *** IN DEVELOPEMENT ***
    
    Find/project axes in `tree` onto `graph`
    
    `tree` is a RootAxialTree
    `graph` is a RootGraph
    `transform` is a affine transformation mapping `graph` frame in `tree` frame
    
    return an AxeList for `graph`
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    graph_axes = {} # stores list of graph segments for each axe id
    unmatch = {}    # store unmatched content
    
    # cost of matching graph segments on tree axes
    # --------------------------------------------
    #  cost is the approximate area between segments and axes
    
    # make a copy of g with transformed node position 
    g = graph.copy()
    g.node = g.node.copy()
    g.node.position = _transform(T=transform, coordinates=graph.node.position)
    g.segment = g.segment.copy()  # copy with transformed _node_list
    g.segment._node_list = node   #    maybe not useful...

    t = tree

    # compute distances from node in g to segment in t
    d,s,p = node_to_axe_distance(g.node.position, t)
    g2t_area = segment_to_projection_area(g, p)        # shape (|gs|,|ta|)
    
    
    # shortest path graph
    # -------------------
    #   compute a sparse matrix representation of connection between g's segments
    #   it is used to compute shortest path for all axe matching
    nbor = g.segment.neighbors
    nbor = nbor.reshape(nbor.shape[0],-1)
    I,J  = nbor.nonzero()
    J    = nbor[I,J]
    sp_graph = csr_matrix((np.ones_like(I),(I,J)))
    
    
    # list of tree axes in priority order
    # --------------------------------------
    #   topological order <=> axe sorted by order <=> subaxe are after their parent
    axe_list = np.argsort(d.tree.axe.order[1:])+1
    ## todo sort by "branching" position, then length
    
    
    # match seed of graph and tree
    # ----------------------------
    seed_match, unmatch_t, unmatch_g = match_seed(g,t)
    seed_match = dict(seed_match)
    unmatch['t_seed'] = unmatch_t
    unmatch['g_seed'] = unmatch_g


    # find/project all t axes into g
    # ==============================
    for axe in axe_list:
        # find possible graph segment to start the projected axe
        # ------------------------------------------------------
        p_axe = t.axe.parent[axe]
        if p_axe==0: # parent is a seed
            starts = (g.segment.seed==t.axe.plant[axe]).nonzero()
        else:
            starts = graph_axes[p_axe]
        
        # compute shortestpath
        # --------------------
        sp_graph.data[:] = g2t_area[J,axe]
        path_cost, predecessor = dijkstra(sp_graph, indices=starts, return_predecessors=True)
        
        # select best path
        path = None
        graph_axe[axe] = path
        
    # contruction AxeList
    graph_axe = None ##
    
    
    
def match_seed(g1,g2):
    """
    Match seeds of graphs `g1` and `g2`
    
    The matching is simply matching the closest seed mean position
    
    :Outputs:
      - the list match pairs [(i0,j0),(i1,j1),...]
        where i,j are the seed ids in `g1` and `g2` respectively 
      - the list of unmatch seed ids in `g1`
      - the list of unmatch seed ids in `g2`
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
    
    ## todo: map (un)match ids from [0,n1/2-1] to pid1/2
    match = [(pid1[s1],pid2[s2]) for s1,s2 in match]
    unmatch1 = [pid1[s1] for s1 in unmatch1]
    unmatch2 = [pid2[s2] for s2 in unmatch2]
    
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
    d_ns, nproj = _distance_to_segment(nodes,t.node.position[:,tsn])[:2]


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
    
