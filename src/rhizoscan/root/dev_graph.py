import numpy as _np


def segment_digraph(segment, cost, callback=None):
    """
    Find an optimised direction for all segments
    
    :Inputs:
      - segment:
         A SegmentList instance that contains the attributes `neighbors`, `seed`
         and `terminal`. If cost is a string, `segment` should also contain the 
         suitable attribute (by default `direction-difference`)
      - cost:
         A SxS array of the cost between segment
      - callback:
         A function that is call at each iteration of the algorithm such as::
        
           callback(i, sdir, sgid, s1,s2)
           
         with `i` is the iteration number, `sdir` is t curent segment direction,
         `sgid` is the current group id of the segments and `s1`and `s2` are the
         id of the processed segments.
       
    :Output:
       - A boolean array of the direction of each segment.
       `False` means as given by input `edge`, `True` means switched direction.
       
       - the segment group id
      
    IN DEVELOPMENT
    """
    nbor = segment.neighbors()
    edge = neighbor_to_edge_list(nbor)
    
    # edges cost
    # ----------
    # edge with seed segment have zero cost
    cost = cost[edge[:,0], edge[:,1]]
    seed = segment.seed>0
    cost[seed[edge[:,0]] | seed[edge[:,1]]] = 0
    
    # sort by cost
    edge = edge[_np.argsort(cost),:]
    
    
    # compute the edge direction
    # --------------------------
    # initialization
    sNum  = segment.node.shape[0]
    group = [[i] for i in xrange(sNum)] # segment group (init one group per segment)
    sgid  = _np.arange(sNum)            # group id of segment   ----- " -----
    gset  = _np.zeros(sNum, dtype=bool) #  group  direction selected (init as unset)
    sdir  = _np.zeros(sNum, dtype=bool) # segment direction w.r.t its group (init as 0:same)
    
    # set seed and terminal segment direction
    #     note: seed must be set after terminal as they also are terminal
    term  = segment.terminal()
    sdir[term] = nbor[term][:,:,1].any(axis=1)  # default means no nbor on side 1
    sdir[seed] = nbor[seed][:,:,0].any(axis=1)  # default means no nbor on side 0
    gset[term] = 1
    gset[seed] = 1
    
    # convert to list for faster 1-element access during iterations
    sdir = sdir.tolist()
    gset = gset.tolist()
    
    # switch: 
    #   True if s1 and s2 must have opposite direction to be part of same group
    #   ie. all edges s.t s1 start (resp. end) == s2 start (resp. end)
    switch = (segment.node[edge[:,0]]==segment.node[edge[:,1]]).any(axis=1) 
    
    from itertools import izip
    for i,(s1,s2),sw in izip(xrange(edge.size), edge.tolist(), switch.tolist()):
        g1 = sgid[s1]
        g2 = sgid[s2]
        
        if g1==g2: continue
        
        g1set = gset[g1]
        g2set = gset[g2]
        
        # don't merge if not possible: both group dir set in unfit direction
        if g1set and g2set and ((sdir[s1]==sdir[s2])==sw):
            continue
    
        # reverse s1,s2 s.t. merging should be g2 into g1
        if (g1set<g2set)  or  ((g2<g1) and (g1set<g2set)):
            g2set,g1set = g1set,g2set
            g2,g1 = g1,g2
            s2,s1 = s1,s2
        
        # merge g2 into g1
        g2s = group[g2]
        sgid[g2s] = g1
        ##print '  merge:', g2s, ' in ', group[g1]
        group[g1].extend(g2s)
        
        # switch g2 segments direction if necessary
        if (sdir[s1]==sdir[s2])==sw:
            for s in g2s: sdir[s] = not sdir[s]
            
        if callback: callback(i, sdir, sgid, s1,s2)
        
    sdir = _np.array(sdir)
    
    return sdir, sgid

def digraph_to_DAG(neighbors, cost, source=None):
    """
    Convert a directed graph to a DAG using a heuristic based on shortest path.
    
    This function first compute the shortest path tree. Then it iteratively add
    edges that were removed by the shortest path but which don't create cycles. 
    This is done following the order of the tree cumulative distance, thus 
    cycles are broken "after" the further away element.

    :Input:
        - neighbors
            an NxK array of the neighbor indices of the K **forward** neighbors
            of N elements. 0 value neighbor are "fake".
        - cost
            Array of (broadcastably) same shape as `neighbors` of the edges cost
             - All "real edges" cost must have **strictly** positive cost -
        - source
            indices, or boolean mask, of the sources for the shortest path
                ex1:   rgraph.segment.seed>0   ;   
                ex2:  -rgraph.segment.neighbors()[...,0].any(axis=1)
            if None, use the list of elements with no incomming edge.
            
    :Output:
        - a list of incomming segment edges (as sets)
        - a list of outgoing  segment edges (as sets)
        - list of removed edges (as pair-tuples)
        - the order of the graph traversal used to break cycles 
          (missing ids were unreachable, see note)
          
    :Note:
        Because it is based on a shortest path, all unreachable elements from 
        the given sources are not processed.
    
    :todo:
      check the case where their is an edge beween the same elements (e,e)
      that is, a self looping root segment (does not append in practice)
    """
    from scipy.sparse.csgraph import dijkstra, reconstruct_path
    
    if source is None:
        source = _np.bincount(neighbors.ravel(),minlength=neighbors.shape[0])==0
        source[0] = True
    if _np.asarray(source).dtype=='bool':
        source = source.nonzero()[0]
    
    # compute shortest path tree
    #   GRAPH_CVT: nbor-type to csgraph
    graph = neighbor_to_csgraph(neighbors, value=cost, omit_bg=1)
    d,parent = dijkstra(graph, indices=source, return_predecessors=1)
    path_id  = d.argmin(axis=0)
    parent   = parent[path_id,_np.arange(d.shape[1])]
    tree  = reconstruct_path(graph,parent)
    
    # get all edges that have been removed from the digraph
    rm_edge = [[] for node in xrange(tree.shape[0])]
    src,dst = (graph-tree).nonzero()                        ## required cost >0
    for si,di in zip(src.tolist(),dst.tolist()):  
        rm_edge[si].append(di)
        
    # get a topological order of the **tree**...
    #    s.t. parent[node_i] always appears before node_i
    d = d[path_id,_np.arange(d.shape[1])]
    dist_order = _np.argsort(d)
    if d.max()==_np.inf:
        dist_order = dist_order[:d[dist_order].argmax()]
    
    # The idea is to add iteratively ancestors following topological order: 
    #   when processing an element, all its parents have already been processed
    # During this iteration, the algorithm adds removed edges when its starting 
    # node has been processed. However it adds it only if this does not create 
    # a cycle: if the second node is not an anscestor of the 1st
    added = []
    removed = []
    ancestors = [set() for node in xrange(tree.shape[0])]
    for e in dist_order:
        # add parent and its ancestors to e ancestors
        p = parent[e]
        if p>=0: 
            ancestors[e].update(ancestors[p])
            ancestors[e].add(p)
        
        # try to add removed edges starting at e
        for c in rm_edge[e]:
            if c not in ancestors[e]:
                ancestors[c].update(ancestors[e])
                ancestors[c].add(e)
                added.append((e,c))
            else:
                removed.append((e,c))
    
    # convert tree to list-of-sets graph type, with 'added' edges added
    #   GRAPH_CVT: csgraph (+) to list-of-set
    incomming = [set() for node in xrange(tree.shape[0])]
    out_going = [set() for node in xrange(tree.shape[0])]
    src,dst = tree.nonzero()
    for si,di in zip(src,dst):
        incomming[di].add(si)
        out_going[si].add(di)
    for si,di in added:
        incomming[di].add(si)
        out_going[si].add(di)
            
    return incomming, out_going, removed, dist_order
    
def topsort(incomming, out_going, source=None, fifo=True):
    """
    Compute the topological order of a DAG
    
    :Inputs:
      - incomming:
            The incomming segment edges in list-of-edge representation 
      - out_going:
            The outgoing segment edges in list-of-edge representation 
      - source:
            Optional list (or mask) of starting elements.
            if None, use the list of elements with no incomming edge.
            *** If given, no path from one source to another should exist ***
      - fifo:
            If True, use a first-in-first-out queue data structure. Otherwise 
            use a last-in-first-out (LIFO) queue. 
            In most cases, topsort is not unique. Using a fifo makes the topsort
            follow some kind of breath-first-order ("old" elements first), while
            a lifo makes it follow some kind of depth-first-order.
            
    :Output:
        The DAG topological order as a list of element ids.
        If given `source` does not allow to reach all elements, then the 
        unreachable one are not included in the returned list.
    
    (*) A. B. Kahn. 1962. Topological sorting of large networks. Commun. ACM 5
    """
    from collections import deque
    
    incomming = [i.copy() for i in incomming] # copy
    parents = [list(i) for i in incomming]   # copy, as convertion set to list
    
    # list of elements ready to be processed
    source  = _init_source(source, lambda: (_np.vectorize(len)(incomming)==0).nonzero())
        
    current = deque(source)
    if fifo: current_append = current.appendleft 
    else:    current_append = current.append 
    
    order   = []        # stores ordered element ids 
    while current:
        e = current.pop()
        order.append(e)
        
        for n in out_going[e]:
            incomming[n].remove(e)
            if len(incomming[n])==0:
                current_append(n)
    
    return order
    
def topsort_node(incomming, out_going, source=None):
    """
    Compute the DAG topological order and return segment order and node list
    
    :Inputs:
      - incomming:
            The incomming segment edges in list-of-edge representation 
      - out_going:
            The outgoing segment edges in list-of-edge representation 
      - source:
            Optional list (or mask) of starting elements.
            if None, use the list of elements with no incomming edge.
            *** If given, sources should not have incomming edges ***
            
    :Output:
      - The order of segments, such as returned by `topsort()`
      - A list in the topological order of the "nodes" which are represented by 
        a pair `(parent, children)`, both being sets of respectively the
        incomming and outgoing segments id.
        So a list of tuples of sets.
        
    :WARNING:
        The following should be true: If two elements share a parent 
        (i.e. have an incomming edge to the same elements), then they have 
        the same set of parents. That is: parent and siblings are all 
        connected through the same (unspecified) node
        
        This means that the directed graph obtained by specifying the direction
        of each segment is a DAG: no cycle need to be broken.
    """
    from collections import deque
    
    incomming = [i.copy() for i in incomming] # copy
    parents = [list(i) for i in incomming]   # copy & convertion set-to-list
    
    # list of elements ready to be processed
    source  = _init_source(source, lambda: (_np.vectorize(len)(incomming)==0).nonzero())
    
    current = set(source)     # element to be processed
    order   = []              # stores ordered node pairs 
    
    def processed(c):
        for child in c:
            current.discard(child)
            if child==101 or 101 in out_going[child]:
                print c, child, out_going[child], [incomming[gc] for gc in out_going[child]]
            for grand_child in out_going[child]:
                incomming[grand_child].remove(child)
                if len(incomming[grand_child])==0:
                    current.add(grand_child)
        
    while current:
        e = current.pop()
        p = parents[e]
        if len(p): c = out_going[p[0]]
        else:      c = set([e])
        order.append((set(p),c))
        if 101 in c:
            print '  *', c,p, len(current)
        processed(c)
    
    return order

def minimum_dag_branching(incomming, cost, invalid=0):
    """
    Compute the minimum branching on the given DAG
    
    The minimum branching is the equivalent of the minimum spanning tree but for 
    digraph. For DAG, which contains no cycle, the algorithm is trivial:
      For all elements, choose its `incomming` element with minimal cost
      
    See the Chu-Liu/Edmonds' algorithm:
      `http://en.wikipedia.org/wiki/Edmonds'_algorithm`
    
    :Inputs:
      - incomming:
          The incomming edges of either a list-of-set type or of neighbor type
      - cost:
          A SxS array of the cost between any pair of segments : S = len(incomming)
      - invalid:
          Value to return for element without parent
          Must be the id of an element: 0<=invalid<=len(incomming)

    :Outputs:
      The selected parent of each graph segment, as a list.
    """
    if isinstance(incomming, list):
        parent_nbor = list_to_neighbor(incomming, invalid=invalid)
    else:
        parent_nbor = incomming
        
    x = _np.arange(parent_nbor.shape[0])
    y = cost[x[:,None],parent_nbor].argmin(axis=1)
    
    parent = parent_nbor[x,y]
    #parent[(parent_nbor==invalid).all(axis=1)] = invalid
    
    return parent
   
def OLD_merge_tree_path(incomming, out_going, top_order, path, spath, priority):
    """
    merge path when possible...
    """
    # dictionary of (tip_element:path_id) of all path
    path_tip = dict([(p[-1] if len(p) else 0,pid) for pid,p in enumerate(path)])
    priority = _np.asarray(priority)
    
    path  = path[:]  # copy
    spath = [set(sp) for sp in spath]  # copy, and cvt to list of sets
    for e in top_order[::-1]:
        if e not in path_tip or len(out_going[e])==0: continue
        
        # find path p1, p2 to merge 
        #   p2ind is the merging position in path p2 
        p1 = path_tip[e]
        p2 = []
        p2ind = []
        ##print e, out_going[e], p1, [spath[e_out] for e_out in out_going[e]]
        for e_out in out_going[e]:
            # find potential path to merge passing through each outgoing element 
            p2sub = list(spath[e_out])
            #if e==1789:
            #    return path
            #    print e_out, [path[p2i] for p2i in p2sub]
            p2sub = [p2sub[i] for i in priority[p2sub].argsort()][:-1]
            p2.append(p2sub)
            p2ind.append([path[p2i].index(e_out) for p2i in p2sub])
            
        p2    = reduce(lambda l1,l2:l1+l2, p2)
        p2ind = reduce(lambda l1,l2:l1+l2, p2ind)
        if len(p2):
            # if there is at least one "free" path
            p2id = priority[p2].argmin()
            p2 = p2[p2id]
            p2ind = p2ind[p2id]
                
            # merge path p1 + end of p2
            for s in path[p2][:p2ind]:
                spath[s].remove(p2)
            path[p1] = path[p1] + path[p2][p2ind:]
            path[p2] = None
            ##pprint '  >', p1, p2, 'merged'
        else:
            pass
            ##print '  no merge available:', e, p1
        
    path = [p for p in path if p is not None]
    
    return path

def OLD_dag_covering_path(dag_edge, cost, source):
    """
    First attempt, local matching (at nodes) of path
        > Conclusion: the path "breaks" too often
    
    return `path` and `path_parent` (edge id or -1 if no parent)
    """
    # convert graph to list, for each elements, of in/out edges set
    #   GRAPH_CVT: sided_nbor to edge-list to set-list
    incomming, out_going = digraph_nbor_to_IO_set(dag_edge)
    out_going = [list(o) for o in out_going]
    parents   = [list(i) for i in incomming]
    
    # initialise data structures
    # --------------------------    
    # path storages
    elt_number = dag_edge.shape[0]
    ##elt_path = [[] for i in xrange(elt_number)]  # list of path for all elements
    ##elt_potl = [[] for i in xrange(elt_number)]  # list of potential path, for all elements
    elt_path = [None]*elt_number  # list of path for all elements
    elt_potp = [None]*elt_number  # list of potential path, for all elements
    
    path = []  # list of element for all path
    potp = []  # list of potential elements for all path
    
    # list of elements to be processed
    source  = _init_source(source, lambda: (_np.vectorize(len)(incomming)==0).nonzero())
    
    # local functions
    # ---------------
    # get an arbitrary element from an (unempty) set
    def set_get(s):
        for e in s: break
        return e
        
    # create a new path
    def new_path():
        path.append([])
        potp.append([])
        return len(path)-1
    
    # remove element e from current list, and update e's children    
    def remove_from_current(e):
        current.remove(e)
        for n in out_going[e]:
            incomming[n].remove(e)
            if len(incomming[n])==0:
                current.add(n)
                
    # add element e to path p
    def add_to_path(e,p):
        elt_path[e].append(p)
        path[p].append(e)
        ##? clean potential ?
    
    # return all the parent and siblings of element e
    #    consider that all siblings shares the same parent set
    def familly_members(e):
        children = out_going[parents[e][0]]
        return parents[e], children
    
    # get the cost of path in parent element to children element
    def transfer_cost(parent, children):
        #if len(children)<len(parent):
        #    return match(children,parent)[::-1]
        #
        ## at this point, lchildrenl>=lparentl
        ppath = [elt_path[p] for p in parent]
        return cost[_np.array(ppath).reshape(-1,1),children]
    
    # find "best match" between parent element and children element
    #   matching by attributing the first found match for both parent to child
    #   and child to parent in the increasing order of pair cost
    from rhizoscan.ndarray import unravel_indices as unravel
    def best_match(parent, children):
        value = cost[_np.array(parent).reshape(-1,1),children]
        p2c = [-1]*value.shape[0]
        c2p = [-1]*value.shape[1]
        
        o = value.ravel().argsort()
        pc = unravel(o,value.shape)
        
        # 1 to 1 match
        for p,c in pc:
            if p2c[p]==-1 and c2p[c]==-1:
                p2c[p] = c
                c2p[c] = p
        direct = [(parent[p],children[c]) for p,c in enumerate(p2c) if c<>-1]
            
        # find match for remaining children, if any
        branch = []
        for p,c in pc:
            if c2p[c]==-1: branch.append((parent[p],children[c]))
            
        return direct, branch

    # Create an intial axial tree as a -disjoint- path covering
    # ---------------------------------------------------------
    init_path = [0]*elt_number
    path_parent = [-1]
    
    def new_init_path(element,parent):
        path_parent.append(parent)
        init_path[element] = len(path_parent)
        
    current = set(source)
    # process source elements
    for e in list(current):
        new_init_path(e,-1)
        remove_from_current(e)
        
    while current:
        # get a "working set", all edges related to a (unspecified) node: 
        parent, children = familly_members(set_get(current))
        
        direct, branch = best_match(parent, children)
        print direct
        
        for p,c in direct: init_path[c] = init_path[p]
        for p,c in branch: new_init_path(c,p)
        for c in children: remove_from_current(c)
    
    return _np.array(init_path), _np.array(path_parent)
    
    # process DAG
    # -----------
    # process source elements
    for e in list(current):
        elt_path[e] = [new_path()]
        remove_from_current(e)
    
    # Second pass ... ?
    while current:
        # get a "working set", all edges related to a (unspecified) node: 
        #     find all parents and siblings of an arbitrary current element
        #     assert all children are in current?
        parent, children = familly_members(set_get(current))
        
        # first attempt: minimum spanning tree
        #    for a dag, this means to simply select the best parent for all children
        c = transfer_cost(parent,children)
        [(pi,ci) for ci,pi in enumerate(np.argmin(c,axis=0))]
        
        
        # solve the path transfer on the working set:
        #     find best 1-1 match from parent(s) to children (1)   - how?
        #     if their is the same number of parents and children
        #        transfer path and potential through these matches (2)
        #     if their are more parents
        #        repeate (1) with left parents until all are matched
        #        apply (2) and merge path and potential when multiple parent to 1 child
        #        [Q: if n-to-1, merge, but if n-to-m distribute ??]
        #     if their are more children
        #        distribute path to children (which?)
        #        distribute potential if possible (which?) & solve potential
        #        add new path if necessary
        #     all other path are set to potential to all children
        #    
        #     solve potential:  (3)
        #           find its best ancestors subpath
        #           remove it from all potential lists
        #           add all found ancestor to path
        #    
        # remove all children from current (see remove_from_current)
        break        
    
    return path
    
def _init_source(source, default):
    """
    return source, as integer indices, or default
    If default is a function, evaluate it (without parameter)
    """
    if source is not None:
        source = _np.asarray(source)
        if source.dtype==bool:
            return source.nonzero()[0]
        else:
            return source
    elif isinstance(default,type(lambda:1)):
        return default()
    else:
        return default
    
    
def init_flow(dag_edge, source=None, incomming=None, out_going=None):
    """
    ABANDONNED
    
    Find a valid flow through DAG represented bu `dag_edge`
    :todo: everything...
    """
    from collections import deque
    flow    = [0]*dag_edge.shape[0]
    source  = _init_source(source,None)
    current = deque() 
    
    # convert graph to list, for each elements, of in/out edges set
    #   GRAPH_CVT: sided_nbor to edge-list to set-list
    incomming, out_going = digraph_nbor_to_IO_set(dag_edge)
    
    def initFlowRight(i):
        pass
    
    for i in source:
        initFlowRight(i)
    while current:
        initFlowLeft(current.pop())
        
    return _np.array(flow)

def neighbor_cost(neighbors, cost):
    """
    Construct the cost relative to `neighbors` graph using array `cost`
    
    :Inputs:
      - neighbors:
         A SxN or SxNx2 neighbor-type array
      
      - cost:
         A SxS array of the cost between all possible segment pairs
         
    :Output:
         An array of same shape as edge, with suitable cost value.
         
    :Example:
        Let `s` be a SegmentList object::
        
        nbor_cost = neighbor_cost(s.neighbors(), s.direction_difference()) 
    """
    return cost[_np.arange(neighbors.shape[0]).reshape((-1,) + (1,)*(neighbors.ndim-1)), neighbors]

def set_downward_segment(graph):
    
    """
    In-place set all segments of RootGraph `graph` downward
    return updated `graph`
    
    *** This function resets the `graph.segment.neighbors()` attribute ***
    """
    upward = _np.diff(graph.node.y()[graph.segment.node],axis=1).ravel()<0
    graph.segment.node[upward] = graph.segment.node[upward][:,::-1]
    graph.segment._neighbors = None # clear precomputed neighbor array
    
    return graph

# conversion between graph representation
# ---------------------------------------
def neighbor_to_csgraph(neighbors, value=1, omit_bg=False, matrix='csr_matrix'):
    """ make a sparse adjacency representation from the `neighbors` graph
    
    :Input:
      - neighbors:
          A Nx[K] array of neighbor indices for N elements (K might be any shape)
      - value:   
          Values of edges. Either 
            - a scalar: all edges have this value
            - an array of the same shape as `neighbors`
      - omit_bg:
          If True, do not add edge from/to background element (i.e. 0)
      - matrix:
          The type of sparse matrix to use (from scipy.sparse)
      
    :Note:
        If the same edge appears several times, their value is summed
    """
    from rhizoscan.ndarray import virtual_array
    from scipy import sparse
    
    nbor = neighbors
        
    nbor   = nbor.reshape(nbor.shape[0], -1) # 2D neighbor array
    matrix = getattr(sparse, matrix)         # matrix constructor
    
    # list of indices pairs
    ij = _np.asarray([(_np.arange(0,nbor.shape[0],1./nbor.shape[1]).astype(int)).ravel(),nbor.ravel()])
    
    # manage scalar value
    if _np.asarray(value).size==1:
        value = virtual_array(nbor.size, value=value)
    else:
        value = _np.broadcast_arrays(nbor,value)[1].ravel()
        
    # remove edges to/from background
    if omit_bg:
        mask  = (ij!=0).all(axis=0)
        ij    = ij[:,mask]
        value = value[mask]
        
    return matrix((value,ij), shape=(nbor.shape[0],)*2)

def list_to_neighbor(edges_in, edges_out=None, invalid=0):
    """
    Convert the a list-of-set type of graph to a neighbor type
    
    If edges_out is None, return a SxN neighbor type array, where missing
    neighbors are filled with value given by `invalid`
    If edges_out is not None, return a SxNx2 array, with same filling rule, and 
    where the returned `neighbor[:,:,0] contains the `edges_in` and 
    `neighbor[:,:,1]` contains the `edges_out`
    """
    
    n = max(map(len,edges_in))
    fill = lambda edge,N: [list(e)+[invalid]*(N-len(e)) for e in edge]
    if edges_out:
        n = max(n,max(map(len,edges_in)))
        edges_in  = fill(edges_in, n)
        edges_out = fill(edges_out,n)
        return _np.dstack((edges_in,edges_out))
    else:
        return _np.array(fill(edges_in,n))
    
def digraph_nbor_to_IO_set(neighbors, incomming=None, out_going=None):
    """
    Convert a neighbor-type digraph `neighbors` into a list of sets representation

    if either incomming or out_going is None, init both 
    otherwise, return them
    
    return incomming, out_going
    """
    elt_number = neighbors.shape[0]
    
    if incomming is None or out_goind is None:
        out_going = [set() for i in xrange(elt_number)]
        incomming = [set() for i in xrange(elt_number)]
        for i,o in neighbor_to_edge_list(neighbors, unique=True, directed=True):
            out_going[i].add(o)
            incomming[o].add(i)
    
    return incomming, out_going
        
def neighbor_to_edge_list(neighbors, unique=True, directed=False):
    """
    Create a list of segment id pairs (as a numpy array)
    
    :Inputs:
      - neighbors:
          A Sx[K] (with possible multidimensional K) array of the indices of the
          neighbors of each segment s in S. Such as returned by 
          `segment.neighbors()` or subpart like `segment.neighbors()[...,0]`
      - unique: 
          if True, sort and remove all repeated and invalid edges
      - directed:
          If False, edge pairs are sorted before duplicate detection is done
          If unique is False, this does not do anything
          
    :todo: remove `unique` keyword?
    """
    nbor = neighbors
    
    # get 'best' neighbors (w.r.t sides) as segments pairs
    # ----------------------------------------------------
    edg   = nbor.reshape(-1,reduce(lambda x,y: x*y,nbor.shape[1:]))
    s1,s2 = edg.nonzero()     # do not keep edge to/from bg segment
    
    # merge segment ids in array `edge`
    #   here edge is a list of segment id pairs  (!= to the edge input)
    if s1.base is s2.base: edge = s1.base   ## seems to be the case, but no proof
    else:                  edge = np.concatenate((s1[:,None],s2[:,None]), axis=1)
    edge[:,1] = edg[edge[:,0],edge[:,1]]
    
    # unique segment pairs
    if unique:
        if directed==False:
            edge.sort(axis=1)
        edge = _np.unique(edge.view(dtype=[('s1', edge.dtype),('s2',edge.dtype)])) \
                       .view(dtype=edge.dtype).reshape(-1,2)
    
    return edge

        
# ploting and user interface stuff
# --------------------------------
def plot_segment_edges(segment, neighbors=None, directed=None):
    """ plot segment graph where `neighbors` define connectivity
    
    neighbors:
      The array of neighbor indices. If None, use `segment.neighbors()`
    directed:
      - if None, draw all edges
      - if 0 or 1, draw only incoming or outgoing edges, respectively
    
    
    plot a node at each segment center, and a link for all valid edges
    """
    from matplotlib import collections as mcol
    from matplotlib import pyplot as plt
    
    nid = segment.node
    ind = (segment.node>0).all(axis=1)
    pos = (segment.node_list.position[:,nid[:,0]] + segment.node_list.position[:,nid[:,1]])/2
    
    if neighbors is None:
        neighbors = segment.neighbors()
    
    if directed is None:
        edge = neighbor_to_edge_list(neighbors, unique=False)
    else:
        edge = neighbor_to_edge_list(neighbors[...,directed], unique=False)
    edge = pos.T[edge]
    
    # plot
    lcol = mcol.LineCollection(edge, color='g')
    plt.gca().add_collection(lcol)
    
    edge = edge.copy()
    edge[:,0,:] = (edge[:,0]+4*edge[:,1])/5
    lcol = mcol.LineCollection(edge, color='r', linewidth=2)
    plt.gca().add_collection(lcol)
    
    plt.plot(pos[0,:],pos[1,:],'.w')
    


