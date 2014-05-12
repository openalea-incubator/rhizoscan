"""
Conversion between graph representation

The central representation in rhizoscan are `RootGraph` and `RootTree`, with 
underlying `NodeList`, `SegmentList` and `AxeList`. These represent the whole 
graph in a general form. However other representation are used in specific 
algorithm to process *graph of segments*.

This module also contain conversion related to the connectivity (digraph, DAG)
and not just between representation.

 
Graph of segments:
------------------
  Graph that represent the neighbors of root segments. Segments can be 
  considered as vertices (here called *elements*) and any pair of connected 
  segments (i.e. which touch each others) represent an edge: each element of 
  the pair is thus a *neighbor* of the other one.
  Because segments have two sides, it is often necessary to distinguished 
  neighbors from each side (in which case graph are said to be *sided*).
  

  Current representations:

  - segment:
     Term use to indicate a SegmentList object, or its `node` attribute

  - list-of-set (los):
     The graph edges are stored as a list (for all elements) of the set of their
     neighbors. Two types are possibles in order to represent graph of segments:
     
      - unsided: a list-of-sets as originaly described
          los[i] is the set of neighbors of segment `i`
          
      - sided: a list-of-tuple-of-2-sets, where each item of the list stores the
        set of neighbor of each segment side: 
          los[i][k] is the set of neighbors of at side `k` of segment `i`
         
  - neighbor:
     Very similar to `list-of-set`, but stored in a numpy array. In order to 
     have a filled array, 0 are appended to neighbor lists.
     
     - unsided: array with shape `(S,N)`, where `S` the number of segments and 
       `N` is the maximum number of neighbors:
         nbor[i,n] is `n`th neighbor of element `i`
         
     - sided: neighbor array have `(S,2,N)` shape, with `S` the number of
       segments, 2 the 2 side of segments and `N` the maximum number of 
       neighbors per side.
         nbor[i,k,n] is `n`th neighbor of at side `k` of element `i`
    
  - edge array:
     Numpy array with shape (E,2) containing the two ids of all `E` edges
  
  - csgraph:
     Representation using scipy sparse matrix which allow the use of scipy 
     sparse graph algorithm (`scipy.sparse.csgraph`). 
     It cannot represent the sided segment graph.
     
"""

import numpy as _np

# representation
# --------------

def segment_to_los(segment_node, node_segment, mask=None):
    """
    Create an sided list-of-set (los) segment graph from a segment type 
    
    :Inputs:
      - segment_node:
          An (S,2) array of the 2 node ids of `S` segments
          I.e. as the `SegmentList.node` attribute 
      - node_segment:
          An array containing the lists of all segments connected to each node.
          I.e. the reverse of `segment_node` as the `NodeList.segment` attribute
      - mask: optional
          If given, it should be a boolean array that flags segments such that
          all nodes that connect only such segments are not treated.
          It is used to manage 'seed' segments of RootGraph.
    
    :Output:
        Return a sided list of sets: a list of a pair of sets. For each "parent"
        segment, it contains a pair (tuple) of the set of connected "children" 
        segments, one for each side the "parent".
        
    :unsided conversion:
        `unsided_los = [nbor[0].union(nbor[1]) for nbor in sided_los]`
        
    :See also: `los_to_neighbor`, `segment_to_neighbor`
    """
    ns   = node_segment.copy()
    if mask is not None:
        invalid_nodes = _np.vectorize(lambda nslist: (mask[nslist]>0).all())(node_segment)
        ns[invalid_nodes] = set()
    ns[0] = set()
    
    # construct nb1 & nb2 the neighbor array of all segments in direction 1 & 2
    nsbor = _np.vectorize(set)(ns)
    snbor = [(s1.difference([i]),s2.difference([i])) for i,(s1,s2) in enumerate(nsbor[segment_node])]

    return snbor

def los_to_neighbor(los, sided):
    """
    Convert a list-of-set (los) type of segment graph to a neighbor array type
    
    :Inputs:
      - los:
          a list-of-set type of segment graph
      - sided:
          Shoulbe True if given `los` is a sided type of graph.
          Or False otherwise.
          
    :Output:
      Return a neighbor type of graph, sided or not depending on arguments.
    
    :See also: `segment_to_los`, `segment_to_neighbor`
    """
    if sided:
        edge_max = max(map(lambda nb:max(len(nb[0]),len(nb[1])),los))
        nbor = _np.zeros((len(los),edge_max,2), dtype='uint32')
        for i,(nb1,nb2) in enumerate(los):
            nbor[i,:len(nb1),0] = list(nb1)
            nbor[i,:len(nb2),1] = list(nb2)
    else:
        edge_max = max(map(len,los))
        nbor = _np.zeros((len(los),edge_max), dtype='uint32')
        for i,nb in enumerate(los):
            nbor[i,:len(nb)] = list(nb)
            
    return nbor
    

def segment_to_neighbor(segment_node, node_segment, mask=None):
    """
    Convert a segment type to a neighbor array type of segment graph
    
    short for::
    
       los  = segment_to_los(....)
       nbor = los_to_neighbor(los, sided=True)
    
    Return an array of shape (S,N,2) with S the number of ("parent") segments, 
    N the maximum number for "children" neighbors per "parent" segment and 2 
    for the 2 "parent" sides. Each neighbors[i,:,k] contains the list of the
    neighboring segments ids on side `k` of segment `i`.
    In order to fill the array, the missing neighbors are set to 0.
    """
    los  = segment_to_los(segment_node=segment_node, node_segment=node_segment, mask=mask)
    return los_to_neighbor(los, sided=True)


def neighbor_to_csgraph(neighbors, value=1, dummy=0, matrix='csr_matrix'):
    """ make a csgraph segment graph type from a `neighbors` type
    
    :Input:
      - neighbors:
          A Nx[K] array of neighbor indices for N elements (K can be any shape)
      - value:
          Values of edges. Either 
            - a scalar: all edges have this value
            - an array of the same shape as `neighbors`
      - dummy:
          index of "dummy" element: no edge is create to/from it
      - matrix:
          The type of sparse matrix to use (from scipy.sparse)

    :Outputs:
        A sparse adjacency matrix with entry for all neighbors 

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
        
    # remove edges to/from dummy elements
    mask  = (ij!=dummy).all(axis=0)
    ij    = ij[:,mask]
    value = value[mask]
        
    return matrix((value,ij), shape=(nbor.shape[0],)*2)

def neighbor_to_edge_array(neighbors, unique=True, directed=False):
    """
    Create an array of segment id pairs
    
    :Inputs:
      - neighbors:
          A Sx[K] (with possible multidimensional K) array of the indices of the
          neighbors of each segment s in S. Such as returned by 
          `segment.neighbors()` or subpart like `segment.neighbors()[...,0]`
      - unique: 
          if True, sort and remove all repeated and invalid edges
      - directed:
          If True, edge pairs are sorted then duplicate detection is done
          
    :Outputs:
      An array with shape (E,2) containing the ids of both segments for all
      neighbor pairs. 0 value entry in `neighbors` are ignored. 
    """
    nbor = neighbors
    
    # get neighbors pairs
    edg   = nbor.reshape(-1,reduce(lambda x,y: x*y,nbor.shape[1:]))
    s1,s2 = edg.nonzero()     # do not keep edge to/from bg segment
    
    # merge segment ids in array `edge`
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

        
# connectivity
# ------------

def segment_to_digraph(segment, direction):
    """
    Create a digraph from `segment` and their given `direction`
    
    :Inputs:
      - segment
          A SegmentList object
      - direction
          A bool vector array with length equal to `segment.number()`.
          True value means the segment direction is reversed.

    :Outputs:
      Return a neighbor type array such that: 
        - neighbors[...,0] are the  incoming neighbors, and
        - neighbors[...,1] are the outcoming neighbors
        
      If `segment` has a 'seed' attribute, no connection is made between them.
    """
    # reverse edge direction
    seg_node = segment.node.copy()
    seg_node[direction] = seg_node[direction][...,::-1]
    node_seg = segment.node_list.segment
    nbor = segment_to_neighbor(seg_node, node_seg, segment.get('seed',None))
        
    # remove edges that are invalid for a directed graph
    # 
    # switch: boolean array with same shape as `nbor` that has True value 
    # where (directed) connection through a neighbors edge requires a change 
    # of one of the segment direction. ie.:
    # 
    # for all edge (i,j) stored in `neighbors`, i.e. j in neighbors[i]: 
    #   - i & j are not in the same relative direction
    #   - i.e. is j a neighbor on side s of i, and i on side s of j ?
    #
    # neighbors that requires switch are invalid in the digraph
    switch = _np.zeros(nbor.shape, dtype=bool)
    sid    = _np.arange(nbor.shape[0])[:,None,None]
    switch[...,0] = (nbor[nbor[...,0],:,0]==sid).any(axis=-1) # side 0
    switch[...,1] = (nbor[nbor[...,1],:,1]==sid).any(axis=-1) # side 1
         
    nbor[switch] = 0
    
    return nbor

def digraph_to_DAG(forward_neighbors, cost, source=None):
    """
    Convert a directed graph to a DAG using a heuristic based on shortest path.
    
    This function first compute the shortest path tree. Then it iteratively add
    edges that were removed by the shortest path but which don't create cycles. 
    This is done following the order of the tree cumulative distance, thus 
    cycles are broken "after" the further away element.

    :Input:
        - forward_neighbors
            an NxK array of the neighbor indices of the K **forward** neighbors
            of N elements. 0 value neighbor not treated (dummy).
        - cost
            Array of (broadcastably) same shape as `neighbors` of their cost
             - All "real" neighbors cost must have **strictly** positive cost -
        - source
            boolean mask of the sources used by the shortest path algorithm
                ex1:   rgraph.segment.seed>0   ;   
                ex2:  -rgraph.segment.neighbors()[...,0].any(axis=1)
            if None, use the list of elements with no incomming edge.
            
    :Output:
        - a list of set (los) type of segment graph where
            los[i][io] is the set of incomming (`io=0`) or outgoing (`io=1`) 
            neighbors of segment `i`
        - list of removed edges (as pair-tuples)
        - the order of the graph traversal used to break cycles 
          (missing ids were unreachable, see note)
          
    :Note:
        Because it is based on a shortest path, all unreachable elements from 
        the given sources are not processed.
    
    :Warning:
      Behavior is not clear if their is an edge beween the same element (e,e)
      that is, a self looping root segment (does not append in practice)
      
    :todo:
      Add unbreakable edges, from initial axe path?
      output might not be a DAG anymore...
    """
    from scipy.sparse.csgraph import dijkstra, reconstruct_path
    
    nbor = forward_neighbors
    
    if source is None:
        source = _np.bincount(nbor.ravel(),minlength=nbor.shape[0])==0
        source[0] = True
    if _np.asarray(source).dtype=='bool':
        source = source.nonzero()[0]
    
    # compute shortest path tree
    graph = neighbor_to_csgraph(nbor, value=cost)
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
    
    # make a list-of-sets graph type, with shortest path tree and 'added' edges
    los = [(set(),set()) for node in xrange(tree.shape[0])]
    src,dst = tree.nonzero()
    for si,di in zip(src,dst):
        los[di][0].add(si)
        los[si][1].add(di)
    for si,di in added:
        los[di][0].add(si)
        los[si][1].add(di)
            
    return los, removed, dist_order
    


# end
