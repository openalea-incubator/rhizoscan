import numpy as _np
import scipy as _sp
from scipy import ndimage as _nd

from rhizoscan.workflow import Struct  as _Struct
from rhizoscan.ndarray  import reshape as _reshape
from rhizoscan.tool     import _property    

class SegmentGraph(_Struct):
    """ A graph where vertices are segment, and so they have two sides
    
    At construction, a Segment graph has two attributs:
      - edge: 
        array of shape (S,N,2) with S the number of segment, N the maximum 
        number for neighbors (per segment side) and 2 for the 2 sides
            > to be an array, missing neighbors are set to 0
      - cost:
        Edge cost (of the same shape), computed from segment.segment_difference
            
                 """
    def __init__(self, rootgraph, invalid_nodes='seed', edge_cost='direction_difference'):
        """ compute a segment graph from RootGraph object 'rootgraph' """
        INDEX_DTYPE = 'uint32'
        
        segment = rootgraph.segment
        node    = rootgraph.node
        
        ns = node.segment.copy()
        if invalid_nodes is not None:
            if invalid_nodes=='seed':
                invalid_nodes = _np.vectorize(lambda ns: (segment.seed[ns]>0).all())(node.segment)
            ns[invalid_nodes] = set()
        ns[0] = set()
        
        # construct nb1 & nb2 the neighbor array of all segments in direction 1 & 2
        nsbor = _np.vectorize(set)(ns)
        snbor = [(s1.difference([i]),s2.difference([i])) for i,(s1,s2) in enumerate(nsbor[segment.node])]
        
        edge_max = max(map(lambda edg:max(len(edg[0]),len(edg[1])),snbor))
        edge = _np.zeros((len(snbor),edge_max,2), dtype=INDEX_DTYPE)
        for i,(nb1,nb2) in enumerate(snbor):
            edge[i,:len(nb1),0] = list(nb1)
            edge[i,:len(nb2),1] = list(nb2)
    
        # compute segment-to-neighbors cost as the direction angle
        if isinstance(edge_cost, basestring):           ##todo: merge SegmentGraph in SegmentList
            edge_cost = getattr(segment,edge_cost)      #         & replace by self[edge_cost]
        edge_cost = edge_cost[_np.arange(edge.shape[0])[:,None,None], edge]  ## should be a tmp property...
        
        # note: by construction, all valid edges are at the begining of axis 1
        self.edge = edge
        self.edge_cost = edge_cost

    @_property
    def edge_switch(self):
        """
        compute if a (directed) connection through edge induce a change of direction
            i.e for all edge (i,j): is j a neighbor on side s of i, and i on side s of j
            thus, i & j are not in the same direction
        """
        if not hasattr(self,'_edge_switch'):
            edge = self.edge
            edge_switch = _np.zeros(edge.shape, dtype=bool)
            sid = _np.arange(edge.shape[0])[:,None,None]
            edge_switch[...,0] = (edge[edge[...,0],:,0]==sid).any(axis=-1) # side 0
            edge_switch[...,1] = (edge[edge[...,1],:,1]==sid).any(axis=-1) # side 1
            self._edge_switch = edge_switch
            self.temporary_attribute.add('_edge_switch')
        return self._edge_switch
        
    @edge_switch.setter
    def edge_switch(self, value):
        if value is None: # delete tmp data
            self.clear_temporary_attribute('_edge_switch')
        else:
            self._edge_switch = value
            self.temporary_attribute.add('_edge_switch')
        
    def segment_chain(self):
        """ merge linear chain of segment 
        
        all valid edges (i.e !=0) must be at the begining of axis 1
        
        return same as extract_chain
        """
        chainable = (self.edge!=0).sum(axis=1)==1
        chain_edge = self.edge[:,0]*chainable
        
        return extract_chain(chain_edge)
        
    def best_edges(self):
        """
        Compute a 'edge' array of the best edge in both sides of all segments
        
        'best' means with lower cost
        :Output:
            - 'edge' array with shape Nx2, for N segments and 2 sides
            - 'best' array with shape Nx2 of the best edges indices in this 
                     SegmentGraph 'edge' attribute
        """
        # compute segment graph, and keep only the best connection for each segment side
        #   => best means "most little cost"
        best = _np.argmin(self.edge_cost,axis=1)
        edge = _np.choose(best, [self.edge[:,i] for i in range(self.edge.shape[1])])
        
        return edge, best
        
    def subaxe(self, max_dcost=None, max_cost=None):
        """ 
        id segment chain s.t. the each chain segment is the minimal edge_cost neighbor 
        of its chain neighbors.
           Not quite clear...
           
        return same as extract_chain
        """
        edge, best = self.best_edges()
    
        if (max_cost is not None) or (max_dcost is not None):
            tested = _np.sum(self.edge>0, axis=1)>1  # keep direct link
            
        if max_cost is not None:
            cost = _np.choose(best, [self.edge_cost[:,i] for i in range(self.edge_cost.shape[1])])
            edge[(cost>max_cost)&tested] = 0
            
        if max_dcost is not None:
            dcost  = _np.sort(self.edge_cost)
            dcost  = dcost[:,0]/dcost[:,1]
            edge[(dcost>max_dcost)&tested] = 0
            
        return extract_chain(edge)
        
    def directed_edge(self, slist, cost='direction_difference', update_graph=True, DAG=True, callback=None):
        """
        Compute a direction of the edge
        
        IN DEVELOPMENT
        """
        # get 'best' neighbors (w.r.t sides) as segments pairs
        # ----------------------------------------------------
        #edg  = self.best_edges()[0]
        edg  = self.edge.reshape(-1,reduce(lambda x,y: x*y,self.edge.shape[1:]))
        s1,s2 = edg.nonzero()     # do not keep edge to bg segment
        
        # merge segment ids in one array
        if s1.base is s2.base: edge = s1.base   # seems to be the case
        else:                  edge = np.concatenate((s1[:,None],s2[:,None]), axis=1)
        edge[:,1] = edg[edge[:,0],edge[:,1]]
        
        # unique segment pairs
        edge.sort(axis=1)
        edge = _np.unique(edge.view(dtype=[('s1', edge.dtype),('s2',edge.dtype)])) \
                       .view(dtype=edge.dtype).reshape(-1,2)
        
        # edges cost
        # ----------
        # edge with seed segment have zero cost
        ##    edge cost = sp.spatial.cosine of slist.direction ?
        ##              = P(edge): cost/sum(cost on ...) ? 
        cost = slist.direction_difference[edge[:,0], edge[:,1]]
        seed = slist.seed>0
        seed[slist.seed>=254] = 0  ## bug that makes fake seeds=254
        cost[seed[edge[:,0]] | seed[edge[:,1]]] = 0
        
        # sort by cost
        edge = edge[_np.argsort(cost),:]
        
        
        # compute the edge direction
        # --------------------------
        # initialization
        sNum  = slist.node.shape[0]
        group = [[i] for i in xrange(sNum)] # segment group (init one group per segment)
        sgid  = _np.arange(sNum)            # group id of segment   ----- " -----
        gset  = _np.zeros(sNum, dtype=bool) #  group  direction selected (init as unset)
        sdir  = _np.zeros(sNum, dtype=bool) # segment direction w.r.t its group (init as 0:same)
        
        # set seed and terminal segment direction
        #     note: seed must be set after terminal as they also are terminal
        term  = slist.terminal
        sdir[term] = self.edge[term][:,:,1].any(axis=1)  # default means no nbor on side 1
        sdir[seed] = self.edge[seed][:,:,0].any(axis=1)  # default means no nbor on side 0
        gset[term] = 1
        gset[seed] = 1
        
        # convert to list for faster 1-element access during iterations
        sdir = sdir.tolist()
        gset = gset.tolist()
        
        # switch: 
        #   True if s1 and s2 must have opposite direction to be part of same group
        #   ie. all edges s.t s1 start (resp. end) == s2 start (resp. end)
        switch = (slist.node[edge[:,0]]==slist.node[edge[:,1]]).any(axis=1) 
        
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
                
            if callback: callback(i, sdir, sgid, gset, s1,s2)
            
        sdir = _np.array(sdir)
        if update_graph:
            self.set_direction(sdir, DAG=DAG)
        
        return sdir, sgid
        
    def set_direction(self, direction, DAG):
        """
        Set direction of segments, and remove invalid edges
        
        The direction are set by reversing the order of this graph edge attribute
            ** edge does not correspond to input segment direction anymore **
            
        :Input:
            - direction 
                should be a boolean vector array of length equal to the number of
                segment, and where True value means the segment should be reverted
            - DAG
                If True, also remove edge such that graph is acyclic 
        """
        # reverse edge direction
        self.edge[direction] = self.edge[direction][...,::-1]
        ## SHOULD reverse self.node too once inherite SegmentList !?
        
        # remove invalid edge,  i.e edges that require switch
        self.edge_switch = None             # delete current edge_switch, if any
        self.edge[self.edge_switch] = 0
        
        # conversion to DAG
        if DAG:
            src = self.edge[...,0].any(axis=1)
        
    def propagation(self, src, get_neighbor, update, max_iterations='len', **kargs):
        """
        Generic propagation meta-algorithm.
        
        Starting with 'src' as current segments, call iteratively:
          - get_neighbor to get indices of the neighbors of segments, to update
          - update to the neighbors
          - set the neighbors as current for next iteration
        
        The propagation stop when either max_iterations is reached or the 
        neighbor list is empty.
        
        :Input:
          - src:
                indices of the segment the propagation start at
                
          - get_neighbor: 
                a function with arguments:
                  * this graph object
                  * the list of current segments
                  * all the kargs
                and returns: 
                  * the array of the current segments id related to 2nd output 
                  * the array of the segment ids to update
                  
                See the get_neighbor method for an example of valid function
                  
          - update:
                a function with arguments:
                  * this graph object
                  * the first  output of get_neighbor
                  * the second output of get_neighbor
                  * all the kargs
                and returns:
                  * 'updated': the array of segments for the next iteration
          
          - max_iterations:
                the maximum number of iterations. If 'len', use the number of 
                segments of this graph
                
          - **kargs: optional key-arguments that are passed to get_neighbor and
                     update
                     
        :Note:
          - to add a stopping criterion, make the 'update' function returns
            an empty list/array
          - get_neighbor and update are called with input arguments in the order
            stated above, passed as non-key arguments but for the kargs
            
        :Output:
            a dictionary of the key-arguments kargs
            
        """
        iteration = 0
        if max_iterations=='len':
            max_iterations = self.edge.shape[0]
            
        new_current = src
        
        while iteration<max_iterations and len(new_current)>0:
            current, next = get_neighbor(self, new_current, **kargs)
            new_current   = update(self, current, next, **kargs)
            iterations += 1
            
        return kargs
        
    def get_neighbor(self, segments, direction=None, **kargs):
        """
        Return the neighbors of 'segments', with optional restriction on direction
        
        :Input:
          - segments: 
              1d integer array of segment ids
          - direction:
              Optional 1D integer array of  this graph segments directions.
              If given, the returned neighbors are restricted on the neighbors
              that are in the "forward" direction. Each segment has two sets of
              neighbors, one for each side. If direction[s] = d, neighbors(s)
              are the neighbors on side d.
              If not given, returns all neighbors, independantly of their side.
                
        'Output':
          - id array from the input 'segment' related to the returned neighbors
          - array of neighbor ids
          Both output arrays have same shape but have arbitrary dimension. 
          However the segments 1d changes only along the 1st dimension. 
        """
        
        #edge = _np.atleast_3d(self.edge)
        if direction: 
            nbor = edge[segments,direction[segments]]#[:,_np.newaxis]
        else:
            nbor = edge[segments]
          
        segment = _reshape(segment,[s if i==0 else -s for i,s in enumerate(nbor.shape)])
        #bg_mask = nbor==0
        
        return segment,nbor #segment[mask], nbor[mask]
        
    def accumulate(self, src, dst, cost, acc, direction=None, filter=None, **properties):
        """
        ...
        """
        pass
    
    def update_direction(self, src, dst, direction):
        """
        Set direction[dst] such that "dst come from src".
        
        If src are 
        """
        pass

def to_csgraph(edge, value=1, omit_bg=False, matrix='csr_matrix'):
    """ make a sparse adjacency representation of a graph from input 'edge'
    
    :Input:
      - edge:    a Nx[K] neighbor indices of N elements (K might be any shape)
      - value:   values of edges. Either 
                    a scalar: all edges have this value
                    an array of the same shape as edge
      - omit_bg: if True, do not add edge from/to background element (0)
      - matrix:  the type of sparse matrix to use (from scipy.sparse)
      
    Note that if the same edge appears several times, their value is summed
    """
    from rhizoscan.ndarray import virtual_array
    from scipy import sparse
        
    edge   = edge.reshape(edge.shape[0], -1) # assert edge is 2D
    matrix = getattr(sparse, matrix)     # matrix constructor
    
    # list of indices pairs
    ij = _np.asarray([(_np.arange(0,edge.shape[0],1./edge.shape[1]).astype(int)).ravel(),edge.ravel()])
    
    # manage scalar value
    if _np.asarray(value).size==1:
        value = virtual_array(edge.size, value=value)
    else:
        value = _np.broadcast_arrays(edge,value)[1].ravel()
        
    # remove edge to/from background
    if omit_bg:
        mask  = (ij!=0).all(axis=0)
        ij    = ij[:,mask]
        value = value[mask]
        
    return matrix((value,ij), shape=(edge.shape[0],)*2)
    
def extract_chain(edge, sorted_order=True):
    """
    Extract the chain identification of all segments w.r.t 'edge' array
    
    :Input:
        edge: Sx2 array of the (maximum) 2 neighbors id of S segments. It should
              be directed: if i&j are connected, then j in edge[i] & i in edge[j]
        
              Invalid neighbors should be set to the "background" segment (ie. 0)
              which should have no neighbors (ie. edge[0,...]==0)
        
        sorted_order: the order is given following chain id: the segments of 
                      chain i come before those of segments of chain j if i<j
                        i.e. chain[order] is sorted
    
    :output:
        graph: the sparse adjacency matrix representing the graph 
        chain: the id of the chain for all segments
        order: an ordering of the segment that garanties that all the segments
               of a chain are contiguous and in a traversal order of the chain. 
               The direction of the traversal is not defined.
    """
    from scipy import sparse
    
    # make the sparse adjacency matrix representation
    g = to_csgraph(edge)               
    
    # connected component of the graph
    #   a segment pair is connected only if there are edges in both direction ('strong')
    #   note: many segments are connected to 0, but 0 is only connected to it-self
    #         which avoid making 1 components all linked to bg
    n,chain = sparse.csgraph.connected_components(g,directed=True,connection='strong')
    
    # Chain tip (and only the tip) are connected to the bg segment (id:0)
    # This fact is used to find the contiguous, sorted-per-chain order of 
    # segments using depth first traversal starting at bg
    order = sparse.csgraph.depth_first_order(g,0,directed=False,return_predecessors=0)
    
    if sorted_order:
        order = order[_np.argsort(chain[order])]

    return g, chain, order
    
def set_downward_segment(graph):
    """
    In-place set all segments of RootGraph 'graph' downward, or horizontal
    
    return updated 'graph'
    """
    upw = _np.diff(graph.node.y[graph.segment.node],axis=1).ravel()<0
    graph.segment.node[upw] = graph.segment.node[upw][:,::-1]
    
    return graph
    
def digraph_to_DAG(edge, cost, source):
    """
    Convert a directed graph to a DAG using a method based on shortest path.
    
    This function first compute the shortest path tree. Then it iteratively add
    edges that were removed by the shortest path but which don't create cycles. 
    This is done following the order of the tree cumulative edge distance.

    :Note:
        Because it is based on a shortest path, all unreachable elements from 
        the given sources are removed.
    
    :Input:
        - edge
            an NxK array of neighbor indices of the K *forward* neighbors of N 
            elements. 0 value edges are "fake edges".
        - cost
            An array of (broadcastably) same shape as edge of the edges cost
            All "real edges" cost must be **strictly** positive 
        - source
            indices or boolean mask of the shortest path sources
            e.g.:  rgraph.segment.seed>0
            
    :Output:
        - updated edge array which is a DAG
        - topological order of the graph nodes (missing ids were unreachable)
           *** topological order is for the tree, not the dag ??? ***
    
    :todo: is topological order for the DAG, if not (probably) remove output
    """
    from scipy.sparse.csgraph import dijkstra, reconstruct_path
    
    if _np.asarray(source).dtype=='bool':
        source = source.nonzero()[0]
    
    # compute shortest path tree
    graph = to_csgraph(edge, value=cost,omit_bg=1)
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
    
    # convert tree to edge-type array, with 'added' edges added
    edge_list = [[] for node in xrange(tree.shape[0])]
    src,dst = tree.nonzero()
    for si,di in zip(src,dst):
        edge_list[si].append(di)
    for si,di in added:
        edge_list[si].append(di)
        
    edge_max = max([len(elist) for elist in edge_list])
    edge = _np.zeros((edge.shape[0],edge_max), dtype=edge.dtype)
    for i,elist in enumerate(edge_list):
        edge[i,:len(elist)] = elist
            
    return edge, dist_order, d ## return removed edges?
    
def dag_path(edge, source, cost=None, callback=None):
    """
    *** NOT DONE and probably FALSE ***
    
    Compute all path in a dag, starting at any node in 'src'
    
    :TODO:
        If cost is provided, must be of same (broadcastable) shape as edge 
        and the sum of edge cost is computed for all path.
    
    :Note:
        No test on the input is done. If inputs are not as expected, the 
        behavior of this function is unknown.
    """
    source = _np.asarray(source)
    if source.dtype=='bool':
        source = source.nonzero()[0]
    
    # total number of path
    ##   to correct: if source have no child, then it's not counted...
    E_out = (edge>0).sum(axis=1)
    pathNum = ((E_out-1)*(E_out>0)).sum()
    
    # in dev...
    all_path = _np.zeros((2*pathNum,edge.shape[0]), dtype=bool)
    all_cost = [0]*pathNum
    path = [False]*edge.shape[0]
    
    edge = edge.tolist() ##[[ei for ei in e if ei>0] for e in edge.tolist()]
    cost = cost.tolist()
    
    def path_traversal(node, current_cost, path_id):
        print node,
        next = [(e,c) for e,c in zip(edge[node], cost[node]) if e>0]
        path[node] = True
        if len(next):
            for next_node, edge_cost in next: 
                path_id = path_traversal(next_node, current_cost+edge_cost, path_id)
        else:
            # end-of-path
            print '.', path_id
            #all_path[path_id, :] = path
            #all_cost[path_id] = current_cost
            path_id += 1
            if callback:
                callback(path_id, path)
        path[node] = False
        return path_id
    
    path_id = 0
    for i in source:
        path_id = path_traversal(i,0, path_id)
    
    return _np.array(all_path), _np.array(all_cost) 
    

def rg_to_at(rg):
    """
    convert general root graph to an axial tree
    
    *** In development ***
    put all steps of in-dev conversion into 1 function 
    """
    set_downward_segment(rg)  # not required, but best to visualise results
    
    sg = SegmentGraph(rg)
    sdir,g = sg.directed_edge(t.segment, update_graph=False, DAG=False) # g?
    
    

