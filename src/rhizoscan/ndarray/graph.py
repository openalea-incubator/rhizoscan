"""
This is some abandoned code which is not updated and nor maintained
"""

import numpy as np
from scipy   import ndimage as nd
from .       import ravel_indices, unravel_indices, add_dim 
from ..tool  import tic, toc

from rhizoscan.workflow.openalea import aleanode as _aleanode # decorator to declare openalea nodes
## how to make a node out of a graph ? make an independant function for each method ?

##TODO: 
##
## => real ArrayGraph idea should be implemented very differently:
##    . edges should be a set of nd-slice-tuple, each for a specific neighbor shift
##          edges dir (eValue?) could be provided, or chain-like alphabet
##          used with a map to dir array (kind of colormap for direction)
##    . multiple node and edge values stored in dictionaries
##    . general concept should be defined, such as:
##          "intelligent" filtering (s.a. bilinear)
##              cell_value update to f(cell_value, set of nbor_value)
##          iterative propagations (with possible transformation along the path)
##              a updated cell value comes from one nbor => shortestPath
##              some kind of filtering with a parent selection
##              parent,cell_value = f(...)
##          update propagation? -> filtering using parenting map ?
##          need to define stop criterion (maxIt, d(value) threshold...)
##    . use of mask to restrict the cells to update ? 
##          needs to be tested
##    . input array dimension should be retained ?
##
## => this graph class should become "VectorGraph"
##    . delete mask nodes, eValue per same nbor, etc...
##    . just keep methods to convert results to original input shape
##          use some node id array (0 - or ? - being mask)
##
## - use mask array instead of indices list !?
##
## - Conversion to/from other graph classes, s.a. networkX:  __init__(nx)  and convert2nx()
##     => how to manage atributes etc... (edge-weight, ???)
## - standardize eValue / distanceMap (real edges value)
##     => eValue should be neighbor_value and distance map eValue ? ...
## - settle shortest path attribute... 
##     => use a structure not stored in graph ?
## - manage several values per node/edge? 
##     => dictionary array ?
##     => or node and footprint has an additional axis for multiple value
##     => or both, convert first to second ?
##     => should be similar to other graph lib
## - pathEnd()   ?
## - childMap()  ?
## - shortestPath update neighbor from current, not current from neighbor
##     => maybe not, cost to reduce the size of vectors might not be worth it
## - add adjency matrix as possible setEdges() arguments
## - define clearly the **stable** API (using _)


# constants
VALUE_DTYPE = 'float32'
INDEX_DTYPE = 'uint32'
INDEX_OUT   = np.array(-1,dtype=INDEX_DTYPE)


class ArrayGraph:
    """
    Graph class designed mostly for ndarray to connect cells to their neighbor. 
    Useful to automatically construct a graph from an array, compute shortest 
    path and distance map. 
    
    For now, the graph store one value per node (the node-value) and one value 
    per edges, the edge value, which are used to compute distances.
    See makeDistanceMap()
    
    Method list:
      makeDistanceMap(...)      construct a distance map used by the shortest path
      shortestPath(...)         compute shortest path map from source points
      update_shortestPath(...)  update a shortest path
      getParentMap(...)         get a map of parent indices for each node
      getPath(...)              retrieve a path -  indices in n-dimension
      getPath1D(...)            --------------- -  indices in 1-dimension
    """
    
    def __init__(self,nodes_value=None,edges=None,edges_value=0, footprint=3):   
        """
        Construct a graph from an array
        
            call:  graph = Graph( nodes_value , edges=None, edges_value=0, footprint=3 )
                  
        :Input:
          nodes_value: Either an n-d array:
                        - each elements is a node, and its values is the node-value.
                        - NaN elements are masked nodes
                      Or a 1D list/tuple of the array shape.
                        - node-value is then zeros for all nodes
                      ***    if None, an empty graph is returned.     ***
                      ***  Use setNodes() and setEdges() to fill it   ***
                        
          edges:      It can be one of the following
                      - None (or not given), use footprint (see below)
                      - a list (or tuple) for all nodes, of list (or tuple) of 
                        all neighbors the including list must have length equal 
                        to the number of nodes but the enclosed lists can be of 
                        any length
                      - a valid edges array for ArrayGraph:
                        a numpy array for size NxK where N is the number of nodes 
                        and K the maximum number of neighbors per nodes. The cells 
                        values are either node indices (integer) or INDEX_OUT

                      If 'edges_value' is use, all neighbors with same position 
                      in the last dimension of edges will receive the same value.
                     
          edges_value: a single value or a 1D array of length equal to 'edges' 
                      last dimension (number of neighbors) These values are shared
                      by all edges with same relative position
                      See makeDistanceMap() 
                      *** edges values can be given directly in footprint  ***
                      
          footprint:  Either a number being the size of the window in all dimension
                      Or a tuple of length n, the size of the window in each dimension
                      Or a numpy ndarray (same dimension as node-array), for which 
                        all non zeros elements represent an edge to be created, 
                        and the elements value is used as the 'edge value'
                      *** if edges is not None, footprint is not used ***
                      
        See also: setNodes(), setEdges()
        """
        ## create attributes used by the shortestPath algorithm 
        self.parent = self.distance = self.distance2 = self.dMap = self.dMap2 = None

        if nodes_value is not None:
            self.setNodes(nodes_value)
            self.setEdges(edges=edges,edges_value=edges_value, footprint=footprint)
        

    def setNodes(self,nodes_value, clean_edges = True):
        """
        Set the node value array of the graph

        :Input:
          nodes_value: Either an n-d array:
                        - each elements is a node, and its values is the node-value.
                        - NaN elements are masked nodes
                      Or a 1D list/tuple of the array shape.
                        - node-value is then zeros for all nodes
                      
          clean_edges: if True, edges array and edges shared value are deleted
        """
        if isinstance(nodes_value,(tuple,list)) and np.isscalar(nodes_value[0]):
            self.nValue = np.zeros(nodes_value,dtype=VALUE_DTYPE)
            self.nSize  = tuple(nodes_value)
        else:
            self.nValue = np.asarray(nodes_value,dtype=VALUE_DTYPE)
            self.nSize  = self.nValue.shape
            
            
        # initialize size and shape attributes
        self.nodeNum = self.nValue.size        # number of nodes
        self.ndim    = self.nValue.ndim        # number of dimension in original data
        
        if clean_edges:
            self.nborNum = 0
            self.edges   = np.zeros(self.nSize + (0,))
            self.eValue  = np.zeros((1,0))
        
        
    def setEdges(self, edges=None,edges_value=0, footprint=3):
        """
        Set the edges between the graph nodes and edges shared value.
        The graph nodes should have been initiated before calling this function

        :Input:
          edges:      It can be one of the following
                      - None (or not given), use footprint (see below)
                      - a list (or tuple) for all nodes, of a list (or tuple) of 
                        the respective node's neighbors
                        The list must have length equal to the number of nodes
                        but the enclosed lists can be of any length
                      - a valid edges array for ArrayGraph:
                        a numpy array for size NxK where N is the number of nodes 
                        and K the maximum number of neighbors per nodes. The cells 
                        values are either node indices (integer) or INDEX_OUT
                      * not done (TODO): an adjacency matrix (boolean, NxN, sparse ?)

                      If 'edges_value' is use, all neighbors with same position 
                      in the last dimension of edges will receive the same value.
                      
          edges_value: a single value or a 1D array of length equal to 'edges' 
                      last dimension (number of neighbors) These values are shared
                      by all edges with same relative position
                      See makeDistanceMap() 
                      *** edges values can be given directly in footprint  ***
                      
          footprint:  Either a number being the size of the window in all dimension
                      Or a tuple of length n, the size of the window in each dimension
                      Or a numpy ndarray (same dimension as node-array), for which 
                        all non zeros elements represent an edge to be created, 
                        and the elements value is used as the 'edge value'
                      *** if edges is not None, footprint is not used ***
                      
        :Warning:
           No test is done on the shape and size of input arguments but the minimum
           that is required to select the suitable edge construction method 
        """
        if edges is not None:
            # pad all node edge list s.t. they have same length
            if isinstance(edges, (tuple,list)): 
                edges  = [sorted(list(set(edg))) for edg in edges]         # uniquify edges lists
                maxEdg = np.max([len(edg) for edg in edges])               # max number of edges
                edges  = [edg+[INDEX_OUT]*(maxEdg-len(edg))  for edg in edges] # pad list with INDEX_OUT
                
            self.edges   = np.asarray(edges,dtype=INDEX_DTYPE)
            self.nborNum = self.edges.shape[-1]                            # max number of neighbors
            self.eValue  = np.asarray(edges_value,dtype=VALUE_DTYPE)       # edge-value
            self.eValue.shape = self.eValue.size,1                         # suitable reshape
            self.edges.shape  = self.nodeNum,self.edges.shape[-1]
            
            # check if edges_value is a scalar (actually, if it hasn't the right size)
            if self.eValue.size != self.edges.shape[-1]:
                self.eValue = np.ones((self.edges.shape[-1],1),dtype=VALUE_DTYPE)*self.eValue[0]
              
              
        else: # make edges and edges value from footprint
            
            # make valid footprint array (if not already valid)
            if isinstance(footprint,int):          footprint = [footprint for d in xrange(self.nValue.ndim)]
            if isinstance(footprint,(tuple,list)): footprint = np.ones(footprint, dtype=VALUE_DTYPE)
            footprint[tuple(np.array(footprint.shape)/2)] = 0    # remove footprint center (a node cannot be its own neighbor)
            
            self.nborNum = (footprint!=0).sum();  # number of neighbors (max)
            
            
            # edges array:  
            # ------------
            # edges is a NxK array where N is the number of nodes, and K the number of neighbor
            self.edges = np.zeros((self.nodeNum,self.nborNum), dtype=INDEX_DTYPE)
            
            K  = np.transpose(footprint.nonzero())   # neighbors are nonzero value of footprint
            K -= (np.array(footprint.shape)/2)       # center footprint 
            
            node = np.arange(self.nodeNum)   # node 1D indices
            node.shape = self.nSize
            
            
            #w = np.zeros_like(footprint)  ## to replace not working 'nice' method, see below
            for i in range(K.shape[0]):
                ## not working... looks like a bug..., look's like it work finally... to check
                self.edges[:,i] = nd.interpolation.shift(node,tuple(-K[i]),order=0,cval=INDEX_OUT).ravel()
                #w[tuple(K[i])] = 1
                #self.edges[:,i] = nd.convolve(node,w,mode='constant',cval=INDEX_OUT).ravel()
                #w[tuple(K[i])] = 0
            
             
            # remove edges connecting node with NaN value (i.e. nodes out of mask)
            mask_ind  = np.nonzero(np.isnan(self.nValue.flat))[0]    # 1D indices of masked nodes
            self.edges[mask_ind,:] = INDEX_OUT                      # remove edges *from* masked nodes
            self.edges.shape = self.edges.size 
            self.edges[np.in1d(self.edges,mask_ind)] = INDEX_OUT   # remove edges  *to*  masked nodes
            self.edges.shape = self.nodeNum,self.nborNum
    
    
            # edges value array (eValue):   
            # ---------------------------
            # the non-zero values of footprint (shape=[1,K], in coherence with 'edges' shape)
            self.eValue = footprint[footprint!=0][np.newaxis,:].astype(VALUE_DTYPE)
            
        # by default all arrays shape are relative to the original data shape 
        self._unravelArrays()
        
        
    @_aleanode({'name':'dmap'})
    def makeDistanceMap(self, method='edges'):
        """
        create distance map used by the shortestPath() method
        
        possible method arguments are:
          'edges', uses the edge value created by the the constructor (from footprint)
          'nodes', uses the node value given to the constructor (as node_array)
           any function that is called with 3 arguments (all with shape nodeNumber x neighborNumber):
               edges, nodes_source, nodes_destination
               ex:   lambda E,S,D: E + (S+D)/2     is the edge value + a "node average" distance map
               
        Note: element of nodes_destination that are out of the image are equal to np.Inf
        """
        
        ## use shifted stride trick for dst ?
        ##   require edges to contain only valid values,
        ##   possible by having a boolean edges_valid array ?
        ## then make dst (edg and src too?) not writable: dst.flags.writable = False

        m = 1 if method=='edges' else 2 if method=='nodes' else 3
        N,K = self.nodeNum, self.nborNum
        
        self._ravelArrays()
        if m>=1: edg = add_dim(self.eValue.flat,0,N)
        if m>=2: src = add_dim(self.nValue.flat,1,K)
        if m>=3: 
            dst = self.nValue.ravel()[np.mod(self.edges,self.nodeNum)]
            dst[self.edges==INDEX_OUT] = np.Inf   # stride trick is thus useless - is it really? :(

        self._unravelArrays()
        
        if m==1: return edg
        if m==2: return src
        if m==3: return method(edg, src, dst)
        
        
    def shortestPath(self, source, distanceMap='edges', distanceMap2=None, **kwargs):
        """
        shortestPath(sourcePixel, distanceMap, dMap2=None, key-arguments)
        
        Compute the shortest path from any source to all nodes using given distance Map.        
        This function create (or overwrite) the 'distance' and 'parent' fields of 
        the graph object, that store for all nodes, the total path distance and 
        the index of the (direct) parent node, respectively.
        
        Input:
        sourcePixels: one index or a list (array/tuple/list) of indices of the nodes 
                      that the distance field is initiated at (zero distance).
                      The indices should be 1D (flat), except if: 
                      it is 1D of the same length (size) as the original data shape
                      it is 2D, and the 2d dimension has size egual to the original data shape
                      In these cases, the indices are converted using tool.ravel_indices
                      
        distanceMap: either an NxK array containing the displacement cost from any N nodes to 
                     any of its K neighbors - This array can be computed with makeDistanceMap()
                     or any of valid argument 'method' of makeDistanceMap()
                     
        dMap2: a second distance map used to compute the graph object 'distance2' field:
               the distance of all nodes to the source node with this distance map.
                 
        other possible arguments:
            maxIteration: maximum number of iteration the algortihm can do
            verbose: if not null, print some descriptif text each 'verbose' (an integer) steps
            callback: if not None, should be a function taking as arguments this graph and the 
                      iteration number (i.e. fct(graph,iter)) called at the end of each iteration
            stat: if not None, should ne a tool.Struct. Three fields 'iter', 'time' and 'node' is 
                  appended which contain the iteration number, the time to compute and the number 
                  of nodes which are updated, at each steps
              
        See also: makeDistanceMap, update_shortestPath, getParent
        """
        
        # starting position, convert to 1D (flat) coordinates if necessary
        source = np.asarray(source,dtype=INDEX_DTYPE)
        source = np.atleast_1d(source)
        if source.ndim>1 or source.size==len(self.nSize):
            source = ravel_indices(source, self.nSize)
        
        # check distanceMap
        if not isinstance(distanceMap,np.ndarray):
            self.dMap = self.makeDistanceMap(distanceMap)
        else:
            self.dMap = distanceMap.astype(VALUE_DTYPE)

        # create distance from source and parent array. By default:
        #    all cells distance is Inf, but the source
        #    all cells parent are them-selfs
        self.parent   = np.arange(self.nodeNum,dtype=INDEX_DTYPE)
        self.distance = np.ones(  self.nodeNum,dtype=VALUE_DTYPE) * np.Inf
        self.distance[source] = 0
        
        # if asked, check distanceMap2
        if distanceMap2 is not None:
            if not isinstance(distanceMap2,np.ndarray):
                self.dMap2 = self.makeDistanceMap(distanceMap2)
            else:
                self.dMap2 = distanceMap2.astype(VALUE_DTYPE)
            self.distance2 = np.ones(  self.nodeNum,dtype=VALUE_DTYPE) * np.Inf
            self.distance2[source] = 0
        
        self._ravelArrays()
        
        # remove source elements that are out of mask (nan node value)
        # and if remaining source element list is empty, quit
        source = source[(-np.isnan(self.nValue[source])).nonzero()[0]]
        if source.size == 0:
            print 'no source' ##
            self._unravelArrays()
            return
        else:
            return self.update_shortestPath(source, **kwargs)


    def update_shortestPath(self, startNode='all', maxIteration=np.Inf, verbose=False, callback=None, stat=None):
        """
        Do the actual computing of the graph shortest path, 
        Or update one if the 'distance' array has changed
          >  graph distance, parent, and dMap arrays should be ready, as it is done by the shortestPath method
          >  curNode should be 'all', 'min' or indices, in which case they should be 1D (flat) indices
        
        :Input:
          curNode: either the list of nodes that the shortest path is starting at
                   or 'min', select nodes with distance equal to the minimum of the distance array (probably 0)
                   or 'all', select all nodes
          maxIteration: maximum number of steps the algortihm will do (max path node number)
          verbose: an integer,  if not null, print some descriptive text each 'verbose' steps
          callback: if not None, should be a function taking as arguments this graph, the 
                    iteration number and the list of node that as been updated at this step
                       i.e.   callback(graph, iter, node)
                    Called at the end of each iteration
          stat: if not None, should be a tool.Struct object. Three fields 'iter', 'time' and 'node' 
                is added to it. They contain the iteration number, the computation time and the number 
                of nodes which have been updated, for each steps
        """
        ## in dev: distanceMap2
        
        # make suitable reshape: flat arrays
        self._ravelArrays()

        # create short name pointer for used arrays
        edges  = self.edges
        eValue = self.eValue
        nValue = self.nValue
        dist   = self.distance
        parent = self.parent
        dMap   = self.dMap
        
        # initialize curNode (current node to be updated) from startNode argument
        if   startNode=='min':
            curNode = np.nonzero(self.distance==self.distance.min())[0].astype(INDEX_DTYPE)
        elif startNode=='all':
            curNode = np.arange(self.distance.size,dtype=INDEX_DTYPE)
        else:
            curNode = np.asarray(startNode,dtype=INDEX_DTYPE).ravel()
            
        # if asked, create the second distance array
        if self.dMap2 is not None:
            dist2 = self.distance2
            dMap2 = self.dMap2
            #dist2 = np.ones_like(dist)*np.inf
            #dist2[curNode] = 0  ## curNode is startNode ? always ? ...

        # initialization for special options
        if verbose or stat:
            startT = tic('imageGraph')
        if stat is not None:
            stat.time = []
            stat.iter = []
            stat.node = []
            

        # start the main algorithm
        iter = 0
        while curNode.size>0 and iter<maxIteration:
            # update current node list: union of neighbors of all current nodes
            curNode = np.setdiff1d(edges[curNode,:],INDEX_OUT)

            # compute dist: from all current nodes to all their neighbors
            # when some of the neighbor are out of the image (i.e.: edges[current,:]==INDEX_OUT)
            #  > d_inc is still updated by considering them acceptable (replaced by mod(INDEX_OUT,nodeNum))
            #  > but afterward these computed fake distances are replaced by Inf
                
            # current node's neighbors 
            #   their are made usable as array indices with the modulus (i.e. remove INDEX_OUT value)
            #   fake indices are later removed
            nbor  = np.mod(edges[curNode,:],self.nodeNum) 

            # compute distance from all neighbor
            #   d_nbor = updated distance *from* all neighbor *to* current node
            #          = neighbor current distance + their distance to the current nodes
            d_nbor = dist[nbor] + dMap[curNode,:]
        
            # replace distance of fake neighbors by Inf
            d_nbor.ravel()[(edges[curNode,:]==INDEX_OUT).ravel()] = np.Inf
            
            # replace nan (invalid by neighbor) by inf  ## why the later nanmin(d_nbor) don't work well ???
            d_nbor[np.isnan(d_nbor)] = np.Inf
            
            
            # update distance arrays: 
            # -----------------------
            #    best_dist = minimum distance from all neighbors
            #    up_index  = indices of the current nodes that should be updated
            #    for nodes that should be updated, i.e. best neighgbor distance is better than current distance
            #       update (reduce) current node list to those only
            #       then, update current distance (dist)
            best_dist     = np.nanmin(d_nbor,1)
            ##if verbose==1: print best_dist
            up_index      = best_dist < dist[curNode].ravel()
            curNode       = curNode[up_index]
            dist[curNode] = best_dist[up_index]
            
            
            # update parent array:
            # --------------------
            #    best_nbor = indices of neighbor with minimal distance (i.e. best_dist)
            #    update parent value from edges array:
            #       use a list of 2 lists indexing : ((all the x's), (all the y's)) 
            ##print d_nbor, up_index
            best_nbor = np.argmin(d_nbor[up_index],1)
            parent[curNode] = edges[curNode, best_nbor] ## curNode & best_node .ravel()
            
            # if asked, update the second distance array
            ## this seems to be wrong: should not be any cal to min - the selection is already done
            if self.dMap2 is not None:
                nbor  = np.mod(edges[curNode,:],self.nodeNum)
                d_nbor = dist2[nbor] + dMap2[curNode,:]
                d_nbor.ravel()[(edges[curNode,:]==INDEX_OUT).ravel()] = np.Inf
                d_nbor[np.isnan(d_nbor)] = np.Inf   ## again, why is this necessary? ...
                best_dist      = np.nanmin(d_nbor,1)
                dist2[curNode] = best_dist

            iter += 1
            
            if verbose and np.mod(iter,verbose)==0:
                print iter, ': n=', curNode.size, '; time=', toc('imageGraph',False)
                
            if callback is not None:
                callback(self,iter,curNode)
                
            if stat is not None:
                stat.time.append(toc('imageGraph',False))
                stat.iter.append(iter)
                stat.node.append(curNode.size)
                
                
        # unravel arrays (back to their initial shape) before return
        self._unravelArrays()
       
        
    #def getChildrenMap(self):
    #    ##! getChildrenMap(): not working !!!
    #    self._ravelArrays()
    #    valid = np.squeeze(np.nonzero(-np.isnan(self.nValue).ravel()))
    #    nborP = self.parent[np.mod(self.edges,self.nodeNum)]      # parent of all neighbors of all nodes
    #    nborP[self.edges==INDEX_OUT] = INDEX_OUT                  # remove invalid ones
    #    
    #    child = self.edges.copy()
    #    child[nborP != add_dim(np.arange(self.nodeNum,dtype=INDEX_DTYPE),1,self.nborNum)] == INDEX_OUT
    #    #child = (nborP,add_dim(np.arange(self.nodeNum,dtype=INDEX_DTYPE),1,self.nborNum))
    #    self._unravelArrays()
    #    child.shape = self.edges.shape
    #    
    #    return child
        
    def parentTree(self, directed='ascend', parent=None):
        """
        Construct the a tree graph based on given parenting map
         
        'parent' is a list containing the node id of the parent cell for all nodes 
        of the graph. If not given, the parent attribute of the graph is used 
        (it must have been created previously using the shortestPath() method)
        
        'directed' indicate which type of graph is made. If it is:
          - 'ascend',  the edges are all pointing from child to parent
          - 'descend', the edges are all pointing from parent to child 
          - 'both' or False: both direction are computed, the graph is not directed
        
        TODO: use this method as a replacement of set edges, and rename makeTree or something
               => self.edges & self.eValue would be None !
        """
        edges = self.edges
        self._ravelArrays()

        if parent is None: 
            parent = self.parent
        else:
            parent = np.asarray(parent,dtype=INDEX_DTYPE)
            parent.shape = parent.size,1
            
        ASCEND  = 1
        DESCEND = 2
        if   directed=='ascend':  directed = ASCEND
        elif directed=='descend': directed = DESCEND
        else:                     directed = ASCEND+DESCEND
        
        child = parent!=np.arange(self.nodeNum)      # indices of nodes that have a parent

        tree_edges  = np.zeros(edges.shape, dtype='bool')  # edges of the tree to be returned
        
        # ascending edges: from all children to their parent
        tree_edges[child,:] = edges[child,:] == add_dim(parent[child],axis=-1,size=edges.shape[-1])
        
        # descending edges: from all parent to their children
        if directed & DESCEND:
            parent_ind = edges[tree_edges]
            child_ind  = (edges[parent_ind,:] == add_dim(child.nonzero()[0],axis=-1,size=edges.shape[-1])).nonzero()[1]
            
            if directed == DESCEND: 
                tree_edges[:] = False  # remove ascending edges if necessary
                
            tree_edges[parent_ind, child_ind] = True
        
        
        # if tree flag, use edges values, otherwise set INDEX_OUT
        tree = ArrayGraph()
        tree.setNodes(self.nValue)
        tree.setEdges(edges=np.choose(tree_edges,[INDEX_OUT, edges]), edges_value=self.eValue)
        
        self._unravelArrays()
        return tree
        
    def subgraph(self,node_indices):
        """
        Return the subgraph induced by the nodes listed in node_indices
        
        node_indices should be either a boolean array of the same size as the 
        graph node, a list of integer indices or a slice object.
        
        The returned graph contains the node value attributs for the respective
        nodes indices and the edges list restricted to the nodes that are in the
        subgraph.
        
        ##todo: optionally keep edges value
        """
        # id of new nodes
        new_nodes = np.zeros(self.nodeNum,dtype=INDEX_DTYPE)
        new_nodes[node_indices] = True
        new_nodes = new_nodes*np.cumsum(new_nodes).astype(INDEX_DTYPE)-1
        new_nodes[new_nodes<0] = INDEX_OUT # just in case INDEX_DTYPE is not uint
        
        # edge mapping to new nodes id
        new_edges = new_nodes[self.edges%self.nodeNum]
        new_edges[self.edges>self.nodeNum] = INDEX_OUT
        
        # sort new edges, and remove unnecessary edges dimension
        #! require that INDEX_OUT is > than any valid index 
        order = np.argsort(new_edges,axis=1)
        new_edges = new_edges[np.arange(self.nodeNum)[:,np.newaxis],order]
        new_edges = new_edges[:,:np.max(np.sum(new_edges!=INDEX_OUT,axis=1))]
        
        g = ArrayGraph()
        g.setNodes(self.nValue[node_indices])
        g.setEdges(new_edges[node_indices])
        
        return g
        
    def connected_components(self):
        components = np.arange(self.nodeNum)
        
        mask  = self.edges!=INDEX_OUT
        nbor  = np.where(mask,self.edges,components[:,np.newaxis])
        new_c = np.min(components[nbor],axis=1)
        new_c = np.minimum(new_c, components)
        
        while np.any(components!=new_c):
            components = new_c
            new_c = np.min(components[nbor],axis=1)
        
        return components
        ##cur_nodes = 0
        ##components    = np.empty(self.nodeNum,dtype=INDEX_DTYPE)
        ##components[:] = INDEX_OUT
        ##
        ##k = 0
        ##while components[cur_nodes]==INDEX_OUT:
        ##    #np.lib.arraysetops.setdiff1d(self.edges[cur_nodes],[INDEX_OUT])
        ##    new_node = set(self.edges[cur_nodes].flat).difference([INDEX_OUT.tolist()]) 
        ##    while np.any(cur_nodes!=next):
        ##        cur_nodes = next
        ##        next = set(self.edges[cur_nodes].flat).difference([INDEX_OUT.tolist()]) #np.lib.arraysetops.setdiff1d(self.edges[cur_nodes],[INDEX_OUT])
        ##        
        ##    components[cur_nodes] = k
        ##    cur_nodes = np.argmax(components==INDEX_OUT)
        ##    k = k+1
            
        return components
        
    def _ravelArrays(self):
        # ravel arrays (except the neighbor dimension), but always keep 2D shape, eg: (N,1)
        self.edges.shape = self.nodeNum,self.nborNum
        #self.eValue.shape = self.nodeNum,self.nborNum
        self.nValue.shape = self.nodeNum,1
        if self.parent is not None:
            self.parent.shape   = self.nodeNum
            self.distance.shape = self.nodeNum
            if self.distance2 is not None:
                self.distance2.shape = self.nodeNum
                
    def _unravelArrays(self):
        # opposite of _ravelArrays: set array back to the original data shape
        # *** parent indices are still stored in 1D, use getParentMap to get indices in ND format ***
        self.edges.shape = self.nSize + (self.nborNum,)
        #self.eValue.shape = self.nSize + (self.nborNum,)
        self.nValue.shape = self.nSize
        if self.parent is not None:
            self.parent.shape   = self.nSize
            self.distance.shape = self.nSize
            if self.distance2 is not None:
                self.distance2.shape = self.nSize
       
    def getParentMap(self):
        # return parenting indices in initial data dimension 
        self._unravelArrays()  # in case it's not already done
        return unravel_indices(self.parent, self.nSize)
        
        
    def getPath(self,destination, stopAt0d=False):
        # return the path to the given destination as a list of pixels position 
        # it requires that the distance and parenting array are computed 
        # using the shortestPath method
        #
        # this method calls getPath1D
        path = self.getPath1D(destination,stopAt0d)
        return unravel_indices(path)

    def getPath1D(self,destination, stopAt0d=False):
        # return the path to the given destination as a list of pixels position 
        # it requires that the distance and parenting array are computed 
        # using the shortestPath method
        #
        # pixel position returned are 1D
        # input position can be 1 or 2D
        
        current = np.asarray(destination,dtype=INDEX_DTYPE)
        if current.size!=1:
            current = ravel_indices(current,self.nSize)
        
        self._ravelArrays()
        parent  = self.parent[current]
        path = [current]
        
        while current!=parent and (not stopAt0d or self.distance[current]!=0)    :
            path.append(parent)
            current = parent
            parent  = self.parent[current]
            
        self._unravelArrays()
        
        return np.array(path)
    
    def getNeighborValue(self,nodes=None, value=None,neighbor=None,default=None):
        """
        Give the values of the 'neighbor' of 'nodes' (or 'default', see below)
        
        :Input:
            nodes: indices of array nodes to look for their neighbors
                   By default, all nodes (in or out of mask)
                   Otherwise, nodes should be a list, tuple or ndarray of 1D indices
                   
            value: array that the returned value are taken from. It is considered flat 
                   By default, the graph node value is taken.
                   
            neighbor: By default, return the value of all neighbors. Otherwise, 
                      it should be a 1D list,tuple or array of neighbor indices
                      
            default: value that is return for all invalid neighbors
                     By default, use numpy NaN. Be careful: NaN cannot 
                     be used if the dtype of value array is not float

        :Output:
            Return an array of shape NxK with N the number of nodes and K the 
            number of neighbors. All neighbors are not valid (out of mask or out
            of value array). In this case, 'default' value used
            
        :Note:
            All indices (nodes and neighbors) should be 1D. For nodes, use 
            unravel_indices() to convert them to 1D. For neighbor, the indices
            should the same as used by the graph.
        """
        if value    is None: value    = self.getNodeValue()
        if neighbor is None: neighbor = slice(None)
        if default  is None: default  = np.NaN
        if nodes    is None: nodes    = slice(None)  # all
        else:
            nodes = np.asarray(nodes)
            if nodes.dtype == 'bool': nodes = nodes.ravel()
            elif nodes.ndim>1:
                nodes = ravel_indices(nodes,value.shape)
        
        self._ravelArrays()
        # get neighbor indices, invalid are converted to valid value (and remove below)
        nbor  = np.mod(self.edges[nodes,neighbor],self.nodeNum) 

        # get neighbor values
        nbor_value = value.ravel()[nbor]
        
        # replace value of invalid neighbors by 'default'
        nbor_value[self.edges[nodes,neighbor]==INDEX_OUT] = default
        
        self._unravelArrays()
        return nbor_value
        
    def getNodesValue(self):
        return self.nValue


@_aleanode({'name':'graph'},description=ArrayGraph.__doc__)
def makeArrayGraph(nodes_value,footprint=3):
    return ArrayGraph(nodes_value,footprint=footprint)

