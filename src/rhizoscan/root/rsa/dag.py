"""
This module contains:
  `graph_to_dag`
      convert a general "segment" graph into a DAG   
  `segment_direction` 
      select a "best" direction for each graph segment
  `minimum_dag_branching`
      Find the best parent of all segment in a dag 
  `dag_topsort`
      Compute a segment partial order on a dag
  `tree_covering_path`
      Compute a path covering from minimum_dag_branching

See the `rhizoscan.root.graph.conversion` module for the description of the 
graph data structure, and related conversion, used in this module. 
    
"""
import numpy as _np

from rhizoscan.root.graph.conversion import segment_to_digraph     as _seg2digraph
from rhizoscan.root.graph.conversion import digraph_to_DAG         as _digraph2dag
from rhizoscan.root.graph.conversion import neighbor_to_edge_array as _nbor2edge
from rhizoscan.root.graph.conversion import neighbor_to_csgraph    as _nbor2csg
from rhizoscan.root.graph.conversion import los_to_neighbor        as _los2nbor


def graph_to_dag(segment, init_axes=None):
    """ Compute a DAG on "segment graph" `segment`
    
    :Outputs:
      - the dag as a list of set (los) segment graph where `dag[i][io]` is the
        set of incomming (`io=0`) and outgoing (`io=1`) neighbors of segment `i`
      - the selected segment direction
    """
    # find segment direction
    direction = segment_direction(segment, init_axes=init_axes)[0]
    
    # construct DAG
    src  = (segment.seed>0)
    length = segment.length()
    digraph = _seg2digraph(segment, direction)
    dag = _digraph2dag(digraph[...,1], length[digraph[...,1]], source=src)[0]

    return dag, direction

def segment_direction(segment, cost=None, group_link=1.05, init_axes=None, callback=None):
    """
    Choose the "best" direction for all segments
    
    :Inputs:
      - segment:
         A SegmentList instance that contains the attribute `seed`. 
      - cost:
         A SxS array of the cost between segment
         The cost of edge with "fixed" segments is set to 0 **in-place**
         By default use `segment.direction_difference()`
      - group_link:
         Start the algorithm by grouping segment that have single connection 
         (i.e. there is no other segment connected at the same node), and that
         have `cost` less than `group_link` value.
      - init_axes:
         optional AxeList which are already set.
         If given, the direction of  those axes segments is *fixed* and selected
         to follow the segment order of each axe. If a segment appears in 
         several axes, the first the `init_axes.partial_order()` has priority. 
      - callback:
         A function that is call at each iteration of the algorithm such as::
        
           callback(i, sdir, sgid, s1,s2)
           
         with `i` is the iteration number, `sdir` is the current segment 
         direction, `sgid` is the current group id of the segments and `s1`and 
         `s2` are the ids of the processed segments.
       
    :Output:
      - A boolean array of the direction of each segment.
        `False` means as given by input `edge`, `True` means switched direction.
      - An array of the segment's group id
      
    :Algorithm:
      - The direction of terminal and seed segments are choosen and fixed. If
        `init_axes` is given, the same is done for the segments in those axes.
      - The algorihtm then parse all edges iteratively and group these segments
        together. The parsing follows the order of lowest `cost` priority 
        (between the segments pairs). Grouped segments have a fixed direction 
        relative to the group. And when a group change direction it does so 
        as a whole: all segments it contains switch direction. If a group 
        contains fixed segments (seed and/or termnial), it is also fixed:
        it cannot change direction anymore.
      - The algorithm terminates when all segments have been parsed. If all
        connected components of the segment graph contain at least one fixed
        segments, then a fixed direction is selected for all segments.
        
    :todo:
      - implement path merging using a class
      - add option to only parse (and attach) segments to a already fixed group. 
        I.e. parse edges only if one of its segments is part of a fixed group. 
        Otherwise, the edge parsing is postponed.
      - use cost* that depend cost to all neighbors it can connect to.
          cost*(i) = sum(cost(s_i,s_j)) for s_j neighbors of s_i
          Q: what about unset neighbor? don't count them?
          Q: use min cost per path side? it should be prop^al to were path goes:
             - axial nbor "weight" max
             - branching: out path (perpendicular?) weight little
      - weight cost by path length
    """
    nbor = segment.neighbors()
    edge = _nbor2edge(nbor)
    
    # variable initialization
    # -----------------------
    sNum  = segment.node.shape[0]
    group = [[i] for i in xrange(sNum)] # segment group (init one group per segment)
    sgid  = _np.arange(sNum)            # group id of segment   ----- " -----
    gset  = _np.zeros(sNum, dtype=bool) # (group) direction selected (init as unset)
    sdir  = _np.zeros(sNum, dtype=bool) # segment direction w.r.t its group (init as same)
    seed  = segment.seed>0
    
    if cost is None:
        cost = segment.direction_difference()

    # Set direction of initial segments
    # ---------------------------------
    # seed & terminal segments
    #   - seed segment should "point" toward the root system
    #   - terminal segment should should "point" outward the root system 
    #     note: seed must be set after terminal as they also are terminal
    term  = segment.terminal()
    sdir[term] = nbor[term][:,:,1].any(axis=1)  # default means no nbor on side 1
    sdir[seed] = nbor[seed][:,:,0].any(axis=1)  # default means no nbor on side 0
    gset[term] = 1
    gset[seed] = 1
    
    # set direction of initial axes
    if init_axes:
        snode = segment.node
        for aid in init_axes.partial_order():
            gset[0] = 0      # just in case  (required if |axe_seg|=1 & pseg=0)
            axe_seg = init_axes.segment[aid]
            
            # empty axe
            if len(axe_seg)==0:
                continue
                
            # axe with 1 segment
            elif len(axe_seg)==1:
                if gset[axe_seg[0]]:
                    # segment dir already set
                    continue
                    
                pseg = init_axes.parent_segment[aid]
                if not gset[pseg]:
                    Warning('Cannot find axe {} segments direction: |axe|=1 and parent segment unset'.format(aid))
                    continue
                    
                sw = (snode[axe_seg[0]]==snode[pseg]).any(axis=1)
                sdir[axe_seg[0]] = sdir[pseg]^sw # pdir=0&switch or pdir=1&no switch
                    
            # general case
            else:
                seg0_dir = (snode[axe_seg[0]][0]==snode[axe_seg[1]]).any()       # s1.n0 touch s2 ##what if s1&s2 loop?
                dir_diff = (snode[axe_seg[:-1]]==snode[axe_seg[1:]]).any(axis=1) # like switch
                axe_sdir = _np.cumsum([seg0_dir]+dir_diff.tolist())%2
            
                # set axe segment dir, if not already set
                mask = gset[axe_seg]
                maxe = [sid for sid in axe_seg if gset[sid]]
                sdir[maxe] = axe_sdir[mask]>0
                gset[maxe] = 1

    
    # convert to list for faster 1-element access during iterations
    sdir = sdir.tolist()
    gset = gset.tolist()
    
    # merge function
    # --------------
    # set segments in same group
    def merge_group(i,s1,s2,sw):
        g1 = sgid[s1]
        g2 = sgid[s2]
        if g1==g2: return
        
        g1set = gset[g1]
        g2set = gset[g2]
        # don't merge if not possible: both group dir set in unfit direction
        if g1set and g2set and ((sdir[s1]==sdir[s2])==sw):
            return
    
        # reverse s1,s2 s.t. merging should be g2 into g1
        if (g1set<g2set)  or  ((g2<g1) and (g1set<g2set)):
            g2set,g1set = g1set,g2set
            g2,g1 = g1,g2
            s2,s1 = s1,s2
        
        # merge g2 into g1
        g2s = group[g2]
        sgid[g2s] = g1
        group[g1].extend(g2s)
        
        # switch g2 segments direction if necessary
        if (sdir[s1]==sdir[s2])==sw:
            for s in g2s: sdir[s] = not sdir[s]
            
        if callback: callback(i, sdir, sgid, s1,s2)

    # merge sequence of segments w/out branching
    # ------------------------------------------
    mask = (nbor>0).sum(axis=1)==1
    I,J = mask.nonzero()
    J = nbor[I,0,J]
    cost_mask = cost[I,J]<group_link
    I = I[cost_mask]
    J = J[cost_mask]
    switch = (segment.node[I]==segment.node[J]).any(axis=1)  # see EdgeIterator.switch
    for i,j, sw in zip(I,J, switch):
        merge_group(-1, i,j, sw)
            
    # iterator
    # --------
    class EdgeIterator(object):
        """ class that implement the iteration """
        def __init__(self, edge, cost):
            # sort by cost
            edge = edge[_np.argsort(cost),:]
            self.edge = edge.tolist()
            
            # switch: 
            #   For all edges (s1,s2), True if s1 and s2 need to have opposite 
            #   direction in order to be part of same group:
            #      True when  s1 start (resp. end) == s2 start (resp. end)
            switch = (segment.node[edge[:,0]]==segment.node[edge[:,1]]).any(axis=1)
            self.switch = switch.tolist()
            self.current = 0
            
        def __iter__(self):
            return self
        def next(self):
            i = self.current
            if i>=len(self.edge): raise StopIteration
            self.current = i+1
            s1,s2 = self.edge[i]
            return i, s1,s2, self.switch[i]

    # edge with seed segment have zero cost
    cost = cost[edge[:,0], edge[:,1]]
    cost[seed[edge[:,0]] | seed[edge[:,1]]] = 0
    iterator = EdgeIterator(edge, cost)


    # compute the edge direction
    # --------------------------
    for i,s1,s2,sw in iterator:
        merge_group(i,s1,s2,sw)
        
    return _np.array(sdir), sgid


def minimum_dag_branching(incomming, cost, init_axes=None):
    """
    Compute the minimum branching on the given DAG
    
    The minimum branching is the equivalent of the minimum spanning tree but for 
    digraph. For DAG, which contains no cycle, the algorithm is trivial:
      For all elements, choose its `incomming` element with minimal cost
      
    See the Chu-Liu/Edmonds' algorithm:
      `http://en.wikipedia.org/wiki/Edmonds'_algorithm`
    
    :Inputs:
      - incomming:
          The incomming edges as a list-of-set type (*)
      - cost:
          A SxS array of the cost between any pair of segments : S = len(incomming)
      - init_axes:
         optional AxeList which are already set.
         If given, the parent of segments on those axes are selected as the 
         previous segment on the respective axes. If their are multiple such
         parent/axes the selection follow `init_axes.partial_order()`.

      (*) See `rhizoscan.root.graph.conversion` doc
    
    :Outputs:
      The selected parent of each graph segment, as an array.
    """
    parent_nbor = _los2nbor(incomming, sided=False)
        
    x = _np.arange(parent_nbor.shape[0])
    y = cost[x[:,None],parent_nbor].argmin(axis=1)
    
    parent = parent_nbor[x,y]
    
    if init_axes is not None:
        for aid in init_axes.partial_order()[::-1]:
            parent[init_axes.segment[aid][1:]] = init_axes.segment[aid][:-1]
            
    return parent

def least_curvature_tree(outgoing, source, angle, length, init_axes=None):
    """
    find a tree that minimize path curvature
    
    Compute parent element using shortest path using curvature as edge cost. 
    The curvature of edge (e1,e2) is computed as::
    
        c(e1,e2) = angle(e1,e2)/length(e2)
        
    It is null for edge starting at a source element
    
    :Inputs:
      outgoing:
        A sided list-of-set graph of the forward neighbor of all elements
      source
        Indices of bool array indicating the possible sources of tree axes 
      angle
        2d array of the angles difference between all elements
      length
        Length of all elements
      init_axes:
        Optional AxeList of previously found axes.
        The parent are taken for all segment of all axes following partial order
        
    :Outputs:
      - array of the parent indices (0 for unset)
      - array of lest cumulative curvature at all elements 
    """
    # curvature neighbor graph
    nbor = _los2nbor(outgoing, sided=False)

    # cost
    x = _np.arange(nbor.shape[0])[:,None]
    a = angle[x,nbor]
    c = a*(length[:,None]+length[nbor])/2           # local curvature
    
    return shortest_path_tree(neighbors=nbor, source=source, cost=c, init_axes=init_axes)

def shortest_axe_tree(outgoing, source, length, init_axes=None):
    """
    Call shortest_path_tree with segment length as cost
    
    :Inputs:
      outgoing:
        A sided list-of-set graph of the forward neighbor of all elements
      source
        Indices of bool array indicating the possible sources of tree axes 
      length
        Length of all elements
      init_axes:
        Optional AxeList of previously found axes.
        The parent are taken for all segment of all axes following partial order
        
    :Outputs:
      - array of the parent indices (0 for unset)
      - array of path length at all elements 
    """
    from rhizoscan.ndarray import add_dim
    # curvature neighbor graph
    nbor = _los2nbor(outgoing, sided=False)

    # cost
    c = add_dim(length.copy(), axis=-1, size=nbor.shape[-1])
    
    return shortest_path_tree(neighbors=nbor, source=source, cost=c, init_axes=init_axes)
    

def shortest_path_tree(neighbors, source, cost, init_axes=None):
    """
    Compute the shortest path tree minimizing given `cost`
    
    Compute parent element using shortest path using curvature as edge cost. 
    The curvature of edge (e1,e2) is computed as::
    
        c(e1,e2) = angle(e1,e2)/length(e2)
        
    It is null for edge starting at a source element
    
    :Inputs:
      neighbor:
        A nieghbor-type graph of the forward neighbor of all elements
      source
        Indices of bool array indicating the possible sources of tree axes 
      cost
        2d array of the cost of all neighbors (same shape as neighbors)
        *** cost of source elements are set to 0 in-place *** 
      init_axes:
        Optional AxeList of previously found axes.
        The parent are taken for all segment of all axes following partial order
        
    :Outputs:
      - array of the parent indices (0 for unset)
      - array of lest cumulative cost at all elements 
    """
    from scipy.sparse.csgraph import dijkstra
    
    # source
    source = _np.asarray(source)
    if source.dtype==bool:
        source = source.nonzero()[0]
    cost[source,:] = 0

    G = _nbor2csg(neighbors, value=cost)
    
    # shortest path
    D,P = dijkstra(G, indices=source, return_predecessors=True)
    
    best = D.argmin(axis=0)
    parent = P[best,_np.arange(best.size)]
    parent[parent<0] = 0
    
    path_cost = D[best,_np.arange(best.size)]
    
    if init_axes is not None:
        for aid in init_axes.partial_order()[::-1]:
            parent[init_axes.segment[aid][1:]] = init_axes.segment[aid][:-1]
            
    return parent, path_cost
    
def dag_topsort(dag, source=None, fifo=True):
    """
    Compute the topological order of a directed acyclic graph `dag`
    
    :Inputs:
      - dag:
          A sided list-of-set segment graph type representing the DAG (*)
      - source:
          Optional list (or boolean mask) of starting elements.
          if None, use the list of elements with no incomming edge.
          *** Note: no path from one source to another should exist ***
      - fifo:
          If True, use a first-in-first-out queue data structure. Otherwise 
          use a last-in-first-out (LIFO) queue. 
          In most cases, topsort is not unique. Using a fifo makes the topsort
          follow some kind of breath-first-order, while a lifo makes it follow
          some kind of depth-first-order.
            
      (*) See `rhizoscan.root.graph.conversion` doc
    
    :Output:
        The DAG topological order as a list of element ids.
        If given `source` does not allow to reach all elements, then the 
        unreachable one are not included in the returned list.
    
    :Reference:
        A. B. Kahn. 1962. Topological sorting of large networks. Com. ACM 5
    """
    from collections import deque
    
    incomming = [nb[0].copy() for nb in dag] # copy
    out_going = [nb[1]        for nb in dag]
    parents   = map(list,incomming)          # copy & convert to list
    
    # list of elements id to start topological sorting
    if source is not None:
        source = _np.asarray(source)
        if source.dtype==bool:
            source = source.nonzero()[0]
    else:
        source = (_np.vectorize(len)(incomming)==0).nonzero()
        
    # prepare queue 
    queue = deque(source)
    if fifo: append_to_queue = queue.appendleft 
    else:    append_to_queue = queue.append 
    
    # sort
    order   = []        # stores ordered element ids 
    while queue:
        e = queue.pop()
        order.append(e)
        
        for n in out_going[e]:
            incomming[n].remove(e)
            if len(incomming[n])==0:
                append_to_queue(n)
    
    return order

def tree_covering_path(parent, top_order, init_axes=None, dummy=0):
    """
    Find the path covering of the tree represented by `parent`
    
    :Inputs:
      - parent:
          `parent[e]` is the parent element of element `e`
      - top_order:
          The topological order of the elements in given tree
      - init_axes:
          Optional axes which are already defined on the graph.
          A dictionary of possible path-to-init_axes map is returned
          See the 'initial axes' section below.
            
    :Output:
      - A list of the list of the (sorted) elements contained in each path
      - A list of the set  of the path going through each elements
      - A dict of (axe-index,list-of-possible-path)
      
    :Algorithm:
      A tree covering path is a set of path that covers all tree elements.
      The algorithm process all tree elements in reverse order (starting at the
      graph leaves). A path is created each time a processed element is not 
      already covered by a path. Otherwise the element is added to all path 
      constructed until the element children (i.e. such that parent[child]=e)
      
      As consequence, all path that share an element, share also all previous 
      elements, except when passing through initial axes.
      
    :Initial axes:
      If initial axes are given (`init_axes`), then when a constructed path 
      reaches an element of such an initial axe, it will follow it (backward)
      until reaching the first element of its further ancestor
      
      Hypothesis are done on given axes:
        - `init_axe.parent_segment` are part of the parent axe, or 0.
        - if multiple axes pass through a segment_parent, the parent axe appears
          first in the initial axes partial order
          
      Expected behaviors:
        - if a new path/axe merge on a segment of an initial axes, it will 
          follow the axe that appears first in the axes partial order
    """
    """
    path are iteratively cosntructed from leaves to root (ie. reverse top_order)
    Path are then reversed before being returned
    """
    
    # construct equivalent path of init_axes
    ##init_path_elt = {}
    init_path_tip = {}  # dict (tip-element, [axes-indices])
    init_elt_path = {}  # keep the "main" init axe for each covered element
    if init_axes is not None:
        for ax_ind in init_axes.partial_order():
            segment = init_axes.segment[ax_ind]
            sparent = init_axes.parent_segment[ax_ind]
            
            ##print ax_ind, sparent, segment
            # if init axe has a parent segment
            if sparent>0:
                segment = init_elt_path[sparent][1] + segment  #! expect parent axe (see hypothesis) 
            ##init_path_elt[ax_ind] = segment
            
            # record tip-2-axes
            init_path_tip.setdefault(segment[-1],[]).append(ax_ind) ## path is not necessary the axes!!

            # record axe index and path for all segments 
            # for which no path have been recorded yet
            for i,s_ind in enumerate(segment):
                if not init_elt_path.has_key(s_ind):
                    init_elt_path[s_ind] = (ax_ind, segment[:i+1])

    # init path data structure
    path_elt = [[]]                                 # elements in path
    elt_path = [set() for i in xrange(len(parent))] # path going through elts
    closed_path = set()                       # path not to be processed anymore
    axe_map = {}
    
    # create a new path starting a element `e`
    def new_path(e):
        path_elt.append([e])
        path_id = len(path_elt)-1
        elt_path[e].add(path_id)
        return path_id
        
    # append element `dst` to path(s) ending at `src`
    def append_path(src, dst):
        src_path = elt_path[src].difference(closed_path)
        for p in src_path:
            path_elt[p].append(dst)
        elt_path[dst].update(src_path)
        
    # path at e are possible axe of init_axes
    def record_axes_map(e):
        e_path = elt_path[e].difference(closed_path)  ##difference(closed)?
        for a in init_path_tip[e]:
            axe_map.setdefault(a,[]).extend(e_path)
        
    # create the covering path
    #   parse all element in inverse top order
    #    - if no axe exist, create one
    #    - add path to the current element parent
    for e in top_order[::-1]:
        
        e_path = elt_path[e]
        if len(e_path)==0: 
            # if no path already exist, create one
            new_path(e)
            
        elif len(e_path.difference(closed_path))==0: 
            # if all path are closed, continue to next element
            continue
        
        if e in init_elt_path:
            # all path reaching an init_axes follows it til the seed
            #   then is set as "closed"
            
            if e in init_path_tip:
                # record as possible axe (to do before closed_path update)
                record_axes_map(e)
                
            # if on init_axes, follow selected path
            init_path = init_elt_path[e][1]
            src = e
            for dst in init_path[:-1][::-1]:
                append_path(src,dst)
                src = dst
            closed_path.update(elt_path[dst])
            
        else:
            p = parent[e]
            if p!=dummy: append_path(e,p)

    return [p[::-1] for p in path_elt], elt_path, axe_map

    

