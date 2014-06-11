"""
Conversion form RootGraph to RootTree

*** IN DEVELOPMENT ***
"""

import numpy as _np

from rhizoscan.root.graph.conversion import segment_to_neighbor as _seg2nbor
from rhizoscan.root.graph.conversion import los_to_neighbor     as _los2nbor
from rhizoscan.root.graph.conversion import neighbor_to_csgraph as _nbor2csgraph
from rhizoscan.root.graph.conversion import neighbor_to_edge_array as _nbor2edge

from rhizoscan.root.graph.conversion import segment_to_digraph as _seg2digraph
from rhizoscan.root.graph.conversion import digraph_to_DAG     as _digraph2dag


def make_tree(graph, axe_selection=[('longest',1),('min_tip_length',10)], init_axes=None):
    """ Construct a RootTree from given RootGraph `graph` """
    segment = graph.segment
    length = segment.length()
    angle = segment.direction_difference()
    src  = (graph.segment.seed>0) 
    
    # graph to DAG
    # ------------
    dag, sdir = graph_to_dag(segment, init_axes=init_axes)

    # tree path convering 
    # -------------------
    parent = minimum_dag_branching(incomming=[nb[0] for nb in dag], cost=angle, init_axes=init_axes)
    top_order = dag_topsort(dag=dag, source=src)
    path_elt,elt_path = tree_covering_path(parent=parent, top_order=top_order, init_axes=init_axes)[:2]

    # dag (mini/optimal) covering
    # ---------------------------
    ##todo: select 1st order, then merge&select 2nd order (etc...?)
    p_length = _np.vectorize(lambda slist:length[slist].sum())(path_elt)
    
    path_elt,elt_path,n,debug = merge_tree_path(dag=dag, top_order=top_order, 
                                   path_elt=path_elt, elt_path=elt_path, 
                                   priority=-p_length,
                                   clean_mask=graph.segment.seed>0)
    
    # Contruct RootTree  ## TODO: finish 
    # -----------------
    # construct AxeList object
    from rhizoscan.root.dev_graph2tree import path_to_axes as p2a
    from rhizoscan.root.graph import RootTree
    graph.segment.parent = parent
    axe = p2a(graph, path_elt, axe_selection=axe_selection)
    
    ##graph.segment.axe = axe.segment_axe                    
    t = RootTree(node=graph.node,segment=graph.segment, axe=axe)
    
    return t
    
class AxeBuilder(object):
    """ use to construct AxeList iteratively """
    def __init__(self, start_id=1):
        self.segment = []
        self.parent  = []
        self.sparent = []
        self.order   = []
        self.plant   = []
        self.ids     = []
        
        self.current_id = start_id
        
    def axe_index(self, axe_id):
        return self.ids.index(axe_id)
        
    def append(self, segment, parent, sparent, order, plant=0, ids=0):
        if ids==0:
            ids = self.current_id
            self.current_id += 1
        
        if plant==0:
            plant = self.plant[self.axe_index(parent)]
        
        self.segment.append(segment)
        self.parent.append(parent)
        self.sparent.append(sparent)
        self.order.append(order)
        self.plant.append(plant)
        self.ids.append(ids)
        
        return ids
        

def make_tree_2(graph, order1='longest', o1_param=1, order2='min_tip_length', o2_param=10, init_axes=None):
    """ Construct a RootTree from given RootGraph `graph` """
    segment = graph.segment
    length = segment.length()
    angle = segment.direction_difference()
    src  = (graph.segment.seed>0) 
    
    # graph to DAG
    # ------------
    dag, sdir = graph_to_dag(segment, init_axes=init_axes)

    # tree path convering 
    # -------------------
    parent = minimum_dag_branching(incomming=[nb[0] for nb in dag], cost=angle, init_axes=init_axes)
    top_order = dag_topsort(dag=dag, source=src)
    path_elt,elt_path,axe_map = tree_covering_path(parent=parent, top_order=top_order, init_axes=init_axes)

    # select init_axes (grown) path
    ##if init_axes:
    ##    map_num = len(axe_map)
    ##    axe_num = init_axes.number()-1
    ##    if map_num!=axe_num:
    ##        missing = set(range(1,axe_num+1)).difference(axe_map.keys())
    ##        print "missing axe in tree covering path: "+str(missing)+" (%d/%d)" % (map_num, axe_num)
    ##
    ##    for ax_ind,path_ind in axe_map.iteritems():
    ##        axe_map[ax_ind] = path_ind[0]  ##! arbitrarily select 1st path
    ##        if len(path_ind)>1:
    ##            print '%d possible path for axe %d' % (len(path_ind), ax_ind) ##

        ## return tmp RootTree for visualization
        #axes = [[] for i in range(init_axes.number())]
        #for ax_ind,path_id in axe_map.iteritems():
        #    axes[ax_ind] = path_elt[path_id]
        #parent = init_axes.parent
        #plant  = init_axes.plant
        #order  = init_axes.order()
        #psegm  = _np.zeros(init_axes.number(),dtype=int)
        #ax_ids = init_axes.set_id()
        #
        #from rhizoscan.root.graph import RootTree
        #from rhizoscan.root.graph import AxeList
        #
        #axe = AxeList(axes, graph.segment, parent=parent, parent_segment=psegm,
        #              plant=plant, order=order, ids=ax_ids)
        #
        #return RootTree(node=graph.node,segment=graph.segment, axe=axe)
        

    # find order 1 axes
    # -----------------
    path_start = [p[0] if len(p) else 0 for p in path_elt]
    path_plant = graph.segment.seed[path_start]
    if order1=='longest':
        o1 = longest_path(path_elt, length=graph.segment.length(),
                          parent=path_plant, number=o1_param,
                          init_axes=init_axes, axe_map=axe_map)
        
    else:
        raise TypeError("unrecognized axe selection method "+str(order1))


    # find order 2 axes
    # -----------------
    
    
    # Contruct RootTree  ## TODO: finish 
    # -----------------
    # construct AxeList object
    from rhizoscan.root.dev_graph2tree import path_to_axes as p2a
    from rhizoscan.root.graph import RootTree
    graph.segment.parent = parent
    axe = p2a(graph, path_elt, axe_selection=axe_selection)
    
    ##graph.segment.axe = axe.segment_axe                    
    t = RootTree(node=graph.node,segment=graph.segment, axe=axe)
    
    return t
    
def graph_to_dag(segment, init_axes=None):
    """ Compute a DAG on "segment graph" `segment` """
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
      add option to only parse (and attach) segments to a already fixed group. 
      I.e. parse edges only if one of its segments is part of a fixed group. 
      Otherwise, the edge parsing is postponed.
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

    # merge "linear" sequence of segments
    # -----------------------------------
    mask = (nbor>0).sum(axis=1)==1
    I,J = mask.nonzero()
    J = nbor[I,0,J]
    cost_mask = cost[I,J]<group_link
    I = I[cost_mask]
    J = J[cost_mask]
    switch = (segment.node[I]==segment.node[J]).any(axis=1)  # see edge_iterator.switch
    for i,j, sw in zip(I,J, switch):
        merge_group(-1, i,j, sw)
            
    # iteration order
    # ---------------
    class edge_iterator(object):
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
    iterator = edge_iterator(edge, cost)


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



def axe_to_path(axe):
    """
    Create a list of path from an AxeList object
    
    All axes in AxeList `axe` are converted into a "tree covering path",
    a list of the axes list of segment id starting at a seed segment.
    In practive:
     - axes which are not connected to any parent gives the their segment list
     - axes with parent gives its segment list with preppend parent start path
       (all segments from seed to the axe parent segment)
       
    axes `parent_segment` should either be part of is `parent` axe, or by `0`
    meaning that the connection is not known.
    """
    pre_seg = {} # segment list to preppend to axe segment list
    
    axe_parent = axe.parent()
    seg_parent = axe.parent_segment()
    for aid in axe.partial_order():
        sparent = seg_parent[aid]
        if sparent==0:
            pre_seg[aid] = []
        else:
            aparent = axe_parent[aid]
            #index of sparent in axe.segment[aparent]
            pass
    
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
      elements, except when passing through inital axes.
      
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
    
    # create a new path starting a element `e`
    def new_path(e):
        path_elt.append([e])
        path_id = len(path_elt)-1
        elt_path[e].add(path_id)
        return path_id
        
    # append element `dst` to path(s) ending at `src`
    axe_map = {}
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
    for e in top_order[::-1]:
        e_path = elt_path[e]
        if len(e_path)==0: 
            new_path(e)
        elif len(e_path.difference(closed_path))==0: 
            continue
        
        if e in init_elt_path:
            # record possible axe path (to do before closed_path update)
            if e in init_path_tip:
                record_axes_map(e)
                
            # if on init_axes, follow selected path 
            init_path = init_elt_path[e][1]
            src = e
            for dst in init_path[len(init_path)-2::-1]:
                append_path(src,dst)
                src = dst
            closed_path.update(elt_path[dst])
            
        else:
            p = parent[e]
            if p!=dummy: append_path(e,p)

    return [p[::-1] for p in path_elt], elt_path, axe_map


def identify_init_path(path_elt, init_axes):
    """ identify which path can be an extension of `init_path` 
    
    :Inputs:
     - path_elt:
         A list of list of element ids for each path, sorted.
     - init_axes:
         An initial AxeList already set. It must have been used to contruct 
         `path_elt` such that a path that contains the last element if an axe
         in `init_axes` also contain all its predecessors.
         Typically, of `path_elt` has been generated with `tree_covering_path`
         that used a `parent` argument constructed with
         ....
    """
    pass

def merge_tree_path(dag, top_order, path_elt, elt_path, priority, clean_mask=None):
    """
    Merge tree covering path on given `dag`, following order of `priority`
    
    :Inputs:
      - dag:
          A direacted acyclic graph as a sided list-of-set graph type
      - top_order:
          List of indices of `dag` element given in topological order
      - path_elt:
          list of list of element in each path of the tree covering path
          See section `Tree covering path` for details
          ** path_elt is edited in place ** 
      - elt_path:
          list of set of path id going through each element of the tree
          See section `Tree covering path` for details
          ** elt_path is edited in place ** 
      - priority:
          A list/array of length equal to the number of path with a value giving
          it priority for merging. The highest priority path are merge first
          See section `Path merging` for details
      - clean_mask:
          optional boolean array indicating element that are insufficient to 
          make a path. If given, path containing only masked elements are 
          removed.
          ## What about invalid path in init_axes ?!
    
    :Tree covering path:
      The tree covering path given by `path_elt` and `elt_path` should:
        - be a graph-covering path set that covers at path start only: 
            p1 = [1,2]
            p2 = [1,3,4]
            p3 = [1,3,5]
        - have at least the last segment of all path not covered by other path
      
    :Path merging:
      Mergin happen if:
        1. a path 'p1' last segment is not terminal: it has at least one 
           outgoing segment 'o1'
        2. a path 'p2' contains 'o1' and all previous segments are covered by at 
           least one other path
           
      Then the end 'p2' starting at 'o1' is appended at the end of 'p1' and
      (the remaining of) 'p2' is removed
    
      Merges is searched for in reverse `top_order`. If at one step, multiple
      merges are possible, the order by which such merges are chosen follows
      given `priority`
    
    :Outputs:
       - updated `path_elt`
       - updated `elt_path`
       - remove path, as a dictionary (path-index,path_elt)
       - a (debug) dictionary giving the merging states that occured at each 
         element of the graph
         
    :todo:
      replace priority by a a better selection method, ex:
       - take into account to relative position (merging direction)
       - take into account estimated/previous order?
      init path should not be allowed to grow too quickly
    """
    # dictionary of (tip_element:path_id) of all path
    path_tip = dict([(p[-1] if len(p) else 0,pid) for pid,p in enumerate(path_elt)])
    
    priority = _np.asarray(priority)
    debug = dict()
    
    # parse dag (from leaves to root) and detect and merge possible path
    for e in top_order[::-1]:
        if len(elt_path[e])==1: 
            debug[e] = 'unique path'
            continue

        child_tip = [(path_tip[c],priority[path_tip[c]],c) for c in dag[e][0] if path_tip.has_key(c)]
        if len(child_tip)==0: 
            debug[e] = 'no ending path on incomming'
            continue
        
        debug[e] = 'merging'  ## no free path/no-endind-on-incoming/merging-possible seems correctly detected
        
        free_path = [(path,priority[path]) for path in elt_path[e]]  # all path in e
        
        child_tip = sorted(child_tip, key=lambda x: x[1], reverse=True)
        free_path = sorted(free_path, key=lambda x: x[1], reverse=True)[1:] # keep "best" path
        
        # 1-to-1 match as many possible tip & path in order of priority
        #    merge= append path p2 starting at e to p1 end
        #           then remove p2
        for (p1,p1p,c),(p2,p2p) in zip(child_tip,free_path):
            merge_pos = path_elt[p2].index(e)
            for s in path_elt[p2][:merge_pos]:
                elt_path[s].remove(p2)
            path_elt[p1] = path_elt[p1] + path_elt[p2][merge_pos:]
            path_elt[p2] = None
            path_tip.pop(c)
        
    # remove empty, and invalid, path
    if clean_mask is not None:
        del_path = dict((i,p) for i,p in enumerate(path_elt) if p is None or _np.all(clean_mask[p]))
    else:
        del_path = dict((i,p) for i,p in enumerate(path_elt) if p is None)
    if 0 in del_path:
        del_path.pop(0)
        
    path_elt = [p for i,p in enumerate(path_elt) if i not in del_path.keys()]
    elt_path = [list(ep.difference(del_path.keys())) for ep in elt_path]
    
    
    return path_elt, elt_path, del_path, debug



def longest_path(path, length, parent, number=1, prev_ids=None):
    """ select the longest `path` of each `parent`
    
    path: list of list of element per path
    length: length of path elements
    parent: integer identifying each path parent (parent=0 are not processed)
    number: the number of path to select per parent
    prev_ids: array of path ids which should be selected first
    
    return a mask array where non-zero item are the selected path, and the 
    value is the path id which taken from `prev_ids` when possible.
    """
    if number!=1:
        raise NotImplementedError("only longest_path with number=1 is implemented")
        
    p_length = _np.vectorize(lambda slist:length[slist].sum())(path)
    parent  = _np.asanyarray(parent)
    
    def masked_argmax(value, mask):
        ind = _np.argmax(value[mask])
        return mask.nonzero()[0][ind]

    if prev_ids is None:
        prev_ids = _np.zeros(p_length.size,dtype=int)
    else:
        prev_ids = _np.asanyarray(prev_ids)
    prev_mask = prev_ids>0

    # select axes
    # -----------
    longest = _np.zeros(p_length.size,dtype=int)
    cur_id  = prev_ids.max()+1
    for par in sorted(set(parent).difference([0])):
        mask = parent==par
        
        # check if longest path is in priority_mask
        pmask = mask&prev_mask
        if pmask.any():
            mask = pmask
            
        # select longest path
        order = masked_argmax(p_length,mask)
        if prev_ids[order]:
            longest[order] = prev_ids[order]
        else:
            longest[order] = cur_id
            cur_id += 1
        
    return longest

def min_path_tip_length(path_elt, elt_path, length, min_length=10):
    """
    Filter path whose tip length are less than `max_length`
    
    the tip is the set of path element that are not covered by any other path
    
    path_elt: list of sorted list of element per path
    elt_path: list of (set of) path going through all elements
    length: length of path elements
    min_length: the minimum length allowed for path tip
    
    return a boolean array
    """
    ptip = path_tip(path_elt,elt_path)
    tip_length = _np.vectorize(lambda elts:length[elts].sum())(ptip)
    return tip_length>=min_length

def path_tip(path_elt, elt_path):
    """
    Return the path tip: last path elements that are not covered by other path
    
    return a list of list of tip elements per path
    """
    path_tip = []
    elt_path_number = _np.vectorize(len)(elt_path)
    for path in path_elt:
        if len(path)==0:
            path_tip.append([])
        else:
            tip_start = elt_path_number[path]==1
            tip_start = tip_start.size - _np.argmin(tip_start[::-1])
            path_tip.append(path[tip_start:])
        
    return path_tip
    
def test_path_tip():
    path = [[],[1,2,3,6],[2,4],[1,7,3,5,8]]
    ep = [[] for i in range(9)]
    for p,elts in enumerate(path):
        for e in elts:
            ep[e].append(p)
    assert path_tip(path,ep)==[[], [6], [4], [5,8]]

def set_downward_segment(graph):
    
    """
    **In-place** turn all segments of RootGraph `graph` downward
    return updated `graph`
    """
    upward = _np.diff(graph.node.y()[graph.segment.node],axis=1).ravel()<0
    graph.segment.node[upward] = graph.segment.node[upward][:,::-1]
    graph.segment.clear_temporary_attribute() # clear precomputed neighbor and possible dependences
    
    return graph

# ploting, mostly for testing purpose
# -----------------------------------
def plot_path(g,path, shift=(.5,5)):
    from scipy.sparse import csr_matrix
    from scipy.sparse import csgraph
    from matplotlib import pyplot as plt
    
    nx=g.node.x(); ny=g.node.y(); sn=g.segment.node
    
    def seglist(p):
        s = _np.bincount(sn[p].ravel()).tolist().index(1)
        cs = csr_matrix((_np.ones(len(p)),zip(*sn[p])), (sn[p].max()+1,)*2)
        return csgraph.depth_first_order(cs,s,directed=0,return_predecessors=0)
    
    g.plot(bg='w', sc='r', linestyle=':')
    shift = _np.arange(-shift[1],shift[1],shift[0])    
    for p in path:
        if len(p)==0: continue
        n = seglist(p)
        s = shift[_np.random.randint(len(shift))]
        h = plt.plot(nx[n]+s,ny[n])[0]
        plt.plot(nx[n[0]]+s,ny[n[0]],'o'+h.get_c())
        