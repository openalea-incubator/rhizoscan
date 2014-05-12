"""
Module to "grow" initial tree axes along unattributed segments

(Might) also manage the detectino of root axes in a RootGraph
"""
import numpy as _np

from rhizoscan.root.graph.conversion import segment_to_neighbor as _seg2nbor
from rhizoscan.root.graph.conversion import los_to_neighbor     as _los2nbor
from rhizoscan.root.graph.conversion import neighbor_to_csgraph as _nbor2csgraph
from rhizoscan.root.graph.conversion import neighbor_to_edge_array as _nbor2edge

from rhizoscan.root.graph.conversion import segment_to_digraph as _seg2digraph
from rhizoscan.root.graph.conversion import digraph_to_DAG     as _digraph2dag


def make_tree(graph):
    """ Construct a RootTree from given RootGraph `graph` 
    
    *** IN DEV ***
    """
    segment = graph.segment
    length = segment.length()
    angle = segment.direction_difference()
    axe = getattr(graph,'axe',None)
    if not hasattr(axe,'segment'): axe = None
    
    # graph to DAG
    # ------------
    # find segment direction
    direction = segment_direction(segment, cost=angle, init_axe=axe)[0]
    
    # construct DAG
    src  = (segment.seed>0) 
    digraph = _seg2digraph(segment, direction)
    dag = _digraph2dag(digraph[...,1], length[digraph[...,1]], source=src)[0]


    # tree path convering 
    # -------------------
    parent = minimum_dag_branching(incomming=[nb[1] for nb in dag], cost=angle)
    top_order = dag_topsort(dag=dag, source=src)
    path_elt,elt_path = tree_covering_path(parent=parent, top_order=top_order, init_axe=axe)


    # dag (minimal) covering
    # ----------------------
    p_length = _np.vectorize(lambda slist:length[slist].sum())(path_elt)
    
    path_elt,elt_path,n,debug = merge_tree_path(dag=dag, top_order=top_order, 
                                   path_elt=path_elt, elt_path=elt_path, 
                                   priority=p_length,
                                   clean_mask=graph.segment.seed>0)
    
    
def simple_axe_growth(dag, axes):
    """
    "Grow" `axes` on `dag` up to any branching **in-place** 
    
    :Inputs:
      - dag:
          A segment graph in sided list-of-set representation
          I.e. a dag[s][1] is the set of forward neighbors of segment `s`
      - axes:
          An AxeList object
          If None, do nothing.
          
    :Outputs:
      - return the updated `axe`
      - the axes growth: a list of appended segment
    """
    if axes is None: return
    
    growth_axes = []
    for i,axe in enumerate(axes.segment):
        if len(axe)==0: 
            growth_axes.append([])
            continue
        
        growth = []
        next = dag[axe[-1]][1]
        while len(next)==1:
            next = next.copy().pop()
            if len(dag[next][0])==1:
                growth.append(next)
                next = dag[next][1]
            else:
                break
        
        growth_axes.append(growth)
        axes.segment[i] = axe+growth
            
    return axes, growth_axes

def simple_tip_growth(dag, mask=set()):
    """
    Create and "grow" axes from `dag` tip (terminal elements) 

    dag tip elements are selected as elements with incomming neighbor(s) but 
    no outgoing one.

    :Inputs:
      - dag:
          A segment graph in sided list-of-set representation
          I.e. a dag[s][1] is the set of forward neighbors of segment `s`
      - mask:
          A set of segment ids that are not to be process
          
    :Outputs:
      - The list tip-axes: a list of sorted segment list
    """
    # detect tip
    tips = [sid for sid in xrange(len(dag)) if len(dag[sid][0])>0 and len(dag[sid][1])==0]
    tips = [tip for tip in tips if tip not in mask]

    # grow tip
    tip_axes = []
    for i,tip in enumerate(tips):
        growth = [tip]
        next = dag[tip][0]
        while len(next)==1:
            next = next.copy().pop()
            if next not in mask and len(dag[next][1])==1:
                growth.append(next)
                next = dag[next][0]
            else:
                break
        tip_axes.append(growth)
            
    return [axes[::-1] for axes in tip_axes]

def segment_direction(segment, cost, init_axes=None, callback=None):
    """
    Choose the "best" direction for all segments
    
    :Inputs:
      - segment:
         A SegmentList instance that contains the attribute `seed`. 
      - cost:
         A SxS array of the cost between segment
         The cost of edge with "fixed" segments is set to 0 **in-place**
      - init_axes:
         AxeList that are already set.
         If given, the direction of the segments of those axes is *fixed* and 
         is selected to follow the segment order of each axe. If a segment 
         appears in several axes, the axe which appears first in the AxeList 
         partial order has priority. 
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
      add option to only parse (and attach) segments to fixed group. I.e. parse
      edges only if one of its segments is part of a fixed group. Otherwise, the
      edge parsing is postponed.
    """
    nbor = segment.neighbors()
    edge = _nbor2edge(nbor)
    
    # edges cost
    # ----------
    # edge with seed segment have zero cost
    cost = cost[edge[:,0], edge[:,1]]
    seed = segment.seed>0
    cost[seed[edge[:,0]] | seed[edge[:,1]]] = 0
    
    # sort by cost
    edge = edge[_np.argsort(cost),:]
    
    
    # variable initialization
    # -----------------------
    sNum  = segment.node.shape[0]
    group = [[i] for i in xrange(sNum)] # segment group (init one group per segment)
    sgid  = _np.arange(sNum)            # group id of segment   ----- " -----
    gset  = _np.zeros(sNum, dtype=bool) # (group) direction selected (init as unset)
    sdir  = _np.zeros(sNum, dtype=bool) # segment direction w.r.t its group (init as same)
    
    # switch: 
    #   For all edges (s1,s2), True if s1 and s2 need to have opposite direction 
    #   in order to be part of same group:
    #      True when  s1 start (resp. end) == s2 start (resp. end)
    switch = (segment.node[edge[:,0]]==segment.node[edge[:,1]]).any(axis=1) 
    
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
                mask = gset[axe]
                maxe = axe[mask]
                sdir[maxe] = axe_sdir[mask]
                gset[maxe] = 1
            
            
    # compute the edge direction
    # --------------------------
    # convert to list for faster 1-element access during iterations
    sdir = sdir.tolist()
    gset = gset.tolist()
    
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


def minimum_dag_branching(incomming, cost):
    """
    Compute the minimum branching on the given DAG
    
    The minimum branching is the equivalent of the minimum spanning tree but for 
    digraph. For DAG, which contains no cycle, the algorithm is trivial:
      For all elements, choose its `incomming` element with minimal cost
      
    See the Chu-Liu/Edmonds' algorithm:
      `http://en.wikipedia.org/wiki/Edmonds'_algorithm`
    
    :Inputs:
      - incomming:
          The incomming edges as a list-of-set type
      - cost:
          A SxS array of the cost between any pair of segments : S = len(incomming)

    :Outputs:
      The selected parent of each graph segment, as an array.
    """
    parent_nbor = _los2nbor(incomming, sided=False)
        
    x = _np.arange(parent_nbor.shape[0])
    y = cost[x[:,None],parent_nbor].argmin(axis=1)
    
    return parent_nbor[x,y]

def dag_topsort(dag, source=None, fifo=True):
    """
    Compute the topological order of a directed acyclic graph `dag`
    
    :Inputs:
      - dag:
          A sided list-of-set segment graph type of the a DAG
      - source:
          Optional list (or boolean mask) of starting elements.
          if None, use the list of elements with no incomming edge.
          *** Note: no path from one source to another should exist ***
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
    
    (*) A. B. Kahn. 1962. Topological sorting of large networks. Com. ACM 5
    """
    from collections import deque
    
    incomming = [nb[0].copy() for nb in dag] # copy
    outgoing  = [nb[1]        for nb in dag]
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
    
def tree_covering_path(parent, top_order, init_path=None, dummy=0):
    """
    Find the path covering of the tree represented by `parent`
    
    :Inputs:
      - parent:
          Id of parent of all element
      - top_order:
          The topological order of the graph
      - init_path: 
          Optional list of path (segment list) which is used as starting base
          ##NOT IMPLEMENTED
            
    :Output:
      - A list of the list of the (sorted) segments contained in each path
      - A list of the set  of the path going through each segment
      
    :Covering path:
      Tree covering path is a set of path that covers all element of the tree.
      Starting at the leaves, a path is created each time an element is not 
      already covered by a path. All path going through the same element
      shares the same path from the root to this element.
    """
    """
    path are iteratively cosntructed from leaves to root (ie. reverse top_order)
    and reverse before being returned
    """
    
    # init path data structure
    path_elt = [[]]                                # elements in path
    elt_path = [[] for i in xrange(len(parent))]   # path going through elements
    
    # create a new path starting a element `e`
    def new_path(e):
        path_elt.append([e])
        path_id = len(path_elt)-1
        elt_path[e].append(path_id)
        return path_id
        
    # append element `dst` to path(s) ending at `src`
    def merge_path(src, dst):
        src_path = elt_path[src]
        for p in src_path:
            path_elt[p].append(dst)
        elt_path[dst].extend(src_path)
        
    # create the covering path
    for e in top_order[::-1]:
        if len(elt_path[e])==0: new_path(e)
        p = parent[e]
        if p!=dummy: merge_path(e,p)

    return [p[::-1] for p in path_elt], map(set,elt_path)

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
          ## What about invalid path in init_axe ?!
    
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
            merge_pos = path_elt[p2].index(e)  ##
            for s in path_elt[p2][:merge_pos]:
                elt_path[s].remove(p2)
            path_elt[p1] = path_elt[p1] + path_elt[p2][merge_pos:]
            path_elt[p2] = None
            path_tip.pop(c)
        
    # remove empty, and invalid, path
    if clean_mask:
        del_path = dict((i,p) for i,p in enumerate(path_elt) if p is None or _np.all(clean_mask[p]))
    else:
        del_path = dict((i,p) for i,p in enumerate(path_elt) if p is None)
    if 0 in del_path:
        del_path.remove(0)
        
    path_elt = [p for i,p in enumerate(path_elt) if i not in del_path.keys()]
    elt_path = [list(ep.difference(del_path.keys())) for ep in elt_path]
    
    
    return path_elt, elt_path, del_path, debug


def path_selection(criteria, parameter):
    """ 
    Create a path selection function used by `path_to_axes`
    """
    pass

def longest_path(path, path_parent, parameter):
    """" select longest path with same parent """
    pass

def path_to_axes(graph, path, priorities):
    """
    Branch path with same start into a hierarchical axe system
    
    :Inputs:
      - graph:
          a RootGraph object
          `graph.segment` should have its 'parent' suitably set
      - path:
          list of ordered list of segment that represent the axes path
      - parent_path:
          id of parent path of all `path`
      - parent_segment
          id of parent segment of all `path`
         
    :Output:
      an AxeList object
    """
    raise NotImplementedError()
    

def path_to_axes_0(graph, path, axe_selection=[('length',1),('min_tip_length',10)]):
    """
    Create an AxeList from a covering path set, selecting path/axe order
    
    :Inputs:
      - graph:
          a RootGraph object
          `graph.segment` should have its 'parent' suitably set
      - path:
          list of order list of segment that represent the covering path
      - axe_selection
         ##...
         
    :Output:
      the constructed AxeList
    
    *** warning: input `path` is changed in-place ***
    """
    from rhizoscan.root.graph import nsa
    
    segment = graph.segment
    axe = path[:]  # to store segment list per axes
    
    sLength = segment.length()
    aLength = _np.vectorize(lambda slist:sLength[slist].sum())(axe)
    aPlant  = segment.seed[[a[0] if len(a) else 0 for a in axe]]
    
    # find axe order
    # --------------
    max_order = len(axe_selection)+1
    aOrder = _np.ones_like(aPlant)*max_order
    
    for order, (method,param) in enumerate(axe_selection):
        order += 1
        
        if method=='length':
            if param==1:
                puid = _np.unique(aPlant)
                if puid[0]==0: puid=puid[1:]
                for plant in puid:
                    aid = _np.argmax(aLength*(aPlant==plant)*(aOrder==max_order))
                    aOrder[aid] = order
                    ##print 'order', order, ': plant', plant, 'axe:', aid
            else:
                raise NotImplementedError('axe order selection with', method, param, 'is not implemented')
                
        elif method=='radius':
            from rhizoscan.stats import cluster_1d
            aRadius = _np.vectorize(lambda slist:segment.radius[slist].mean())(axe)
            aRadius[_np.isnan(aRadius)] = 0
            main_axes = cluster_1d(aRadius, classes=2, bins=256)
            aOrder[main_axes] = order
            
        elif method=='seminal':
            from rhizoscan.stats import cluster_1d
            adist = _axe_distance_to_seed(graph=graph,axe=axe,aPlant=aPlant, a2process=aOrder==max_order)
            if param==1 or param=='max':
                puid = _np.unique(aPlant)
                if puid[0]==0: puid=puid[1:]
                for plant in puid:
                    aid = _np.argmax(adist*(aPlant==plant))
                    aOrder[aid] = order
            else:
                seminal = cluster_1d(adist,classes=2, bins=256)
                aOrder[seminal>0] = order
           
        elif method=='min_tip_length':
            saxe = segment_axe_list(axe, segment.number())
            saxe_num = _np.vectorize(len)(saxe)
            
            # test tip length all axes in order of decreasing length
            sorted_axe = sorted(enumerate(axe),key=lambda x:aLength[x[0]],reverse=1)
            for aid,slist in sorted_axe:
                if aOrder[aid]<max_order: continue
                    
                # find contiguous axe parts that are owned only by this axe 
                # these are considered potential axe tips 
                own_seg   = _np.ma.masked_not_equal(saxe_num[slist],1)
                own_slice = _np.ma.notmasked_contiguous(own_seg)
                if isinstance(own_slice,slice):
                    own_slice = [own_slice]

                # For each tip, test if it has suffisent length
                keep = False
                for tip in own_slice[::-1]:
                    if segment.length()[slist[tip]].sum()>param: 
                        keep = True
                        break
                        
                # if at least one tip is correct, set this axe to required order
                if keep:
                    axe[aid] = slist[:tip.stop]
                    aOrder[aid] = order
                    saxe_num[slist[tip.stop:]] -= 1
                else:
                    saxe_num[slist] -= 1

        elif axe_selection[0]=='all':
            aOrder[:] = order
    aOrder[0] = 0
    
    
    # convert path to axes:
    # ---------------------
        # find axe order of segment: order of the axe passing with lowest order 
    sOrder = _np.ones(segment.seed.size,dtype='uint8')*max_order
    #sOrder[segment.seed>0] = 0 # remove seed from path
    for i,a in enumerate(axe):
        if i==0: continue
        sOrder[a] = _np.minimum(sOrder[a],aOrder[i])
    
        # crop start of path that are also part of an axe with lower order
    for i,a in enumerate(axe):
        if i==0: continue
        start = _np.argmax(sOrder[a]>=aOrder[i]) # 1st element with correct order
        axe[i] = a[start:]
        
        # find parent segment and parent axe  ## not robust!?!
    parent_seg = nsa.parent_segment(axe,segment.parent)
    parent_axe = nsa.parent_axe(axe,parent_seg)
    
    ## for 2ndary+ axe with no parent, set (1st) axe with order-1 as parent axe 
    invalid_parent = [aid for aid, parent in enumerate(parent_axe) if aOrder[aid]>1 and parent==0]
    for aid in invalid_parent:
        pid = aPlant[aid]
        porder = aOrder[aid]-1
        parent = [p for p in range(len(axe)) if aOrder[p]==porder and aPlant[p]==pid]
        parent_axe[aid] = parent[0] if len(parent) else 0 ## if/else not necessary?

    ##    # sort axes by decreasing length (after cropping), for priority in AxeList
    ##aLength = _np.vectorize(lambda slist:segment.length()[slist].sum())(axe)
    ##l_sort  = _np.roll(_np.argsort(aLength)[::-1],1)
    ##axe    = [axe[i]    for i in l_sort]
    ##aOrder = [aOrder[i] for i in l_sort]
    ##aPlant = [aPlant[i] for i in l_sort]

    axelist = _AxeList(axes=axe, segment_list=segment,
                       parent=parent_axe, parent_segment=parent_seg,
                       order=aOrder, plant=aPlant)

    return axelist
    

