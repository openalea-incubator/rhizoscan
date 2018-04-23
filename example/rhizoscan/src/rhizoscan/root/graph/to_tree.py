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

    return path_elt, elt_path
    
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
    def __init__(self, graph, path_elt, elt_path, start_id=1):
        self.graph = graph
        
        # stores path 
        self.path_elt = path_elt
        self.elt_path = elt_path
        self._count_path()
        
        # stores axe (and properties)
        self.axes    = [None]  # index of path of axes
        self.start   = [0]     # axe first segment in path
        self.parent  = [0]     # index of axe parent
        self.sparent = [0]     # parent segment index
        self.order   = [0]     # axe order
        self.plant   = [0]     # axe plant
        self.ids     = [0]     # axe id
        
        self.current_id = start_id # counter of axe id
        
    def _count_path(self):
        self.path_count = _np.vectorize(len)(self.elt_path)
        
    def path_index_of(self, path, elt):
        """ return index of `elt` in `path` """
        try:
            return self.path_elt[path].index(elt)
        except ValueError:
            raise TypeError('element %d is not in path %d' % (elt,path))
    
    def path_at(self, elt, cutable=False):
        """ return path going through element `elt` 
        
        If cutable=True: return only path for which element previous to `elt`
        can be remove: where there are other path going through  
        """
        path = self.elt_path[elt]
        
        if cutable:
            pelt     = self.path_elt
            pcount   = self.path_count
            index_of = self.path_index_of
            path = [p for p in path if (pcount[pelt[p][:index_of(p,elt)]]>1).all()]
            
        return path
        
    def path_plant(self):
        start = [p[0] if len(p) else 0 for p in self.path_elt]
        return self.graph.segment.seed[start]

    def path_tip(self, uncovered=False):
        """ return the path tip
        
        If uncovered is True, return the list of list of uncovered tip elements
        If False, reutnr the list of path tip element, or 0 for empty path
        """
        if uncovered:
            path_tip = []
            for elts in self.path_elt:
                if len(elts)==0:
                    path_tip.append([])
                else:
                    tip_start = self.path_count[elts]==1
                    if tip_start.all():
                        path_tip.append(elts)
                    else:
                        tip_start = tip_start.size - _np.argmin(tip_start[::-1])
                        path_tip.append(elts[tip_start:])
        else:
            path_tip = [elts[-1] if len(elts) else 0 for elts in self.path_elt]
            
        return path_tip

    def empty_path(self):
        """ return a boolean mask of empty path """
        return _np.vectorize(len)(self.path_elt)==0
        
    def remove_path(self, index):
        """ emtpy path `index` """
        for e in self.path_elt[index]:
            self.elt_path[e].discard(index)
            self.path_count[e] -= 1
        self.path_elt[index] = []

    def cut_path(self, path, elt):
        """ remove `path` element previous to `e` """
        elt_ind = self.path_index_of(path,elt)
        for e in self.path_elt[path][:elt_ind]:
            self.elt_path[e].remove(path)
            self.path_count[e] -= 1
        self.path_elt[path] = self.path_elt[path][elt_ind:]
        
    def merge_path(self, path1, path2):
        """ append `path2` at the end of `path1` (and remove `path2`) """
        self.path_elt[path1] += self.path_elt[path2]
        for elt in self.path_elt[path2]:
            self.elt_path[elt].remove(path2)
            self.elt_path[elt].add(path1)
        self.path_elt[path2] = []
        
        
    def axe_index(self, axe_id):
        return self.ids.index(axe_id)
        
    def new_id(self):
        """ return a new axe id """
        id = self.current_id
        self.current_id += 1
        return id
        
    def add_axe(self, path_index, order, plant=0, id=0, parent_id=0):
        """ add path `path_index` to axe """
        if id==0: id = self.new_id()
        segment = self.path_elt[path_index]

        if plant==0:
            segment = self.path_elt[path_index]
            plant = self.graph.segment.seed[segment[0]]
                
        if order==1:
            parent  = 0
            sparent = 0
            start = 0
        else:
            def non_overlap(path1,path2):
                """ return index of 1st element of path1 that is not in path2 """
                path_cmp = map(lambda x,y: x!=y, path1,path2)
                if True in path_cmp:
                    return path_cmp.index(True)
                else:
                    return 0
                
            segment = self.path_elt[path_index]
            start_parent = {}
            if parent_id:
                possible_parent = [self.ids.index(parent_id)]
            else:
                possible_parent = self.path_indices(order=1)
                
            for parent_ind in possible_parent:
                s = non_overlap(segment,self.path_elt[parent_ind])
                start_parent[s] = parent_ind
            start  = max(start_parent.keys())
            
            # record axe id of parent
            if parent_id:
                parent = possible_parent[0]            
            elif start:
                pp_ind = start_parent[start]
                parent = [i for i,pind in enumerate(self.axes) if pind==pp_ind][0]
            else:
                parent = 0 ## parent of secondary axe emerging from seed???
            
            # remove overlap
            sparent = segment[start-1] if start else 0
            
        self.axes.append(path_index)
        self.parent.append(parent)
        self.sparent.append(sparent)
        self.start.append(start)
        self.order.append(order)
        self.plant.append(plant)
        self.ids.append(id)
        
        ##print 'axe added: path:%d with axe id:%d' % (path_index,id)
        return id
        
    def append(self, segment, parent_id, sparent, order, path_index, plant=0, id=0):
        if id==0:
            id = self.current_id
            self.current_id += 1
        
        if plant==0:
            if parent_id:
                plant = self.plant[self.axe_index(parent_id)]
            else:
                plant = self.graph.segment.seed[segment[0]]
        
        self.segment.append(segment)
        self.parent.append(parent_id)
        self.sparent.append(sparent)
        self.order.append(order)
        self.plant.append(plant)
        self.ids.append(id)
        
        self.path_ind.append(path_index)
        ##print 'append path:%d as axe id:%d' % (path_index,id)
        
        return id
    
    def update(self, path_elt):
        for i,path in enumerate(self.path_ind):
            if path is None: continue
            elts = path_elt[path]
            if elts!=self.segment[i]:
                ##print 'update axe id:%d, path:%d' % (i, path) 
                self.segment[i]=elts

    def get_segment(self, axe_id):
        return self.segment[self.axe_index(axe_id)]
        
    def path_indices(self, order='all'):
        if order=='all': return self.axes
        
        ind = [k for k in range(len(self.ids)) if self.order[k]==order]
        return [self.axes[i] for i in ind]
        
    def make(self):
        """ Construct the AxeList """
        from rhizoscan.root.graph import AxeList
        from rhizoscan.root.graph import RootTree
        segments = (self.path_elt[i] if i else [] for i in self.axes)
        segments = [seg[start:] for seg,start in zip(segments,self.start)] 
        
        axe = AxeList(axes=segments, segment_list=self.graph.segment, 
                      parent=_np.array(self.parent), 
                      parent_segment=_np.array(self.sparent),
                      plant=_np.array(self.plant), 
                      order=_np.array(self.order), 
                      ids=_np.array(self.ids))
        
        return RootTree(node=self.graph.node,segment=self.graph.segment, axe=axe)

class PathSet(object):
    """ manage set of path """
    def __init__(self, path_elt, elt_path):
        self.path_elt = path_elt
        self.elt_path = elt_path
        self._count_path()
        
    def _count_path(self):
        self.path_count = _np.vectorize(len)(self.elt_path)
        
    def path_index_of(self, path, elt):
        """ return index of `elt` in `path` """
        try:
            return self.path_elt[path].index(elt)
        except ValueError:
            raise TypeError('element %d is not in path %d' % (elt,path))
    
    def path_at(self, elt, cutable=False):
        """ return path going through element `elt` 
        
        If cutable=True: return only path for which element previous to `elt`
        can be remove: where there are other path going through  
        """
        path = self.elt_path[elt]
        
        if cutable:
            pelt     = self.path_elt
            pcount   = self.path_count
            index_of = self.path_index_of
            path = [p for p in path if (pcount[pelt[p][:index_of(p,elt)]]>1).all()]
            
        return path
        
    def path_tip(self, uncovered=False):
        """ return the path tip
        
        If uncovered is True, return the list of list of uncovered tip elements
        If False, reutnr the list of path tip element, or 0 for empty path
        """
        if uncovered:
            path_tip = []
            for elts in self.path_elt:
                if len(elts)==0:
                    path_tip.append([])
                else:
                    tip_start = self.path_count[elts]==1
                    if tip_start.all():
                        path_tip.append(elts)
                    else:
                        tip_start = tip_start.size - _np.argmin(tip_start[::-1])
                        path_tip.append(elts[tip_start:])
        else:
            path_tip = [elts[-1] if len(elts) else 0 for elts in self.path_elt]
            
        return path_tip

    def cut_path(self, path, elt):
        """ remove `path` element previous to `e` """
        elt_ind = self.path_index_of(path,elt)
        for e in self.path_elt[path][:elt_ind]:
            self.elt_path[e].remove(path)
            self.path_count[e] -= 1
        self.path_elt[path] = self.path_elt[path][elt_ind:]
        
    def merge_path(self, path1, path2):
        """ append `path2` at the end of `path1` (and remove `path2`) """
        self.path_elt[path1] += self.path_elt[path2]
        for elt in self.path_elt[path2]:
            self.elt_path[elt].remove(path2)
            self.elt_path[elt].add(path1)
        self.path_elt[path2] = []
        
    def clean(self, mask=None):
        if mask is not None:
            del_path = set(i for i,p in enumerate(self.path_elt) if len(p)==0 or mask[p].all())
        else:
            del_path = set(i for i,p in enumerate(path_elt) if len(p)==0)
        del_path.discard(0)
            
        self.path_elt = [p for i,p in enumerate(self.path_elt) if i not in del_path]
        self.elt_path = [pset.difference(del_path) for pset in self.elt_path]
        self._count_path()
        
        return del_path
        


def make_tree_2(graph, order1='longest', o1_param=1, order2='min_tip_length', o2_param=10, prune=15, init_axes=None):
    """ Construct a RootTree from given RootGraph `graph` """
    from rhizoscan.misc import printError
    from rhizoscan.root.graph import RootTree
    
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

    # make axe builder structure
    # --------------------------
    builder = AxeBuilder(graph, path_elt, elt_path,
                         start_id=1 if init_axes is None else init_axes.set_id().max()+1)

    # prune "fake" path
    # -----------------
    empty_path = builder.empty_path()
    slength = length.copy()
    slength[graph.segment.seed>0] = 0
    def min_tip():
        ptip = builder.path_tip(True)
        tip_len = _np.array([slength[slist].sum() for slist in ptip])
        tip_len[empty_path] = _np.inf
        path_min = tip_len.argmin()
        ##print path_min, tip_len[path_min]
        return path_min, tip_len[path_min]
        
    while True:
        path_min, tip_len = min_tip()
        if tip_len>prune: break
        builder.remove_path(path_min)
        empty_path[path_min] = True
          

    def non_overlap(path1,path2):
        """ return index of 1st element of path1 that is not in path2 """
        return map(lambda x,y: x!=y, path1,path2).index(True)


    # select (grown) path for init_axes ##todo: improve simplistic method
    # ---------------------------------
    if init_axes:
        map_num = len(axe_map)
        axe_num = init_axes.number()-1
        
        # rm pruned path
        empty_path = builder.empty_path()
        for ax_ind,path_ind in axe_map.iteritems():
            path = [p for p in path_ind if not empty_path[p]]
            if len(path)==0:
                printError("[TRK] all path removed for axe:"+str(ax_ind)+" (rm:"+str(path_ind)+")")
                axe_map.pop(ax_ind)
            else:
                axe_map[ax_ind] = path
            
        if map_num!=axe_num:
            missing = set(range(1,axe_num+1)).difference(axe_map.keys())
            printError("missing axe in tree covering path: "+str(missing)+" (%d/%d)" % (map_num, axe_num),stack=0)
    
        for ax_ind,path_ind in axe_map.iteritems():
            axe_map[ax_ind] = path_ind[0]  ##! arbitrarily select 1st path
            if len(path_ind)>1:
                printError('%d possible path for axe %d' % (len(path_ind), ax_ind), stack=0)

        # add mapped path to builder
        init_id = init_axes.set_id()
        for axe_index in init_axes.partial_order():
            parent = init_axes.parent[axe_index]
            parent = init_id[parent]
            order  = init_axes.order()[axe_index]
            plant  = init_axes.plant[axe_index]
            path_ind = axe_map[axe_index]
                
            builder.add_axe(path_index=path_ind, order=order, plant=plant,
                            id=init_id[axe_index], parent_id=parent)

    # find order 1 axes
    # -----------------
    path_plant = builder.path_plant()
    if order1=='longest':
        selected = builder.path_indices(order=1)
        o1 = longest_path(path_elt, graph=graph, parent=path_plant, 
                          number=o1_param, selected=selected)
        for path_ind in set(o1):##.difference(selected):
            ##print 'new axe', path_ind, builder.current_id
            builder.add_axe(path_index=path_ind, order=1, 
                            plant=path_plant[path_ind])
            
    else:
        raise TypeError("unrecognized axe selection method "+str(order1))

    # merge axes
    # ----------
    # segment direction in radian
    # (corrected by selected direction)
    seg_dir = (segment.direction()+(2-sdir)*_np.pi)%(2*_np.pi)
    merge_tree_path_3(builder, dag, top_order, seg_dir, length)
    
    # add all remaining path as 2nd order axes
    # ----------------------------------------
    o1_path = builder.path_indices(order=1)
    for path_ind in set(range(len(path_elt))).difference(o1_path):
        if len(path_elt[path_ind]):
            builder.add_axe(path_ind, order=2)
            
    return builder.make()

  
def make_tree(graph, axe_selection=[('longest',1),('min_tip_length',10)], init_axes=None):
    return make_tree_2(graph=graph, order1=axe_selection[0][0], o1_param=axe_selection[0][1],
                                    order2=axe_selection[1][0], o2_param=axe_selection[1][1])

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

    
def merge_tree_path_3(builder, dag, top_order, sdirection, slength):
    """
    Merge tree covering path in builder on given `dag`
    
    sdirection: the dag segment direction in **[0,2pi]** radian
    """
    debug = dict()
    
    # dictionary of (tip_element,path_id) of all path
    path_tip = dict((tip,pid) for pid,tip in enumerate(builder.path_tip()))
    
    # to keep track of path order 
    path_history = {} # angle diff and elt length from tip to current element
    
    # parse dag (from leaves to root) and detect and merge possible path
    existing_path = builder.path_indices()
    for elt in top_order[::-1]:
        # update path_history
        # -------------------
        edir = sdirection[elt]
        for out_elt in dag[elt][1]:
            # angle diff
            dangle = sdirection[out_elt]-edir
            dangle = min(dangle, 2*_np.pi-dangle)
            path = set(builder.path_at(out_elt))
            #path.intersection_update(mergeable_path)
            for p in path:
                path_history.setdefault(p,[]).append((dangle,slength[out_elt], out_elt))
        
        # mergeable paths (on elt)
        # ------------------------
        mergeable_path = builder.path_at(elt, cutable=True)
        mergeable_path = set(mergeable_path).difference(existing_path)
        
        if len(mergeable_path)==0:
            debug[elt] = 'unique path'
            continue

        # incomming path
        # --------------
        incomming_path = [(c,path_tip[c]) for c in dag[elt][0] if path_tip.has_key(c)]
        
        if len(incomming_path)==0: 
            debug[elt] = 'no incomming path'
            continue
            
        # sort mergeable path
        # -------------------
        # note: path tip are not in path_angle, 
        #   but these does not reach here in practice
        mergeable_hist = dict((p,path_history[p][::-1]) for p in mergeable_path)
        mergeable_path = sorted(mergeable_path,key=mergeable_hist.get)
        mergeable_hist = sorted(mergeable_hist.values())
        ##print '\n'.join([''.join(['(%1.2f,%3.2f,%d)' % h if h else 'None' for h in hist]) for hist in mergeable_hist])
        
        # match mergeable to incomming w.r.t segment direction
        # ----------------------------
        in_dir = {}
        for in_elt in dag[elt][0]:
            # angle diff
            dangle = sdirection[in_elt]-edir
            dangle = min(dangle, 2*_np.pi-dangle)
            in_dir[in_elt] = dangle
        sorted_in = sorted(in_dir.keys(), key=in_dir.__getitem__) 
        
        
        
        # merge incomming & mergeable path
        # --------------------------------
        for (c,p1),p2 in zip(incomming_path,mergeable_path):
            ##print '---merge path', p1, p2, 'at', elt,'---'
            ##print builder.path_elt[p1]
            ##print builder.path_elt[p2]
            builder.cut_path(p2,elt)
            builder.merge_path(p1,p2)
            
        debug[elt] = 'merged: '+str(zip(incomming_path,mergeable_path))
        
    return debug

    
def merge_tree_path_2(dag, top_order, path_elt, elt_path, priority, fixed_path=[], clean_mask=None):
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
        - have at least the last segment of all path not covered by another path
      
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
    path_set = PathSet(path_elt,elt_path)
    
    # dictionary of (tip_element,path_id) of all path
    path_tip = dict((tip,pid) for pid,tip in enumerate(path_set.path_tip()))
    
    priority = _np.asarray(priority)
    debug = dict()
    
    # parse dag (from leaves to root) and detect and merge possible path
    for elt in top_order[::-1]:
        # find set of path (on elt) that can be merged
        # --------------------------------------------
        mergeable_path = path_set.path_at(elt, cutable=True)
        mergeable_path = set(mergeable_path).difference(fixed_path)
        
        if len(mergeable_path)==0:
            debug[elt] = 'unique path'
            continue

        # find incomming path to merge with free path
        incomming_path = [(path_tip[c],c) for c in dag[elt][0] if path_tip.has_key(c)]
        
        if len(incomming_path)==0: 
            debug[elt] = 'no incomming path'
            continue
        
        
        # 1-to-1 match and merge incomming & free path in order of priority
        mergeable_path = sorted(mergeable_path, key=lambda p:  priority[p])
        incomming_path = sorted(incomming_path, key=lambda ip: priority[ip[0]])
        for (p1,c),p2 in zip(incomming_path,mergeable_path):
            ##print '---merge path', p1, p2, 'at', c, elt,'---'
            path_set.cut_path(p2,elt)
            path_set.merge_path(p1,p2)
            ##path_tip.pop(c)
            
        debug[elt] = 'merge: '+str([(p1,p2) for (p1,c),p2 in zip(incomming_path,mergeable_path)])
        
    
    # remove empty, and invalid, path
    del_path = None#path_set.clean(mask=clean_mask)
    
    return path_set.path_elt, path_set.elt_path, del_path, debug



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
        - have at least the last segment of all path not covered by another path
      
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
    # dictionary of (tip_element,path_id) of all path
    path_tip = dict([(p[-1] if len(p) else 0,pid) for pid,p in enumerate(path_elt)])
    
    priority = _np.asarray(priority)
    debug = dict()
    
    
    # parse dag (from leaves to root) and detect and merge possible path
    for e in top_order[::-1]:
        # find set of path (on e) that can be merged
        # ------------------------------------------
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
    del_path.pop(0,None)
        
    path_elt = [p for i,p in enumerate(path_elt) if i not in del_path.keys()]
    elt_path = [list(ep.difference(del_path.keys())) for ep in elt_path]
    
    return path_elt, elt_path, del_path, debug



merge_tree_path = merge_tree_path_2
def longest_path(path, graph, parent, number=1, selected=[], masked=[]):
    """ select the longest `path` of each `parent`
    
    path: list of list of element per path
    graph: RootGraph on which path are defined
    parent: integer identifying each path parent (parent=0 are not processed)
    number: the number of path to select per parent
    
    selected: optional list of already selected path
    masked: optional list of path not to select
    
    return a list of selected path indices
    """
    if number!=1:
        raise NotImplementedError("only longest_path with number=1 is implemented")
        
    length = graph.segment.length()
    p_length = _np.vectorize(lambda slist:length[slist].sum())(path)
    parent  = _np.asanyarray(parent)
    
    def masked_argmax(value, mask):
        ind = _np.argmax(value[mask])
        return mask.nonzero()[0][ind]

    masked_path = _np.ones(p_length.size,dtype=bool)
    masked_path[masked] = 0

    # select axes
    # -----------
    longest = selected[:]
    for par in sorted(set(parent).difference([0])):
        mask = (parent==par)&masked_path
        
        if not mask[selected].any():
            order = masked_argmax(p_length,mask)
            longest.append(order)
        
    return longest

def min_path_tip_length(path, graph, parent, min_length=10, selected=[], masked=[]):
    """
    select path whose tip length are less than `min_length`
    
    the tip is the set of path element that are not covered by any other path
    
    path: list of sorted list of element per path
    graph: RootGraph on which path are defined
    parent: not used
    min_length: the minimum length allowed for path tip
    selected: optional list of already selected path
    masked: optional list of path not to select
    
    return a list of selected path indices (as a 1d numpy array)
    """
    ptip = path_tip(path)
    length = graph.segment.length().copy()
    length[graph.segment.seed>0] = 0
    tip_length = _np.vectorize(lambda elts:length[elts].sum())(ptip)

    selection = tip_length>=min_length
    selection[selected] = 1
    selection[masked] = 0
    
    return selection.nonzero()[0]

def path_tip(path):
    """
    Return the path tip: last path elements that are not covered by other path
    
    return a list of list of tip elements per path
    """
    path_tip = []
    elt_num  = max(map(max,filter(len,path)))+1
    elt_path_number = _np.zeros(elt_num,dtype=int)
    for p in path:
        elt_path_number[p]+=1
        
    for p in path:
        if len(p)==0:
            path_tip.append([])
        else:
            tip_start = elt_path_number[p]==1
            if tip_start.all():
                path_tip.append(p)
            else:
                tip_start = tip_start.size - _np.argmin(tip_start[::-1])
                path_tip.append(p[tip_start:])
        
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
        