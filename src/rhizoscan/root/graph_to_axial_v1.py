"""
Initial version of axial tree construction from root graph

*** Some old code is copied, but not made to work ***
"""

@_node('tree')
def graph2axial(graph, to_tree=0, to_axe=0, single_order1_axe=True):
    """
    graph: a RootGraph instance (with NodeList and SegmentList)
    to_tree, to_axe: method to use
    single_order1_axe: if True, keep only the longest order 1 axe, other are set to 2nd order
    """
    seed = graph.segment.seed
    ##seed[_np.any(self.segment.node==0,axis=1)] = _UNREACHABLE  ###BUG!!!
    sid = _np.arange(self.segment.size+1)
    pid = _nd.minimum(self.sid,seed,_np.arange(seed.max()+1))  # plant(seed) unique id
    sid[seed>0] = self.pid[seed[seed>0]]
    pid = self.pid[1:]
    
    stype   = _np.zeros(self.sid.size,dtype=int)    # 0:UNSET
    stype[seed>0] = _SEED
    stype[seed<0] = _UNREACHABLE
    
    # to finish
    make_tree(graph, method=to_tree)
    find_axes(graph, method=to_axe, single_order1_axe=single_order1_axe)
    #self.set_axes(single_order1_axe=single_order1_axe)

    return tree ## ?

# OLD CODE TO BE REVWRITEN
# ========================
def make_tree(self, method): 
    """
    method: 1 for length, 2 for direction, 3 for both
    """
    # compute cost to compute shortest-path-tree
    tree_cost  = 1
    if method&1: tree_cost  = self.sgraph.makeDistanceMap('nodes')
    if method&2: tree_cost *= 1 - _np.cos(self.dtheta) + 2**-20
    
    # compute shortest tree
    self.sgraph.shortestPath(_np.nonzero(self.stype==_SEED)[0],distanceMap=tree_cost)#,distanceMap2=self.sgraph.makeDistanceMap('nodes'))
    self.segment.order  = _np.argsort(self.sgraph.distance)  ## make it a tmp property, computed using segment.parent ?

    # set segment parent id, and update segment type
    self.segment.parent = self.sid[self.sgraph.parent]
    self.stype[self.stype>=_UNSET]     = _UNSET
    self.segment.parent[self.stype==_SEED]    = 0   # seeds have no parent
    self.stype[self.segment.parent==self.sid] = _UNREACHABLE
    self.segment.parent[self.stype==_UNREACHABLE] = 0

def find_axes(self, method, single_order1_axe=True):
    """
    method = 1 for length, 2 for direction, 3 for both
    method = 'arabido' => use find_arabido_axes
    """
    if method=='arabido':
        self.find_arabido_axes()
        return
        
    # find terminal segment
    sterm = self.segment.terminal
    
    # compute cost/gain to compute shortest-tree/axe label respectively
    axial_gain = 1
    if method&1: axial_gain  = self.sgraph.makeDistanceMap('nodes')
    if method&2: axial_gain *= 1 + _np.cos(self.dtheta)

    gain = self.segment.length.copy()#_np.zeros(self.sid.size)
    gain[0] = 0
    for i in self.segment.order:
        #if self.segment.parent[i]!=0 and gain[i]==0:
        #    set_dist(i)
        if self.segment.parent[i]!=0:
            p = self.segment.parent[i]
            j = _np.argmax(self.edges[p,:]==i)
            gain[i] += gain[p] + axial_gain[p,j]
    
    # axes of segments: 
    # -----------------
    #    init: unique id for all terminal segments
    self.segment.axe   = _np.cumsum(sterm)*sterm
    self.segment.axe[self.stype==_UNREACHABLE] = _UNREACHABLE
    self.segment.axe[self.stype==_SEED] = _SEED
    axes = self.segment.axe
    
    # for all unset segments, select it iteratively starting further away from seeds
    index = self.segment.order[::-1]  # indices of segments in decreasing order for their distance
    #index = _np.argsort(gain)[::-1]        
    index = index[axes[index]==0]     # keep only reachable segments with no axes selected  

    for i in index:
        nb = self.edges[i]
        nb = nb[nb>0]
        nb = nb[self.segment.parent[nb]==i]
        
        if nb.size==0:
            axes[i]=axes.max()+1
            ##print 'new axe needed:', i,axes[i]
        else:
            j = _np.argmax(gain[nb])
            axes[i] = axes[nb[j]]
            gain[i] = gain[nb[j]]
            
    self.axial_gain = gain
    self.stype[axes>_UNSET] = _SET
    
    self.set_axes(single_order1_axe=single_order1_axe)
    
def set_axes(self,s_axe=None, s_parent=None, a_segment=None, single_order1_axe=True):
    """ 
    Finalise creation of root axe - automatically called by find_axes
    
    :Input:
      - s_axe:
          array of the "main" axe id of all segment   - main if there is overlapping
          if None, use self.segment.axe
      - s_parent:  
          the parent segment of all segment
          if None, use self.segment.parent
      - a_segment: 
          list, for all axes, of the their (sorted) segment list
          if None, compute it considering the graph does not contain loop
        single_order1_axe: 
          if True, keep only the longest axe touching a seed as the main axe 
    """
        
    # manage axe & parent arguments
    if s_axe is None: s_axe = self.segment.axe
    else:             self.segment.add_property('axe',    s_axe)
    if s_parent is None: s_parent = self.segment.parent
    else:                self.segment.add_property('parent', s_parent)
    
        
    # make the axe structure
    # ----------------------
    self.axe = AxeList()
    axe = self.axe
    axe.set_segment_list(self.segment)
    
    # set axe.segment, or compute it
    if a_segment is None:
        a_segment = [[] for i in range(s_axe.max()+1)]
        direct_child = (s_axe==s_axe[s_parent])
        ends = _np.setdiff1d(self.sid,s_parent[direct_child])  # segment with no parent
        ends = ends[s_axe[ends]>0]                             # don't process segment with unset axes (unreachable, seed?)
        for sid in ends:
            aid = s_axe[sid]
            slist = []
            while s_axe[sid]==aid:
                slist.append(sid)
                sid = s_parent[sid]
            a_segment[aid].extend(slist[::-1])
        
    axe.segment = _np.array(a_segment,dtype=object)
    
    # compute the axe length and arc length of segment w.r.t their axe
    # ----------------------------------------------------------------
    arc_length = _np.zeros_like(self.segment.length)
    axe.length = _np.zeros(len(axe.segment))
    ##axe.size   = _np.zeros(len(axe.segment),dtype=int)   ## does not respect GraphList standart: size is the number of elements (-1) !
    for i,slist in enumerate(axe.segment):
        if len(slist)==0: continue
        slist = _np.asarray(slist)
        arcL = _np.cumsum(self.segment.length[slist])
        main_axe = self.segment.axe[slist]==i            # if axis are overloaping, update
        arc_length[slist[main_axe]] = arcL[main_axe]     # arc length if axe i is the segment "main" axe 
        axe.length[i] = arcL[-1]
        ##axe.size[i]   = len(arcL)
    self.segment.add_property('axelength',arc_length)
    
    # compute the axes parent, order, and plant id 
    # --------------------------------------------
    axe.parent = _np.array([_UNREACHABLE if len(sl)==0 else s_axe[s_parent[sl[0]]] for sl in axe.segment])
    order = _np.zeros(len(axe.segment),dtype=int)
    plant = _np.zeros(len(axe.segment),dtype=int)
    
    o1 = (axe.parent==_SEED).nonzero()[0]
    order[o1] = 1
    plant[o1] = [self.segment.seed[s_parent[slist[0]]] for slist in axe.segment[o1]]
    
    if single_order1_axe:
        plant_id = _np.unique(plant[o1])
        plant_id = plant_id[plant_id>0]
        for pid in plant_id:
            main = (plant==pid).nonzero()[0]
            if len(main)>1:
                second = main[axe.length[main]<_np.max(axe.length[main])]
                order[second] = 2
    
    ax_unset = (order==_UNSET) & (axe.parent>0)
    unset_tot = -1
    while _np.any(ax_unset) and (unset_tot!=_np.sum(ax_unset)):
        unset_tot = _np.sum(ax_unset)
        #print order
        #print axe.parent
        p_order = order[axe.parent[ax_unset]]
        p_plant = plant[axe.parent[ax_unset]]
        order[ax_unset] = _np.choose(p_order==_UNSET,(p_order+1,_UNSET))
        plant[ax_unset] = p_plant
        ax_unset = (order==_UNSET) & (axe.parent>0)
        
    axe.order = order
    axe.plant = plant


def find_arabido_axes(self): 
    """ IN DEVELOPMENT: find axes allowing overlap, for arabido model """
    sterm  = self.segment.terminal
    sseed  = self.segment.seed
    parent = self.segment.parent
    d2seed = self.segment.distance_to_seed

    sseed[sseed>=254] = 0     ## there is some bug that make unreachable 254 instead of 0
    
    # construct all axes
    # ------------------
    s_axe = [[] for i in range(sseed.size)]  # list of axes for each segment
    a_seg = [[0]]                            # list of segments for each axe
    a_pid = [ 0 ]                            # plant id (seed) of each axe
    s_axe[0] = [0]                           #    bg(0) axe = [bg segment]
    
    # First pass: find all (order 1) axes that cover all segments
    #   process terminal segments first, then from further away
    #   for each (not already set) segment, store the path to the seed 
    order = _np.argsort(d2seed + d2seed.max()*sterm)[::-1] 
    for sid in _np.arange(sseed.size)[order]:
        if len(s_axe[sid])>0 or sseed[sid]>0: 
            continue
        cur_axe = len(a_seg)   # id of new axe
        ax_seg  = []
        while sseed[sid]==0:
            ax_seg.append(sid)
            sid = parent[sid]
            if sid==0: break
            
        if sid>0: # new axe only if it reach somewhere
            # add seed segment to axe (?)
            ##ax_seg.append(sid)
            # add axe id to all path segments
            for s in ax_seg:
                s_axe[s].append(cur_axe)
            # append axe
            a_seg.append(ax_seg)
            a_pid.append(sseed[sid])
            
    a_seg[0] = [0]
    a_pid    = _np.array(a_pid)
    
    # compute length of axe
    slength = self.segment.length
    a_length = _np.vectorize(lambda sid: slength[sid].sum())(a_seg)
    
    # set the main axe of each segment as the longest that pass through it
    for sid, alist in enumerate(s_axe):
        if len(alist)==0: 
            s_axe[sid] = 0
        else:
            s_axe[sid] = alist[_np.argmax(a_length[alist])]
    s_axe = _np.array(s_axe)
    
    # for each plant id, find the longest axe and make it primary
    ##   redondant with the single_order1_axe of set_axes...
    a_order = _np.ones(a_pid.size)*2
    a_pid[a_pid>=254] = 0     ## there is some bug that make unreachable 254 instead of 0
    upid = _np.unique(a_pid)
    for pid in upid[upid>0]:
        mask = a_pid==pid
        main = mask.nonzero()[0][_np.argmax(a_length[mask])]          
        a_order[main] = 1
    
    # remove all segments that are in one of the main axe (order 1
    #    and inverse list order to be seed to tip
    #reduce_axe = lambda slist: slist[_np.cumprod(a_order[s_axe[slist]]>1)]
    #a_seg = _np.vectorize(reduce_axe)(a_seg)
    for aid, slist in enumerate(a_seg):
        slist   = _np.array(slist)
        slist   = slist[a_order[s_axe[slist]]==a_order[aid]]
        #sl_mask = (_np.cumprod(a_order[s_axe[slist]]==a_order[aid])>0)
        #slist   = slist[sl_mask]
        
        a_seg[aid] = slist[::-1] #[sid for i,sid in enumerate(slist) if sl_mask[i]]
        
    a_seg = _np.array(a_seg)
    
    s_axe[sseed>0] = _SEED
    ################################
    ## adapted copy of set_axes... #
    ################################
    s_parent = self.segment.parent
    self.segment.add_property('axe',    s_axe)
    self.axe = AxeList()
    self.axe.set_segment_list(self.segment)
    self.axe.segment = _np.array(a_seg,dtype=object)
    axe = self.axe
    
    # compute the axe length and arc length of segment w.r.t their axe
    # ----------------------------------------------------------------
    arc_length = _np.zeros_like(self.segment.length)
    axe.length = _np.zeros(len(axe.segment))
    ##axe.size   = _np.zeros(len(axe.segment),dtype=int)   ## doesn not respect GraphList standart: size is he number of elements (-1) !
    for i,slist in enumerate(axe.segment):
        if len(slist)==0: continue
        arcL = _np.cumsum(self.segment.length[slist])
        main_axe = self.segment.axe[slist]==i            # if axis are overloaping, update
        arc_length[slist[main_axe]] = arcL[main_axe]     # arc length if axe i is segment "main" axe 
        axe.length[i] = arcL[-1]
        ##axe.size[i]   = len(arcL)
    self.segment.add_property('axelength',arc_length)
    
    # compute the axes parent, order, and plant id 
    # --------------------------------------------
    ## w.r.t set_axes: here seed axes are part or axes
    #     thus parent[axe_seg[0]] is always 0
    axe.parent = _np.array([_UNREACHABLE if len(sl)==0 else s_axe[s_parent[sl[0]]] for sl in axe.segment])
    ##plant = self.segment.seed[[sl[0] for sl in t.axe.segment]]
    order = _np.zeros(len(axe.segment),dtype=int)
    plant = _np.zeros(len(axe.segment),dtype=int)
    
    o1 = (axe.parent==_SEED).nonzero()[0]
    order[o1] = 1
    plant[o1] = [self.segment.seed[s_parent[slist[0]]] for slist in axe.segment[o1]]
    ##o1 = plant>0
    ##order[o1] = 1
    ##plant[o1] = [self.segment.seed[s_parent[slist[0]]] for slist in axe.segment[o1]]
    
    if 1: ##single_order1_axe:
        plant_id = _np.unique(plant[o1])
        plant_id = plant_id[plant_id>0]
        for pid in plant_id:
            main = (plant==pid).nonzero()[0]
            if len(main)>1:
                second = main[axe.length[main]<_np.max(axe.length[main])]
                order[second] = 2
    
    ax_unset = (order==_UNSET) & (axe.parent>0)
    unset_tot = -1
    while _np.any(ax_unset) and (unset_tot!=_np.sum(ax_unset)):
        unset_tot = _np.sum(ax_unset)
        #print order
        #print axe.parent
        p_order = order[axe.parent[ax_unset]]
        p_plant = plant[axe.parent[ax_unset]]
        order[ax_unset] = _np.choose(p_order==_UNSET,(p_order+1,_UNSET))
        plant[ax_unset] = p_plant
        ax_unset = (order==_UNSET) & (axe.parent>0)
        
    axe.order = order
    axe.plant = plant
    

