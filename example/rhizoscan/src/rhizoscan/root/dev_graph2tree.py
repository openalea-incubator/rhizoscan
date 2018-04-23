"""
Development of algorithms to convert RootGraph to a RootAxialGraph

It uses mainly graph.py and dev_graph.py


##TODO:
 - make a "virtual" edge graph where too short edges are removed (i.e. their 
   nodes are treated as 1 unique
     * there can be neighboring edges detected thus groups of nodes to merge  
     * if one node can be "suitably" selected: then an actual merge is possible
        ? a node that don't rotate edges too much?
        ? this might create edge between the same node. let merge node 'nm' and
          a nighbor node nn, then edge (nm,nm) might appear, and pairs of (nn,nm)
          
     * or simply "avoid" them in the following algorithm, and add them back
       once axes are computed
        ? how to manage segment-segment edges, and direction_difference
        * detect all node-pairs that are too close to be meaningful
           - maybe remove node-pair that are segments but not "cross segments"
             (cross seg=seg that touch >=2 other on both size) 
           - find cluster of those (connected components of node-pairs graph)
           - create a virtual segment.node where id of detected nodes are 
             replaced by those of the cluster (ex: min of node id in cluster)
           ? check for the computation of direction diff - > is it valid
           - after axe detection, fill axes with missing segment
             ? how...?
             ? what about inexistant segment: create a new SegmentList?
 - in the same ligne: detect "missing" edge: unconnected close nodes for which
   there is a pair of edges (one on each side) that point toward each other
"""
import numpy as _np

from . import dev_graph as _dg
from .graph import AxeList as _AxeList
from .graph import RootTree as _RootTree

# robust graph: virtually remove/add "little" segments
# ----------------------------------------------------
##todo:
## - robust RootGraph, where 
##     * segments are added between close enough nodes
##     * too little segment are merge/removed
## - make a "robust" SegmentList subclass with
##     * .node are ids of "node group", i.e. min(nid) for nid in all merge node group
##     * .base_node are the id of real node
##     * .direction is still computed from base_node
##     * what about length?
## - reconstruction of general RT from AxeList made using "robust" SegmentList
##     * generate new 'smooth curve' (node/segments) between clustered nodes?
##
def make_robust_graph(node,segment, merge_distance):
    """
    IN DEV
    return a RootGraph where nodes that are close enough are merged
    
    the output graph segments have same length and direction as original ones 
    """
    pairs = _close_pairs(node.position.T, max_d=merge_distance)
    
    

def _close_pairs(X,max_d):
    """
    find all pairs of `x`i in `X` that are closer than `max_d`
    
    `X` is a (n,k) array of `n` points in `k` coordinates
    
    outputs is a (m,2) array of m pairs of `X` elements
    """
    from scipy.spatial.distance import cdist
    d = cdist(X,X)

    I,J = (d<max_d).nonzero()
    IJ  = _np.sort(_np.vstack((I,J)), axis=0)

    # remove diagonal element
    IJ  = IJ[:,_np.diff(IJ,axis=0).ravel()!=0]

    # remove duplicate
    dt = _np.dtype([('i',int),('j',int)])
    pairs = _np.unique(IJ.T.view(dtype=dt)).view(int).reshape(-1,2)

    return pairs

def dag_covering_path(incomming, out_going, parent, segment_length, node_order=None):
    """
    Find a path set that cover the dag
    ...
    fails if the directed rootgraph had broken cycles...
        actually, it is `topsort_node` that fails
    """
    #if parent=='minimum_cost':
    #    parent = _dg.minimum_dag_branching(incomming=incomming, cost=cost, invalid=-1)

    # find element with no direct child: no other element have them in parent
    parent_set = set(parent)
    no_dchild  = [i not in parent for i in xrange(len(parent))]
    
    if node_order is None:
        node_order = _dg.topsort_node(incomming=incomming, out_going=out_going)
        
    ## for now, priority is not used: 'minimum length' is assumed
    
    
    # init path data structure
    path_elt = [[]]                                # elements in path
    path_length = [[]]
    elt_path = [[] for i in xrange(len(parent))]   # path in elements
    def new_path(e):
        path_elt.append([e])
        path_length.append(segment_length[e])
        path_id = len(path_elt)-1
        elt_path[e].append(path_id)
        return path_id
        
    def add_to_path(p,e):
        path_elt[p].append(e)
        elt_path[e].append(p)
        path_length[p] += segment_length[e]
        
    # create a path for all graph leaves: element without out_going
    leaves = (_np.vectorize(len)(out_going)==0) & \
             (_np.vectorize(len)(incomming)>0)
    for i in leaves.nonzero()[0]:
        new_path(i)
    
    redirect_count = 0
    
    # compute path
    for parents, children in node_order[::-1]:
        ##print children,parents, [elt_path[c] for c in children]
        if len(parents)==0: continue
        
        p_tip = [p for p in parents if no_dchild[p]]
        if len(p_tip)==0:
            # transfer of path to element's direct parent
            #    no possible "new" path
            for c in children:
                p = parent[c]
                if p<0: continue
                for path in elt_path[c]:
                    add_to_path(path, p)
        else:
            # find "main" path of each child branch, and store remaining
            #    i.e. for all children branches, transfer the "best" path
            #         keep the others for alternate transfer
            free_path = []
            match = dict()
            for c in children:
                p = parent[c]
                if p<0: continue
                    
                c_path = [(path,path_length[path]) for path in elt_path[c]]
                c_path = sorted(c_path,key=lambda x:x[1])
                add_to_path(c_path[-1][0],p)
                free_path.extend(c_path[:-1])
                match.update((path,p) for path,length in c_path[:-1])
            
            # redirect "free path" to branches with no direct child
            free_path = sorted(free_path,key=lambda x:x[1])
            for p,(path,length) in zip(p_tip,free_path):
                add_to_path(path,p)
                match.pop(path)
                redirect_count += 1
            
            # transfer remaining unset path to direct parent
            for path,p in match.iteritems():
                add_to_path(path,p)
            
            # create new path for remaning parent branch
            for p in p_tip[len(free_path):]:
                new_path(p)
                
    return [p[::-1] for p in path_elt], elt_path, redirect_count
            
def tree_path(parent, top_order, invalid=-1):
    """
    Find all the path from graph roots to leaves
    
    :Inputs:
      - parent:
            Id of parent of all element
      - top_order:
            The topological order of the graph
            
    :Output:
      - A list of the list of the segments contained in each path
      - A list of the list of the path going through each segment
    """
    # init path data structure
    path = [[]]                                    # elements in path
    elt_path = [[] for i in xrange(len(parent))]   # path in elements
    def new_path(e):
        path.append([e])
        path_id = len(path)-1
        elt_path[e].append(path_id)
        return path_id
    def merge_path(src, dst):
        src_path = elt_path[src]
        for p in src_path:
            path[p].append(dst)
        elt_path[dst].extend(src_path)
        
    # compute path
    for e in top_order[::-1]:
        if len(elt_path[e])==0: new_path(e)
        p = parent[e]
        if p!=invalid: merge_path(e,p)

    return [p[::-1] for p in path], elt_path
    
def merge_tree_path(incomming, out_going, top_order, path_elt, elt_path, priority, smask, clean_path=True):
    """
    merge path when possible...`
    
    input path: 
     - they should be a graph-covering path set that covers at path start only: 
         p1 = [1,2]
         p2 = [1,3,4]
         p3 = [1,3,5]
     - At least the last segment of all path are not cover by any other path
      
    What is path merging:
      if a path 'p1' last segment is not terminal 
         i.e. it has at least one out going segment 'o1'
      if a path 'p2' contains 'o1' and all previous segments are covered by at 
         least one other path
      then the end 'p2' starting at 'o1' is attached to the end of 'p1'
         and (the remaining of) 'p2' is removed
    
    The order by which such merges are selected 
    
    return updated path_elt, elt_path, and the number of merged path
    """
    # dictionary of (tip_element:path_id) of all path
    path_tip = dict([(p[-1] if len(p) else 0,pid) for pid,p in enumerate(path_elt)])
    
    priority = _np.asarray(priority)
    
    path_elt = path_elt[:]  # copy
    elt_path = [set(path_list) for path_list in elt_path]  # copy, and cvt to list of sets
    
    tmp = dict()
    
    for e in top_order[::-1]:
        if len(elt_path[e])==1: 
            tmp[e] = 'unique path'
            continue

        child_tip = [(path_tip[c],priority[path_tip[c]],c) for c in incomming[e] if path_tip.has_key(c)]
        if len(child_tip)==0: 
            tmp[e] = 'no ending path on incomming'
            continue
        
        tmp[e] = 'merging'  ## no free path/no-endind-on-incoming/merging-possible seems correctly detected
        
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
        
    if clean_path:
        del_path = [i for i,p in enumerate(path_elt) if p is None or _np.all(smask[p])]
        if 0 in del_path: del_path.remove(0)
        del_num  = len(del_path)
        path_elt = [p for i,p in enumerate(path_elt) if i not in del_path]
        elt_path = [list(ep.difference(del_path)) for ep in elt_path]
    else:
        del_num = len([i for i,p in enumerate(path_elt) if p is None])
    
    return path_elt, elt_path, del_num, tmp

def path_to_axes(graph, path, axe_selection=[('longest',1),('min_tip_length',10)]):
    """
    Create an AxeList from a covering path set, selecting path/axe order
    
    `segment` should have its 'parent' suitably set
    
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
        
        if method=='longest':
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
    sOrder = _np.ones(segment.number(),dtype='uint8')*max_order
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
    

# general rgraph to axial tree function
# -------------------------------------
def make_root_tree(graph, axe_selection=[('length',1),('min_tip_length',10)], verbose=False):
    seed = graph.segment.seed
    src  = (graph.segment.seed>0) 
    angle  = graph.segment.direction_difference()
    length = graph.segment.length()
    
    graph.node.set_segment(graph.segment)     ## just in case
    
    # convert graph to DAG
    # --------------------
    gs = graph.segment
    direction = _dg.segment_digraph(gs, gs.direction_difference())[0]
    digraph = graph.segment.digraph(direction)
    e_in, e_out = _dg.digraph_to_DAG(digraph[...,1], length[digraph[...,1]], source=src)[:2]

    # create a tree path convering from select "best" segment parent 
    # --------------------------------------------------------------
    parent = _dg.minimum_dag_branching(incomming=e_in, cost=angle, invalid=0)
    top_order = _dg.topsort(incomming=e_in, out_going=e_out, source=src)
    path,spath = tree_path(parent=parent, top_order=top_order, invalid=0)
    
    #node_order = _dg.topsort_node(incomming=e_in, out_going=e_out, source=src)
    #path, spath, n = dag_covering_path(incomming=e_in,out_going=e_out,
    #                                   parent=parent, segment_length=length, 
    #                                   node_order=node_order)
    m = len(path)
    pLength = _np.vectorize(lambda slist:length[slist].sum())(path)
    
    path,spath,n,tmp = merge_tree_path(incomming=e_in, out_going=e_out, top_order=top_order, 
                                   path_elt=path, elt_path=spath, priority=pLength,
                                   smask=graph.segment.seed>0, clean_path=True)
    
    if verbose:
        print ' axial tree:', n, 'path connected on ', m, '(%2.1f percent)' % (100*float(n)/m) ##
    #return dict(incomming=e_in, out_going=e_out, top_order=top_order, 
    #            path=path, spath=spath, priority=pLength.argsort()[::-1])
    
    # Contruct RootTree
    # -----------------
    # construct AxeList object
    graph.segment.parent = parent
    axe = path_to_axes(graph, path, axe_selection=axe_selection)
    
    ##graph.segment.axe = axe.segment_axe                    
    t = _RootTree(node=graph.node,segment=graph.segment, axe=axe)
    
    return t


def segment_axe_list(axe, segment_number):
    """ list of axe id passing through segments """
    saxe = [[] for i in xrange(segment_number)]
    for aid, slist in enumerate(axe):
        for s in slist:
            saxe[s].append(aid)
    return saxe


# ploting and user interface tools
# --------------------------------
def plot_path(graph, path, value):
    pLength = _np.vectorize(lambda slist:graph.segment.length()[slist].sum())(path)
    
    # find "best" path of each segment
    spath = _np.zeros(graph.segment.node.shape[0], dtype=int)
    for p,slist in enumerate(path):
        if len(slist)==0: continue
        mask = pLength[spath[slist]]<pLength[p]
        spath[[s for s,m in zip(slist,mask) if m>0]] = p
    
    print spath
    print (value[spath]).max()
    graph.plot(bg='k', sc=_np.asarray(value)[spath])

def interactive_path_plot(rgraph, path=None, spath=None):
    """
    To quit, ctrl+c in the python shell
    """
    from .dev_graph import _segdist_env, _segment_distance
    from matplotlib import pyplot as plt
    
    if path is None:
        path  = rgraph.axe.segment
        spath = [[] for i in xrange(rgraph.segment.number())]
        for pid,p in enumerate(path):
            for s in p:
                spath[s].append(pid)
    valid = _np.vectorize(len)(spath)>0
    sc = _np.zeros(rgraph.segment.number(), dtype=int)
   
    env = _segdist_env(rgraph)
    
    if hasattr(rgraph,'axe') and hasattr(rgraph.axe,'order'):
        rgraph.plot(bg='k', sc='order')
    else:
        sc[valid] = 1
        rgraph.plot(bg='k', sc=sc)
        
    while True:
        p = plt.ginput()
        
        axis = plt.axis()
        if len(p):
            d = _segment_distance(p,env)
            sc[:] = 0
            sc[valid] = 1
            for p in spath[d.argmin()]:
                sc[path[p]] += 1
                
            print 'segment id:', d.argmin(), 'axes id:', spath[d.argmin()]
            rgraph.plot(bg='k', sc=sc)
            plt.axis(axis)
        else:
            break


def _axe_distance_to_seed(graph, axe=None, aPlant=None, a2process=None):
    """
    Cumulative distance to seed of all axes
    
    return cumulative distance
    """
    from scipy import ndimage as nd
    
    if axe is None:       axe    = graph.axe.segment
    if aPlant is None:    aPlant = graph.axe.plant
    if a2process is None: a2process = _np.ones(graph.axe.number(),dtype=bool)
    
    segment = graph.segment
    
    # find the position of the seed of each plant
    nseed = _np.vectorize(lambda slist: segment.seed[slist].min() if len(slist) else 0)(graph.node.segment)
    plant_ind = _np.arange(aPlant.max()+1)
    seed_x = nd.mean(graph.node.x(),labels=nseed, index=plant_ind)
    seed_y = nd.mean(graph.node.y(),labels=nseed, index=plant_ind)
    seed_pos = _np.vstack((seed_x,seed_y))                    # shape: 2xSeed#
    
    # find segment which has only one axe passing through
    saxe = segment_axe_list(axe, segment.number())
    uniq_axe = _np.array([(sid,alist[0]) for sid,alist in enumerate(saxe) if len(alist)==1 and a2process[alist[0]]])
    slist = uniq_axe[:,0]
    alist = uniq_axe[:,1]
    
    # compute distance to seed 
    npos  = graph.node.position
    snode = segment.node
    ndist = ((npos[:,snode[slist]]-seed_pos[:,aPlant[alist]][:,:,None])**2).sum(axis=0)**.5  #segment x node_1/2
    sdist = _np.abs(ndist[:,1]-ndist[:,0])
    adist = nd.sum(sdist,labels=alist,index=_np.arange(len(axe)))
            
    return adist
