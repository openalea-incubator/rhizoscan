"""
Package for tracking root architecture

Main steps of roor tracking:
 1. project RootTree `t0` onto RootGraph `g1`
     => Generate a base root tree labeling: RootTree t1 = g U estiamted AxeList
 2. Growth of this initial tree
 
Algorithms used for step 1 are found in the `projection` module
Those of step 2 are in the `growth` module

 - node|segment(t+1) to axe(t) distance
    - distance from node(t+1) to node(t): cdist
    - distance from node(t+1) to root segment(t) - listed by axe
    - distance from node(t+1) to root axes(t) - min d(n,s) per axes
    - input: axialtree_t1, graph-t2
    - output:  distance matrix, ???
 
 - todo: match plant seeds
    - distance from seeds centers
    - match closest 1-to-1
    
 - todo: init first axial tree
 
 - todo: iterative shortest path to find all axes (lower order 1st)
    - start at parent axe (init with leaves)
    - distance sum should be (an approximation of) a suitable curve distance
    - select "best path", max-cover/min-distance
    
 - todo: root "growth":
    - use (& arrange) the graph-2-axial method?
"""
import numpy as _np

from rhizoscan.opencv   import descriptors as _descriptors
from rhizoscan.geometry import transform   as _transform

from .projection import axe_projection

def track_root(dseq, update=False, verbose=True, plot=False):
    """
    TESTING / IN DEV
    
    track root in dataset sequence `dseq`
    
    each item in `dseq` should have attributes graph,tree,key_point,descriptor
    
    use rhizoscan.opencv stuff to estimate affine transform between image
    """
    from .projection import match_seed
    from .projection import node_to_axe_distance
    d0 = dseq[0].load()
    d1 = dseq[1].load()
    
    # image tracking
    if not update and d1.has_key('image_transform'):
        T = d1.image_transform
    else:
        kp0   = d0.key_point
        desc0 = d0.descriptor
        kp1   = d1.key_point
        desc1 = d1.descriptor
        
        T = _descriptors.affine_match(kp0,desc0, kp1,desc1, verbose=verbose)
        d1.image_transform = T

    t = d0.tree
    g = d1.graph

    # make a copy of g with transformed node position 
    gnpos = _transform(T=T, coordinates=g.node.position)
    g_input = g
    g = g.copy()
    g.node = g.node.copy()
    g.node.position = gnpos

    # compute g node to t segment distance
    d,s,p = node_to_axe_distance(g.node.position, t)

    if plot:
        from matplotlib import pyplot as plt
        g.plot(bg='k',sc=(g.segment.seed>0)+1)
        t.plot(bg=None)
        n = gnpos
        I = _np.arange(p.shape[1])  # list of nodes index
        k = _np.argmin(d,axis=1)    # best axe match for each node
        plt.plot([n[0,:],p[0,I,k]],[n[1,:],p[1,I,k]], 'b')
    
    
    # match seed
    seed_match, unmatch_t, unmatch_g = match_seed(t,g)
    
    # get ids of first nodes of t 1st order axe 
    
    return t,g,seed_match
    # match axe 0
    # match axe i>0
    
def load_test_ds():
    """ return tracking dataset for testing purpose """
    from rhizoscan.root.pipeline.dataset import make_dataset
    ds,inv,out = make_dataset('/Users/diener/root_data/nacry/AR570/pando/pando_rsa.ini')
    cds = ds.group_by(['genotype.name','nitrate','plate'],key_base='metadata')
    return cds
    
def test_axe_projection(sds, start=0):
    """ call axe_projection on dataset items in ds - for dev purpose - """
    from rhizoscan.root.pipeline import compute_tree
    from rhizoscan.root.graph.to_tree import graph_to_dag
    from rhizoscan.root.tracking.growth import simple_axe_growth
    from rhizoscan.root.tracking.growth import simple_tip_growth
    
    d1 = sds[start]
    d2 = sds[start+1]
    d1.load()
    d2.load()
    g1 = d1.graph
    g2 = d2.graph
    t1 = compute_tree(g1)
    t2 = axe_projection(t1, g2, d2.image_transform)
    
    dag2, sdir = graph_to_dag(t2.segment, t2.axe) 
    axe2,daxe = simple_axe_growth(dag2, t2.axe) # update t2.axe in place
    mask = set()
    map(mask.update, axe2.segment)
    tip_axes  = simple_tip_growth(dag2, mask=mask)
    
    return g1,t1,g2,t2, sdir, daxe, tip_axes, dag2
    

def mtest_axe_projection(cds, group=0, max_shift=2):
    from scipy.sparse.csgraph import connected_components
    from scipy import ndimage as nd
    from scipy.linalg import inv
    from matplotlib import pyplot as plt
    
    from rhizoscan.root.graph.conversion import neighbor_to_csgraph
    from rhizoscan.root.graph.nsa import AxeList
    
    def unreachable(g):
        """ bool array of segments reachable from seeds """
        nbor = g.segment.neighbors()
        cs = neighbor_to_csgraph(nbor)
        n, lab = connected_components(cs)
        unreachable_lab = _np.ones(n,dtype=bool)
        unreachable_lab[lab[g.segment.seed>0]] = 0
        return unreachable_lab[lab]
        
    
    def uncover(t, tip_axe):
        """ return a array of segment uncovered by any axes:
            0:covered, 1:unreachable, 2:reachable 
        """
        sc = 2-unreachable(t)
        
        for slist in t.axe.segment:
            sc[slist] = 0
        for slist in tip_axe:
            sc[slist] = 0
        
        return sc
    
    start = 0
    while isinstance(group,int):
        g1,t1,g2,t2, sdir, daxes, taxes, dag = test_axe_projection(cds[group], start)
        T1 = cds[group][start].image_transform
        T2 = cds[group][start+1].image_transform
        
        # draw t1
        ax = plt.subplot(2,1,1)
        ax.set_position([0,.51,1,.49])
        sc = unreachable(t1)
        t1.plot(sc=sc, indices=sc&(t1.segment.node!=0).all(axis=1),linestyle=':', linewidth=2, transform=T1)
        t1.plot(bg=None,max_shift=max_shift, linewidth=2, transform=T1)
        
        # draw t2
        ax = plt.subplot(2,1,2, sharex=ax, sharey=ax)
        ax.set_position([0,0,1,.49])
        
        ##sc = _np.zeros(t2.segment.number(),dtype=int)
        ##for i,daxe in enumerate(daxes):
        ##    sc[daxe] = i
        ##t2.plot(sc=7*(sc>0), linewidth=3, transform=T2)
        
        sc = uncover(t2, taxes)
        t2.plot(sc=sc, indices=(sc>0)&(t2.segment.node!=0).all(axis=1),linestyle=':', linewidth=2, transform=T2)
        t2.plot(bg=None,max_shift=max_shift, linewidth=2, transform=T2)
        
        # draw new tip
        taxes = [] + taxes
        tmp = _np.zeros(len(taxes))
        taxe = AxeList(taxes, segment_list=t2.segment,
                       parent=tmp, parent_segment=tmp,
                       plant=tmp, order=tmp)
        t2.axe = taxe
        t2.plot(bg=None,max_shift=2, linewidth=1, transform=T2)
        
        
        g = raw_input('group:')
        if g=='q':
            group = g
        elif g.startswith('+'):
            try:
                start = int(g[1:])
            except:
                print "  Error: unrecognized start '"+g[1:]+"'"
                start = 0
        else:
            try:
                group = int(g)
            except:
                group += 1
            start = 0


