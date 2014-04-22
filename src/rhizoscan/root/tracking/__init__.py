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
    ds = ds.group_by(['genotype.name','nitrate','plate'],key_base='metadata')[0]
    ds[0].load()
    ds[1].load()
    return ds
    
def test_axe_projection(ds):
    """ call axe_projection on dataset items in ds - for dev purpose - """
    from .projection import axe_projection
    return axe_projection(ds[0].tree, ds[1].graph, ds[1].image_transform)#, interactive=True)


