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

from rhizoscan import geometry as _geo
from .projection import axe_projection

def track_root(dseq, verbose=True, **kargs):
    """
    TESTING / IN DEV
    
    track root in dataset (sorted) sequence `dseq`, with:
     dseq[0].tree
     dseq[i>0].graph
     dseq[i].image_transform
    
    **kargs to be passed to pipeline
    
    use load_test_ds('simple') for a simple example
    """
    from rhizoscan.root.graph.to_tree import graph_to_dag
    ##from rhizoscan.root.graph.to_tree import set_downward_segment
    from rhizoscan.root.tracking.growth import simple_axe_growth
    from rhizoscan.root.graph.to_tree import make_tree_2
    
    from rhizoscan.root.pipeline.arabidopsis import pipeline
    
    from rhizoscan.misc import printWarning
    
    # retrieve dseq[0], and run pipeline
    d1 = dseq[0].copy().load(attempt=True)
    pipeline.run(namespace=d1,verbose=verbose, **kargs)
    
    if not d1.has_key('image_transform'):
        printWarning("1st item of dataset has no 'image_transform': use identity")
    
    t1 = make_tree_2(d1.graph)
    # add axe.id if not already set (deprecated?)
    if not t1.axe.has_key('id'):
        t1.axe.id = _np.arange(t1.axe.number())
        d1.dump()
    
    for i,d2 in enumerate(dseq[1:]):
        i +=1 
        
        if verbose:
            print 'root tracking of '+d1.__key__+' > '+d2.__key__
            
        # load d2
        d2 = d2.copy().load(attempt=True)
        pipeline.run(namespace=d2,verbose=verbose, **kargs)
        
        if not d2.has_key('image_transform'):
            printWarning("item %d of dataset has no 'image_transform': eye identity" % i)
        g2 = d2.graph
        
        #project d1.tree axes onto d2.graph
        I = _np.eye(3)
        T = _geo.dot(_geo.inv(d1.get('image_transform',I)), d2.get('image_transform',I))
        t2 = axe_projection(t1, g2, T)
    
        #"simple" axe growth  ## add as option of axe_projection?
        dag2, sdir = graph_to_dag(t2.segment, t2.axe) 
        axe2,daxe = simple_axe_growth(dag2, t2.axe) # update t2.axe in place
        
        t2 = make_tree_2(g2, init_axes=t2.axe)
        
        #todo: "simple" add new branch
        
        #todo: store d2.tree_trk
        d2.set('tree_trk',t2,store=1)
        d2.dump()
        
        # d2.tree_trk become d1.tree for next iteration
        d1 = d2
        t1 = t2

def plot_track_root(dseq):
    from matplotlib import pyplot as plt
    
    ac='id'
    i = 1
    while True:
        d1 = dseq[i-1].copy().load()
        d2 = dseq[i].copy().load()
        t1 = d1.get('tree_trk')
        if t1 is None:
            print '*',
            t1 = d1.tree
        plt.subplot(1,2,1)
        t1.plot(ac=ac, max_shift=4)
        
        t2 = d2.get('tree_trk')
        plt.subplot(1,2,2)
        if t2 is None:
            plt.cla()
        else:
            t2.plot(ac=ac, max_shift=4)
        
        k = raw_input(d1.__key__+' > '+d2.__key__+':')
        if k=='q':
            return
        elif k=='o':
            ac = 'order' if ac is 'id' else 'id'
        else:
            try:
                i = int(k)
            except:
                i += 1
            finally:
                i = (i-1)%(len(dseq)-1) +1

def load_test_ds(name='simple'):
    """ return tracking dataset for testing purpose 
    
    name can be 'simple' or 'AR570'
    """
    from rhizoscan.root.pipeline.dataset import make_dataset
    
    if name=='AR570':
        ds,inv,out = make_dataset('/Users/diener/root_data/nacry/AR570/pando/pando_rsa.ini')
        cds = ds.group_by(['genotype.name','nitrate','plate'],key_base='metadata')
        
    elif name=='simple':
        ds,inv,out = make_dataset('/Users/diener/root_data/test/tracking/simple/simple.ini')
        ds.ksort()
        cds = [ds]
    elif name=='superposition':
        ds,inv,out = make_dataset('/Users/diener/root_data/test/tracking/superposition/superposition.ini')
        ds.ksort()
        cds = [ds]
        
    return cds
    
def test_axe_projection(sds, start=0):
    """ call axe_projection on dataset items in ds - for dev purpose - """
    from rhizoscan.root.graph.to_tree import make_tree
    from rhizoscan.root.graph.to_tree import graph_to_dag
    from rhizoscan.root.graph.to_tree import set_downward_segment
    from rhizoscan.root.tracking.growth import simple_axe_growth
    from rhizoscan.root.tracking.growth import simple_tip_growth
    
    d1 = sds[start]
    d2 = sds[start+1]
    d1.load()
    d2.load()
    g1 = d1.graph
    g2 = d2.graph
    t1 = make_tree(g1)
    
    g2 = set_downward_segment(g2)  ## for debug visualisation
    from rhizoscan import geometry as geo
    T = geo.dot(geo.inv(d1.image_transform), d2.image_transform)
    t2 = axe_projection(t1, g2, T)
    
    dag2, sdir = graph_to_dag(t2.segment, t2.axe) 
    axe2,daxe = simple_axe_growth(dag2, t2.axe) # update t2.axe in place
    mask = set()
    map(mask.update, axe2.segment)
    tip_axes  = simple_tip_growth(dag2, mask=mask)
    
    return g1,t1,g2,t2, sdir, daxe, tip_axes, dag2
    

def mtest_axe_projection(cds, group=0, start=0, max_shift=2):
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
        
    
    def uncover(t, tip_axe, sdir):
        """ return a array of segment uncovered by any axes:
            0:covered, 1:unreachable, 2:reachable-downward, 3:reachable-upward 
        """
        sc = 2-unreachable(t)
        sc[sc==2] += sdir[sc==2]
        
        for slist in t.axe.segment:
            sc[slist] = 0
        for slist in tip_axe:
            sc[slist] = 0
        
        return sc
    
    corder=1
    while isinstance(group,int):
        g1,t1,g2,t2, sdir, daxes, taxes, dag = test_axe_projection(cds[group], start)
        T1 = cds[group][start].image_transform
        T2 = cds[group][start+1].image_transform
        
        # draw t1
        ax = plt.subplot(2,1,1)
        ax.set_position([0,.51,1,.49])
        sc = unreachable(t1)&(t1.segment.node!=0).all(axis=1)
        
        bg='w'
        if any(sc):
            t1.plot(sc=sc, indices=sc&(t1.segment.node!=0).all(axis=1),linestyle=':', linewidth=2, transform=T1)
            bg=None
        t1.plot(bg=bg,max_shift=max_shift, corder=corder, linewidth=2, transform=T1)
        
        # draw t2
        ax = plt.subplot(2,1,2, sharex=ax, sharey=ax)
        ax.set_position([0,0,1,.49])
        
        ##sc = _np.zeros(t2.segment.number(),dtype=int)
        ##for i,daxe in enumerate(daxes):
        ##    sc[daxe] = i
        ##t2.plot(sc=7*(sc>0), linewidth=3, transform=T2)
        
        sc = uncover(t2, taxes, sdir)
        sc_mask = (sc>0)&(t2.segment.node!=0).all(axis=1)
        bg='w'
        if any(sc_mask):
            t2.plot(sc=sc, indices=(sc>0)&(t2.segment.node!=0).all(axis=1),linestyle=':', linewidth=2, transform=T2)
            bg=None
        t2.plot(bg=bg,max_shift=max_shift, corder=corder, linewidth=2, transform=T2)
        
        # draw new tip
        taxes = [] + taxes
        tmp = _np.zeros(len(taxes))
        taxe = AxeList(taxes, segment_list=t2.segment,
                       parent=tmp, parent_segment=tmp,
                       plant=tmp, order=tmp)
        t2.axe = taxe
        t2.plot(bg=None,max_shift=max_shift, linewidth=1, transform=T2)
        
        
        g = raw_input('group:%d,start:%d=>' % (group,start))
        if g=='q':
            group = g
        elif g.startswith('+'):
            try:
                start = int(g[1:])
            except:
                print "  Error: unrecognized start '"+g[1:]+"'"
                start = 0
        elif g.startswith('s'):
            try:
                max_shift = int(g[1:])
            except:
                print "  Error: unrecognized shift '"+g[1:]+"'"
        elif g.startswith('c'):
            try:
                corder = int(g[1:])
            except:
                print "  Error: unrecognized corder '"+g[1:]+"'"
        else:
            try:
                group = int(g)
            except:
                group += 1
            start = 0


