"""
dev on tracking
"""
#from rhizoscan.root.graph.conversion import segment_to_los as _seg2los

import numpy as _np

from .projection import node_to_axe_distance
from .projection import segment_to_projection_area


# track tree
# ----------

def test_s2a(ds, group=2, start=0):
    from rhizoscan.root.graph.to_tree import make_tree
    from rhizoscan.root.graph.to_tree import set_downward_segment
    from rhizoscan import geometry as geo
    
    d1 = ds[group][start]
    d2 = ds[group][start+1]
    d1.load()
    d2.load()
    g1 = d1.graph
    g2 = d2.graph
    t1 = make_tree(g1)

    T = geo.dot(geo.inv(d1.image_transform), d2.image_transform)

    # set graph into the same frame as tree
    # -------------------------------------
    g2 = g2.copy()
    g2.node = g2.node.copy()
    g2.node.position = geo.transform(T=T, coordinates=g2.node.position)
    g2.segment = g2.segment.copy()    # copy with transformed _node_list
    g2.segment._node_list = g2.node   #    maybe not useful...


    return (t1,g2,T) + seg_to_axe_distance(t1,g2)
    
    g2 = set_downward_segment(g2)  ## for debug visualisation
    t2 = axe_projection(t1, g2, T)
    
def seg_to_axe_distance(t,g):
    ### get the list of segment-to-segment connections
    ##g_sedges = _seg2los(g.segment.node,g.node.segment,mask=g.segment.seed)
    ##g_sedges = [sedge[0].union(sedge[1]) for sedge in g_sedges]

    # compute distances from segment in g to axe in t
    dn,na,np = node_to_axe_distance(g.node.position, t) # d(gn,ta),a(gn),p(gn/a)
    
    # mean segment-to-axe distance 
    ds = dn[g.segment.node]#.mean(axis=1)                # d(gs,ta)
    
    g2t_area = segment_to_projection_area(g, np)       # shape (|gs|,|ta|)
    g2t_area += g2t_area[1:].min()/g.segment.number()  # assert strict >0

    return dn,ds, g2t_area
    

def plot_criteria(g,ds,epsilon,sigma, show=0, t=None):
    # definitions
    Es = ds.min(axis=1)       # min d(s,a) for all a
    a_ = ds.argmin(axis=1)    # best axe for assignment 
    As = ds==Es[:,None]       # axis which are at min dist Es as bool array
    Es2 = (ds+_np.inf*Es[:,None]).min(axis=1)
    
    # criteria of assigment 
    I1 = As.sum(axis=1)==1    # unicity
    I2 = Es<=epsilon          # proximity
    I3 = (Es/Es2)<sigma       # separability
    
    I = (I1 & I2 & I3)*a_     # 0:unrespected criteria, 1+: best axes
    
    # criteria of neighborhood
    #   instead of computing each criteria, label suitable connected segments
    nbor = g.segment.neighbors().copy()    # II1: neighborhood
    mask = (nbor>0).sum(axis=1)!=1         # II2: linearity (of neighborhood)
    m2 = a_[nbor] # II3...
    for nb in range(nbor.shape[1]):        # remove nbor unrespectful of II*
        nbor[:,nb][mask] = 0
    
    from rhizoscan.root.graph.conversion import neighbor_to_csgraph
    from scipy.sparse.csgraph import connected_components
    csg = neighbor_to_csgraph(nbor)
    n,II = connected_components(csg)
    
    return I,II, m2, a_
    
    # display
    # -------
    bg = 'w'
    if t is not None:
        from rhizoscan.geometry import translation
        t.plot(bg=bg,ac='g',transform=translation((5,0)))
        bg = None
    
    if show==0:
        show = eps_max[:,0]
    elif show==1:
        show = eps_max[:,1]
    else:
        show = eps_max[:,0]/eps_max[:,0]        
                                       
    #sc = _np.minimum(show,emax)*(g.segment.reachable()+0)
    sc = (show<emax)*(g.segment.reachable()+0)
    g.plot(bg=bg,sc=sc,indices=g.segment.reachable())
    from matplotlib import pyplot as plt
    if not disp_t and Tgt is not None:
        plt.ylim(plt.ylim()[::-1])
    #plt.ylim(plt.ylim()[::-1]);
    
    
# improve image tracking
# ----------------------
def node_los_graph(g):
    """ neighbor nodes for all nodes as a list of set """
    nlos = [set(g.segment.node[slist].ravel().tolist()) for slist in g.node.segment]
    for i,nb in enumerate(nlos):
        nb.discard(i)                       
    return nlos

def node_csgraph(g):
    from rhizoscan.root.graph import conversion as cvs
    nlos  = node_los_graph(g)
    nnbor = cvs.los_to_neighbor(nlos,sided=0)
    return cvs.neighbor_to_csgraph(nnbor)
    
def crossing_nodes(g):
    smask = (-g.segment.reachable())#|(g.segment.seed>0)
    smask = smask.nonzero()[0]
    ns = map(set,g.node.segment)
    ns = [slist.difference(smask) for slist in ns]
    return _np.vectorize(len)(ns)>2
    
def linear_segments(g):
    """ segment labeled by linear sequence (connected through 1-1 node) """
    # find node with only 2 segments
    nlin = _np.vectorize(len)(g.node.segment)==2
    nlin = _np.array(g.node.segment[nlin].tolist())
    
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    lin_graph = csr_matrix((_np.ones(nlin.shape[0]),(nlin[:,0],nlin[:,1])), 
                            shape=(g.segment.number(),)*2)
    return connected_components(lin_graph, directed=False)
    
def lin_segment_length(g,label=None):
    if label is None:
        n,label = linear_segments(g)
    else:
        n = label.max()+1
    
    from scipy import ndimage as nd
    lab_len = nd.sum(g.segment.reachable()*g.segment.length(),
                     labels=label,
                     index=_np.arange(n))
    
    return lab_len[label], label
    
def graph_feature_node(g):
    seg_lin_len,seg_lin_lab = lin_segment_length(g)
    cnode = crossing_nodes(g)
    
    kp_pos = g.node.position[:,cnode].T  # position of cross nodes
    
    seg_dir = g.segment.direction()
    bin = _np.linspace(-_np.pi,_np.pi,9) # edges of [-pi,pi] in 8 bins
    kp_desc = []
    for slist in g.node.segment[cnode]:
        kp_desc.append(_np.histogram(seg_dir[slist],bins=bin,weights=seg_lin_len[slist])[0])
        
    kp_desc = _np.array([d.tolist() for d in kp_desc],dtype='float32')
            
    return kp_pos, kp_desc
