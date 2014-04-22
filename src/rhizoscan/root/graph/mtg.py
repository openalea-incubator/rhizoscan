"""
MTG to represent Root System Architecture
"""
import numpy as _np
from scipy import ndimage as _nd

try:
    from openalea.mtg import MTG as _MTG
except ImportError:
    raise ImportWarning("Could not import openalea.mtg")

from rhizoscan.workflow import node as _node # to declare workflow nodes

@_node('mtg')
def tree_to_mtg(tree):
    """ create a mtg from this axial tree """
    
    # initialization
    # ==============
    # node sequence of each axe, node coordinates, ...
    node_list = tree.axe.get_node_list()[0]
    node_pos  = [(xi,yi,0) for xi,yi in tree.node.position.T]
    max_order = tree.axe.order().max()
    
    # compute seed position
    seed_mask  = tree.segment.seed>0
    seed_label = tree.segment.seed[seed_mask]
    seed_id    = _np.unique(seed_label)
    
    seed_pos   = tree.node.position[:,tree.segment.node[seed_mask]]
    seed_pos   = seed_pos.reshape(seed_pos.shape[0],-1)
    seed_label = (seed_label[:,None]*_np.ones((1,2),dtype=int)).ravel()
    x = _nd.mean(seed_pos[0], labels=seed_label, index=seed_id)
    y = _nd.mean(seed_pos[1], labels=seed_label, index=seed_id)
    seed_pos = _np.vstack((x,y,_np.zeros(x.size))).T
    seed_pos = dict(zip(seed_id,seed_pos))

    # create mtg tree
    # ===============
    #   parse axe (adding nodes) in min-order, then max-length order
    #   1st time a node is parsed, its current axe is set as its main
    #   hypothesis: 1st axe node (branching node) "main axe" is the parent axe
    g = _MTG()
    
    mtg_pid = {}  # plant id in mtg - keys are tree plant id  
    mtg_nid = {}  # node  id in mtg - keys are tree nodes id -> set the 1st processed
    
    
    # add plant
    # ---------
    for pid in seed_id:
        properties = dict(plant=pid)
        
        # add the plant: scale 1
        mtg_id = g.add_component(g.root, plant=pid, label='P%d'%pid)
        ##mtg_pid[pid] = mtg_id
        
        # add an axe (scale 2) 
        v = g.add_component(mtg_id, label='G', order=0)
        
        # add a node in its center (scale 3)
        n = g.add_component(v, position=seed_pos[pid], label='g', **properties)
        mtg_pid[pid] = n
        
    # To select the parent axe: 
    #   - axe parsing follows axe asc. order, then length desc. order
    #   - the 1st time a node is added, the current axe is set as its 'main' 
    #     (stored mtg_nid)
    #   - when a new axe is processed, its first node is set as its parent:
    #      - if its first node has already been parsed: it is selected as
    #        its parent node, which induces its parent axe
    #      - otherwise, choose the seed node as its parent node
    for order in range(1,max_order+1):
        axe_mask  = tree.axe.order()==order
        axe_order = _np.argsort(tree.axe.length()[axe_mask])[::-1]
        
        # add axe
        # -------
        g_order = g.property('order')
        for aid in axe_mask.nonzero()[0][axe_order]:
            # find parent node id in mtg
            nlist  = node_list[aid]
            if len(nlist)==0: continue
            
            node_0 = nlist[0]
            properties = dict(order=order, plant=pid, axe_id=aid, radius=1)
            
            if not mtg_nid.has_key(node_0):  #order==1 and axes connected to seed 
                parent_node = mtg_pid[tree.axe.plant[aid]]
                parent_node,cur_axe = g.add_child_and_complex(parent_node, 
                    position=node_pos[nlist[0]], 
                    edge_type='+',
                    **properties)
                g.node(parent_node).label='S'
                g.node(cur_axe).label='A'
                g_order[cur_axe] = order
                mtg_nid.setdefault(nlist[0],parent_node)
                
            else:
                parent_node = mtg_nid[node_0]
                # add 1st and current axe
                position=node_pos[nlist[1]]
                parent_node,cur_axe = g.add_child_and_complex(parent_node, 
                        position=position,
                        edge_type='+', **properties)
                g.node(parent_node).label='S'
                g.node(cur_axe).label='A'
                g_order[cur_axe] = order
                mtg_nid.setdefault(nlist[1],parent_node)
                nlist = nlist[1:]
                
            # add nodes
            # ---------
            properties.pop('order')
            for node in nlist[1:]:
                position=node_pos[node]
                parent_node = g.add_child(parent_node, 
                       position=position, 
                       x=position[0], 
                       y=-position[2], 
                       edge_type='<', 
                       label='S', 
                       **properties)
                mtg_nid.setdefault(node,parent_node)
                _p = g.parent(parent_node)
                if _p is None: print '**** parent None %d ****' % parent_node
        
    # add edge_type to axe vertex: '+' for all axe but 1st that have no parent
    ## second (or more) order axes which starts at seeds are not topologically second order !!!
    edge_type = g.property('edge_type')
    for v in g.vertices(scale=2):
        if g.parent(v) is not None:
            #g[v]['edge_type'] = '+' # don't work
            edge_type[v] = '+'

    return g
    

## tests
def test_to_mtg(t):
    g = tree_to_mtg(t)
    
    # test if plant property is correct
    # ---------------------------------
    # split mtg by plants
    g = map(g.sub_mtg, g.vertices(scale=1))
    
    pl_axe = [list(set(gi.property('axe').values())) for gi in g]
    pl_axe = [[a for a in alist if a>0] for alist in pl_axe]      ## remove negativ axe...
    
    ac = np.zeros(ref.axe.number(),dtype=int)
    for i,al in enumerate(pl_axe): ac[al] = i+1
    
    assert (t.axe.plant==ac).all()
