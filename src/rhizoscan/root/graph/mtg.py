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

@_node('rsa')
def tree_to_mtg(tree):
    """ create a mtg from given RootTree `tree` """
    # - parse axe in partial order
    # - keep mapping (plant/axe/node id in tree) to (vertex id in mtg)
    #   for parent and complex look up
   
    g = root_mtg()
    
    # mapping tree-to-mtg ids
    mtg_pid = {}  # plant id in mtg - keys are tree plant id  
    mtg_aid = {}  # axe   id in mtg - keys are tree axe   id  
    mtg_nid = {}  # node  id in mtg - keys are (axe,nodes id)
    
    
    # add plant
    # ---------
    for pid in set(tree.axe.plant).difference([0]):
        mtg_id = add_plant(g, plant_id=pid)
        mtg_pid[pid] = mtg_id
        
    # for all axes (in their partial order),
    #   - add the axe to mtg
    #   - add the axe segments
    axes_order = tree.axe.order()
    axes_nodes = tree.axe.get_node_list()[0]
    tree_pos = tree.node.position
    for axe_id in tree.axe.partial_order():
        # add axes
        # --------
        parent_axe = tree.axe.parent[axe_id]
        axe_order  = axes_order[axe_id]
        if parent_axe==0:
            mtg_plant = mtg_pid[tree.axe.plant[axe_id]]
            mtg_axe = add_axe(g, plant=mtg_plant, order=int(axe_order))
            mtg_aid[axe_id] = mtg_axe
        else:
            mtg_axe = add_axe(g, parent=mtg_aid[parent_axe], order=int(axe_order))
            mtg_aid[axe_id] = mtg_axe
            
        # add segments
        # ------------
        axe_nodes = axes_nodes[axe_id]
        node_0 = axe_nodes[0]
        if parent_axe==0:
            # add 1st axe segment
            pos = tuple(tree_pos[:,node_0])+(0,)
            vid = add_segment(g, axe=mtg_axe, position=pos)
            mtg_nid[(axe_id,node_0)] = vid
            mtg_axe = None # next node is a successor
        else:
            # id of mtg parent vertex
            vid = mtg_nid[(parent_axe,node_0)]
            
        # add all successor descendants
        seg_nid = axe_nodes[1:]
        seg_pos = tree_pos[:,seg_nid]
        for nid,(x,y) in zip(seg_nid, seg_pos.T):
            vid = add_segment(g, parent=vid, axe=mtg_axe, position=(x,y,0)) 
            mtg_nid[(axe_id,nid)] = vid
            mtg_axe = None # next node is a successor
            
    g.__serializer__ = '.rsml'
    return g



class RSMLSerializer(object):
    """ Class to serialize/deserialize root mtg into rsml file """
    extension = '.rsml'
    def __init__(self):
        pass
    
    def dump(self, mtg, stream):
        from rsml import io
        from rsml.continuous import discrete_to_continuous
        cmtg = discrete_to_continuous(mtg.copy())
        io.mtg2rsml(cmtg, stream)
    
    def load(self,stream):
        from rsml import io
        from rsml.continuous import continuous_to_discrete
        rsml = io.rsml2mtg(stream)
        return continuous_to_discrete(rsml)


# API for root-mtg
# ----------------
##todo: root-mtg doc
def root_mtg():
    """ create an empty mtg structure to stores root system """
    g = _MTG()
    prop = g.graph_properties()
    prop['type'] = 'RootMTG'
    prop['scales'] = ['Plant','Axe','Segment']
    
    return g

def add_plant(g, **properties):
    """ Add a plant vertex to (root)mtg `g` """
    return g.add_component(g.root, label='P', edge_type='/', **properties)
    
def add_axe(g, parent=None, plant=None, **properties):
    """ 
    Add a root axe to root mtg `g`
    
    Either `parent` or `plant` should be given:
     - parent for secondary axes, the same plant as parent is selected
     - plant for primary axes
    """
    if parent:
        return g.add_child(parent, label='A', edge_type='+', **properties)
    else:
        return g.add_component(plant, label='A', edge_type='/', **properties)
        
def add_segment(g, parent=None, axe=None, **properties):
    """
    Add a root segment to root mtg `g`
    
    Either `parent`, `axe` or both should be given:
     - axe only for the 1st segment of an axe
     - parent only for successor segment in same axe as parent
     - both for branching axes, the `parent` axe should be the `axe` parent
    """
    if parent is not None:
        # successor, same axe
        if axe is None:
            seg_id = g.add_child(parent, label='S', edge_type='<', **properties)
            
        # branch
        else:
            seg_id = g.add_child(parent, label='S', edge_type='+', **properties)
            g.add_component(axe, component_id=seg_id)
            
    else:
        # 1st axe segment
        seg_id = g.add_component(axe, label='S', edge_type='/', **properties)
        
    return seg_id


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
