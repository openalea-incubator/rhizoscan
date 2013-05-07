from rhizoscan.workflow import Data
from openalea.mtg import algo, traversal
from openalea.plantgl.all import Vector3 as V3, norm

def test1():
    """ load a pickled RootAxialTree object and convert it to mtg """
    tree = Data.load('data/rootaxialtree.data')
    
    g = tree.to_mtg()
    
    # orders
    orders = algo.orders(g)
    
    max_order = max(orders.itervalues())
    assert max_order == 4
    
    max_scale = g.max_scale()
    assert max_scale == 2
    nb_plants = len(g.component_roots_at_scale(g.root, scale=max_scale))
    
    nb_axes = sum(1 for v in g.vertices_iter(scale=max_scale) if g.edge_type(v) == '+')+nb_plants
    assert nb_axes == 190
    
    def length(n):
        if not n.parent(): return 0.
        pos1, pos2 = n.parent().position, n.position
        return norm(V3(pos2)-V3(pos1))
    lengthes = {v:length(g.node(v)) for v in g.vertices(scale=max_scale)}
    g.properties()['length'] = lengthes
    
    assert 26067 <= sum(lengthes.itervalues()) <= 26067.1
    assert 31 <= g.node(6).length <= 32  # assert no change in structure
