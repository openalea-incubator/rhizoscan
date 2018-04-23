"""
Test import reference RSA from neuronJ file
"""

def test_load_ndf():
    import os
    from rhizoscan.root.neuronJ import NJ_loader
    
    filename = os.path.abspath('test/data/chl15_1_1.ndf')
    
    nj_tree = NJ_loader(filename, tree=False)
    nj_tree.to_tree()
    tree = nj_tree.make_axial_tree()
    
    assert hasattr(tree.node, 'segment'), 'node list has not segment attribute set'
    
    return tree
