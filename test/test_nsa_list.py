"""
Test for NodeLise, SegmentList and AxeList
"""

def test_constructors():
    import numpy as np
    from rhizoscan.root.graph import nsa
    from rhizoscan.root.graph import RootGraph
    
    n = nsa.NodeList(np.random.rand(2,100))
    s = nsa.SegmentList(np.random.randint(0,100,(50,2)), n)
    n.set_segment(s)
    
    # call property
    n.terminal 
    s.terminal # call property that compute termnial segments
    assert n.terminal.size==n.number
    assert s.terminal.size==s.number
    assert s.length.size==s.number
    
    
    g = RootGraph(node=n,segment=s)
    
    return g