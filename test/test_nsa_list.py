"""
Test for NodeLise, SegmentList and AxeList
"""

def test_constructors():
    import numpy as np
    from rhizoscan.root.graph import nsa
    from rhizoscan.root.graph import RootGraph, RootTree
    
    n = nsa.NodeList(np.array([[0,1,1,1,1,2,0],[0,0,1,2,3,1.5,2.5]]))
    s = nsa.SegmentList(np.array([[0,1,2,3,2,3],[0,2,3,4,5,6]]).T, n)
    s.seed = np.array([0,1,0,0])
    n.set_segment(s)
    
    assert n.number()==7, 'incorrect number of nodes'
    assert s.number()==6, 'incorrect number of segments'
    assert n.terminal().sum()==4, 'incorrect number of termnial nodes'+str(n.terminal())
    assert s.terminal().sum()==4, 'incorrect number of termnial segments'+str(s.terminal())
    
    
    axes = [[],[1,2,3],[4],[5]]
    segment_parent = [0,0,1,2,1,2]
    seg_parent = nsa.parent_segment(axes,segment_parent)
    axe_parent = nsa.parent_axe(axes,seg_parent)

    assert (seg_parent==[0,0,1,2]).all(), 'incorrect parent segment of axes'
    assert (axe_parent==[0,0,1,1]).all(), 'incorrect parent axes'
    
    a = nsa.AxeList(axes, segment_list=s,
                    parent=axe_parent, parent_segment=seg_parent,
                    plant=[0,1,1,1],
                    order=[0,1,2,2])
        
    assert a.order().max()==2, 'incorrect axe max order'
    assert (a.position_on_parent()==[0,0,1,2]).all()
    assert (a.partial_order()==[0,1,2,3]).all()
        
    g = RootGraph(node=n,segment=s)
    t = RootTree(node=n,segment=s, axe=a)
    
    return t