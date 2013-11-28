

def test_node_decoration_and_call():
    from rhizoscan.workflow import node
    
    @node('a','b')
    def f(n1,n2): return n2,n1+n2

    # assert f is still the correct function
    assert f(2,3)==(3,5)
    
    # assert node.run behavior
    d = node.run(f, n1=2,n2=3)
    assert d.has_key('a') and d.has_key('b')
    assert d['a']==3 and d['b']==5
    
def test_pipeline_creation_and_call():
    from rhizoscan.workflow import node, pipeline
    
    @node('c','d')
    def f(a): return a+2, a+3
    @node('e')
    def g(b,c,d): return b+c+d
    
    @pipeline(nodes=[f,g])
    def h(): pass
    
    # assert name of pipeline input are correct
    assert [i['name'] for i in node.get_attribute(h, 'inputs')]==['a','b']
    
    # assert restricted function computation ability 
    n = h._nodes_to_compute('missing',False,dict(a=1, c=2,d=3))
    assert len(n)==1   # only g to be computed
    assert node.get_attribute(n[0], 'name')=='g'
    
    # assert Pipeline.run call
    ns = h.run(a=1,b=2)
    assert ns.has_key('e')
    assert ns['e']==9
    
    ## unsuitable behavior when called as a node: needs to define __node__.outputs
    #assert node.run(h,...) 

