
def test_mapping_io():
    from rhizoscan.datastructure import Mapping
    from tempfile import mkstemp
    
    fid, fname = mkstemp()
    
    m = Mapping(a=1,b=2,c=3)
    m.set_file(fname)
    m.__loader_attributes__ = ['a']
    
    loader = m.dump()
    assert 'a'     in loader.__dict__.keys()
    assert 'b' not in loader.__dict__.keys()
    
    n = loader.load()
    assert all(map(hasattr,[n]*3,['a','b','c'])) # all keys reloaded
    assert n.a==m.a and n.b==m.b and n.c==m.c
    
    #todo: check private attributes are suitable
    
    import os
    os.remove(fname)
