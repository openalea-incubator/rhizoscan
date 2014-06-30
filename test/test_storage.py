
def test_write_delete_tempfile():
    from tempfile import mkstemp
    fid, fname = mkstemp()
    import os
    os.close(fid)
    os.remove(fname)    

def test_mapping_read():
    import os
    from rhizoscan.datastructure import Mapping
    
    filename = os.path.abspath('test/data/zen.map')
    m = Mapping()
    m.load(filename)
    
    assert m.has_key('zen'), "missing attribute 'zen'"
    assert len(m.zen)==836
    

def test_mapping_io():
    from rhizoscan.datastructure import Mapping
    from tempfile import mkdtemp
    import os 
    
    dname = mkdtemp()
    fname = os.path.join(dname, 'test_mapping.save')
    
    try:
        m = Mapping(a=1,b=2)
        m.set_file(fname)
        m.set('c',Mapping(c=3),store=True)
        url = m.get_file().get_url()
        ctn = m.get_file().get_container()
        assert url==fname, 'not the correct file:'+repr(url)+'!='+fname
        assert ctn==dname, 'not the correct directory:'+repr(ctn)+'!='+dname
        
        m.__loader_attributes__ = ['a']
        
        loader = m.dump()
        
        url = m.get_file().get_url()
        ctn = m.get_file().get_container()
        assert url==fname, 'not the correct file:'+repr(url)+'!='+fname
        assert ctn==dname, 'not the correct directory:'+repr(ctn)+'!='+dname
        assert m.get_file().exists(), 'storage file does not exist:'+repr(m.get_file())
        
        assert 'a'     in loader.__dict__.keys()
        assert 'b' not in loader.__dict__.keys()
        assert 'c' not in loader.__dict__.keys()
        
        n = loader.load()
        assert all(map(hasattr,[n]*3,['a','b','c'])) # all keys reloaded
        assert n.a==m.a and n.b==m.b and n.get('c').c==m.c.c
        
        #todo: check private attributes are suitable
    finally:
        import shutil
        shutil.rmtree(dname)
        
def test_mapping_move():
    from rhizoscan.datastructure import Data, Mapping
    from tempfile import mkdtemp
    import os 
    import shutil
    
    dname  = mkdtemp()
    dname1 = os.path.join(dname,'dir1')
    dname2 = os.path.join(dname,'dir2')
    fname1 = os.path.join(dname1, 'test_mapping.save')
    fname2 = os.path.join(dname2, 'test_mapping.save')

    try:
        m1 = Mapping(a=1,b=2)
        m1.set_file(fname1)
        m1.set('c',Mapping(c=3),store=True)
        
        loader = m1.dump()
        shutil.copytree(dname1,dname2)
        shutil.rmtree(dname1)
        
        m2 = Data.load(fname2)
        
        assert m2.get_file().get_url()==fname2
        assert m2.get_file().get_container()==dname2
        
        assert all(map(hasattr,[m2]*3,['a','b','c'])) # all keys reloaded
        assert m1.a==m2.a and m2.get('c').c==m1.c.c
        
    finally:
        shutil.rmtree(dname)

def test_windows_path():
    from rhizoscan.storage import FileObject
    from rhizoscan.storage import FileEntry
    
    filename = 'c:\tmp\yo.txt'
    f = FileObject(filename)
    assert isinstance(f.entry, FileEntry)
    
    filename = 'c:/tmp/yo.txt'
    f = FileObject(filename)
    assert isinstance(f.entry, FileEntry)


