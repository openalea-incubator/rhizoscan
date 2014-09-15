import os

image_file = 'test/data/pipeline/arabido.png'
project_file = 'test/data/pipeline/arabidopsis/database.ini'

def arabidopsis_pipeline(output=None):
    from rhizoscan.root.pipeline.arabidopsis import pipeline
    from rhizoscan.datastructure import Mapping
    
    filename = os.path.abspath(image_file)
    assert os.path.exists(filename), "could not find test image file:"+filename 
    
    d = Mapping(filename=filename, plant_number=2,
                fg_smooth=1, border_width=.08,leaf_bbox=[0,0,1,.4],root_max_radius=5, verbose=1)
    
    if output:
        d.set_storage(output)
        pipeline.run(namespace=d, store=['pmask','rmask','seed_map','tree','rsa'])
    else:
        pipeline.run(namespace=d)
        
    assert d.has_key('image')   , "pipeline did not compute 'image'" 
    assert d.has_key('pmask')   , "pipeline did not compute 'pmask'"
    assert d.has_key('rmask')   , "pipeline did not compute 'rmask'"
    assert d.has_key('seed_map'), "pipeline did not compute 'seed_map'"
    assert d.has_key('graph')   , "pipeline did not compute 'graph'"
    assert d.has_key('tree')    , "pipeline did not compute 'tree'"
    assert d.has_key('rsa')     , "pipeline did not compute 'rsa'"
    
    # test tree
    import numpy as np
    t = d.tree
    assert t.axe.number()==8, "not the correct number of axes:"+str(t.axe.number())
    # problem: there is an axe with only a seed segment ??!
    assert (np.unique(t.axe.plant)==[0,1,2]).all(), "not the correct number of plants"+str(np.unique(t.axe.plant))
    
    pos_on_parent = t.axe.position_on_parent()
    assert abs(np.sort(pos_on_parent[:8])-[0,0,0,35,36,76,95,129]).max()<2, 'incorrect axe position_on_parent'

    # test mtg
    g = d.rsa
    plant_number = len(g.vertices(scale=1))
    axe_number = len(g.vertices(scale=2))
    assert plant_number==2, "not the correction number of plants in mtg"+str(plant_number)
    assert axe_number==7,   "not the correction number of axes in mtg"+str(axe_number)
    
    if output:
        # test rsml serialization
        from rhizoscan.root.graph.mtg import RSMLSerializer
        with d.rsa.__file_object__.entry.open('r') as f:
            t = RSMLSerializer().load(f)
        assert len(g.vertices())==len(t.vertices())
        
        # test file extension
        def ext(attr):
            return d[attr].__file_object__.get_extension()
        assert ext('pmask')   =='.png',    "stored pmask has not the '.png' extension"
        assert ext('rmask')   =='.png',    "stored rmask has not the '.png' extension"
        assert ext('seed_map')=='.png',    "stored seed_map has not the '.png' extension"
        assert ext('tree')    =='.pickle', "stored tree has not the '.pickle' extension"
        assert ext('rsa')     =='.rsml',   "stored rsa has not the '.rsml' extension"

    return d

def test_arabido_pipeline_no_storage():
    arabidopsis_pipeline()
    
def test_arabido_pipeline_with_storage():
    import os
    from tempfile import mkdtemp
    tmp = mkdtemp()
    try:
        arabidopsis_pipeline(tmp)
    finally:
        os.rmdir(tmp)

def test_load_dataset():
    from rhizoscan.misc.path import abspath
    from rhizoscan.root.pipeline.dataset import make_dataset
    
    pfile = os.path.abspath(project_file)
    pdir  = os.path.dirname(pfile)
    
    assert os.path.exists(pfile), "could not find test project file:"+pfile 
    
    def test_loaded_dataset(ds,invalid,out,exp_out):
        assert len(ds)==4, "invalid number of image file in test project"
        assert len(invalid)==1, "invalid number of invalid file in test project"
        assert hasattr(ds[0], 'filename'), "dataset item has not 'filename' attribute"
        assert hasattr(ds[0], 'metadata'), "dataset item has not 'metadata' attribute"
        assert hasattr(ds[0], '__storage__'), "dataset item has not external storage set"
        
        assert out==exp_out

    ds, invalid, output = make_dataset(ini_file=pfile, out_dir='test_out', verbose=0)
    test_loaded_dataset(ds,invalid,output, os.path.join(pdir,'test_out'))

    ds, invalid, output = make_dataset(ini_file=pfile, verbose=0)
    test_loaded_dataset(ds,invalid,output, os.path.join(pdir,'outputs'))
        
    return ds
    


